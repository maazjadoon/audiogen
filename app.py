"""
OmniVoice Studio — FastAPI application entry-point.

Architecture
────────────
  POST /api/v2/generate          — sync (wait for result)
  POST /api/v2/generate/async    — submit job, returns job_id immediately
  GET  /api/v2/jobs/{job_id}     — poll job status
  GET  /api/v2/jobs/batch        — batch status check (multiple job IDs)
  GET  /api/v2/history           — paginated past jobs
  GET  /api/v2/health            — model + system health
  GET  /metrics                  — Prometheus scrape endpoint
  GET  /outputs/{filename}       — serve generated audio files
  GET  /                         — serve the HTML frontend

Features
────────
  • Graceful shutdown with SIGTERM handling
  • Request queuing with semaphore-based concurrency
  • Rate limiting (per-IP)
  • API key authentication (optional)
  • Streaming TTS via Server-Sent Events
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import signal
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import aiofiles
from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Form, HTTPException,
    Request, Response, UploadFile, status, Header,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from database import GenerationJob, get_db, init_db
from generation_service import create_job, generate_audio
from karaoke_service import create_kvc_job, process_karaoke
from logger import get_logger, setup_logging
from metrics import get_system_snapshot, prometheus_output
from model_manager import model_status, unload_model
from schemas import (
    BatchGenerationRequest, GenerationRequest, GenerationResponse,
    HealthResponse, HistoryResponse, JobStatusResponse, VoiceBlendRequest,
)

# ── Bootstrap ──────────────────────────────────────────────────────────────────
cfg = get_settings()
setup_logging(
    debug=cfg.debug,
    log_file=cfg.log_file,
    log_max_bytes=cfg.log_max_bytes,
    log_backup_count=cfg.log_backup_count,
)
log = get_logger(__name__)

OUTPUT_DIR = cfg.output_dir

# ── Graceful shutdown state ────────────────────────────────────────────────────
_shutdown_event = asyncio.Event()
_active_requests: set[str] = set()
_request_lock = asyncio.Lock()

# ── Rate limiting state ─────────────────────────────────────────────────────────
_rate_limit_store: dict[str, list[float]] = {}
_rate_limit_lock = asyncio.Lock()

# ── Security ───────────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[START] OmniVoice Studio starting up...")
    await init_db()
    log.info(f"[DIR] Output dir: {OUTPUT_DIR.resolve()}")
    log.info(f"[DB] Database: {cfg.db_url}")
    
    # Model warmup (optional)
    if cfg.model_warmup:
        from model_manager import warmup_model
        await warmup_model()
    
    # Setup signal handlers for graceful shutdown
    def _signal_handler(signum, frame):
        log.info(f"[SIGNAL] Received signal {signum}, initiating graceful shutdown...")
        _shutdown_event.set()
    
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    
    yield
    
    # Shutdown: wait for active requests to complete
    log.info("[STOP] Shutting down... waiting for active requests")
    timeout = 30.0
    start = time.time()
    while _active_requests and (time.time() - start) < timeout:
        log.info(f"[STOP] Waiting for {len(_active_requests)} active requests...")
        await asyncio.sleep(0.5)
    
    if _active_requests:
        log.warning(f"[STOP] {len(_active_requests)} requests did not complete in time")
    
    unload_model()
    log.info("[STOP] Shutdown complete")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OmniVoice Studio",
    description="Production-grade AI voice cloning & synthesis API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    # Check if shutdown is in progress
    if _shutdown_event.is_set():
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Server is shutting down"}
        )
    
    req_id = str(uuid.uuid4())[:8]
    request.state.req_id = req_id
    
    async with _request_lock:
        _active_requests.add(req_id)
    
    try:
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        response.headers["X-Request-Id"] = req_id
        response.headers["X-Response-Time-Ms"] = str(elapsed)
        return response
    finally:
        async with _request_lock:
            _active_requests.discard(req_id)


# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════════
async def check_rate_limit(request: Request) -> bool:
    """Return True if request is within rate limit."""
    if cfg.rate_limit <= 0:
        return True
    
    # Get client IP
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    client_ip = client_ip.split(",")[0].strip() if "," in client_ip else client_ip
    
    now = time.time()
    window = 60.0  # 1 minute window
    
    async with _rate_limit_lock:
        if client_ip not in _rate_limit_store:
            _rate_limit_store[client_ip] = []
        
        # Remove old entries outside window
        _rate_limit_store[client_ip] = [
            ts for ts in _rate_limit_store[client_ip] if (now - ts) < window
        ]
        
        # Check limit
        if len(_rate_limit_store[client_ip]) >= cfg.rate_limit:
            return False
        
        # Record this request
        _rate_limit_store[client_ip].append(now)
        return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to API endpoints."""
    if request.url.path.startswith("/api/"):
        if not await check_rate_limit(request):
            return JSONResponse(
                status_code=429,
                content={"status": "error", "message": f"Rate limit exceeded: {cfg.rate_limit} requests/minute"}
            )
    return await call_next(request)


# ═══════════════════════════════════════════════════════════════════════════════
# API AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Verify API key if authentication is enabled."""
    if not cfg.api_key_enabled:
        return True
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include 'Authorization: Bearer <api_key>' header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != cfg.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True


# ── Static files ───────────────────────────────────────────────────────────────
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# ── Root — serve HTML frontend ─────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("AudioGenGUI.html", media_type="text/html")


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/v2/health", response_model=HealthResponse, tags=["System"])
async def health():
    mstat = model_status()
    snap = get_system_snapshot()
    overall = "ok" if mstat.get("loaded") else "degraded"
    return HealthResponse(
        status=overall,
        model=mstat,
        system=snap,
        version="2.0.0",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PROMETHEUS METRICS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/metrics", include_in_schema=False)
async def metrics():
    data, ctype = prometheus_output()
    return Response(content=data, media_type=ctype)


# ══════════════════════════════════════════════════════════════════════════════
# SYNCHRONOUS GENERATION  (waits for result — best for frontend)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v2/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate_sync(
    request: Request,
    # JSON body path
    body: Optional[GenerationRequest] = None,
    # Form/multipart path
    text: Optional[str] = Form(None),
    settings_json: Optional[str] = Form(None, alias="settings"),
    ref_audio: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """
    Submit a TTS generation request and wait for the result.
    Accepts either:
      • `application/json` with a GenerationRequest body
      • `multipart/form-data` with text, settings, and optional ref_audio file
    """
    # ── Parse inputs ────────────────────────────────────────────────────────
    if body is not None:
        # JSON path
        from schemas import GenerationSettings
        gen_text = body.text
        gen_settings = body.settings
        ref_path: Optional[str] = None
    else:
        # Form path
        if not text:
            raise HTTPException(400, "text is required")
        gen_text = text.strip()
        try:
            raw = json.loads(settings_json or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid settings JSON: {exc}")
        from schemas import GenerationSettings
        gen_settings = GenerationSettings(**raw)
        ref_path = None

        if ref_audio and ref_audio.filename:
            ts = int(time.time())
            safe_name = f"ref_{ts}_{ref_audio.filename}"
            orig_ref_path = str(OUTPUT_DIR / safe_name)
            content = await ref_audio.read()
            async with aiofiles.open(orig_ref_path, "wb") as f:
                await f.write(content)
            log.info(f"Saved reference audio → {orig_ref_path}")

            # Convert any format (AAC/MP3/M4A/…) → WAV so soundfile can load it
            def _to_wav(src: str, dst: str) -> str:
                try:
                    import torchaudio
                    wav, sr = torchaudio.load(src)
                    torchaudio.save(dst, wav, sr)
                    return dst
                except Exception:
                    pass
                try:
                    from pydub import AudioSegment
                    AudioSegment.from_file(src).export(dst, format="wav")
                    return dst
                except Exception:
                    pass
                return src  # fallback: use original if conversion fails

            wav_ref_path = str(OUTPUT_DIR / f"ref_{ts}.wav")
            loop2 = asyncio.get_event_loop()
            ref_path = await loop2.run_in_executor(None, _to_wav, orig_ref_path, wav_ref_path)
            log.info(f"Reference audio ready → {ref_path}")

    # ── Create DB record ────────────────────────────────────────────────────
    job_id = await create_job(
        text=gen_text,
        gen_settings=gen_settings,
        ref_audio_path=ref_path if body is None else None,
    )

    # ── Run generation ──────────────────────────────────────────────────────
    result = await generate_audio(
        job_id=job_id,
        text=gen_text,
        gen_settings=gen_settings,
        ref_audio_path=ref_path if body is None else None,
    )

    http_status = 200 if result["status"] == "success" else 500
    return JSONResponse(content=result, status_code=http_status)


# ══════════════════════════════════════════════════════════════════════════════
# ASYNC GENERATION  (fire-and-forget, poll /jobs/{id})
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v2/generate/async", tags=["Generation"])
async def generate_async(
    body: GenerationRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key),
):
    """
    Submit job asynchronously — returns immediately with a job_id.
    Poll GET /api/v2/jobs/{job_id} for status.
    """
    gen_settings = body.settings
    job_id = await create_job(
        text=body.text,
        gen_settings=gen_settings,
    )
    background_tasks.add_task(
        generate_audio,
        job_id=job_id,
        text=body.text,
        gen_settings=gen_settings,
    )
    return {"status": "pending", "job_id": job_id}


# ── BATCH GENERATION ───────────────────────────────────────────────────────────
@app.post("/api/v2/generate/batch", tags=["Generation"])
async def generate_batch(
    body: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key),
):
    """Submit up to 10 generation requests in one call. Returns list of job_ids."""
    job_ids = []
    for item in body.items:
        job_id = await create_job(text=item.text, gen_settings=item.settings)
        background_tasks.add_task(
            generate_audio,
            job_id=job_id,
            text=item.text,
            gen_settings=item.settings,
        )
        job_ids.append(job_id)
    return {"status": "pending", "job_ids": job_ids, "count": len(job_ids)}


# ══════════════════════════════════════════════════════════════════════════════
# JOB STATUS POLLING
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/v2/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    job = await db.get(GenerationJob, job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=str(job.id),
        status=str(job.status),
        audio_url=job.audio_url,
        generation_time=job.generation_time,
        error_message=job.error_message,
        created_at=job.created_at.isoformat() if job.created_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BATCH JOB STATUS
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v2/jobs/batch", tags=["Jobs"])
async def get_batch_jobs(
    job_ids: list[str],
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """Get status for multiple jobs at once."""
    from sqlalchemy import select
    from database import GenerationJob

    result = await db.execute(
        select(GenerationJob).where(GenerationJob.id.in_(job_ids))
    )
    jobs = result.scalars().all()

    job_map = {str(j.id): {
        "job_id": str(j.id),
        "status": j.status,
        "audio_url": j.audio_url,
        "generation_time": j.generation_time,
        "error_message": j.error_message,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "finished_at": j.finished_at.isoformat() if j.finished_at else None,
    } for j in jobs}

    # Include missing job IDs as "not_found"
    for jid in job_ids:
        if jid not in job_map:
            job_map[jid] = {"job_id": jid, "status": "not_found"}

    return {"jobs": list(job_map.values()), "count": len(job_ids)}


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING TTS (Server-Sent Events)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v2/generate/stream", tags=["Generation"])
async def generate_stream(
    request: Request,
    text: str = Form(...),
    settings_json: Optional[str] = Form("{}"),
    ref_audio: Optional[UploadFile] = File(None),
    _: bool = Depends(verify_api_key),
):
    """
    Streaming TTS via Server-Sent Events.
    Yields progress updates during generation, then final audio URL.
    """
    import json as _json
    from schemas import GenerationSettings

    try:
        raw = _json.loads(settings_json or "{}")
    except _json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid settings JSON: {exc}")

    gen_settings = GenerationSettings(**raw)

    # Save reference audio if provided
    ref_path = None
    if ref_audio and ref_audio.filename:
        ts = int(time.time())
        safe_name = f"ref_{ts}_{ref_audio.filename}"
        orig_ref_path = str(OUTPUT_DIR / safe_name)
        content = await ref_audio.read()
        async with aiofiles.open(orig_ref_path, "wb") as f:
            await f.write(content)

        # Convert to WAV
        def _to_wav(src: str, dst: str) -> str:
            try:
                import torchaudio
                wav, sr = torchaudio.load(src)
                torchaudio.save(dst, wav, sr)
                return dst
            except Exception:
                pass
            try:
                from pydub import AudioSegment
                AudioSegment.from_file(src).export(dst, format="wav")
                return dst
            except Exception:
                pass
            return src

        wav_ref_path = str(OUTPUT_DIR / f"ref_{ts}.wav")
        loop = asyncio.get_event_loop()
        ref_path = await loop.run_in_executor(None, _to_wav, orig_ref_path, wav_ref_path)

    # Create job
    job_id = await create_job(
        text=text,
        gen_settings=gen_settings,
        ref_audio_path=ref_path,
    )

    async def event_generator():
        """Yield SSE events: progress updates and final result."""
        import asyncio

        # Yield job creation event
        yield f"event: job_created\ndata: {_json.dumps({'job_id': job_id, 'status': 'pending'})}\n\n"
        await asyncio.sleep(0.1)

        # Yield progress updates
        yield f"event: progress\ndata: {_json.dumps({'job_id': job_id, 'status': 'running', 'progress': 10})}\n\n"
        await asyncio.sleep(0.1)

        # Run generation
        try:
            result = await generate_audio(
                job_id=job_id,
                text=text,
                gen_settings=gen_settings,
                ref_audio_path=ref_path,
            )

            if result["status"] == "success":
                yield f"event: complete\ndata: {_json.dumps({'job_id': job_id, 'status': 'done', 'audio_url': result.get('audio_url')})}\n\n"
            else:
                yield f"event: error\ndata: {_json.dumps({'job_id': job_id, 'status': 'error', 'message': result.get('message', 'Generation failed')})}\n\n"

        except Exception as exc:
            yield f"event: error\ndata: {_json.dumps({'job_id': job_id, 'status': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/v2/history", response_model=HistoryResponse, tags=["History"])
async def list_history(
    page: int = 1,
    page_size: int = 20,
    status_filter: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    page_size = min(page_size, 100)
    q = select(GenerationJob).order_by(desc(GenerationJob.created_at))
    if status_filter:
        q = q.where(GenerationJob.status == status_filter)

    count_q = select(GenerationJob)
    if status_filter:
        count_q = count_q.where(GenerationJob.status == status_filter)

    from sqlalchemy import func
    total_res = await db.execute(select(func.count()).select_from(count_q.subquery()))
    total = total_res.scalar_one()

    q = q.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(q)
    jobs = result.scalars().all()

    items = [
        {
            "job_id": str(j.id),
            "status": j.status,
            "mode": j.mode,
            "text_preview": (j.text or "")[:80],
            "audio_url": j.audio_url,
            "generation_time": j.generation_time,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "gpu_mem_mb": j.gpu_mem_mb,
        }
        for j in jobs
    ]
    return HistoryResponse(total=total, page=page, page_size=page_size, items=items)


@app.delete("/api/v2/history/{job_id}", tags=["History"])
async def delete_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    job = await db.get(GenerationJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # Optionally remove file from disk
    if job.filepath and Path(job.filepath).exists():
        Path(job.filepath).unlink(missing_ok=True)
    await db.delete(job)
    await db.commit()
    return {"deleted": job_id}


# ── Export history as CSV ──────────────────────────────────────────────────────
@app.get("/api/v2/history/export/csv", tags=["History"])
async def export_history_csv(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    import csv, io
    result = await db.execute(
        select(GenerationJob).order_by(desc(GenerationJob.created_at)).limit(1000)
    )
    jobs = result.scalars().all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["job_id", "status", "mode", "text_preview",
                     "generation_time_s", "created_at", "audio_url"])
    for j in jobs:
        writer.writerow([
            j.id, j.status, j.mode, (j.text or "")[:120],
            j.generation_time, j.created_at, j.audio_url,
        ])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=omnivoice_history.csv"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# KARAOKE VOICE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v2/karaoke", tags=["Karaoke"])
async def karaoke_voice_conversion(
    ref_audio: UploadFile = File(..., description="Reference audio — voice identity (3–10s)"),
    karaoke_audio: UploadFile = File(..., description="Karaoke / performance audio to convert"),
    settings_json: Optional[str] = Form("{}"),
    audio_type: str = Form("full_song"),          # full_song|vocals_only|karaoke_track|speech
    manual_lyrics: Optional[str] = Form(None),    # used when audio_type=karaoke_track
    pitch_shift_override: Optional[int] = Form(None),
    match_pitch_auto: bool = Form(True),
    mix_with_instrumental: bool = Form(False),
    vocal_gain: float = Form(1.0),
    inst_gain: float = Form(0.80),
    language: str = Form("auto"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: bool = Depends(verify_api_key),
):
    """
    Karaoke Voice Conversion.

    Upload Audio A (reference voice) + Audio B (karaoke/performance).
    Returns Audio C: the performance from B sung in the voice of A.

    - **ref_audio**: 3–10s clean reference audio of the target voice
    - **karaoke_audio**: the track to convert (vocals, speech, or singing)
    - **pitch_shift_override**: manual semitone shift (-12..+12), overrides auto
    - **match_pitch_auto**: automatically align pitch registers
    - **mix_with_instrumental**: mix converted vocals with separated BG music
    - **vocal_gain / inst_gain**: mix levels (0.0–1.5)
    - **language**: Whisper transcription language (auto, en, ur, ar, etc.)
    """
    # ── Parse settings ─────────────────────────────────────────────────────────
    try:
        raw = json.loads(settings_json or "{}")
    except json.JSONDecodeError:
        raw = {}
    from schemas import GenerationSettings
    gen_settings = GenerationSettings(**raw)
    gen_settings.mode = "karaoke"  # type: ignore[assignment]

    # ── Save uploaded files ────────────────────────────────────────────────────
    ts = int(time.time())

    # Save originals, then convert to WAV for reliable loading
    orig_ref_path = str(OUTPUT_DIR / f"kvc_ref_{ts}_{ref_audio.filename}")
    ref_content = await ref_audio.read()
    async with aiofiles.open(orig_ref_path, "wb") as f:
        await f.write(ref_content)

    orig_kar_path = str(OUTPUT_DIR / f"kvc_kar_{ts}_{karaoke_audio.filename}")
    kar_content = await karaoke_audio.read()
    async with aiofiles.open(orig_kar_path, "wb") as f:
        await f.write(kar_content)

    # Convert to WAV so librosa/soundfile can load any format
    import asyncio as _asyncio
    def _to_wav(src, dst):
        try:
            import torchaudio
            wav, sr = torchaudio.load(src)
            torchaudio.save(dst, wav, sr)
            return dst
        except Exception:
            pass
        try:
            from pydub import AudioSegment
            AudioSegment.from_file(src).export(dst, format="wav")
            return dst
        except Exception:
            pass
        return src  # fallback: use original if conversion fails

    ref_path = str(OUTPUT_DIR / f"kvc_ref_{ts}.wav")
    kar_path = str(OUTPUT_DIR / f"kvc_kar_{ts}.wav")
    loop2 = asyncio.get_event_loop()
    ref_path = await loop2.run_in_executor(None, _to_wav, orig_ref_path, ref_path)
    kar_path = await loop2.run_in_executor(None, _to_wav, orig_kar_path, kar_path)

    log.info(f"[KVC] ref={ref_audio.filename}  karaoke={karaoke_audio.filename}")

    # ── Create DB job & run ────────────────────────────────────────────────────
    job_id = await create_kvc_job(
        ref_audio_path=ref_path,
        karaoke_audio_path=kar_path,
        gen_settings=gen_settings,
    )

    result = await process_karaoke(
        job_id=job_id,
        ref_audio_path=ref_path,
        karaoke_audio_path=kar_path,
        gen_settings=gen_settings,
        audio_type=audio_type,
        manual_lyrics=manual_lyrics,
        pitch_shift_override=pitch_shift_override,
        match_pitch_auto=match_pitch_auto,
        mix_with_instrumental=mix_with_instrumental,
        vocal_gain=vocal_gain,
        inst_gain=inst_gain,
        language=language,
    )

    http_status = 200 if result["status"] == "success" else 500
    return JSONResponse(content=result, status_code=http_status)


@app.post("/api/v2/karaoke/async", tags=["Karaoke"])
async def karaoke_async(
    ref_audio: UploadFile = File(...),
    karaoke_audio: UploadFile = File(...),
    settings_json: Optional[str] = Form("{}"),
    pitch_shift_override: Optional[int] = Form(None),
    match_pitch_auto: bool = Form(True),
    mix_with_instrumental: bool = Form(False),
    vocal_gain: float = Form(1.0),
    inst_gain: float = Form(0.80),
    language: str = Form("auto"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: bool = Depends(verify_api_key),
):
    """Async karaoke VC — submit job, returns job_id immediately."""
    try:
        raw = json.loads(settings_json or "{}")
    except json.JSONDecodeError:
        raw = {}
    from schemas import GenerationSettings
    gen_settings = GenerationSettings(**raw)

    ts = int(time.time())
    ref_path = str(OUTPUT_DIR / f"kvc_ref_{ts}_{ref_audio.filename}")
    async with aiofiles.open(ref_path, "wb") as f:
        await f.write(await ref_audio.read())

    kar_path = str(OUTPUT_DIR / f"kvc_kar_{ts}_{karaoke_audio.filename}")
    async with aiofiles.open(kar_path, "wb") as f:
        await f.write(await karaoke_audio.read())

    job_id = await create_kvc_job(
        ref_audio_path=ref_path,
        karaoke_audio_path=kar_path,
        gen_settings=gen_settings,
    )
    background_tasks.add_task(
        process_karaoke,
        job_id=job_id,
        ref_audio_path=ref_path,
        karaoke_audio_path=kar_path,
        gen_settings=gen_settings,
        pitch_shift_override=pitch_shift_override,
        match_pitch_auto=match_pitch_auto,
        mix_with_instrumental=mix_with_instrumental,
        vocal_gain=vocal_gain,
        inst_gain=inst_gain,
        language=language,
    )
    return {"status": "pending", "job_id": job_id}


# ══════════════════════════════════════════════════════════════════════════════
# VOICE BLENDING / MIXING
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v2/blend", response_model=GenerationResponse, tags=["Generation"])
async def blend_voices(
    request: Request,
    text: str = Form(...),
    blend_weight: float = Form(0.5),  # 0.0 = 100% voice B, 1.0 = 100% voice A
    settings_json: Optional[str] = Form("{}"),
    ref_audio_a: Optional[UploadFile] = File(None, description="Primary voice (A)"),
    ref_audio_b: Optional[UploadFile] = File(None, description="Secondary voice (B) — optional for blending"),
    _: bool = Depends(verify_api_key),
):
    """
    Voice blending — mix two reference voices at a given ratio.
    Upload Audio A + Audio B → generate text with blended voice characteristics.

    - **blend_weight**: 0.0 = 100% voice B, 0.5 = equal mix, 1.0 = 100% voice A
    - **ref_audio_a**: Primary voice reference (required)
    - **ref_audio_b**: Secondary voice reference (optional, if omitted uses voice A only)
    """
    import json as _json
    from schemas import GenerationSettings

    try:
        raw = _json.loads(settings_json or "{}")
    except _json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid settings JSON: {exc}")

    gen_settings = GenerationSettings(**raw)
    gen_settings.mode = "blend"
    gen_settings.voice_blend_weight = blend_weight

    # Save reference audio A
    ref_path_a = None
    if ref_audio_a and ref_audio_a.filename:
        ts = int(time.time())
        safe_name = f"blend_a_{ts}_{ref_audio_a.filename}"
        orig_ref_path = str(OUTPUT_DIR / safe_name)
        content = await ref_audio_a.read()
        async with aiofiles.open(orig_ref_path, "wb") as f:
            await f.write(content)

        # Convert to WAV
        def _to_wav(src: str, dst: str) -> str:
            try:
                import torchaudio
                wav, sr = torchaudio.load(src)
                torchaudio.save(dst, wav, sr)
                return dst
            except Exception:
                pass
            try:
                from pydub import AudioSegment
                AudioSegment.from_file(src).export(dst, format="wav")
                return dst
            except Exception:
                pass
            return src

        wav_ref_path = str(OUTPUT_DIR / f"blend_a_{ts}.wav")
        loop = asyncio.get_event_loop()
        ref_path_a = await loop.run_in_executor(None, _to_wav, orig_ref_path, wav_ref_path)
        log.info(f"Blend: Primary voice saved → {ref_path_a}")

    # Save reference audio B (optional)
    ref_path_b = None
    if ref_audio_b and ref_audio_b.filename:
        ts = int(time.time())
        safe_name = f"blend_b_{ts}_{ref_audio_b.filename}"
        orig_ref_path = str(OUTPUT_DIR / safe_name)
        content = await ref_audio_b.read()
        async with aiofiles.open(orig_ref_path, "wb") as f:
            await f.write(content)

        # Convert to WAV
        def _to_wav_b(src: str, dst: str) -> str:
            try:
                import torchaudio
                wav, sr = torchaudio.load(src)
                torchaudio.save(dst, wav, sr)
                return dst
            except Exception:
                pass
            try:
                from pydub import AudioSegment
                AudioSegment.from_file(src).export(dst, format="wav")
                return dst
            except Exception:
                pass
            return src

        wav_ref_path = str(OUTPUT_DIR / f"blend_b_{ts}.wav")
        loop = asyncio.get_event_loop()
        ref_path_b = await loop.run_in_executor(None, _to_wav_b, orig_ref_path, wav_ref_path)
        gen_settings.ref_audio_secondary = ref_path_b
        log.info(f"Blend: Secondary voice saved → {ref_path_b}")

    if not ref_path_a:
        raise HTTPException(400, "Primary reference audio (A) is required")

    # Create and run job
    job_id = await create_job(
        text=text,
        gen_settings=gen_settings,
        ref_audio_path=ref_path_a,
    )

    result = await generate_audio(
        job_id=job_id,
        text=text,
        gen_settings=gen_settings,
        ref_audio_path=ref_path_a,
    )

    http_status = 200 if result["status"] == "success" else 500
    return JSONResponse(content=result, status_code=http_status)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=cfg.host,
        port=cfg.port,
        workers=cfg.workers,
        reload=cfg.debug,
        log_level="debug" if cfg.debug else "info",
        access_log=cfg.debug,
    )
