"""
OmniVoice Studio — Core generation service.

Handles:
  • Input validation & preprocessing
  • Async model inference (offloaded to thread executor)
  • Audio post-processing & disk write
  • Database persistence
  • Prometheus metrics recording
  • Redis cache (optional)
  • Concurrency semaphore & generation timeouts
  • OOM recovery (safe_generate)
  • Webhook callback delivery
  • Output format conversion (WAV/MP3/FLAC/OGG)
  • Emotion, SSML, pronunciation lexicon processing
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── Cooperative cancellation + UI progress (polling) ───────────────────────────
_cancel_jobs: set[str] = set()
_job_progress: dict[str, tuple[str, int]] = {}  # job_id -> (phase_message, 0..100)


def request_cancel(job_id: str) -> None:
    """Mark job for cooperative cancellation (checked before/after GPU work)."""
    _cancel_jobs.add(job_id)


def is_cancel_requested(job_id: str) -> bool:
    return job_id in _cancel_jobs


def set_job_progress(job_id: str, message: str, percent: int) -> None:
    pct = max(0, min(100, int(percent)))
    _job_progress[job_id] = (message, pct)


def get_job_progress(job_id: str) -> Optional[tuple[str, int]]:
    return _job_progress.get(job_id)


def clear_job_tracking(job_id: str) -> None:
    _cancel_jobs.discard(job_id)
    _job_progress.pop(job_id, None)


async def finalize_cancelled_job(job_id: str) -> dict[str, Any]:
    """Persist cancelled status and clear tracking."""
    from datetime import datetime, timezone

    clear_job_tracking(job_id)
    async with AsyncSessionLocal() as db:
        job = await db.get(GenerationJob, job_id)
        if job and job.status not in ("done", "error", "cancelled"):
            job.status = "cancelled"
            job.error_message = "Cancelled by user"
            job.finished_at = datetime.now(timezone.utc)
            job.generation_time = 0.0
            await db.commit()
    return {
        "status": "error",
        "job_id": job_id,
        "message": "Generation cancelled",
    }

from config import get_settings
from database import AsyncSessionLocal, GenerationJob, SystemEvent
from logger import get_logger
from metrics import (
    get_system_snapshot, record_generation,
    record_active_job_start, record_active_job_end, check_disk_space,
)
from model_manager import get_model, safe_generate
from schemas import GenerationSettings

log = get_logger(__name__)
settings = get_settings()

# ── Concurrency semaphore ──────────────────────────────────────────────────────
_generation_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)

# ── Optional Redis ──────────────────────────────────────────────────────────────
_redis: Any = None


async def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    if not settings.redis_url:
        return None
    try:
        import redis.asyncio as aioredis
        _redis = await aioredis.from_url(settings.redis_url, decode_responses=False)
        return _redis
    except Exception as exc:
        log.warning(f"Redis unavailable ({exc}), caching disabled.")
        return None


# ── Text preprocessing helpers ─────────────────────────────────────────────────
def _apply_pronunciation_lexicon(text: str, lexicon: dict[str, str]) -> str:
    """Replace words in text with their phonetic equivalents from the lexicon."""
    if not lexicon:
        return text
    for word, phonetic in lexicon.items():
        # Case-insensitive whole-word replacement
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(phonetic, text)
    return text


def _process_ssml_breaks(text: str) -> str:
    """
    Convert simple SSML <break time="500ms"/> tags into silence placeholders.
    OmniVoice doesn't support SSML natively, so we approximate by inserting
    ellipsis pauses which most TTS models interpret as natural breaks.
    """
    # <break time="500ms"/> → "..."
    # <break time="1s"/>    → "...... "
    def _replace_break(m):
        time_str = m.group(1)
        if time_str.endswith("ms"):
            ms = int(time_str[:-2])
        elif time_str.endswith("s"):
            ms = int(float(time_str[:-1]) * 1000)
        else:
            ms = 500
        dots = max(1, ms // 200)
        return "." * dots + " "

    text = re.sub(
        r'<break\s+time="([^"]+)"\s*/?>',
        _replace_break,
        text,
        flags=re.IGNORECASE,
    )
    # Remove any other unsupported SSML tags
    text = re.sub(r"<[^>]+>", "", text)
    return text


def _style_to_emotion(style: str) -> str:
    """Map GUI style chip selection → emotion for generation."""
    style_map = {
        "neutral": "neutral",
        "formal": "steady",
        "casual": "warm",
        "enthusiastic": "excited",
        "calm": "warm",
        "storytelling": "expressive",
        "news_anchor": "steady",
    }
    return style_map.get(style, "neutral")


# Valid OmniVoice instruct tokens that map to emotions
_EMOTION_INSTRUCT_TOKENS = {
    "whispering": "whisper",
}


def _build_emotion_text_prefix(emotion: str) -> str:
    """Return a text prefix that guides the model's emotional delivery.

    OmniVoice's `instruct` field only accepts specific voice-descriptor tokens
    (e.g. 'male', 'british accent', 'whisper').  Free-form emotion instructions
    must be prepended to the `text` field instead.
    """
    emotion_text_map = {
        "happy": "[happy tone] ",
        "sad": "[sad tone] ",
        "angry": "[angry tone] ",
        "surprised": "[surprised tone] ",
        "whispering": "[whispered] ",
        "excited": "[excited tone] ",
        "expressive": "[expressive storytelling tone] ",
        "warm": "[warm gentle tone] ",
        "dynamic": "[dynamic varied tone] ",
        "intense": "[intense dramatic tone] ",
        "steady": "[calm steady tone] ",
        "neutral": "",
    }
    return emotion_text_map.get(emotion, "")


def prepare_omnivoice_text(
    text: str,
    gen_settings: GenerationSettings,
    *,
    job_id: str = "",
) -> tuple[str, str, str]:
    """
    Shared preprocessing for OmniVoice: lexicon, SSML, style→emotion, text prefixes, instruct.
    Returns (processed_text, instruct, resolved_emotion).
    """
    processed_text = text

    if gen_settings.pronunciation_lexicon:
        processed_text = _apply_pronunciation_lexicon(
            processed_text, gen_settings.pronunciation_lexicon
        )

    if gen_settings.ssml_enabled:
        processed_text = _process_ssml_breaks(processed_text)

    resolved_emotion = gen_settings.emotion
    if gen_settings.style != "neutral" and gen_settings.emotion == "neutral":
        resolved_emotion = _style_to_emotion(gen_settings.style)
        if job_id:
            log.info(f"[{job_id}] Style='{gen_settings.style}' → mapped emotion='{resolved_emotion}'")
    elif gen_settings.style != "neutral":
        if job_id:
            log.info(
                f"[{job_id}] Style='{gen_settings.style}' specified but emotion='{gen_settings.emotion}' "
                "explicitly set — using explicit emotion"
            )
    else:
        if job_id:
            log.info(f"[{job_id}] Style=neutral, emotion='{gen_settings.emotion}'")

    emotion_prefix = _build_emotion_text_prefix(resolved_emotion)
    if emotion_prefix:
        processed_text = emotion_prefix + processed_text
        if job_id:
            log.info(f"[{job_id}] Emotion text prefix applied: '{emotion_prefix.strip()}'")
    instruct = _build_emotion_instruct(resolved_emotion, gen_settings.instruct)

    return processed_text, instruct, resolved_emotion


def _build_emotion_instruct(emotion: str, existing_instruct: str) -> str:
    """Build the instruct string using only valid OmniVoice tokens.

    If the emotion maps to a valid instruct token (e.g. whispering → whisper),
    prepend it.  Otherwise, only use the user's existing instruct.
    """
    token = _EMOTION_INSTRUCT_TOKENS.get(emotion, "")
    if token and existing_instruct:
        return f"{token}, {existing_instruct}"
    return token or existing_instruct


# ── Audio preprocessing helpers ────────────────────────────────────────────────
def _preprocess_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalise, ensure shape is (samples,) or (samples, channels)."""
    if audio_data.ndim > 2:
        audio_data = audio_data.squeeze()
    # If shape is (channels, samples) → transpose
    if audio_data.ndim == 2 and audio_data.shape[0] < audio_data.shape[1]:
        audio_data = audio_data.T
    
    # Sanitize: Handle NaNs/Infs
    if not np.isfinite(audio_data).all():
        log.warning("Detected non-finite values (NaN/Inf) in generated audio. Fixing...")
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Peak normalise to prevent clipping
    peak = np.abs(audio_data).max()
    if peak > 0:
        gain = 0.95 / peak
        # Cap gain for very quiet signals to avoid noise floor amplification
        if peak < 0.001:
            gain = min(gain, 20.0)
            log.warning(f"Very low peak ({peak:.2e}) in generated audio. Capping gain.")
        audio_data = audio_data * gain
    return audio_data.astype(np.float32)


def _save_audio(
    audio_data: np.ndarray,
    sr: int,
    prefix: str,
    output_format: str = "wav",
) -> tuple[str, str]:
    """Write audio to disk in requested format, return (filename, abs_filepath)."""
    import soundfile as sf

    # Final sanity check
    if not np.isfinite(audio_data).all():
        audio_data = np.nan_to_num(audio_data)

    timestamp = int(time.time())
    uid = uuid.uuid4().hex[:6]

    # Always save WAV first
    wav_filename = f"{prefix}_{timestamp}_{uid}.wav"
    wav_filepath = settings.output_dir / wav_filename
    sf.write(str(wav_filepath), audio_data, sr, subtype="PCM_24")

    if output_format == "wav":
        return wav_filename, str(wav_filepath.resolve())

    # Convert to requested format
    out_filename = f"{prefix}_{timestamp}_{uid}.{output_format}"
    out_filepath = settings.output_dir / out_filename
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_wav(str(wav_filepath))
        export_params = {}
        if output_format == "mp3":
            export_params = {"bitrate": "192k"}
        seg.export(str(out_filepath), format=output_format, **export_params)
        # Remove intermediate WAV
        wav_filepath.unlink(missing_ok=True)
        return out_filename, str(out_filepath.resolve())
    except Exception as exc:
        log.warning(f"Format conversion to {output_format} failed ({exc}), keeping WAV")
        return wav_filename, str(wav_filepath.resolve())


# ── Audio cache helpers ────────────────────────────────────────────────────────
def _cache_key(text: str, ref_path: Optional[str], gen_settings: GenerationSettings) -> str:
    """Generate a deterministic cache key from generation parameters."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(str(gen_settings.model_dump()).encode("utf-8"))
    if ref_path and Path(ref_path).exists():
        h.update(str(Path(ref_path).stat().st_size).encode())
        h.update(str(Path(ref_path).stat().st_mtime).encode())
    return h.hexdigest()[:16]


async def _check_cache(key: str) -> Optional[str]:
    """Check if a cached audio file exists for this key."""
    r = await _get_redis()
    if r is None:
        return None
    try:
        cached = await r.get(f"omnivoice:cache:{key}")
        if cached:
            filepath = cached.decode("utf-8")
            if Path(filepath).exists():
                return filepath
    except Exception:
        pass
    return None


async def _set_cache(key: str, filepath: str, ttl: int = 3600) -> None:
    """Cache a generation result for the given key."""
    r = await _get_redis()
    if r is None:
        return
    try:
        await r.set(f"omnivoice:cache:{key}", filepath.encode(), ex=ttl)
    except Exception:
        pass


# ── Webhook delivery ──────────────────────────────────────────────────────────
async def _deliver_webhook(url: str, payload: dict) -> None:
    """POST the result JSON to the webhook URL. Fire-and-forget."""
    if not url:
        return
    try:
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=settings.webhook_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                log.info(f"[WEBHOOK] Delivered to {url} — status={resp.status}")
    except Exception as exc:
        log.warning(f"[WEBHOOK] Failed to deliver to {url}: {exc}")


# ── Core inference function (runs in executor) ─────────────────────────────────
def _run_inference(model: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """OOM-safe blocking TTS call — executed in a thread pool."""
    return safe_generate(model, kwargs)


# ── Main service function ──────────────────────────────────────────────────────
async def generate_audio(
    *,
    job_id: str,
    text: str,
    gen_settings: GenerationSettings,
    ref_audio_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Full generation pipeline with concurrency control, timeouts, and caching.
    """
    t0 = time.perf_counter()
    step_timings: dict[str, float] = {}

    # ── Disk space check ──────────────────────────────────────────────────────
    if not check_disk_space(settings.disk_min_free_gb):
        return {
            "status": "error",
            "job_id": job_id,
            "message": f"Insufficient disk space (need {settings.disk_min_free_gb} GB free)",
        }

    # ── Audio cache check ─────────────────────────────────────────────────────
    cache_key = _cache_key(text, ref_audio_path, gen_settings)
    cached_path = await _check_cache(cache_key)
    if cached_path:
        filename = Path(cached_path).name
        log.info(f"[{job_id}] Cache hit → {filename}")
        return {
            "status": "success",
            "job_id": job_id,
            "message": "Audio served from cache",
            "audio_url": f"/outputs/{filename}",
            "filepath": cached_path,
            "generation_time": 0.0,
            "text": text,
            "metadata": {"cached": True},
        }

    # ── Concurrency gate ──────────────────────────────────────────────────────
    set_job_progress(job_id, "Queued — waiting for GPU…", 5)
    record_active_job_start()
    try:
        async with _generation_semaphore:
            if is_cancel_requested(job_id):
                return await finalize_cancelled_job(job_id)
            set_job_progress(job_id, "Acquired GPU — starting…", 10)
            return await _generate_audio_inner(
                job_id=job_id,
                text=text,
                gen_settings=gen_settings,
                ref_audio_path=ref_audio_path,
                t0=t0,
                cache_key=cache_key,
            )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t0
        clear_job_tracking(job_id)
        log.error(f"[{job_id}] Generation timed out after {elapsed:.1f}s")
        async with AsyncSessionLocal() as db:
            job = await db.get(GenerationJob, job_id)
            if job:
                from datetime import datetime, timezone
                job.status = "error"
                job.error_message = "Generation timed out"
                job.finished_at = datetime.now(timezone.utc)
                job.generation_time = round(elapsed, 2)
                await db.commit()
        return {
            "status": "error",
            "job_id": job_id,
            "message": f"Generation timed out after {settings.generation_timeout_s}s",
        }
    finally:
        record_active_job_end()


async def _generate_audio_inner(
    *,
    job_id: str,
    text: str,
    gen_settings: GenerationSettings,
    ref_audio_path: Optional[str],
    t0: float,
    cache_key: str,
) -> dict[str, Any]:
    """Inner generation logic, called inside the semaphore."""
    step_timings: dict[str, float] = {}

    async with AsyncSessionLocal() as db:
        # Mark job as running
        job = await db.get(GenerationJob, job_id)
        if job:
            job.status = "running"
            await db.commit()

    try:
        if is_cancel_requested(job_id):
            return await finalize_cancelled_job(job_id)
        set_job_progress(job_id, "Preparing text & voice settings…", 16)

        # ── 1. Preprocess text ────────────────────────────────────────────────
        processed_text, instruct, resolved_emotion = prepare_omnivoice_text(
            text, gen_settings, job_id=job_id
        )

        # ── 2. Build inference kwargs ─────────────────────────────────────────
        kwargs: dict[str, Any] = {
            "text": processed_text,
            "num_steps": gen_settings.steps,
            "cfg_scale": gen_settings.cfg,
            "temperature": gen_settings.temp,
            "sampler": gen_settings.sampler,
            "speed": gen_settings.speed,
            "pitch_shift": gen_settings.pitch,
            "similarity_weight": gen_settings.sim,
        }
        if ref_audio_path:
            kwargs["ref_audio"] = ref_audio_path
        if gen_settings.ref_text:
            kwargs["ref_text"] = gen_settings.ref_text
        if gen_settings.duration:
            kwargs["max_duration"] = gen_settings.duration
        if gen_settings.seed is not None:
            kwargs["seed"] = gen_settings.seed
        if gen_settings.lang and gen_settings.lang != "auto":
            kwargs["language"] = gen_settings.lang
        if instruct:
            kwargs["instruct"] = instruct

        # Voice blending: secondary reference audio and blend weight
        if gen_settings.ref_audio_secondary and Path(gen_settings.ref_audio_secondary).exists():
            kwargs["ref_audio_secondary"] = gen_settings.ref_audio_secondary
            kwargs["voice_blend_weight"] = gen_settings.voice_blend_weight
            log.info(f"[{job_id}] Voice blend enabled — weight={gen_settings.voice_blend_weight:.2f}")

        log.info(f"[{job_id}] Generating — mode={gen_settings.mode} text_len={len(processed_text)} style={gen_settings.style} emotion={resolved_emotion}")

        if is_cancel_requested(job_id):
            return await finalize_cancelled_job(job_id)

        # ── 3. Load model ─────────────────────────────────────────────────────
        set_job_progress(job_id, "Loading OmniVoice model…", 26)
        model = await get_model(gen_settings.model_dump())

        if is_cancel_requested(job_id):
            return await finalize_cancelled_job(job_id)

        # ── 4. Run inference with timeout ─────────────────────────────────────
        set_job_progress(job_id, "Synthesizing audio (diffusion — may take a while)…", 48)
        loop = asyncio.get_running_loop()
        t_gen = time.perf_counter()
        # Adaptive timeout: base + per-char allowance so long texts don't time out
        _timeout = settings.generation_timeout_s + len(processed_text) * 0.3
        log.info(f"[{job_id}] Inference timeout={_timeout:.0f}s (base={settings.generation_timeout_s} + {len(processed_text)} chars × 0.3s)")
        out = await asyncio.wait_for(
            loop.run_in_executor(None, _run_inference, model, kwargs),
            timeout=_timeout,
        )
        step_timings["generation"] = round(time.perf_counter() - t_gen, 3)

        if is_cancel_requested(job_id):
            return await finalize_cancelled_job(job_id)

        set_job_progress(job_id, "Post-processing & saving file…", 90)

        # ── 5. Post-process ───────────────────────────────────────────────────
        sr = gen_settings.sr
        if isinstance(out, dict):
            audio_data = out.get("audio") or out.get("wav") or out.get("waveform")
            sr = out.get("sample_rate", out.get("sr", gen_settings.sr))
        elif isinstance(out, (list, tuple)):
            audio_data = out[0]
            if len(out) > 1 and isinstance(out[1], int):
                sr = out[1]
        else:
            audio_data = out

        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        audio_data = _preprocess_audio(np.asarray(audio_data))

        # ── 6. Save audio (with format conversion) ────────────────────────────
        t_save = time.perf_counter()
        filename, filepath = _save_audio(
            audio_data, sr, gen_settings.prefix, gen_settings.output_format
        )
        step_timings["save"] = round(time.perf_counter() - t_save, 3)

        # Cache the result
        await _set_cache(cache_key, filepath)

        elapsed = time.perf_counter() - t0
        snap = get_system_snapshot()
        gpu_mem = int(snap.get("gpu_mem_used_mb", 0))
        audio_duration_s = len(audio_data) / sr if sr > 0 else 0

        # ── 7. Persist to DB ──────────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            job = await db.get(GenerationJob, job_id)
            if job:
                from datetime import datetime, timezone
                job.status = "done"
                job.audio_url = f"/outputs/{filename}"
                job.filepath = filepath
                job.generation_time = round(elapsed, 2)
                job.finished_at = datetime.now(timezone.utc)
                job.gpu_mem_mb = gpu_mem
                job.cpu_pct = snap.get("cpu_pct", 0)
                job.step_timings_json = json.dumps(step_timings)
                await db.commit()

        record_generation(
            success=True, elapsed=elapsed, gpu_mem_mb=gpu_mem,
            audio_duration_s=audio_duration_s, step_timings=step_timings,
        )

        log.info(f"[{job_id}] Done in {elapsed:.2f}s → {filename}")

        clear_job_tracking(job_id)

        # ── 8. Cleanup temporary intermediate files ───────────────────────────
        if ref_audio_path:
            try:
                path_obj = Path(ref_audio_path)
                # Only delete if it's one of our temp-generated WAVs in output dir
                if (path_obj.parent.resolve() == settings.output_dir.resolve() and 
                    path_obj.suffix.lower() == ".wav" and 
                    path_obj.name.startswith("ref_")):
                    path_obj.unlink(missing_ok=True)
                    log.debug(f"[{job_id}] Cleaned up temp file: {path_obj.name}")
            except Exception as e:
                log.warning(f"[{job_id}] Temp cleanup failed: {e}")

        result = {
            "status": "success",
            "job_id": job_id,
            "message": "Audio generated successfully",
            "audio_url": f"/outputs/{filename}",
            "filepath": filepath,
            "generation_time": round(elapsed, 2),
            "text": text,
            "metadata": {
                "sample_rate": sr,
                "gpu_mem_mb": gpu_mem,
                "cpu_pct": snap.get("cpu_pct", 0),
                "output_format": gen_settings.output_format,
                "emotion": gen_settings.emotion,
                "step_timings": step_timings,
                "audio_duration_s": round(audio_duration_s, 2),
            },
        }

        # Deliver webhook if configured
        if gen_settings.webhook_url:
            asyncio.create_task(_deliver_webhook(gen_settings.webhook_url, result))

        return result

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        clear_job_tracking(job_id)
        log.error(f"[{job_id}] Generation failed: {exc}")
        traceback.print_exc()
        record_generation(success=False, elapsed=elapsed)

        async with AsyncSessionLocal() as db:
            job = await db.get(GenerationJob, job_id)
            if job:
                from datetime import datetime, timezone
                job.status = "error"
                job.error_message = str(exc)
                job.finished_at = datetime.now(timezone.utc)
                job.generation_time = round(elapsed, 2)
                await db.commit()

        error_result = {
            "status": "error",
            "job_id": job_id,
            "message": f"Generation failed: {exc}",
        }

        # Deliver webhook even on error
        if gen_settings.webhook_url:
            asyncio.create_task(_deliver_webhook(gen_settings.webhook_url, error_result))

        return error_result


async def create_job(
    *,
    text: str,
    gen_settings: GenerationSettings,
    ref_audio_path: Optional[str] = None,
) -> str:
    """Create a pending DB job row and return its ID."""
    job_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as db:
        job = GenerationJob(
            id=job_id,
            mode=gen_settings.mode,
            text=text,
            ref_audio_path=ref_audio_path,
            settings_json=json.dumps(gen_settings.model_dump()),
            status="pending",
            webhook_url=gen_settings.webhook_url or None,
        )
        db.add(job)
        await db.commit()
    return job_id
