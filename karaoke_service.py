"""
OmniVoice Studio — Karaoke Voice Conversion Service
====================================================
Pipeline
--------
1. Load & separate vocals from karaoke audio (librosa HPSS)
2. Transcribe vocals with Whisper (already loaded by OmniVoice)
3. Estimate F0 pitch delta between reference voice and karaoke track
4. Generate new audio with OmniVoice (ref voice identity + karaoke lyrics)
5. Time-stretch output to match original karaoke duration (optional)
6. Mix with instrumental backing track (optional)
7. Persist and return result

Enhancements
------------
• Transcript caching by audio file hash — avoids re-transcribing same audio
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import get_settings
from database import AsyncSessionLocal, GenerationJob
from logger import get_logger
from metrics import get_system_snapshot, record_generation, check_disk_space
from generation_service import (
    prepare_omnivoice_text, set_job_progress, clear_job_tracking
)
from model_manager import get_model, safe_generate
from schemas import GenerationSettings

log = get_logger(__name__)
settings = get_settings()

# ── Audio constants ────────────────────────────────────────────────────────────
_DEFAULT_SR = 24_000

# ── Transcript cache (audio_hash -> transcript) ────────────────────────────────
_transcript_cache: dict[str, str] = {}
_transcript_cache_lock = asyncio.Lock()


def _compute_audio_hash(audio_path: str) -> str:
    """Compute SHA256 hash of audio file content for caching."""
    try:
        h = hashlib.sha256()
        with open(audio_path, "rb") as f:
            # Read in chunks to handle large files
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()[:16]  # 16 chars is sufficient for dedup
    except Exception:
        # Fallback to file size + mtime if can't read
        try:
            stat = Path(audio_path).stat()
            return f"{stat.st_size:x}_{stat.st_mtime:.0f}"
        except Exception:
            return str(uuid.uuid4())[:16]


async def _get_cached_transcript(audio_hash: str) -> Optional[str]:
    """Get cached transcript if available."""
    async with _transcript_cache_lock:
        return _transcript_cache.get(audio_hash)


async def _set_cached_transcript(audio_hash: str, transcript: str) -> None:
    """Cache transcript for future use."""
    async with _transcript_cache_lock:
        _transcript_cache[audio_hash] = transcript


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _load_audio(path: str, sr: int = _DEFAULT_SR) -> tuple[np.ndarray, int]:
    """
    Load any audio file -> mono float32 numpy array at target sample rate.
    Backend cascade:
      1. librosa/soundfile  -- WAV, FLAC, OGG (fast, no ffmpeg)
      2. torchaudio         -- AAC, M4A, MP3, MP4 via Windows MediaFoundation
      3. pydub              -- everything else (needs ffmpeg in PATH)
    """
    import warnings

    # Backend 1: librosa / soundfile (handles WAV/FLAC/OGG natively)
    try:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio.astype(np.float32), sr
    except Exception:
        pass

    # Backend 2: torchaudio for decoding, scipy for resampling (CPU-only, no VRAM)
    try:
        import torchaudio
        waveform, native_sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_np = waveform.squeeze(0).numpy().astype(np.float32)
        if native_sr != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(native_sr, sr)
            audio_np = resample_poly(audio_np, sr // g, native_sr // g)
        return audio_np.astype(np.float32), sr
    except Exception:
        pass

    # Backend 3: pydub (requires ffmpeg in PATH -- winget install Gyan.FFmpeg)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        seg = seg.set_channels(1).set_frame_rate(sr)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples = samples / float(2 ** (seg.sample_width * 8 - 1))
        return samples, sr
    except Exception:
        pass

    raise RuntimeError(
        f"Cannot load '{Path(path).name}'. Supported: WAV, FLAC, OGG, MP3, AAC, M4A. "
        f"Install ffmpeg for broadest support: winget install Gyan.FFmpeg"
    )


def _save_audio(audio: np.ndarray, sr: int, prefix: str) -> tuple[str, str]:
    """Write WAV to output dir, return (filename, abs_filepath)."""
    import soundfile as sf
    ts = int(time.time())
    uid = uuid.uuid4().hex[:6]
    filename = f"{prefix}_{ts}_{uid}.wav"
    filepath = settings.output_dir / filename

    # Sanitize: Handle NaNs/Infs that might come from model or processing
    if not np.isfinite(audio).all():
        log.warning(f"Detected non-finite values (NaN/Inf) in {prefix} audio. Zeroing out.")
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalise before write (Optimized in-place)
    peak = np.abs(audio).max()
    if peak > 0:
        # If signal is extremely quiet, don't blast it with 1000x gain (prevents noise floor amplification)
        gain = 0.92 / peak
        if peak < 0.001:
            gain = min(gain, 20.0)  # max 26dB gain for near-silence
            log.warning(f"Very low peak ({peak:.2e}) detected in {prefix} audio. Capping normalization gain to 20x.")
        
        audio *= gain

    sf.write(str(filepath), audio, sr, subtype="PCM_24")
    return filename, str(filepath.resolve())


def _separate_vocals(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Harmonic-Percussive Source Separation (HPSS).
    Returns (vocal_approx, instrumental_approx) — both mono float32.
    For production quality use Demucs; HPSS is a fast built-in fallback.
    """
    import librosa
    # Sanity check
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio)

    # Slightly tighter margin → faster HPSS with acceptable separation for mixing
    harmonic, percussive = librosa.effects.hpss(audio, margin=2.5)
    return harmonic, audio - harmonic   # harmonic ≈ vocals, residual ≈ drums+percussion


def _estimate_pitch_semitones(audio: np.ndarray, sr: int) -> float:
    """
    Estimate median voiced F0 in semitones (A4=69).

    Uses librosa YIN on a short downsampled excerpt — typically **much** faster than
    probabilistic YIN (pyin) on full 24 kHz tracks, with adequate accuracy for
    coarse karaoke pitch alignment.
    """
    import librosa

    if audio.size == 0:
        return 69.0
    
    # Sanitize input
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio)

    # Analyze up to ~40 s from the centre (vocals often clearest mid-track)
    max_sec = 40.0
    max_samples = int(min(len(audio), sr * max_sec))
    mid = len(audio) // 2
    half = max_samples // 2
    y = audio[max(0, mid - half): min(len(audio), mid + half)].astype(np.float32, copy=False)

    target_sr = 16_000
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if len(y) < sr // 4:
        return 69.0

    fmin_hz = librosa.note_to_hz("E2")
    fmax_hz = librosa.note_to_hz("C6")
    # Larger hop + modest frame → fewer frames, faster YIN
    hop = 512
    frame = 2048
    try:
        f0 = librosa.yin(y, fmin=fmin_hz, fmax=fmax_hz, sr=sr, frame_length=frame, hop_length=hop)
        voiced = f0[(f0 > fmin_hz * 0.9) & (f0 < fmax_hz * 1.1) & np.isfinite(f0)]
        if len(voiced) < 3:
            return 69.0
        median_hz = float(np.median(voiced))
        if median_hz <= 0:
            return 69.0
        return float(12 * np.log2(median_hz / 440.0) + 69)
    except Exception as e:
        log.warning(f"Pitch estimation failed: {e}")
        return 69.0


def _pitch_shift_audio(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift audio pitch by n semitones without changing tempo."""
    import librosa
    if abs(semitones) < 0.1:
        return audio
    # Sanitize
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Stretch audio by `rate` (>1 = slower output = sped-up playback to fit shorter target)."""
    import librosa

    if abs(rate - 1.0) < 0.01:
        return audio

    # Ensure audio is C-contiguous for faster processing in librosa
    if not audio.flags.c_contiguous:
        audio = np.ascontiguousarray(audio)

    # Sanitize
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio)

    # Use more standard n_fft=2048 for better quality; hop_length=512 is 1/4 overlap
    return librosa.effects.time_stretch(audio, rate=rate, n_fft=2048, hop_length=512)


def _mix_tracks(
    vocal: np.ndarray,
    instrumental: np.ndarray,
    vocal_gain: float = 1.0,
    inst_gain: float = 0.85,
) -> np.ndarray:
    """Mix vocal and instrumental to same length, return mix."""
    min_len = min(len(vocal), len(instrumental))

    # Optimization: Use in-place operations to avoid large intermediate copies.
    # We allocate the mix array once and then add the weighted tracks.
    mix = vocal[:min_len] * vocal_gain
    mix += instrumental[:min_len] * inst_gain

    # Normalisation removed here — it is handled once during the final _save_audio step
    # to avoid redundant full-track passes.
    return mix.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER TRANSCRIPTION (uses the model already loaded by OmniVoice)
# ══════════════════════════════════════════════════════════════════════════════
def _transcribe_audio_sync(audio_path: str, language: str = "auto") -> str:
    """
    Transcribe audio using Whisper.
    Loads audio as numpy array first (via torchaudio/librosa — no ffmpeg needed),
    then passes {"array": ..., "sampling_rate": 16000} directly to the pipeline
    so Whisper never needs to decode the file itself.

    Strategy (in order of preference):
      1. OmniVoice's already-loaded asr_model pipeline (zero extra VRAM)
      2. transformers AutomaticSpeechRecognition pipeline (reloads if needed)
    """
    _WHISPER_SR = 16_000  # Whisper expects 16 kHz

    lang_kw: dict = {}
    if language and language != "auto":
        lang_kw = {"generate_kwargs": {"language": language}}

    # Pre-load audio as numpy array so Whisper never has to touch ffmpeg
    audio_input: Any
    try:
        audio_np, _ = _load_audio(audio_path, sr=_WHISPER_SR)
        audio_input = {"array": audio_np, "sampling_rate": _WHISPER_SR}
        log.info(f"[KVC] Pre-loaded audio for Whisper ({len(audio_np)//_WHISPER_SR}s @ {_WHISPER_SR}Hz)")
    except Exception as load_exc:
        log.warning(f"[KVC] Audio pre-load failed ({load_exc}), falling back to filename path")
        audio_input = audio_path  # last-resort: let the pipeline try

    # ── Strategy 1: reuse OmniVoice's internal Whisper pipeline ───────────────
    try:
        import model_manager as mm
        model = mm._model
        if model is not None:
            asr = getattr(model, "asr_model", None)
            if asr is not None:
                result = asr(audio_input, **lang_kw)
                text = result.get("text", "") if isinstance(result, dict) else str(result)
                if text.strip():
                    log.info(f"[KVC] Whisper (OmniVoice internal) transcript: '{text[:80]}'")
                    return text.strip()
    except Exception as e:
        log.warning(f"[KVC] OmniVoice asr_model access failed: {e}")

    # ── Strategy 2: standalone transformers pipeline ───────────────────────────
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        # Use GPU if available for significantly faster transcription
        dev_idx = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device=dev_idx,
            chunk_length_s=30,
            stride_length_s=5,
        )
        result = pipe(audio_input, **lang_kw)
        text = result["text"].strip() if isinstance(result, dict) else ""
        log.info(f"[KVC] Whisper (transformers) transcript: '{text[:80]}'")
        return text
    except Exception as exc:
        log.warning(f"[KVC] Whisper transcription failed: {exc}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
def _run_omnivoice_sync(model: Any, kwargs: dict[str, Any]) -> Any:
    """Blocking OmniVoice call — runs in executor with OOM safety."""
    return safe_generate(model, kwargs)


def _parse_omnivoice_output(out: Any, default_sr: int) -> tuple[np.ndarray, int]:
    """
    OmniVoice can return a list, dict, or raw tensor.
    Always returns (audio_np_float32, sample_rate).
    """
    sr = default_sr
    if isinstance(out, dict):
        audio = out.get("audio") or out.get("wav") or out.get("waveform")
        sr = out.get("sample_rate", out.get("sr", default_sr))
    elif isinstance(out, (list, tuple)):
        audio = out[0]
        if len(out) > 1 and isinstance(out[1], int):
            sr = out[1]
    else:
        audio = out

    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    if audio.ndim > 1:
        audio = audio.squeeze()
    return audio, int(sr)


# ── Concurrency semaphore for chunks ───────────────────────────────────────────
_chunk_semaphore = asyncio.Semaphore(1)  # Only 1 chunk on GPU at a time to be safe with VRAM


def _split_text_into_chunks(text: str, max_chars: int = 400) -> list[str]:
    """Split long text into natural chunks (sentences/phrases) for TTS."""
    if len(text) <= max_chars:
        return [text]
    
    # Split by common sentence/phrase delimiters
    delimiters = r"[.!?|;]\s+|(?<=\w)\s{2,}"
    raw_chunks = re.split(delimiters, text)
    
    chunks = []
    current = ""
    for c in raw_chunks:
        c = c.strip()
        if not c: continue
        if len(current) + len(c) < max_chars:
            current += (c + ". ")
        else:
            if current: chunks.append(current.strip())
            # If a single sentence is too long, hard cut it
            if len(c) > max_chars:
                for i in range(0, len(c), max_chars):
                    chunks.append(c[i : i + max_chars])
                current = ""
            else:
                current = c + ". "
    if current:
        chunks.append(current.strip())
    return chunks


async def _generate_chunk(
    model: Any, 
    text: str, 
    kwargs: dict, 
    idx: int, 
    total: int, 
    job_id: str
) -> tuple[int, np.ndarray, int]:
    """Generate a single chunk with VRAM safety and progress reporting."""
    async with _chunk_semaphore:
        log.info(f"[{job_id}] Generating chunk {idx+1}/{total} ({len(text)} chars)")
        set_job_progress(job_id, f"Synthesizing vocals (chunk {idx+1}/{total})…", 50 + int((idx/total) * 30))
        
        chunk_kwargs = kwargs.copy()
        chunk_kwargs["text"] = text
        
        loop = asyncio.get_running_loop()
        raw_out = await loop.run_in_executor(None, _run_omnivoice_sync, model, chunk_kwargs)
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
            
        audio, sr = _parse_omnivoice_output(raw_out, kwargs.get("sr", _DEFAULT_SR))
        return idx, audio, sr


# ══════════════════════════════════════════════════════════════════════════════
# MAIN KARAOKE SERVICE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
async def process_karaoke(
    *,
    job_id: str,
    ref_audio_path: str,
    karaoke_audio_path: str,
    gen_settings: GenerationSettings,
    audio_type: str = "full_song",      # full_song | vocals_only | karaoke_track | speech
    manual_lyrics: Optional[str] = None, # used when audio_type=karaoke_track
    pitch_shift_override: Optional[int] = None,
    match_pitch_auto: bool = True,
    mix_with_instrumental: bool = False,
    vocal_gain: float = 1.0,
    inst_gain: float = 0.80,
    language: str = "auto",
) -> dict[str, Any]:
    """
    Full karaoke voice conversion pipeline with chunked parallel generation.
    """
    t0 = time.perf_counter()

    # ── 0. Disk space check ──────────────────────────────────────────────────────
    if not check_disk_space(settings.disk_min_free_gb):
        return {
            "status": "error",
            "job_id": job_id,
            "message": f"Insufficient disk space (need {settings.disk_min_free_gb} GB free)",
        }

    # Mark running
    async with AsyncSessionLocal() as db:
        job = await db.get(GenerationJob, job_id)
        if job:
            job.status = "running"
            await db.commit()

    try:
        loop = asyncio.get_running_loop()

        # ── 1. Load audios ─────────────────────────────────────────────────────
        set_job_progress(job_id, "Loading audio files…", 5)
        log.info(f"[{job_id}] [KVC] Loading audio files...")
        karaoke_audio, karaoke_sr = await loop.run_in_executor(
            None, _load_audio, karaoke_audio_path, _DEFAULT_SR
        )
        ref_audio, ref_sr = await loop.run_in_executor(
            None, _load_audio, ref_audio_path, _DEFAULT_SR
        )

        karaoke_duration = len(karaoke_audio) / karaoke_sr
        log.info(f"[{job_id}] [KVC] audio_type={audio_type}  duration={karaoke_duration:.1f}s")

        # ── 2. Transcription — strategy depends on audio_type ────────────────────
        set_job_progress(job_id, "Transcribing lyrics…", 15)
        transcript_source = "whisper"

        if audio_type == "karaoke_track":
            if manual_lyrics and manual_lyrics.strip():
                transcript = manual_lyrics.strip()
                transcript_source = "manual"
            else:
                transcript = Path(karaoke_audio_path).stem.replace("_", " ").replace("-", " ")
                transcript_source = "filename"
        elif manual_lyrics and manual_lyrics.strip():
            transcript = manual_lyrics.strip()
            transcript_source = "manual"
        else:
            audio_hash = _compute_audio_hash(karaoke_audio_path)
            cached_transcript = await _get_cached_transcript(audio_hash)

            if cached_transcript:
                transcript = cached_transcript
                transcript_source = "cache"
            else:
                log.info(f"[{job_id}] [KVC] Transcribing with Whisper...")
                transcript = await loop.run_in_executor(
                    None, _transcribe_audio_sync, karaoke_audio_path, language
                )
                await _set_cached_transcript(audio_hash, transcript)

            if not transcript.strip():
                transcript = Path(karaoke_audio_path).stem.replace("_", " ").replace("-", " ")
                transcript_source = "filename"

        # ── 3. Pitch analysis ──────────────────────────────────────────────────
        set_job_progress(job_id, "Analyzing pitch…", 30)
        pitch_semitones = 0.0
        if pitch_shift_override is not None:
            pitch_semitones = float(pitch_shift_override)
        elif match_pitch_auto:
            ref_task = loop.run_in_executor(None, _estimate_pitch_semitones, ref_audio, ref_sr)
            kar_task = loop.run_in_executor(None, _estimate_pitch_semitones, karaoke_audio, karaoke_sr)
            ref_pitch, kar_pitch = await asyncio.gather(ref_task, kar_task)
            pitch_semitones = round(ref_pitch - kar_pitch, 1)

        # ── 4. HPSS separation ─────────────────────────────────────────────────
        instrumental: Optional[np.ndarray] = None
        if audio_type == "karaoke_track":
            instrumental = karaoke_audio
        elif mix_with_instrumental and audio_type == "full_song":
            set_job_progress(job_id, "Separating vocals from music…", 40)
            _, instrumental = await loop.run_in_executor(
                None, _separate_vocals, karaoke_audio, karaoke_sr
            )

        # ── 5. Chunked OmniVoice Generation ────────────────────────────────────
        set_job_progress(job_id, "Synthesizing new vocals…", 50)
        processed_text, instruct, resolved_emotion = prepare_omnivoice_text(
            transcript, gen_settings, job_id=job_id
        )
        
        # Split processed text into smaller chunks for faster diffusion
        text_chunks = _split_text_into_chunks(processed_text, max_chars=settings.max_text_length // 4)
        log.info(f"[{job_id}] [KVC] Text split into {len(text_chunks)} chunks")

        model = await get_model(gen_settings.model_dump())

        base_gen_kwargs: dict[str, Any] = {
            "ref_audio": ref_audio_path,
            "num_steps": gen_settings.steps,
            "cfg_scale": gen_settings.cfg,
            "temperature": gen_settings.temp,
            "sampler": gen_settings.sampler,
            "speed": gen_settings.speed,
            "similarity_weight": gen_settings.sim,
            "sr": gen_settings.sr,
        }
        total_pitch = int(round(pitch_semitones + gen_settings.pitch))
        total_pitch = max(-12, min(12, total_pitch))
        base_gen_kwargs["pitch_shift"] = total_pitch

        if gen_settings.ref_text:
            base_gen_kwargs["ref_text"] = gen_settings.ref_text
        if gen_settings.seed is not None:
            base_gen_kwargs["seed"] = gen_settings.seed
        if gen_settings.lang and gen_settings.lang != "auto":
            base_gen_kwargs["language"] = gen_settings.lang
        if instruct:
            base_gen_kwargs["instruct"] = instruct

        # Run chunks through GPU (sequential with semaphore, but logic is ready for parallel if we increase semaphore)
        chunk_tasks = [
            _generate_chunk(model, chunk_text, base_gen_kwargs, i, len(text_chunks), job_id)
            for i, chunk_text in enumerate(text_chunks)
        ]
        chunk_results = await asyncio.gather(*chunk_tasks)
        
        # Sort results by index and concatenate
        chunk_results.sort(key=lambda x: x[0])
        audio_chunks = [res[1] for res in chunk_results]
        out_sr = chunk_results[0][2]
        generated_audio = np.concatenate(audio_chunks)

        # ── 6. Time-stretch ────────────────────────────────────────────────────
        set_job_progress(job_id, "Adjusting timing…", 80)
        time_stretch_factor: Optional[float] = None
        gen_duration = len(generated_audio) / out_sr
        if karaoke_duration > 0 and gen_duration > 0:
            stretch_rate = gen_duration / karaoke_duration
            if 0.33 < stretch_rate < 3.0 and abs(stretch_rate - 1.0) > 0.08:
                time_stretch_factor = round(float(stretch_rate), 4)
                generated_audio = await loop.run_in_executor(
                    None, _time_stretch, generated_audio, stretch_rate
                )

        # ── 7. Mix ─────────────────────────────────────────────────────────────
        final_audio = generated_audio
        if mix_with_instrumental and instrumental is not None:
            set_job_progress(job_id, "Mixing tracks…", 90)
            if karaoke_sr != out_sr:
                import librosa
                instrumental = librosa.resample(instrumental, orig_sr=karaoke_sr, target_sr=out_sr)
            final_audio = await loop.run_in_executor(
                None, _mix_tracks, generated_audio, instrumental, vocal_gain, inst_gain
            )

        # ── 8. Save ────────────────────────────────────────────────────────────
        set_job_progress(job_id, "Saving result…", 95)
        filename, filepath = await loop.run_in_executor(
            None, _save_audio, final_audio, out_sr, "kvc"
        )

        # No temp file to cleanup anymore (we transcribe raw karaoke directly)

        elapsed = time.perf_counter() - t0
        snap = get_system_snapshot()
        gpu_mem = int(snap.get("gpu_mem_used_mb", 0))

        # ── 10. Persist to DB ──────────────────────────────────────────────────
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
                await db.commit()

        record_generation(success=True, elapsed=elapsed, gpu_mem_mb=gpu_mem)
        log.info(f"[{job_id}] [KVC] Done in {elapsed:.2f}s → {filename}")

        clear_job_tracking(job_id)

        # ── 11. Cleanup temporary intermediate files ──────────────────────────
        try:
            for p in [ref_audio_path, karaoke_audio_path]:
                path_obj = Path(p)
                # Only delete if it's one of our temp-generated WAVs in output dir
                if (path_obj.parent.resolve() == settings.output_dir.resolve() and 
                    path_obj.suffix.lower() == ".wav" and 
                    (path_obj.name.startswith("kvc_ref_") or path_obj.name.startswith("kvc_kar_"))):
                    path_obj.unlink(missing_ok=True)
                    log.debug(f"[KVC] Cleaned up temp file: {path_obj.name}")
        except Exception as e:
            log.warning(f"[KVC] Temp cleanup failed: {e}")

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Karaoke voice conversion complete",
            "audio_url": f"/outputs/{filename}",
            "filepath": filepath,
            "generation_time": round(elapsed, 2),
            "transcript": transcript,
            "transcript_source": transcript_source,
            "pitch_delta_semitones": pitch_semitones,
            "karaoke_duration_s": round(karaoke_duration, 2),
            "time_stretch_factor": time_stretch_factor,
            "metadata": {
                "sample_rate": out_sr,
                "gpu_mem_mb": gpu_mem,
                "mixed_with_instrumental": mix_with_instrumental,
                "style": gen_settings.style,
                "emotion": gen_settings.emotion,
            },
        }

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        clear_job_tracking(job_id)
        log.error(f"[{job_id}] [KVC] Failed: {exc}")
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

        return {
            "status": "error",
            "job_id": job_id,
            "message": f"Karaoke voice conversion failed: {exc}",
        }


# ── Job creation helper (reuses existing DB model) ────────────────────────────
async def create_kvc_job(
    *,
    ref_audio_path: str,
    karaoke_audio_path: str,
    gen_settings: GenerationSettings,
) -> str:
    """Create a pending DB row for a karaoke VC job."""
    job_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as db:
        job = GenerationJob(
            id=job_id,
            mode="karaoke",
            text=f"[KVC] ref={Path(ref_audio_path).name} kar={Path(karaoke_audio_path).name}",
            ref_audio_path=ref_audio_path,
            settings_json=json.dumps(gen_settings.model_dump()),
            status="pending",
        )
        db.add(job)
        await db.commit()
    return job_id
