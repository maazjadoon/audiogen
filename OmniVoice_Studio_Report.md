# OmniVoice Studio — Technical Report & Software Description

> **Purpose of this document:** A complete reference for any AI assistant or developer who needs to understand, extend, debug, or integrate with OmniVoice Studio. All file names, API routes, data flows, and design decisions are documented here.

---

## 1. What Is OmniVoice Studio?

**OmniVoice Studio** is a local, production-grade AI voice generation and voice conversion web application built on top of the `OmniVoice` model (`k2-fsa/omnivoice` on HuggingFace).

It runs as a **FastAPI backend** (Python) with a **single-page HTML/JS frontend**. Everything runs locally on the user's Windows machine with an NVIDIA GPU (optimized for 6 GB VRAM using 4-bit NF4 quantization).

### Core Capabilities

| Feature | Description |
|---|---|
| **Voice Cloning (TTS)** | Upload 3–10s of a person's voice → generate any text in that voice |
| **Voice Design** | Describe a voice in text (e.g., "female, young, British accent") → generate speech |
| **Auto Voice** | Model selects a voice internally from its prior — no reference needed |
| **Karaoke Voice Conversion (KVC)** | Upload voice A + audio track B → re-perform track B entirely in voice A |

---

## 2. Project File Structure

```
d:\Labs\ANN\audiogen\
│
├── app.py                  # FastAPI application entry point — all HTTP routes
├── generation_service.py   # Core TTS pipeline (build kwargs → run → save → DB)
├── karaoke_service.py      # Karaoke Voice Conversion pipeline (full 8-step flow)
├── model_manager.py        # Singleton OmniVoice model loader with quantization
├── schemas.py              # Pydantic v2 request/response models
├── config.py               # Settings loaded from .env (pydantic-settings)
├── database.py             # SQLAlchemy async DB (SQLite) — job persistence
├── logger.py               # Structured JSON logger
├── metrics.py              # Prometheus metrics + system snapshot
│
├── AudioGenGUI.html        # Complete single-file frontend (HTML + CSS + JS)
├── requirements.txt        # All Python dependencies
├── .env                    # Runtime configuration (DO NOT commit)
├── .env.example            # Template for .env
│
├── omnivoice_outputs/      # All generated audio files saved here
├── hf_cache/               # HuggingFace model cache (offline mode)
├── omnivoice.db            # SQLite database (job history)
│
├── Dockerfile              # Docker support
├── docker-compose.yml      # Multi-service compose (app + prometheus + grafana)
├── monitoring/             # Prometheus/Grafana configs
└── tests/                  # Pytest test suite
```

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| **Web framework** | FastAPI (async) with Uvicorn ASGI server |
| **AI model** | `k2-fsa/omnivoice` (OmniVoice diffusion TTS) |
| **Model loading** | `transformers` + `bitsandbytes` (4-bit NF4 quantization) |
| **ASR / Transcription** | `openai/whisper-large-v3-turbo` (via transformers pipeline) |
| **Audio loading** | Multi-backend cascade: `torchaudio` → `librosa` → `pydub` |
| **Pitch analysis** | `librosa.pyin` (probabilistic YIN F0 estimator) |
| **Vocal separation** | `librosa.effects.hpss` (Harmonic-Percussive Source Separation) |
| **Database** | SQLite via `sqlalchemy` + `aiosqlite` (async) |
| **Metrics** | Prometheus (custom counters + system snapshot via `psutil`) |
| **Config** | `pydantic-settings` reading from `.env` file |
| **Frontend** | Vanilla HTML + CSS + JavaScript (single file: `AudioGenGUI.html`) |
| **Caching** | Optional Redis (`redis-py` async) |

---

## 4. Configuration (`.env`)

The app is configured entirely through environment variables, loaded from `.env`:

```ini
# HuggingFace — model cache location and auth token
HF_TOKEN=hf_XXXX           # optional, needed only for private/gated models
HF_HOME=./hf_cache         # local cache directory

# Offline mode — prevents HF from checking the internet on every startup
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1
DEBUG=false

# Paths
OUTPUT_DIR=./omnivoice_outputs
DB_URL=sqlite+aiosqlite:///./omnivoice.db

# Redis (optional caching — leave blank to disable)
REDIS_URL=redis://localhost:6379/0

# Model defaults
DEFAULT_QUANT=4bit          # fp32 | fp16 | 8bit | 4bit
DEFAULT_DEVICE=cuda:0
```

---

## 5. API Routes (FastAPI)

All routes are served from `app.py`. Base URL: `http://localhost:8000`

### TTS / Voice Generation

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v2/generate` | **Synchronous TTS** — waits for result, returns audio URL |
| `POST` | `/api/v2/generate/async` | **Async TTS** — returns `job_id` immediately |
| `POST` | `/api/v2/generate/batch` | **Batch TTS** — up to 10 items, all async |

**Request format** for `/api/v2/generate` (multipart/form-data):
```
text           = "Hello world"           # Required — text to synthesize
settings       = {...}                   # JSON string of GenerationSettings
ref_audio      = <file>                  # Optional — reference audio file (any format)
```

The `ref_audio` file is automatically **converted to WAV** before being passed to OmniVoice (torchaudio → pydub fallback chain), so AAC, MP3, M4A, FLAC, etc. all work.

**`GenerationSettings` (JSON):**
```json
{
  "quant": "4bit",          // fp32 | fp16 | 8bit | 4bit
  "qtype": "nf4",           // nf4 | fp4 (4-bit quantization type)
  "dquant": true,           // double quantization (saves ~0.4 GB)
  "device": "cuda:0",       // target device
  "steps": 50,              // diffusion steps (10–200)
  "cfg": 3.0,               // CFG scale (0.5–10.0)
  "temp": 1.0,              // temperature (0.1–2.0)
  "sampler": "dpmsolver++", // dpmsolver++ | euler | ddim
  "speed": 1.0,             // speech speed (0.5–2.0x)
  "pitch": 0,               // pitch shift in semitones (-12 to +12)
  "sim": 0.8,               // speaker similarity weight (0–1)
  "duration": null,         // fixed output duration in seconds (optional)
  "seed": null,             // random seed for reproducibility
  "sr": 24000,              // output sample rate
  "prefix": "clone",        // filename prefix for output files
  "ref_text": "",           // what the reference audio says (optional, for Whisper skip)
  "lang": "auto",           // output language hint
  "mode": "clone",          // clone | design | auto
  "instruct": ""            // voice design instruction string
}
```

### Karaoke Voice Conversion

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v2/karaoke` | **Synchronous KVC** — waits for result |
| `POST` | `/api/v2/karaoke/async` | **Async KVC** — returns `job_id` |

**Request format** for `/api/v2/karaoke` (multipart/form-data):
```
ref_audio             = <file>           # Voice identity (3–10s)
karaoke_audio         = <file>           # Source audio to re-perform
audio_type            = "full_song"      # full_song | vocals_only | karaoke_track | speech
manual_lyrics         = "..."            # Optional — overrides Whisper for ALL audio types
match_pitch_auto      = true             # Auto pitch alignment
pitch_shift_override  = null             # Manual semitone shift (overrides auto)
mix_with_instrumental = false            # Mix output with separated BG music
vocal_gain            = 1.0             # Vocal level in mix
inst_gain             = 0.80            # Instrumental level in mix
language              = "auto"           # Whisper language hint
settings_json         = {...}            # GenerationSettings JSON
```

Both uploaded audio files (ref + karaoke) are **converted to WAV** before any processing.

### Job Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v2/jobs/{job_id}` | Poll job status |
| `GET` | `/api/v2/history` | Paginated job history |
| `DELETE` | `/api/v2/history/{job_id}` | Delete a job + its audio file |
| `GET` | `/api/v2/history/export/csv` | Export history as CSV |

### System

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v2/health` | Model + system health check |
| `GET` | `/metrics` | Prometheus scrape endpoint |
| `GET` | `/outputs/{filename}` | Serve generated audio files |
| `GET` | `/` | Serve the HTML frontend |

---

## 6. TTS Generation Pipeline (`generation_service.py`)

```
POST /api/v2/generate
        │
        ▼
1. Parse request (JSON or multipart)
2. Save ref_audio to disk (if provided)
3. Convert ref_audio → WAV  [torchaudio → pydub fallback]
4. Create DB job record (status = "pending")
5. Build model.generate() kwargs
6. get_model() — load OmniVoice singleton (lazy, thread-safe)
7. run_in_executor → model.generate(**kwargs)  [blocking, off async loop]
8. Parse output tensor → numpy float32
9. Peak-normalise audio
10. Write WAV to omnivoice_outputs/ (PCM 24-bit)
11. Update DB job (status = "done", audio_url, filepath, timing)
12. Record Prometheus metrics
13. Return JSON response with audio_url
```

**Output files** are named: `{prefix}_{unix_timestamp}_{uid6}.wav`  
Example: `clone_1777696661_31cb5e.wav`

---

## 7. Karaoke Voice Conversion Pipeline (`karaoke_service.py`)

The KVC pipeline has 8 steps:

```
POST /api/v2/karaoke
        │
        ▼
1. UPLOAD & CONVERT
   - Save ref_audio + karaoke_audio
   - Both converted to WAV (torchaudio → pydub)

2. LOAD AUDIO
   - _load_audio() cascade:
     a. librosa/soundfile  (WAV, FLAC, OGG — no ffmpeg)
     b. torchaudio         (MP3, AAC, M4A via Windows MediaFoundation)
     c. pydub              (fallback, needs ffmpeg)
   - Both loaded to mono float32 numpy array at 24kHz

3. TRANSCRIPTION (skipped if manual_lyrics provided)
   - _transcribe_audio_sync():
     a. Pre-load audio as numpy array at 16kHz (avoids ffmpeg dependency)
     b. Pass {"array": ..., "sampling_rate": 16000} to Whisper pipeline
     c. Strategy 1: Reuse OmniVoice's internal asr_model (zero extra VRAM)
     d. Strategy 2: Standalone transformers Whisper pipeline
   - If manual_lyrics is provided → skip Whisper entirely, use the text as-is

4. PITCH ANALYSIS
   - librosa.pyin() on both ref_audio and karaoke_audio
   - Compute median F0 in semitones (MIDI scale)
   - delta = ref_pitch - karaoke_pitch → applied as pitch_shift

5. VOCAL SEPARATION (only for full_song + mix_with_instrumental)
   - librosa.effects.hpss() → (harmonic ≈ vocals, percussive+residual ≈ instrumental)

6. OMNIVOICE GENERATION
   - kwargs: text=transcript, ref_audio=ref_wav_path, pitch_shift=total_pitch, ...
   - Runs in thread executor (non-blocking)
   - VRAM cleared after generation

7. TIME STRETCH
   - librosa.effects.time_stretch() to match original karaoke duration
   - Applied if |stretch_rate - 1.0| > 5% and stretch_rate is in (0.33, 3.0)

8. MIX & SAVE
   - Optionally mix generated vocals with separated instrumental
   - Peak normalise to 0.92
   - Write WAV to omnivoice_outputs/ as kvc_{timestamp}_{uid}.wav
   - Update DB + return response
```

### Audio Type Modes

| `audio_type` | Transcription | HPSS | Notes |
|---|---|---|---|
| `full_song` | Whisper on raw file | Yes (if mixing) | Vocals + music — Whisper transcribes through music |
| `vocals_only` | Whisper on raw file | No | Clean isolated vocals |
| `karaoke_track` | **None** — manual_lyrics **required** | No (whole file IS instrumental) | No vocals to transcribe |
| `speech` | Whisper on raw file | No | Spoken audio, dialogue |

---

## 8. Model Management (`model_manager.py`)

- **Singleton pattern** with `asyncio.Lock` (double-checked locking)
- Model is loaded **lazily** on first generation request (not at startup)
- Blocking `OmniVoice.from_pretrained()` runs in a **thread executor** to avoid blocking the async event loop
- Supports quantization via `bitsandbytes`:
  - `4bit` NF4 (default, ~4.2 GB VRAM) — recommended for 6 GB GPUs
  - `4bit` FP4
  - `8bit` (~7.5 GB VRAM)
  - `fp16` (~14 GB VRAM)
  - `fp32` (~16 GB VRAM)
- `unload_model()` frees VRAM and clears CUDA cache — called on server shutdown

---

## 9. Database Schema (`database.py`)

SQLite via `aiosqlite` + SQLAlchemy async. Single table: `generation_jobs`

| Column | Type | Description |
|---|---|---|
| `id` | UUID (str) | Job ID |
| `mode` | str | clone / design / auto / karaoke |
| `text` | str | Input text |
| `ref_audio_path` | str | Path to saved reference audio |
| `settings_json` | str | JSON dump of GenerationSettings |
| `status` | str | pending → running → done / error |
| `audio_url` | str | `/outputs/{filename}` URL |
| `filepath` | str | Absolute disk path of output WAV |
| `generation_time` | float | Seconds to generate |
| `gpu_mem_mb` | int | VRAM used at completion |
| `cpu_pct` | float | CPU usage at completion |
| `error_message` | str | Error string if failed |
| `created_at` | datetime | Job creation timestamp |
| `finished_at` | datetime | Job completion timestamp |

---

## 10. Frontend (`AudioGenGUI.html`)

A **single HTML file** (~1800 lines) containing all UI, CSS, and JavaScript. No build step required — served directly by FastAPI.

### Navigation Panels

| Panel ID | Label | Purpose |
|---|---|---|
| `panel-clone` | Voice cloning | Upload ref audio + text → TTS |
| `panel-design` | Voice design | Describe voice in text → TTS |
| `panel-auto` | Auto voice | Model picks voice → TTS |
| `panel-karaoke` | Karaoke VC | Two-audio conversion pipeline |
| `panel-hw` | Model & hardware | Quantization, VRAM settings |
| `panel-adv` | Advanced params | Diffusion, prosody, speaker controls |
| `panel-out` | Output & export | Format, sample rate, post-processing |
| `panel-codegen` | Code generator | Auto-generates Python script from settings |
| `panel-history` | History | Paginated generation history |

### Key JavaScript Functions

| Function | Purpose |
|---|---|
| `runGen()` | Submits TTS request via `POST /api/v2/generate` |
| `runKVC()` | Submits KVC request via `POST /api/v2/karaoke` |
| `kvcSetType(t)` | Updates audio type card styling + shows/hides lyrics hint |
| `pollHealth()` | Polls `/api/v2/health` every 15s for status bar |
| `loadAPIHistory()` | Fetches recent done jobs from `/api/v2/history` on load |
| `updateCode()` | Regenerates the Python code snippet in the Code generator panel |
| `readSettings()` | Reads all UI inputs into global `S` settings object |

### Global State (`S` object)
```js
var S = {
    device, quant, qtype, dquant, offload, vramFrac,
    steps, cfg, temp, sampler, speed, pitch, duration, sim, seed,
    fmt, sr, bd, normalize, trimSilence, clearCache,
    outdir, prefix, refText, instruct, lang
};
```

---

## 11. Audio Loading Architecture (No ffmpeg Required)

A critical design decision: **ffmpeg is not required** for normal operation on Windows.

The multi-backend cascade used everywhere:

```
Input file (any format: WAV, MP3, AAC, M4A, FLAC, OGG, ...)
        │
        ▼ Backend 1: librosa/soundfile
        │   Handles: WAV, FLAC, OGG natively (no ffmpeg)
        │   Fails for: MP3, AAC, M4A
        │
        ▼ Backend 2: torchaudio
        │   Handles: MP3, AAC, M4A, MP4 via Windows MediaFoundation
        │   CPU-only, no VRAM used
        │   Resampled with scipy.signal.resample_poly
        │
        ▼ Backend 3: pydub (fallback)
            Handles: everything else
            Requires ffmpeg in PATH
            Used only if both above fail
```

For **Whisper transcription** specifically, audio is pre-loaded to a numpy array at 16 kHz and passed as `{"array": ..., "sampling_rate": 16000}` — this bypasses Whisper's internal ffmpeg decoder entirely.

---

## 12. Known Issues & Resolutions

| Issue | Root Cause | Resolution |
|---|---|---|
| `soundfile.LibsndfileError: Format not recognised` for `.aac` | soundfile doesn't support AAC | Convert to WAV using torchaudio before passing to OmniVoice |
| `ffmpeg was not found but is required to load audio files` from Whisper | HF Whisper pipeline decodes from filename using ffmpeg | Pre-load audio as numpy array, pass dict instead of filename |
| `NameError: name 'asyncio' is not defined` | `asyncio` was imported inline as `_asyncio` but used as `asyncio` | Added `import asyncio` at module top level |
| `[Errno 10048] Only one usage of each socket address` | Old server process still holding port 8000 | Kill old process: `Stop-Process -Name python -Force` |
| HuggingFace network requests on every startup | HF Hub checks for model updates online | Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` in `.env` |
| Whisper transcript = `हाँ, हाँ, हाँ...` for music | Whisper picks up repeated syllable from music background | User can provide `manual_lyrics` to override Whisper |

---

## 13. Running the Application

### Prerequisites
- Python 3.10
- NVIDIA GPU with CUDA (6 GB+ VRAM recommended)
- Virtual environment at `.venv/`

### Start Server
```powershell
# From d:\Labs\ANN\audiogen\
& .venv\Scripts\python.exe app.py
```

Server starts at: `http://localhost:8000`  
Frontend UI: `http://localhost:8000/`  
API docs: `http://localhost:8000/docs`  
Prometheus metrics: `http://localhost:8000/metrics`

### Stop Server
```powershell
Stop-Process -Name python -Force
```

---

## 14. Generation Settings — Quick Reference

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `steps` | 50 | 10–200 | Diffusion steps — higher = better quality, slower |
| `cfg` | 3.0 | 0.5–10.0 | Classifier-free guidance — higher = more text-adherent |
| `temp` | 1.0 | 0.1–2.0 | Sampling temperature — higher = more varied/creative |
| `sampler` | `dpmsolver++` | 3 options | Diffusion sampler algorithm |
| `speed` | 1.0 | 0.5–2.0x | Speech rate |
| `pitch` | 0 | -12 to +12 | Pitch shift in semitones |
| `sim` | 0.8 | 0–1 | Speaker similarity weight |
| `quant` | `4bit` | 4 options | Model quantization — `4bit` fits 6 GB GPU |

---

## 15. Output Files

All output WAVs saved to `omnivoice_outputs/`:

| Pattern | Source |
|---|---|
| `clone_{ts}_{uid}.wav` | TTS voice cloning |
| `kvc_{ts}_{uid}.wav` | Karaoke voice conversion |
| `ref_{ts}_{original_filename}` | Saved reference audio (original format) |
| `ref_{ts}.wav` | Converted WAV version of reference audio |
| `kvc_ref_{ts}.wav` | Converted WAV of KVC reference audio |
| `kvc_kar_{ts}.wav` | Converted WAV of KVC karaoke audio |

---

*Report generated: 2026-05-02 | Project path: `d:\Labs\ANN\audiogen\` | Version: 2.0.0*
