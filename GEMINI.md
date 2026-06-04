# OmniVoice Studio Project Context

## Project Overview
**OmniVoice Studio** is a local, production-grade AI voice generation and voice conversion web application built on top of the `OmniVoice` model (`k2-fsa/omnivoice`). It runs via a **FastAPI backend** (Python) with a **single-page HTML/JS frontend** and is optimized for consumer NVIDIA GPUs (e.g., 6 GB VRAM) using 4-bit NF4 quantization.

## Tech Stack
- **Web framework:** FastAPI (async) + Uvicorn
- **Frontend:** Vanilla HTML, CSS, JavaScript (Single file: `AudioGenGUI.html`)
- **AI Models:** `k2-fsa/omnivoice` (TTS), `openai/whisper-large-v3-turbo` (ASR)
- **Audio Processing:** Multi-backend cascade (`torchaudio` → `librosa` → `pydub`), `librosa.pyin` (Pitch), `librosa.effects.hpss` (Vocal separation)
- **Model Loading:** `transformers`, `bitsandbytes` (4-bit NF4)
- **Database:** SQLite (`aiosqlite`, `sqlalchemy`)

## Directory Structure & Important Files
- `app.py`: FastAPI application entry point with all HTTP routes.
- `generation_service.py`: Core TTS generation pipeline.
- `karaoke_service.py`: Karaoke Voice Conversion (KVC) pipeline logic.
- `model_manager.py`: Singleton model loader (thread-safe, lazy-loaded, handles quantization/offloading).
- `schemas.py`: Pydantic request/response models.
- `AudioGenGUI.html`: The complete frontend interface.
- `OmniVoice_Studio_Report.md`: **Crucial reference document containing the full technical report, architecture, API routes, and known issues.** Read this file if deep context is required.

## Core Architectural Principles

1. **FFmpeg Independence for Audio Pipelines:**
   - Always try to use `torchaudio` or `soundfile`/`librosa` before falling back to `pydub` (which requires ffmpeg).
   - Whisper ASR runs on a pre-loaded numpy array (16kHz) to avoid internal ffmpeg subprocess calls.

2. **Non-Blocking Model Inference:**
   - Any `model.generate()` or heavy audio processing (e.g., librosa) MUST be run in a thread executor (`asyncio.get_event_loop().run_in_executor`) to prevent blocking the async event loop.

3. **Offline-First Capabilities:**
   - The project relies heavily on `.env` variables (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`) to prevent unexpected internet lookups from huggingface libraries during startup.

4. **Resource Management:**
   - Memory is cleared effectively (`torch.cuda.empty_cache()`, `gc.collect()`) after inference to keep the app viable on 6GB VRAM hardware.

## How to use this project with AI
- For extending the project, refer to the Pydantic schemas in `schemas.py` and ensure the frontend payload in `AudioGenGUI.html` matches.
- For new audio processing, ensure you test across different file formats (.wav, .mp3, .m4a) using the existing cascade in `karaoke_service.py` or `generation_service.py`.
