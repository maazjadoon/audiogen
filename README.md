# 🎙️ OmniVoice Studio — AI Voice Cloning & Audio Generation

> A **production-grade, fully local** voice cloning and AI audio generation system. Clone any voice from 5–10 seconds of audio, generate speech, and perform karaoke vocal conversion — all running on your own hardware with zero cloud dependency.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Async%20API-009688?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)
![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-orange?style=flat-square&logo=prometheus)
![SQLite](https://img.shields.io/badge/SQLite-Job%20Store-lightblue?style=flat-square)

---

## 🚀 Features

- **🎤 Voice Cloning** — Clone a target voice from just 5–10 seconds of reference audio using speaker embedding transfer
- **🎵 Karaoke Vocal Conversion (KVC)** — Convert vocals in karaoke tracks to a cloned voice
- **⚡ Async Job Queue** — Submit jobs and poll for progress; non-blocking processing
- **📊 Prometheus Metrics** — Real-time system and model performance monitoring
- **🌐 Web GUI** — Built-in HTML frontend for easy interaction
- **🐳 Docker Support** — One-command deployment
- **💾 Job History** — Paginated job history with full status tracking

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│               OmniVoice Studio API                  │
│                  (FastAPI + Uvicorn)                 │
└──────────┬────────────────────────────┬─────────────┘
           │                            │
    ┌──────▼──────┐            ┌────────▼────────┐
    │ Generation  │            │    Karaoke VC   │
    │  Service    │            │    Service      │
    │ (TTS/Clone) │            │  (KVC Pipeline) │
    └──────┬──────┘            └────────┬────────┘
           │                            │
    ┌──────▼────────────────────────────▼──────┐
    │           Model Manager                  │
    │     (Speaker Embeddings + TTS Model)     │
    └──────────────────────┬───────────────────┘
                           │
                    ┌──────▼──────┐
                    │  SQLite DB  │
                    │ (Job Store) │
                    └─────────────┘
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v2/generate` | Sync voice generation |
| `POST` | `/api/v2/generate/async` | Submit async job (JSON body) |
| `POST` | `/api/v2/generate/async/form` | Async multipart upload (GUI) |
| `GET` | `/api/v2/jobs/{job_id}` | Poll job status + progress |
| `POST` | `/api/v2/jobs/{job_id}/cancel` | Cancel a running job |
| `GET` | `/api/v2/jobs/batch` | Batch status for multiple jobs |
| `GET` | `/api/v2/history` | Paginated job history |
| `GET` | `/api/v2/health` | Model + system health check |
| `GET` | `/metrics` | Prometheus scrape endpoint |
| `GET` | `/` | Serve HTML frontend |

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- ~4–8 GB disk space for models

### Option A: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/maazjadoon/audiogen.git
cd audiogen

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Run the server
python app.py
# or on Windows:
run.bat
```

Open `http://localhost:8000` in your browser.

### Option B: Run with Docker

```bash
docker compose up --build
```

---

## ⚙️ Configuration (`.env`)

```env
# Server
HOST=0.0.0.0
PORT=8000

# Model settings
DEVICE=cuda          # or 'cpu'
MAX_CONCURRENT_JOBS=2

# Storage
OUTPUT_DIR=omnivoice_outputs
DB_PATH=omnivoice.db
```

---

## 📦 Project Structure

```
audiogen/
├── app.py                  # FastAPI application entry-point
├── generation_service.py   # Core TTS + voice cloning logic
├── karaoke_service.py      # Karaoke vocal conversion pipeline
├── model_manager.py        # Model loading + speaker embeddings
├── database.py             # SQLite job store (async)
├── schemas.py              # Pydantic request/response models
├── config.py               # App configuration
├── logger.py               # Structured logging
├── metrics.py              # Prometheus metrics definitions
├── AudioGenGUI.html        # Frontend web interface
├── monitoring/
│   └── prometheus.yml      # Prometheus scrape config
├── tests/
│   └── test_omnivoice.py   # Test suite
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 👤 Author

**Muhammad Maaz Jadoon**
- GitHub: [@maazjadoon](https://github.com/maazjadoon)
- Email: mohazjadoon@gmail.com
- University: PAF-IAST, B.S. Artificial Intelligence (2023–2027)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
