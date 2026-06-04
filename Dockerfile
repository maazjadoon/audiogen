# ═══════════════════════════════════════════════════════════════════════════════
# OmniVoice Studio — Production Dockerfile
# Multi-stage: builder → runtime
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: builder ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for audio libs and BitsAndBytes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source
COPY . .

# Create non-root user
RUN useradd -m -u 1001 omnivoice && \
    mkdir -p /app/omnivoice_outputs /app/hf_cache && \
    chown -R omnivoice:omnivoice /app

USER omnivoice

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v2/health || exit 1

# Launch with Uvicorn
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
