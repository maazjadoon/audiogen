"""
OmniVoice Studio — Production Configuration
Centralised settings loaded from environment / .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── HuggingFace ───────────────────────────────────────────────────────────
    hf_token: str = ""
    hf_home: str = "./hf_cache"
    hf_hub_offline: int = 0
    transformers_offline: int = 0

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: Path = Path("./omnivoice_outputs")
    db_url: str = "sqlite+aiosqlite:///./omnivoice.db"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = ""          # empty = no Redis cache

    # ── Security ──────────────────────────────────────────────────────────────
    secret_key: str = "change-me-in-production"
    api_key_enabled: bool = False
    api_key: str = ""

    # ── Model defaults ────────────────────────────────────────────────────────
    default_quant: Literal["fp32", "fp16", "8bit", "4bit"] = "4bit"
    default_device: str = "cuda:0"
    model_warmup: bool = False        # preload model at startup (adds 30-60s startup)

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit: int = 20              # requests / minute / IP

    # ── Concurrency ───────────────────────────────────────────────────────────
    max_concurrent_jobs: int = 2      # max parallel generation jobs
    generation_timeout_s: int = 300   # per-job timeout (seconds)

    # ── Text limits ───────────────────────────────────────────────────────────
    max_text_length: int = 2000       # hard character limit per request

    # ── Storage limits ────────────────────────────────────────────────────────
    disk_min_free_gb: float = 2.0     # minimum free disk space before rejecting jobs

    # ── Webhook ───────────────────────────────────────────────────────────────
    webhook_timeout_s: int = 10       # seconds to wait for webhook HTTP response

    # ── Log rotation ──────────────────────────────────────────────────────────
    log_file: str = ""                # write JSON logs to file if set
    log_max_bytes: int = 10_485_760   # 10 MB per log file
    log_backup_count: int = 5         # keep 5 rotated log files

    # ── Observability ─────────────────────────────────────────────────────────
    metrics_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    """Singleton settings — call get_settings() everywhere."""
    cfg = Settings()
    # Apply HF env vars immediately so every import afterwards picks them up
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ["HF_HOME"] = cfg.hf_home
    if cfg.hf_token:
        os.environ["HF_TOKEN"] = cfg.hf_token
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg
