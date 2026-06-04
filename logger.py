"""
OmniVoice Studio — Structured logging setup.
Uses Python's standard logging with a rich handler in dev
and JSON-formatted lines in production.

Supports:
  • Rich console output (dev mode)
  • JSON structured logging (production)
  • Rotating file handler (log_file + log_max_bytes + log_backup_count in .env)
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from typing import Any

# ── Try rich for pretty dev output ────────────────────────────────────────────
try:
    from rich.logging import RichHandler
    _RICH = True
except ImportError:
    _RICH = False


class _JsonFormatter(logging.Formatter):
    """Single-line JSON log records for production / log aggregators."""

    import json as _json

    def format(self, record: logging.LogRecord) -> str:
        import json, traceback
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = traceback.format_exception(*record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)  # type: ignore[arg-type]
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    debug: bool = False,
    log_file: str = "",
    log_max_bytes: int = 10_485_760,
    log_backup_count: int = 5,
) -> None:
    """Configure root logger — call once at startup."""
    level = logging.DEBUG if debug else logging.INFO

    handlers: list[logging.Handler] = []

    # ── Console handler ────────────────────────────────────────────────────────
    if debug and _RICH:
        console_handler: logging.Handler = RichHandler(
            rich_tracebacks=True, markup=True, show_path=False
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_JsonFormatter())
    handlers.append(console_handler)

    # ── Rotating file handler (optional) ──────────────────────────────────────
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(_JsonFormatter())
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
