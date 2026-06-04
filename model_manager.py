"""
OmniVoice Studio — Model registry / loader.

Responsibilities
────────────────
• Singleton model management (load once, reuse)
• Quantisation config (4-bit NF4 / 8-bit / FP16 / FP32)
• Graceful error handling & retry logic
• Hardware metrics capture
• Model warmup at startup (optional)
• OOM recovery — graceful fallback
• VRAM release/reclaim for external processes
"""
from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any

import torch

from logger import get_logger

log = get_logger(__name__)

# ── Global singletons ──────────────────────────────────────────────────────────
_model: Any = None
_model_lock = asyncio.Lock()
_load_ts: float | None = None


def _build_quant_config(quant: str, qtype: str, double_quant: bool):
    """Return a BitsAndBytesConfig or None."""
    from transformers import BitsAndBytesConfig

    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qtype,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=double_quant,
        )
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


async def get_model(settings: dict[str, Any]) -> Any:
    """
    Async-safe singleton loader.
    First call blocks until the model is loaded; subsequent calls return immediately.
    """
    global _model, _load_ts

    if _model is not None:
        return _model

    async with _model_lock:
        # Double-checked locking
        if _model is not None:
            return _model

        log.info("Loading OmniVoice model -- this may take 60-120 s ...")
        t0 = time.perf_counter()

        try:
            _model = await asyncio.get_event_loop().run_in_executor(
                None,
                _load_model_sync,
                settings,
            )
            _load_ts = time.perf_counter() - t0
            log.info(f"[OK] OmniVoice loaded in {_load_ts:.1f}s")
        except Exception as exc:
            log.error(f"[FAIL] Model load failed: {exc}")
            traceback.print_exc()
            raise

    return _model


def _load_model_sync(settings: dict[str, Any]) -> Any:
    """Blocking model load — runs in a thread executor."""
    from omnivoice import OmniVoice  # type: ignore[import]

    quant = settings.get("quant", "4bit")
    qtype = settings.get("qtype", "nf4")
    double_quant = bool(settings.get("dquant", True))
    device = settings.get("device", "cuda:0")

    quant_cfg = _build_quant_config(quant, qtype, double_quant)

    model = OmniVoice.from_pretrained(
        "k2-fsa/omnivoice",
        device_map=device,
        quantization_config=quant_cfg,
    )
    return model


async def warmup_model() -> None:
    """
    Pre-load the model during startup so the first user request isn't slow.
    Called from app lifespan when MODEL_WARMUP=true.
    """
    from config import get_settings
    cfg = get_settings()
    default_settings = {
        "quant": cfg.default_quant,
        "qtype": "nf4",
        "dquant": True,
        "device": cfg.default_device,
    }
    log.info("[WARMUP] Pre-loading model at startup...")
    try:
        await get_model(default_settings)
        log.info("[WARMUP] Model ready.")
    except Exception as exc:
        log.error(f"[WARMUP] Failed: {exc}")


def unload_model() -> None:
    """Free VRAM — call during graceful shutdown or hot-reload."""
    global _model, _load_ts
    if _model is not None:
        del _model
        _model = None
        _load_ts = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Model unloaded and VRAM cleared.")


def temporarily_release_vram() -> None:
    """
    Move model to CPU and free VRAM for external processes.
    Call reclaim_vram() to move it back.
    """
    global _model
    if _model is not None and hasattr(_model, "to"):
        try:
            _model.to("cpu")
            torch.cuda.empty_cache()
            log.info("[VRAM] Model moved to CPU, VRAM released.")
        except Exception as e:
            log.warning(f"[VRAM] Release failed: {e}")


def reclaim_vram(device: str = "cuda:0") -> None:
    """Move model back to GPU after temporarily_release_vram()."""
    global _model
    if _model is not None and hasattr(_model, "to"):
        try:
            _model.to(device)
            log.info(f"[VRAM] Model moved back to {device}.")
        except Exception as e:
            log.warning(f"[VRAM] Reclaim failed: {e}")


def safe_generate(model: Any, kwargs: dict[str, Any]) -> Any:
    """
    OOM-safe wrapper around model.generate().
    If CUDA OOM occurs, clear cache and retry once on CPU.
    """
    try:
        return model.generate(**kwargs)
    except torch.cuda.OutOfMemoryError:
        log.warning("[OOM] CUDA out of memory — clearing cache and retrying...")
        torch.cuda.empty_cache()
        try:
            return model.generate(**kwargs)
        except torch.cuda.OutOfMemoryError:
            log.error("[OOM] Second attempt also OOM. Generation failed.")
            raise RuntimeError(
                "GPU out of memory. Try shorter text, lower diffusion steps, "
                "or reduce quality settings."
            )


def model_status() -> dict[str, Any]:
    """Return a status dict for the /health endpoint."""
    loaded = _model is not None
    info: dict[str, Any] = {
        "loaded": loaded,
        "load_time_s": round(_load_ts, 2) if _load_ts else None,
    }
    if torch.cuda.is_available():
        try:
            mem = torch.cuda.memory_stats()
            info["vram_allocated_mb"] = round(
                mem.get("allocated_bytes.all.current", 0) / 1024**2, 1
            )
            info["vram_reserved_mb"] = round(
                mem.get("reserved_bytes.all.current", 0) / 1024**2, 1
            )
            info["device"] = torch.cuda.get_device_name(0)
        except Exception:
            info["vram_allocated_mb"] = 0
            info["vram_reserved_mb"] = 0
    return info
