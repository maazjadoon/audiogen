"""
OmniVoice Studio — System metrics collector.
Exposes Prometheus metrics and a /metrics/snapshot JSON endpoint.

Enhancements:
  • Per-step timing histograms (whisper, pitch, hpss, generation, stretch, mix)
  • Active jobs gauge for backpressure monitoring
  • Disk space gauge for storage alerting
  • Audio minutes counter for cost analytics
"""
from __future__ import annotations

import shutil
import time
from typing import Any

import psutil

try:
    import GPUtil
    _GPU = True
except ImportError:
    _GPU = False

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    _PROM = True
    _registry = CollectorRegistry(auto_describe=True)

    REQ_TOTAL = Counter(
        "omnivoice_requests_total", "Total generation requests",
        ["status"], registry=_registry
    )
    GEN_LATENCY = Histogram(
        "omnivoice_generation_seconds", "Audio generation latency",
        buckets=[1, 2, 5, 10, 20, 30, 60, 120],
        registry=_registry,
    )
    GPU_MEM = Gauge(
        "omnivoice_gpu_mem_mb", "GPU memory allocated (MB)",
        registry=_registry
    )
    CPU_USAGE = Gauge(
        "omnivoice_cpu_pct", "CPU utilisation %",
        registry=_registry
    )
    RAM_USAGE = Gauge(
        "omnivoice_ram_mb", "RAM used (MB)",
        registry=_registry
    )

    # ── NEW: Per-step timing ──────────────────────────────────────────────────
    STEP_LATENCY = Histogram(
        "omnivoice_step_seconds", "Per-pipeline-step latency",
        ["step"],   # whisper, pitch, hpss, generation, stretch, mix, save
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
        registry=_registry,
    )

    # ── NEW: Active jobs (for backpressure) ───────────────────────────────────
    ACTIVE_JOBS = Gauge(
        "omnivoice_active_jobs", "Currently running generation jobs",
        registry=_registry
    )

    # ── NEW: Disk space ───────────────────────────────────────────────────────
    DISK_FREE_GB = Gauge(
        "omnivoice_disk_free_gb", "Free disk space on output volume (GB)",
        registry=_registry
    )

    # ── NEW: Audio minutes generated (cost analytics) ─────────────────────────
    AUDIO_MINUTES = Counter(
        "omnivoice_audio_minutes_total", "Total audio minutes generated",
        registry=_registry
    )

except ImportError:
    _PROM = False


def get_system_snapshot() -> dict[str, Any]:
    """Return CPU, RAM, GPU stats as a dict."""
    snap: dict[str, Any] = {
        "cpu_pct": psutil.cpu_percent(interval=0.1),
        "ram_mb": round(psutil.Process().memory_info().rss / 1024**2, 1),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "ram_used_pct": psutil.virtual_memory().percent,
        "timestamp": time.time(),
    }

    # Disk space
    try:
        from config import get_settings
        disk = shutil.disk_usage(str(get_settings().output_dir))
        snap["disk_free_gb"] = round(disk.free / 1024**3, 2)
        snap["disk_total_gb"] = round(disk.total / 1024**3, 2)
        if _PROM:
            DISK_FREE_GB.set(snap["disk_free_gb"])
    except Exception:
        snap["disk_free_gb"] = -1

    if _GPU:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                g = gpus[0]
                snap["gpu_name"] = g.name
                snap["gpu_mem_used_mb"] = g.memoryUsed
                snap["gpu_mem_total_mb"] = g.memoryTotal
                snap["gpu_load_pct"] = g.load * 100
                snap["gpu_temp_c"] = g.temperature
        except Exception:
            pass
    return snap


def record_generation(
    success: bool,
    elapsed: float,
    gpu_mem_mb: int | None = None,
    audio_duration_s: float = 0.0,
    step_timings: dict[str, float] | None = None,
) -> None:
    """Update Prometheus counters after a generation completes."""
    if not _PROM:
        return
    label = "success" if success else "error"
    REQ_TOTAL.labels(status=label).inc()
    GEN_LATENCY.observe(elapsed)

    # Per-step timings
    if step_timings:
        for step_name, step_time in step_timings.items():
            STEP_LATENCY.labels(step=step_name).observe(step_time)

    # Audio minutes generated
    if audio_duration_s > 0:
        AUDIO_MINUTES.inc(audio_duration_s / 60.0)

    snap = get_system_snapshot()
    CPU_USAGE.set(snap["cpu_pct"])
    RAM_USAGE.set(snap["ram_mb"])
    if gpu_mem_mb is not None:
        GPU_MEM.set(gpu_mem_mb)


def record_active_job_start() -> None:
    """Increment active jobs gauge."""
    if _PROM:
        ACTIVE_JOBS.inc()


def record_active_job_end() -> None:
    """Decrement active jobs gauge."""
    if _PROM:
        ACTIVE_JOBS.dec()


def check_disk_space(min_free_gb: float = 2.0) -> bool:
    """Return True if enough disk space is available."""
    try:
        from config import get_settings
        disk = shutil.disk_usage(str(get_settings().output_dir))
        free_gb = disk.free / 1024**3
        return free_gb >= min_free_gb
    except Exception:
        return True  # can't check, assume ok


def prometheus_output() -> tuple[bytes, str]:
    """Return (bytes, content_type) for the /metrics endpoint."""
    if not _PROM:
        return b"# prometheus-client not installed\n", "text/plain"
    return generate_latest(_registry), CONTENT_TYPE_LATEST
