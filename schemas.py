"""
OmniVoice Studio — Pydantic v2 request/response schemas.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Generation request ─────────────────────────────────────────────────────────
class GenerationSettings(BaseModel):
    """Parameters passed alongside the text payload."""

    # Quantisation / hardware
    quant: Literal["fp32", "fp16", "8bit", "4bit"] = "4bit"
    qtype: Literal["nf4", "fp4"] = "nf4"
    dquant: bool = True
    device: str = "cuda:0"

    # Synthesis
    steps: int = Field(50, ge=1, le=200)
    cfg: float = Field(3.0, ge=0.5, le=10.0)
    temp: float = Field(1.0, ge=0.1, le=2.0)
    sampler: Literal["dpmsolver++", "euler", "ddim"] = "dpmsolver++"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    pitch: int = Field(0, ge=-12, le=12)
    sim: float = Field(0.8, ge=0.0, le=1.0)

    # Optional overrides
    duration: Optional[float] = Field(None, ge=1.0, le=60.0)
    seed: Optional[int] = None
    sr: int = Field(24000, ge=8000, le=48000)
    prefix: str = Field("clone", max_length=32, pattern=r"^[a-zA-Z0-9_-]+$")

    # Voice design / clone extras
    ref_text: str = ""
    lang: str = "auto"
    mode: Literal["clone", "design", "auto", "karaoke", "blend"] = "clone"
    instruct: str = ""

    # ── NEW: Style selection (from GUI tone/style chips) ────────────────────────
    style: Literal[
        "neutral", "formal", "casual", "enthusiastic", "calm", "storytelling", "news_anchor"
    ] = "neutral"

    # ── NEW: Emotion control ──────────────────────────────────────────────────
    emotion: Literal[
        "neutral", "happy", "sad", "angry", "surprised", "whispering", "excited",
        "expressive", "warm", "dynamic", "intense", "steady"
    ] = "neutral"

    # ── NEW: Output format conversion ─────────────────────────────────────────
    output_format: Literal["wav", "mp3", "flac", "ogg"] = "wav"

    # ── NEW: Pronunciation lexicon — key=word, value=phonetic ─────────────────
    pronunciation_lexicon: dict[str, str] = Field(default_factory=dict)

    # ── NEW: SSML pause/break support ─────────────────────────────────────────
    ssml_enabled: bool = False  # When True, text is parsed for <break/> tags

    # ── NEW: Webhook callback URL ─────────────────────────────────────────────
    webhook_url: str = ""  # POST result JSON here when async job completes

    # ── NEW: Voice blending (mix two reference voices) ─────────────────────────
    # Primary voice weight: 0.0-1.0 (1.0 = 100% voice A, 0.0 = 100% voice B)
    voice_blend_weight: float = Field(0.5, ge=0.0, le=1.0)
    # Secondary reference audio path (for blending, uploaded via API)
    ref_audio_secondary: str = ""


class VoiceBlendRequest(BaseModel):
    """Request to blend two voices with a given weight."""
    text: str = Field(..., min_length=1, max_length=5000)
    ref_audio_a: str  # Path or URL to primary voice
    ref_audio_b: str  # Path or URL to secondary voice
    blend_weight: float = Field(0.5, ge=0.0, le=1.0, description="0.0 = 100% voice B, 1.0 = 100% voice A")
    settings: GenerationSettings = Field(default_factory=GenerationSettings)


class GenerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


# ── Batch request ──────────────────────────────────────────────────────────────
class BatchGenerationRequest(BaseModel):
    items: list[GenerationRequest] = Field(..., min_length=1, max_length=10)


# ── Response models ────────────────────────────────────────────────────────────
class GenerationResponse(BaseModel):
    status: Literal["success", "error", "pending"]
    job_id: str
    message: str = ""
    audio_url: Optional[str] = None
    filepath: Optional[str] = None
    generation_time: Optional[float] = None
    text: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    audio_url: Optional[str] = None
    generation_time: Optional[float] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
    # Populated for in-flight jobs (GET /jobs/{id} polling)
    phase: Optional[str] = None
    progress_pct: Optional[int] = Field(None, ge=0, le=100)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    model: dict[str, Any]
    system: dict[str, Any]
    checks: dict[str, bool] = Field(default_factory=dict)  # NEW: dependency checks
    version: str = "2.1.0"


class HistoryResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[dict[str, Any]]
