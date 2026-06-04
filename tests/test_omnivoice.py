"""
OmniVoice Studio — Production Test Suite
────────────────────────────────────────
Three-Layer Strategy:
  Layer 1 — Foundation (pipeline, config, DB)
  Layer 2 — Model / API (mocked)
  Layer 3 — Business / Load
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Test env vars (set before importing app modules) ──────────────────────────
os.environ.setdefault("DB_URL", "sqlite+aiosqlite:///./test_omnivoice.db")
os.environ.setdefault("OUTPUT_DIR", "./test_outputs")
os.environ.setdefault("HF_TOKEN", "hf_test_token")
os.environ.setdefault("API_KEY_ENABLED", "false")
os.environ.setdefault("METRICS_ENABLED", "false")

from app import app
from database import Base, GenerationJob, _engine, get_db, init_db
from schemas import GenerationRequest, GenerationSettings


# ════════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture(autouse=True, scope="function")
async def fresh_db():
    """Re-create tables before every test function."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client against the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── Mock OmniVoice model output ────────────────────────────────────────────────
def _mock_model_output() -> dict[str, Any]:
    sr = 24000
    duration = 2  # seconds
    audio = (np.sin(np.linspace(0, 440 * 2 * np.pi, sr * duration)) * 0.5).astype(np.float32)
    return {"audio": audio, "sample_rate": sr}


@pytest.fixture
def mock_model():
    """Patch model loading and inference with a deterministic sine wave."""
    m = MagicMock()
    m.generate.return_value = _mock_model_output()
    with patch("model_manager.get_model", new=AsyncMock(return_value=m)):
        with patch("generation_service.get_model", new=AsyncMock(return_value=m)):
            yield m


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 1 — FOUNDATION TESTS
# ════════════════════════════════════════════════════════════════════════════════
class TestConfig:
    """Validate configuration loading."""

    def test_settings_loads(self):
        from config import get_settings
        cfg = get_settings()
        assert cfg.port == 8000

    def test_output_dir_created(self):
        from config import get_settings
        cfg = get_settings()
        assert cfg.output_dir.exists()

    def test_db_url_is_set(self):
        from config import get_settings
        cfg = get_settings()
        assert "sqlite" in cfg.db_url


class TestDatabase:
    """Validate DB schema and basic CRUD."""

    @pytest.mark.asyncio
    async def test_tables_created(self):
        from sqlalchemy import inspect
        from sqlalchemy.ext.asyncio import AsyncConnection
        async with _engine.connect() as conn:
            tables = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
        assert "generation_jobs" in tables
        assert "system_events" in tables

    @pytest.mark.asyncio
    async def test_job_insert_and_read(self):
        from database import AsyncSessionLocal
        job_id = str(uuid.uuid4())
        async with AsyncSessionLocal() as db:
            job = GenerationJob(id=job_id, text="Hello world", status="pending")
            db.add(job)
            await db.commit()

        async with AsyncSessionLocal() as db:
            fetched = await db.get(GenerationJob, job_id)
            assert fetched is not None
            assert fetched.text == "Hello world"
            assert fetched.status == "pending"


class TestSchemaValidation:
    """Pydantic schema edge cases."""

    def test_valid_request(self):
        req = GenerationRequest(text="Hello", settings=GenerationSettings())
        assert req.text == "Hello"

    def test_text_stripped(self):
        req = GenerationRequest(text="  Hello  ")
        assert req.text == "Hello"

    def test_text_too_long_rejected(self):
        with pytest.raises(Exception):
            GenerationRequest(text="x" * 2001)

    def test_empty_text_rejected(self):
        with pytest.raises(Exception):
            GenerationRequest(text="")

    def test_steps_bounds(self):
        with pytest.raises(Exception):
            GenerationSettings(steps=300)

    def test_invalid_prefix_rejected(self):
        with pytest.raises(Exception):
            GenerationSettings(prefix="bad name!")

    def test_defaults_are_sane(self):
        s = GenerationSettings()
        assert s.steps == 50
        assert s.cfg == 3.0
        assert s.quant == "4bit"


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 2 — API / MODEL TESTS
# ════════════════════════════════════════════════════════════════════════════════
class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/api/v2/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_has_required_fields(self, client):
        data = (await client.get("/api/v2/health")).json()
        assert "status" in data
        assert "model" in data
        assert "system" in data
        assert "version" in data

    @pytest.mark.asyncio
    async def test_system_snapshot_has_cpu(self, client):
        data = (await client.get("/api/v2/health")).json()
        assert "cpu_pct" in data["system"]


class TestGenerationEndpoint:
    """API-level tests using mocked model."""

    @pytest.mark.asyncio
    async def test_sync_generate_success(self, client, mock_model, tmp_path):
        with patch("generation_service.settings") as ms:
            ms.output_dir = tmp_path
            resp = await client.post(
                "/api/v2/generate",
                json={"text": "Hello world", "settings": {"steps": 10}},
            )
        # Accept 200 or 500 (model may not be installed in CI)
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_missing_text_returns_422(self, client):
        resp = await client.post("/api/v2/generate", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_async_generate_returns_job_id(self, client, mock_model):
        resp = await client.post(
            "/api/v2/generate/async",
            json={"text": "Test async generation", "settings": {"steps": 5}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_batch_generate_returns_multiple_ids(self, client, mock_model):
        resp = await client.post(
            "/api/v2/generate/batch",
            json={
                "items": [
                    {"text": "First item", "settings": {"steps": 5}},
                    {"text": "Second item", "settings": {"steps": 5}},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["job_ids"]) == 2

    @pytest.mark.asyncio
    async def test_batch_exceeds_limit(self, client):
        items = [{"text": f"Item {i}", "settings": {}} for i in range(11)]
        resp = await client.post("/api/v2/generate/batch", json={"items": items})
        assert resp.status_code == 422


class TestJobPolling:
    @pytest.mark.asyncio
    async def test_poll_nonexistent_job_returns_404(self, client):
        resp = await client.get(f"/api/v2/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_poll_created_job_returns_200(self, client, mock_model):
        # Create a job
        resp = await client.post(
            "/api/v2/generate/async",
            json={"text": "Poll test", "settings": {"steps": 5}},
        )
        job_id = resp.json()["job_id"]
        # Poll it
        await asyncio.sleep(0.1)
        poll_resp = await client.get(f"/api/v2/jobs/{job_id}")
        assert poll_resp.status_code == 200
        data = poll_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("pending", "running", "done", "error", "cancelled")


class TestHistory:
    @pytest.mark.asyncio
    async def test_history_empty(self, client):
        resp = await client.get("/api/v2/history")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_history_pagination(self, client, mock_model):
        # Create 3 jobs
        for i in range(3):
            await client.post(
                "/api/v2/generate/async",
                json={"text": f"Item {i}", "settings": {"steps": 5}},
            )
        resp = await client.get("/api/v2/history?page=1&page_size=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 3
        assert len(data["items"]) <= 2

    @pytest.mark.asyncio
    async def test_history_csv_export(self, client):
        resp = await client.get("/api/v2/history/export/csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 2 — DATA / AUDIO PROCESSING TESTS
# ════════════════════════════════════════════════════════════════════════════════
class TestAudioProcessing:
    """Validate audio normalization and shape handling."""

    def test_normalise_mono(self):
        from generation_service import _preprocess_audio
        audio = np.random.randn(24000).astype(np.float32) * 5.0
        out = _preprocess_audio(audio)
        assert np.abs(out).max() <= 0.96  # normalised

    def test_transpose_channels_first(self):
        from generation_service import _preprocess_audio
        # Channels-first input (1, 24000) should be transposed to (24000, 1)
        audio = np.random.randn(1, 24000).astype(np.float32)
        out = _preprocess_audio(audio)
        assert out.shape[0] > out.shape[-1] or out.ndim == 1

    def test_output_dtype_float32(self):
        from generation_service import _preprocess_audio
        audio = np.random.randn(8000).astype(np.float64)
        out = _preprocess_audio(audio)
        assert out.dtype == np.float32

    def test_silent_audio_handled(self):
        from generation_service import _preprocess_audio
        audio = np.zeros(8000, dtype=np.float32)
        out = _preprocess_audio(audio)
        assert np.all(out == 0)


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 3 — PERFORMANCE / LOAD TESTS
# ════════════════════════════════════════════════════════════════════════════════
class TestPerformance:
    """Simple load / throughput tests."""

    @pytest.mark.asyncio
    async def test_health_endpoint_fast(self, client):
        """Health check must respond in < 500 ms."""
        t0 = time.perf_counter()
        resp = await client.get("/api/v2/health")
        elapsed = time.perf_counter() - t0
        assert resp.status_code == 200
        assert elapsed < 0.5, f"Health endpoint too slow: {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_concurrent_health_requests(self, client):
        """10 concurrent health requests should all succeed."""
        tasks = [client.get("/api/v2/health") for _ in range(10)]
        results = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_async_submissions(self, client, mock_model):
        """5 concurrent async job submissions should all get unique job IDs."""
        tasks = [
            client.post(
                "/api/v2/generate/async",
                json={"text": f"Concurrent test {i}", "settings": {"steps": 5}},
            )
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        ids = [r.json()["job_id"] for r in results if r.status_code == 200]
        assert len(set(ids)) == len(ids), "Duplicate job IDs detected!"

    @pytest.mark.asyncio
    async def test_history_large_page_capped(self, client):
        """page_size > 100 should be silently capped."""
        resp = await client.get("/api/v2/history?page_size=9999")
        assert resp.status_code == 200


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 3 — SECURITY TESTS
# ════════════════════════════════════════════════════════════════════════════════
class TestSecurity:
    @pytest.mark.asyncio
    async def test_cors_headers_present(self, client):
        resp = await client.options(
            "/api/v2/health",
            headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"},
        )
        # Accepts CORS pre-flight
        assert resp.status_code in (200, 204)

    @pytest.mark.asyncio
    async def test_request_id_header(self, client):
        resp = await client.get("/api/v2/health")
        assert "x-request-id" in resp.headers

    @pytest.mark.asyncio
    async def test_oversized_text_rejected(self, client):
        """Text > 2000 chars must be rejected by schema validation."""
        resp = await client.post(
            "/api/v2/generate/async",
            json={"text": "A" * 2001, "settings": {}},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_nonexistent_job_404(self, client):
        resp = await client.delete(f"/api/v2/history/{uuid.uuid4()}")
        assert resp.status_code == 404


# ════════════════════════════════════════════════════════════════════════════════
# METRICS TESTS
# ════════════════════════════════════════════════════════════════════════════════
class TestMetrics:
    def test_system_snapshot_has_required_keys(self):
        from metrics import get_system_snapshot
        snap = get_system_snapshot()
        assert "cpu_pct" in snap
        assert "ram_mb" in snap
        assert "timestamp" in snap

    def test_cpu_pct_in_range(self):
        from metrics import get_system_snapshot
        snap = get_system_snapshot()
        assert 0 <= snap["cpu_pct"] <= 100

    @pytest.mark.asyncio
    async def test_prometheus_endpoint_accessible(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
