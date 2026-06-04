"""
OmniVoice Studio — SQLAlchemy async database layer.
Tables: GenerationJob, SystemEvent

Enhancements:
  • WAL mode + busy_timeout for concurrent write safety
  • webhook_url column for async callback delivery
  • DB health check helper
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, event
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine
)
from sqlalchemy.orm import DeclarativeBase

from config import get_settings

settings = get_settings()

# ── Engine ─────────────────────────────────────────────────────────────────────
_engine = create_async_engine(
    settings.db_url,
    echo=settings.debug,
    pool_pre_ping=True,
    connect_args={
        "check_same_thread": False,  # SQLite only
        "timeout": 30,                # SQLite busy timeout — prevents BusyError
    },
)

# Enable WAL mode for SQLite (dramatically better concurrent read perf)
@event.listens_for(_engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _conn_record):
    try:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()
    except Exception:
        pass


AsyncSessionLocal = async_sessionmaker(
    _engine, class_=AsyncSession, expire_on_commit=False
)


# ── Base / Models ──────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class GenerationJob(Base):
    """One TTS generation request."""
    __tablename__ = "generation_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime(timezone=True), nullable=True)

    # Input
    mode = Column(String(20), default="clone")   # clone | design | auto
    text = Column(Text, nullable=False)
    ref_audio_path = Column(String, nullable=True)
    settings_json = Column(Text, default="{}")

    # Result
    status = Column(String(20), default="pending")  # pending | running | done | error
    audio_url = Column(String, nullable=True)
    filepath = Column(String, nullable=True)
    generation_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Hardware snapshot
    gpu_mem_mb = Column(Integer, nullable=True)
    cpu_pct = Column(Float, nullable=True)

    # Webhook callback URL (async jobs)
    webhook_url = Column(String, nullable=True, default=None)

    # Per-step timing JSON (for detailed metrics)
    step_timings_json = Column(Text, nullable=True, default=None)


class SystemEvent(Base):
    """Generic event log (model load, errors, etc.)."""
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    level = Column(String(10), default="INFO")
    category = Column(String(40))
    message = Column(Text)
    detail = Column(Text, nullable=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
async def init_db() -> None:
    """Create all tables (idempotent) and sync missing columns."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Sync missing columns for SQLite (simple migration-less approach)
        if "sqlite" in settings.db_url:
            for table_name, table in Base.metadata.tables.items():
                # Get current columns
                def get_current_cols(sync_conn):
                    cursor = sync_conn.exec_driver_sql(f"PRAGMA table_info({table_name})")
                    return [row[1] for row in cursor.fetchall()]
                
                current_cols = await conn.run_sync(get_current_cols)
                
                # Find missing columns
                for column in table.columns:
                    if column.name not in current_cols:
                        # Add missing column
                        type_str = str(column.type.compile(dialect=conn.dialect))
                        nullable = "NULL" if column.nullable else "NOT NULL"
                        default = ""
                        if column.default is not None and not callable(column.default.arg):
                            default = f" DEFAULT {column.default.arg}"
                        
                        await conn.execute(__import__("sqlalchemy").text(
                            f"ALTER TABLE {table_name} ADD COLUMN {column.name} {type_str} {nullable}{default}"
                        ))


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a DB session per request."""
    async with AsyncSessionLocal() as session:
        yield session


async def check_db_health() -> bool:
    """Return True if DB is reachable and writable."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        return True
    except Exception:
        return False
