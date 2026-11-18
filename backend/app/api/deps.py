"""Shared FastAPI dependencies.

Endpoints obtain database sessions, settings and health helpers from here so
that route modules stay thin and free of connection logic.
"""

from __future__ import annotations

import time

from fastapi import Depends, Query
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import DatabaseUnavailableError
from app.database.session import get_db
from app.services.tides.namria import NAMRIATideProvider

__all__ = [
    "get_db",
    "get_settings",
    "Settings",
    "require_db",
    "database_health",
    "get_tide_provider",
    "pagination_params",
]


def require_db(db: Session) -> Session:
    """Return ``db`` or raise :class:`DatabaseUnavailableError`."""
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:  # pragma: no cover - exercised in integration
        raise DatabaseUnavailableError("Database is not reachable") from exc
    return db


def database_health(db: Session) -> dict[str, object]:
    """Collect database / extension health information.

    Returns a dict matching :class:`app.schemas.health.DatabaseHealthResponse`.
    Never raises: connection problems are reported in-band so the endpoint can
    return a structured 200/503 instead of an opaque 500.
    """
    info: dict[str, object] = {
        "status": "ok",
        "database": "postgresql",
        "reachable": True,
        "latency_ms": None,
        "server_version": None,
        "timescaledb_version": None,
        "postgis_version": None,
        "error": None,
    }

    started = time.perf_counter()
    try:
        info["server_version"] = db.execute(text("SHOW server_version")).scalar_one()
        info["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    except SQLAlchemyError as exc:
        info.update(status="error", reachable=False, error=str(exc))
        return info

    # Extensions are expected but reported as their own signal.
    for key, ext in (("timescaledb_version", "timescaledb"), ("postgis_version", "postgis")):
        try:
            info[key] = db.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = :name"),
                {"name": ext},
            ).scalar_one_or_none()
        except SQLAlchemyError:
            info[key] = None

    return info


def get_tide_provider(db: Session = Depends(get_db)) -> NAMRIATideProvider:
    """Provide a request-scoped NAMRIA tide provider."""
    return NAMRIATideProvider(db)


def pagination_params(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> tuple[int, int]:
    """Common limit/offset pagination parameters."""
    return limit, offset
