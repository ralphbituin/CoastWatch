"""Liveness and readiness health checks."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app import __version__
from app.api.deps import database_health, get_db, get_settings
from app.core.config import Settings
from app.schemas.health import DatabaseHealthResponse, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service liveness")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Basic liveness probe — does not touch the database."""
    return HealthResponse(
        status="ok",
        service=settings.project_name,
        version=__version__,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
    )


@router.get(
    "/health/db",
    response_model=DatabaseHealthResponse,
    summary="Database readiness",
    responses={503: {"model": DatabaseHealthResponse}},
)
def health_db(db: Session = Depends(get_db)) -> JSONResponse:
    """Verify the database connection and report extension versions."""
    info = database_health(db)
    model = DatabaseHealthResponse(**info)  # type: ignore[arg-type]
    status_code = 200 if model.reachable else 503
    return JSONResponse(status_code=status_code, content=model.model_dump(mode="json"))
