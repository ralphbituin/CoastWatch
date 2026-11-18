"""CoastWatch FastAPI application entry point.

Run locally with::

    uvicorn app.main:app --reload --port 8000

All module routes are mounted under ``/api/v1``. Schema changes are handled by
Alembic, never by ``create_all`` at runtime.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestLoggingMiddleware
from app.database.session import dispose_engine

logger = get_logger("coastwatch.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks."""
    configure_logging()
    logger.info(
        "startup project=%s version=%s environment=%s",
        settings.project_name,
        __version__,
        settings.environment,
    )
    yield
    logger.info("shutdown")
    dispose_engine()


def create_app() -> FastAPI:
    """Application factory."""
    configure_logging()

    app = FastAPI(
        title=settings.project_name,
        version=__version__,
        description=(
            "Tide-aware shoreline monitoring and anomaly detection. "
            "A monitoring / decision-support system — not an operational "
            "tsunami warning service."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    register_exception_handlers(app)

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", tags=["meta"], summary="Service metadata")
    def root() -> dict[str, str]:
        return {
            "service": settings.project_name,
            "version": __version__,
            "environment": settings.environment,
            "api": settings.api_v1_prefix,
            "docs": "/docs",
        }

    return app


app = create_app()
