"""API router assembly.

All module routers are mounted under the versioned prefix (``/api/v1``) in
``main.py``. Keeping the assembly here means routes never import ``main``.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    anomalies,
    cameras,
    health,
    shorelines,
    stations,
    tides,
    websocket,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(stations.router, prefix="/stations")
api_router.include_router(tides.router, prefix="/tides")
api_router.include_router(cameras.router, prefix="/cameras")
api_router.include_router(shorelines.router, prefix="/shorelines")
api_router.include_router(anomalies.router, prefix="/anomalies")
api_router.include_router(websocket.router)

__all__ = ["api_router"]
