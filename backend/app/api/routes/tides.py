"""Tide routes (stations, datasets, CSV import, predictions).

Implemented in Phase 2.
"""

from __future__ import annotations

from app.api.routes._stub import phase_stub_router

router = phase_stub_router(
    module="tides",
    phase=2,
    planned_endpoints=[
        "GET /tides/stations",
        "POST /tides/stations",
        "POST /tides/imports/preview",
        "POST /tides/imports",
        "GET /tides/imports",
        "GET /tides/predictions",
        "GET /tides/high-low-waters",
    ],
)
