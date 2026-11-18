"""Shoreline routes (observations, calibrations, baselines).

Implemented in Phases 4-5.
"""

from __future__ import annotations

from app.api.routes._stub import phase_stub_router

router = phase_stub_router(
    module="shorelines",
    phase=4,
    planned_endpoints=[
        "GET /shorelines/observations",
        "GET /shorelines/observations/latest",
        "GET /shorelines/calibrations",
        "POST /shorelines/calibrations",
        "GET /shorelines/baselines",
    ],
)
