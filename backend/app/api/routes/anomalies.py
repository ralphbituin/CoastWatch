"""Anomaly routes (events, scoring, system state).

Implemented in Phase 6.
"""

from __future__ import annotations

from app.api.routes._stub import phase_stub_router

router = phase_stub_router(
    module="anomalies",
    phase=6,
    planned_endpoints=[
        "GET /anomalies/events",
        "GET /anomalies/events/{event_id}",
        "GET /anomalies/current",
        "POST /anomalies/events/{event_id}/acknowledge",
    ],
)
