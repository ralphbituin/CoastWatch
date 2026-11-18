"""Camera routes (connection, status, frames).

Implemented in Phase 3.
"""

from __future__ import annotations

from app.api.routes._stub import phase_stub_router

router = phase_stub_router(
    module="cameras",
    phase=3,
    planned_endpoints=[
        "GET /cameras",
        "POST /cameras",
        "GET /cameras/{camera_id}",
        "GET /cameras/{camera_id}/status",
        "POST /cameras/{camera_id}/test-connection",
        "GET /cameras/{camera_id}/frame",
    ],
)
