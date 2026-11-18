"""Station routes (monitoring stations, tide stations, countries).

Implemented in Phase 2 (tide-station management) and fleshed out further in
Phase 7. Registered now so the versioned API surface is stable.
"""

from __future__ import annotations

from app.api.routes._stub import phase_stub_router

router = phase_stub_router(
    module="stations",
    phase=2,
    planned_endpoints=[
        "GET /stations",
        "POST /stations",
        "GET /stations/{station_id}",
        "GET /stations/{station_id}/tide-stations",
        "GET /countries",
    ],
)
