"""Helper for building explicit phase-stub routers.

CoastWatch develops strictly one phase at a time. Module routes that are not
yet implemented still register their API surface (so versioning and the
frontend route map are stable) but return HTTP 501 with an honest payload
naming the phase that implements them.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.common import NotImplementedPayload


def phase_stub_router(*, module: str, phase: int, planned_endpoints: list[str]) -> APIRouter:
    """Build a router whose single ``GET`` endpoint announces its phase."""
    router = APIRouter()
    payload = NotImplementedPayload(
        module=module,
        phase=phase,
        implemented=False,
        message=(
            f"The '{module}' module is scheduled for Phase {phase} " f"and is not implemented yet."
        ),
        planned_endpoints=planned_endpoints,
    ).model_dump()

    @router.get("", summary=f"{module} (phase {phase} — not implemented)")
    def _status() -> JSONResponse:
        return JSONResponse(status_code=501, content=payload)

    return router
