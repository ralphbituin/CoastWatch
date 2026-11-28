"""DEMO MODE API routes (feature-gated, in-memory, read-only).

Every route returns 404 unless ``DEMO_INJECTION_ENABLED=true``.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.demo.registry import get_registry
from app.demo.service import (
    DemoCapacityError,
    DemoValidationError,
    build_presets,
    compare_recordings,
    get_simulation_manager,
)
from app.demo.summary import get_summary_manager

router = APIRouter(tags=["demo-injection"])


def require_demo_enabled() -> None:
    if not bool(settings.demo_injection_enabled):
        raise HTTPException(status_code=404, detail="Not Found")


class SimulationRequest(BaseModel):
    recording_id: str = Field(min_length=1)
    amplitude_pct: float
    ramp_s: float
    hold_s: float = 300.0
    shape: str = "retreat_return"
    sign: int = 1
    speed: int = 30
    start: str | int | None = "random"
    seed: int | None = None


def _recording_or_404(recording_id: str):
    try:
        return get_registry().get(recording_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="recording not found") from exc


@router.get("/recordings", dependencies=[Depends(require_demo_enabled)])
def list_recordings() -> dict[str, Any]:
    recordings = get_registry().list()
    return {
        "banner": "SIMULATION - not real data",
        "recordings": [recording.to_dict() for recording in recordings],
        "recordings_root_note": (
            "Only run directories with observations and anomalies tables under "
            "ml/reports/live_eval are eligible."
        ),
    }


@router.get("/recordings/{recording_id}/presets", dependencies=[Depends(require_demo_enabled)])
def recording_presets(recording_id: str) -> dict[str, Any]:
    recording = _recording_or_404(recording_id)
    return {
        "recording_id": recording.id,
        "status": recording.status,
        "reason": recording.reason,
        "presets": build_presets(recording),
        "banner": "SIMULATION - not real data",
    }


@router.get("/recordings/{recording_id}/summary", dependencies=[Depends(require_demo_enabled)])
def recording_summary(recording_id: str) -> dict[str, Any]:
    recording = _recording_or_404(recording_id)
    if not recording.replayable:
        return {
            "recording_id": recording.id,
            "status": recording.status,
            "reason": recording.reason,
            "result": None,
            "banner": "SIMULATION - not real data",
        }
    job = get_summary_manager().get(recording)
    return {
        **job.progress(),
        "result": job.result,
        "recording": recording.to_dict(),
        "banner": "SIMULATION - not real data",
    }


@router.get("/simulations/{simulation_id}", dependencies=[Depends(require_demo_enabled)])
def simulation_status(simulation_id: str) -> dict[str, Any]:
    simulation = get_simulation_manager().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="simulation not found")
    return simulation.status_payload()


@router.post("/simulations", dependencies=[Depends(require_demo_enabled)])
def create_simulation(payload: SimulationRequest = Body(...)) -> dict[str, Any]:
    recording = _recording_or_404(payload.recording_id)
    body = payload.model_dump()
    if body.get("seed") is None:
        body.pop("seed", None)
    try:
        simulation = get_simulation_manager().create(recording, body)
    except DemoValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DemoCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return {
        "id": simulation.id,
        "recording_id": recording.id,
        "status": simulation.status,
        "params": simulation.params,
        "warnings": simulation.warnings,
        "stream": f"/api/v1/demo/injection/simulations/{simulation.id}/stream",
        "banner": "SIMULATION - not real data",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(_json_safe(payload), default=str, allow_nan=False)}\n\n"


@router.get("/simulations/{simulation_id}/stream", dependencies=[Depends(require_demo_enabled)])
async def stream_simulation(simulation_id: str) -> StreamingResponse:
    manager = get_simulation_manager()
    simulation = manager.get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="simulation not found")
    manager.mark_streaming(simulation_id)
    speed = max(1, int(simulation.params.get("speed") or 30))

    async def generate() -> AsyncIterator[str]:
        previous: datetime | None = None
        try:
            for message in list(simulation.messages):
                if simulation.stop_event.is_set():
                    break
                if message.get("type") == "sample":
                    timestamp = message.get("time")
                    try:
                        moment = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                    except ValueError:
                        moment = None
                    if previous is not None and moment is not None:
                        delay = max(0.0, (moment - previous).total_seconds() / float(speed))
                        if delay > 0:
                            await asyncio.sleep(delay)
                    previous = moment
                yield _sse(message)
            if simulation.stop_event.is_set():
                yield _sse(
                    {
                        "type": "stopped",
                        "sim_id": simulation.id,
                        "banner": "SIMULATION - not real data",
                    }
                )
            else:
                manager.mark_done(simulation.id)
        except asyncio.CancelledError:
            manager.disconnected(simulation.id)
            raise
        finally:
            manager.disconnected(simulation.id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/simulations/{simulation_id}/stop", dependencies=[Depends(require_demo_enabled)])
def stop_simulation(simulation_id: str) -> dict[str, Any]:
    simulation = get_simulation_manager().stop(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="simulation not found")
    return {
        "id": simulation.id,
        "status": simulation.status,
        "banner": "SIMULATION - not real data",
    }


@router.delete("/simulations/{simulation_id}", dependencies=[Depends(require_demo_enabled)])
def delete_simulation(simulation_id: str) -> dict[str, Any]:
    if not get_simulation_manager().delete(simulation_id):
        raise HTTPException(status_code=404, detail="simulation not found")
    return {"deleted": True, "id": simulation_id, "banner": "SIMULATION - not real data"}


@router.get("/compare", dependencies=[Depends(require_demo_enabled)])
def compare(
    amplitude_pct: float = Query(..., ge=5.0, le=60.0),
    ramp_s: float = Query(..., gt=0.0),
) -> dict[str, Any]:
    return compare_recordings(get_registry().list(), amplitude_pct=amplitude_pct, ramp_s=ramp_s)
