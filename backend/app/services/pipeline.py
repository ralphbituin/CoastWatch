"""Pipeline orchestration.

From Phase 4 onward this module owns the end-to-end flow:

    camera → frames → vision(waterline) → baseline(expected) →
    anomaly(score) → persistence → WebSocket push

Route handlers must **not** embed this flow; they call into
:class:`CoastWatchPipeline`. Phase 1 defines the seam and the data contract
only — the individual stages are implemented in their own modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineResult:
    """Outcome of a single observation cycle."""

    station_id: int
    camera_id: int
    observed_at: datetime
    # "where is the waterline now" — produced by vision/shoreline.py
    actual_shoreline_m: float | None
    # "where should the waterline be given the tide" — baseline/expected_shoreline.py
    expected_shoreline_m: float | None
    residual_m: float | None
    retreat_velocity_m_s: float | None
    anomaly_score: float | None
    state: str
    details: dict[str, Any]


class CoastWatchPipeline:
    """Coordinates the camera → vision → baseline → anomaly flow.

    Implemented incrementally from Phase 4. Phase 1 provides the class shape so
    that routes and tests depend on a stable orchestration entry point.
    """

    def __init__(self, session: Any | None = None) -> None:
        self._session = session

    def process_once(self, *, station_id: int, camera_id: int) -> PipelineResult:
        raise NotImplementedError("Pipeline orchestration is implemented from Phase 4 onward.")

    def start(self) -> None:
        raise NotImplementedError(
            "Continuous pipeline execution is implemented from Phase 4 onward."
        )

    def stop(self) -> None:
        raise NotImplementedError(
            "Continuous pipeline execution is implemented from Phase 4 onward."
        )
