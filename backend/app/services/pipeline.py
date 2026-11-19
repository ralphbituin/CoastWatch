"""Pipeline orchestration.

From Phase 4 onward this module owns the end-to-end flow:

    camera → frames → vision(waterline) → baseline(expected) →
    anomaly(score) → persistence → WebSocket push

Phase 4 wires the camera → vision seam: frame acquisition, quality assessment,
segmentation (when an operator supplies or a trained model provides a mask) and
persistence of :class:`ShorelineObservation` rows. Baseline (Phase 5) and
anomaly (Phase 6) remain unimplemented; ``PipelineResult.expected_shoreline_m``
and later stages stay ``None`` until those phases land.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.services.camera import manager as camera_manager
from app.services.camera.base import CameraError
from app.services.vision.calibration import CalibrationProfile
from app.services.vision.segmentation import ShorelineSegmenter
from app.services.vision.shoreline import (
    ShorelineExtractionError,
    ShorelineObservationResult,
    extract_shoreline,
)

logger = get_logger(__name__)


def record_observation(
    session: Any,
    camera: Any,
    *,
    observed_at: datetime,
    result: ShorelineObservationResult,
    calibration_id: int | None,
    quality_score: float | None,
    quality_flags: list[str],
    source_frame_path: str | None,
) -> Any:
    """Persist one ShorelineObservation row and return it."""
    from app.models.shoreline import ShorelineObservation

    observation = ShorelineObservation(
        camera_id=camera.id,
        observed_at=observed_at,
        monitoring_station_id=camera.monitoring_station_id,
        calibration_id=calibration_id,
        shoreline_geometry=result.points,
        calibrated_position_m=result.calibrated_position_m,
        segmentation_confidence=result.segmentation_confidence,
        frame_quality_score=quality_score,
        frame_quality_flags=list(quality_flags),
        source_frame_path=source_frame_path,
    )
    session.add(observation)
    return observation


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

    Phase 4 implements :meth:`process_once` for one camera. ``segmenter_factory``
    exists so tests (and future validated models) can inject a segmenter; the
    default uses :class:`ShorelineSegmenter`, which — until a model artifact is
    trained and validated — always raises :class:`ModelUnavailableError`.
    """

    def __init__(
        self,
        session: Any | None = None,
        *,
        segmenter_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._session = session
        self._segmenter_factory = segmenter_factory or ShorelineSegmenter

    def process_once(self, *, station_id: int, camera_id: int) -> PipelineResult:
        """Run one observation cycle for ``camera_id``.

        Never fabricates a waterline: when no mask/model is available the
        result is ``state="DATA_UNRELIABLE"`` with an explicit reason —
        unavailable data is never reported as normal.
        """
        if self._session is None:
            raise ValueError("CoastWatchPipeline requires a database session (session=...)")

        from app.core.exceptions import ModelUnavailableError, NotFoundError
        from app.models.camera import Camera

        camera = self._session.get(Camera, camera_id)
        if camera is None:
            raise NotFoundError(f"Camera not found: {camera_id}")
        if camera.monitoring_station_id != station_id:
            raise NotFoundError(f"Camera {camera_id} is not part of station {station_id}")

        try:
            frame, quality = camera_manager.capture_frame(camera)
        except CameraError as exc:
            return PipelineResult(
                station_id=station_id,
                camera_id=camera_id,
                observed_at=datetime.now(UTC),
                actual_shoreline_m=None,
                expected_shoreline_m=None,
                residual_m=None,
                retreat_velocity_m_s=None,
                anomaly_score=None,
                state="DATA_UNRELIABLE",
                details={"reason": "frame_unavailable", "message": str(exc)},
            )

        try:
            segmentation = self._segmenter_factory().predict(frame.image)
        except ModelUnavailableError as exc:
            logger.info("pipeline: segmentation unavailable for camera %s", camera_id)
            return PipelineResult(
                station_id=station_id,
                camera_id=camera_id,
                observed_at=frame.captured_at,
                actual_shoreline_m=None,
                expected_shoreline_m=None,
                residual_m=None,
                retreat_velocity_m_s=None,
                anomaly_score=None,
                state="DATA_UNRELIABLE",
                details={
                    "reason": "segmentation_model_unavailable",
                    "message": str(exc),
                    "frame_quality_score": quality.score,
                    "frame_quality_flags": list(quality.flags),
                },
            )

        # A mask exists (operator-supplied now, or a trained model later).
        calibration = self._active_calibration(camera)
        try:
            result = extract_shoreline(
                segmentation.mask,
                calibration=calibration,
                confidence=segmentation.confidence,
            )
        except ShorelineExtractionError as exc:
            return PipelineResult(
                station_id=station_id,
                camera_id=camera_id,
                observed_at=frame.captured_at,
                actual_shoreline_m=None,
                expected_shoreline_m=None,
                residual_m=None,
                retreat_velocity_m_s=None,
                anomaly_score=None,
                state="DATA_UNRELIABLE",
                details={"reason": "shoreline_extraction_failed", "message": str(exc)},
            )

        calibration_id = camera.active_calibration_id if calibration else None
        record_observation(
            self._session,
            camera,
            observed_at=frame.captured_at,
            result=result,
            calibration_id=calibration_id,
            quality_score=quality.score,
            quality_flags=list(quality.flags),
            source_frame_path=None,
        )
        try:
            self._session.commit()
        except Exception as exc:
            self._session.rollback()
            logger.exception("pipeline: failed to persist observation for camera %s", camera_id)
            return PipelineResult(
                station_id=station_id,
                camera_id=camera_id,
                observed_at=frame.captured_at,
                actual_shoreline_m=None,
                expected_shoreline_m=None,
                residual_m=None,
                retreat_velocity_m_s=None,
                anomaly_score=None,
                state="DATA_UNRELIABLE",
                details={"reason": "persistence_failed", "message": str(exc)},
            )

        return PipelineResult(
            station_id=station_id,
            camera_id=camera_id,
            observed_at=frame.captured_at,
            actual_shoreline_m=result.calibrated_position_m,
            expected_shoreline_m=None,
            residual_m=None,
            retreat_velocity_m_s=None,
            anomaly_score=None,
            state="OBSERVED",
            details={
                "reason": "observation_persisted",
                "points": len(result.points),
                "calibration_version": result.calibration_version,
                "frame_quality_score": quality.score,
                "frame_quality_flags": list(quality.flags),
                "note": "baseline/anomaly stages land in Phases 5-6; no verdict yet",
            },
        )

    def _active_calibration(self, camera: Any) -> CalibrationProfile | None:
        """Build a CalibrationProfile from the camera's active calibration row."""
        if not camera.active_calibration_id:
            return None
        from app.models.camera import CameraCalibration

        session = self._session
        if session is None:  # pragma: no cover - guarded by process_once
            return None
        row = session.get(CameraCalibration, camera.active_calibration_id)
        if row is None:
            return None
        return CalibrationProfile(
            camera_id=camera.id,
            version=row.version,
            homography=row.homography_matrix,
            meters_per_pixel=row.meters_per_pixel,
        )

    def start(self) -> None:
        raise NotImplementedError(
            "Continuous pipeline execution is implemented from Phase 4 onward."
        )

    def stop(self) -> None:
        raise NotImplementedError(
            "Continuous pipeline execution is implemented from Phase 4 onward."
        )
