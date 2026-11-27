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
from app.services.vision.roi import apply_roi_to_class_mask, load_active_roi
from app.services.vision.segmentation import InvalidFrameError, ShorelineSegmenter
from app.services.vision.shoreline import (
    ShorelineExtractionError,
    ShorelineObservationResult,
    extract_shoreline,
    extract_shoreline_from_classes,
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
    recording_id: str | None = None,
    frame_index: int | None = None,
    elapsed_seconds: float | None = None,
    stream_session_id: str | None = None,
    ingest_latency_ms: float | None = None,
    timestamp_source: str | None = None,
    timestamp_confidence: float | None = None,
    extraction_status: str | None = None,
    extraction_reason: str | None = None,
    class_fractions: dict[str, Any] | None = None,
    inference_time_ms: float | None = None,
    horizon_y_px: float | None = None,
    frame_shape: tuple[int, int] | None = None,
    previous_signed_position: float | None = None,
) -> Any:
    """Persist one observation through the single Phase 7 recording service.

    ``(camera_id, observed_at)`` is the primary key. An exact duplicate timestamp
    is dropped (and reported) instead of being nudged into a fabricated time;
    ``(recording_id, frame_index)`` reprocessing updates the existing row.
    """
    from app.services.recording.service import write_observation

    outcome = write_observation(
        session,
        camera,
        observed_at=observed_at,
        result=result,
        calibration_id=calibration_id,
        quality_score=quality_score,
        quality_flags=quality_flags,
        source_frame_path=source_frame_path,
        recording_id=recording_id,
        frame_index=frame_index,
        elapsed_seconds=elapsed_seconds,
        stream_session_id=stream_session_id,
        ingest_latency_ms=ingest_latency_ms,
        timestamp_source=timestamp_source,
        timestamp_confidence=timestamp_confidence,
        extraction_status=extraction_status,
        extraction_reason=extraction_reason,
        class_fractions=class_fractions,
        inference_time_ms=inference_time_ms,
        horizon_y_px=horizon_y_px,
        frame_shape=frame_shape,
        previous_signed_position=previous_signed_position,
    )
    if outcome.status == "out_of_order":
        raise ValueError(f"out-of-order shoreline observation rejected: {outcome.reason}")
    return outcome.observation


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
        except InvalidFrameError as exc:
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
                    "reason": "invalid_frame",
                    "message": str(exc),
                    "frame_quality_score": quality.score,
                    "frame_quality_flags": list(quality.flags),
                },
            )

        calibration = self._active_calibration(camera)
        outline = self._active_roi(camera)
        class_mask = getattr(segmentation, "class_mask", None)
        try:
            if class_mask is not None:
                # Active path: Ocean/Beach boundary from the 3-class mask.
                result = extract_shoreline_from_classes(
                    class_mask,
                    calibration=calibration,
                    confidence=segmentation.confidence,
                    monitoring_polygon=outline,
                )
            else:
                # Compatibility for injected binary-mask segmenters.
                result = extract_shoreline(
                    segmentation.mask,
                    calibration=calibration,
                    confidence=segmentation.confidence,
                    monitoring_polygon=outline,
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

        if not getattr(result, "shoreline_available", True):
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
                    "reason": "no_shoreline",
                    "message": result.reason or "no Ocean/Beach contact in this frame",
                    "valid_columns": getattr(result, "valid_columns", 0),
                    "segmentation_confidence": segmentation.confidence,
                    "inference_ms": getattr(segmentation, "inference_ms", None),
                    "architecture": getattr(segmentation, "architecture", None),
                    "checkpoint_sha256": getattr(segmentation, "checkpoint_sha256", None),
                    "frame_quality_score": quality.score,
                    "frame_quality_flags": list(quality.flags),
                },
            )

        calibration_id = camera.active_calibration_id if calibration else None
        from app.services.video.pipeline import _horizon_y

        # Horizon describes the monitored area, not a distant false shoreline.
        monitored_mask = (
            apply_roi_to_class_mask(class_mask, outline) if class_mask is not None else None
        )
        horizon = _horizon_y(monitored_mask) if monitored_mask is not None else None
        # Provenance is honest per source. Uploaded stills only have file mtime,
        # which is labelled as such (low confidence) rather than presented as a
        # capture time. Virtual video-file clocks are labelled likewise.
        if camera.camera_type == "upload":
            timestamp_source, timestamp_confidence = "file_mtime", 0.3
        elif camera.camera_type == "video":
            timestamp_source, timestamp_confidence = "provider_virtual_clock", 0.3
        else:
            timestamp_source, timestamp_confidence = "stream_wall_clock", 0.6
        # Class fractions remain full-frame model diagnostics. The extracted
        # shoreline and horizon above are restricted to the monitoring outline.
        class_fractions = {
            name: row.get("fraction")
            for name, row in getattr(segmentation, "class_statistics", {}).items()
        }
        from app.services.recording.service import write_observation

        outcome = write_observation(
            self._session,
            camera,
            observed_at=frame.captured_at,
            result=result,
            calibration_id=calibration_id,
            quality_score=quality.score,
            quality_flags=list(quality.flags),
            source_frame_path=None,
            recording_id=f"adhoc-{camera.id}",
            frame_index=self._next_adhoc_frame_index(camera.id),
            elapsed_seconds=None,
            timestamp_source=timestamp_source,
            timestamp_confidence=timestamp_confidence,
            class_fractions=class_fractions,
            inference_time_ms=getattr(segmentation, "inference_ms", None),
            horizon_y_px=horizon,
            frame_shape=frame.image.shape[:2],
        )
        observation = outcome.observation
        if observation is None:
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
                    "reason": "out_of_order_observation",
                    "message": outcome.reason,
                    "timestamp_source": timestamp_source,
                },
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

        # Feed the rolling detector exactly once for the stored row. The
        # detector is warm-started from stored history so a one-off cycle is not
        # blind. A persistence failure above never reaches this point.
        rolling_row = None
        rolling_error = None
        if outcome.status in {"persisted", "updated"}:
            try:
                rolling_row = self._feed_rolling_once(camera, observation)
                self._session.commit()
            except Exception as exc:  # pragma: no cover - defensive
                self._session.rollback()
                rolling_error = str(exc)
                logger.warning("pipeline: rolling update failed for camera %s: %s", camera_id, exc)
        elif outcome.status == "duplicate":
            rolling_error = "duplicate_timestamp: rolling detector not fed twice"

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
                "segments": len(getattr(result, "segments", []) or []),
                "valid_columns": getattr(result, "valid_columns", 0),
                "orientation_used": getattr(result, "orientation_used", None),
                "bridge_kernel_px": getattr(result, "bridge_kernel_px", None),
                "median_kernel_px": getattr(result, "median_kernel_px", None),
                "max_gap_px": getattr(result, "max_gap_px", None),
                "margin_px": getattr(result, "margin_px", None),
                "calibration_version": result.calibration_version,
                "segmentation_confidence": segmentation.confidence,
                "inference_ms": getattr(segmentation, "inference_ms", None),
                "architecture": getattr(segmentation, "architecture", None),
                "checkpoint_sha256": getattr(segmentation, "checkpoint_sha256", None),
                "class_fractions": class_fractions,
                "frame_quality_score": quality.score,
                "frame_quality_flags": list(quality.flags),
                "timestamp_source": timestamp_source,
                "timestamp_confidence": timestamp_confidence,
                "signed_position_px": getattr(observation, "signed_position_px", None),
                "rolling_baseline_px": (
                    None if rolling_row is None else rolling_row.rolling_baseline_px
                ),
                "rolling_classification": (
                    None if rolling_row is None else rolling_row.classification
                ),
                "rolling_error": rolling_error,
                "note": "one canonical observation row; rolling row persisted separately",
            },
        )

    def _next_adhoc_frame_index(self, camera_id: int) -> int:
        from sqlalchemy import func, select

        from app.models.shoreline import ShorelineObservation

        if self._session is None:  # pragma: no cover - guarded by process_once
            return 1
        current = self._session.scalar(
            select(func.max(ShorelineObservation.frame_index)).where(
                ShorelineObservation.recording_id == f"adhoc-{camera_id}"
            )
        )
        return int(current or 0) + 1

    def _feed_rolling_once(self, camera: Any, observation: Any) -> Any:
        """Warm-start from stored history and update the detector once."""
        from app.services.anomaly.rolling_baseline import RollingBaselineConfig
        from app.services.recording.service import (
            load_rolling_history,
            observation_to_rolling,
            persist_rolling_row,
        )

        config = RollingBaselineConfig(
            nominal_interval_seconds=30.0,
            frame_height=(float(camera.resolution_height) if camera.resolution_height else None),
        )
        history = (
            []
            if self._session is None
            else load_rolling_history(
                self._session,
                camera.id,
                window_seconds=config.window_seconds,
                before=observation.observed_at,
            )
        )
        from app.services.anomaly.rolling_baseline import RollingBaselineDetector

        detector = RollingBaselineDetector(config)
        detector.warm_start([observation_to_rolling(item) for item in history])
        row = detector.update(observation_to_rolling(observation))
        if self._session is not None:  # pragma: no cover - guarded by process_once
            persist_rolling_row(
                self._session,
                camera,
                row,
                recording_id=f"adhoc-{camera.id}",
                frame_index=observation.frame_index,
            )
        return row

    def _active_roi(self, camera: Any) -> dict[str, Any] | None:
        """The active calibration's operator monitoring outline, if any."""
        if self._session is None:  # pragma: no cover - guarded by process_once
            return None
        return load_active_roi(self._session, camera)

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
