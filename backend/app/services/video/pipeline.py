"""Phase 5/6 prototype orchestration: video → shoreline → tide → baseline → anomaly.

This module is session-free. It uses the production SegFormer segmenter and the
production class-aware shoreline extractor; it does not implement a second
model or a second shoreline algorithm. Database persistence lives in
``app.services.video.persistence`` and is called by the API route/offline script.

The output is deliberately explicit about what is unknown:

* each sampled frame records its extraction status and reason;
* a missing shoreline or tide yields ``DATA_UNRELIABLE`` downstream;
* the baseline is fitted only when verified tide pairs support it;
* movement-only mode never emits ``NORMAL``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.logging import get_logger
from app.services.anomaly.detector import (
    AnomalyDetector,
    AnomalyObservation,
    Detection,
    DetectorConfig,
    derive_movement_thresholds,
    resolve_detector_mode,
    validate_detector_mode,
)
from app.services.anomaly.rolling_baseline import (
    BASELINE_STATUS_ROLLING,
    DETECTOR_MODE_ROLLING,
    RollingBaselineConfig,
    RollingEvaluation,
    RollingObservation,
    RollingRowResult,
    evaluate_rolling_baseline,
)
from app.services.baseline.expected_shoreline import (
    BaselineFitConfig,
    BaselineFitResult,
    PairedObservation,
    fit_tide_conditioned_baseline,
)
from app.services.video.sampling import (
    SampledFrame,
    VideoMetadata,
    frame_sha256,
    iter_sampled_frames,
    probe_video,
    stable_recording_id,
)
from app.services.video.timestamps import ClockAnchor, derive_timestamp
from app.services.vision.position import measure_shoreline_position
from app.services.vision.roi import apply_roi_to_class_mask
from app.services.vision.segmentation import ShorelineSegmenter
from app.services.vision.shoreline import (
    ShorelineExtractionError,
    extract_shoreline_from_classes,
)

logger = get_logger(__name__)

TideLookup = Callable[[datetime | None], Any]


@dataclass(slots=True)
class PrototypeConfig:
    sample_interval_seconds: float = 30.0
    start_elapsed: float = 0.0
    end_elapsed: float | None = None
    max_samples: int | None = None
    timezone: str = "Asia/Bangkok"
    country_iso: str | None = "TH"
    latitude: float | None = None
    longitude: float | None = None
    max_station_distance_km: float = 120.0
    camera_id: int | None = None
    monitoring_station_id: int | None = None
    min_baseline_samples: int = 30
    min_tide_range_m: float = 0.20
    min_phase_transitions: int = 3
    require_both_phases: bool = True
    # Near-shore region: lower half for a land_below camera. The full segments
    # are still retained; this only selects the primary monitored waterline.
    primary_region_fraction: float | None = 0.5
    persistence: int = 3
    z_threshold: float = 3.5
    min_confidence: float | None = None
    save_frames: bool = False
    frame_dir: str | None = None
    save_frame_compression: int = 3
    hash_frames: bool = True
    # Operational mode: auto | rolling_baseline | tide_conditioned | movement_only
    detector_mode: str = "auto"
    rolling: RollingBaselineConfig = field(default_factory=RollingBaselineConfig)
    frame_height: float | None = None
    # Active calibration monitoring outline (resolved once by the caller).
    roi_polygon: dict[str, Any] | None = None


@dataclass(slots=True)
class AnalysisStats:
    sampled: int = 0
    processed: int = 0
    available: int = 0
    unavailable: int = 0
    extraction_failures: int = 0
    segmentation_failures: int = 0
    total_inference_ms: float = 0.0
    total_processing_ms: float = 0.0
    inference_count: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    class_fraction_sums: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        count = max(1, self.inference_count)
        return {
            "sampled": self.sampled,
            "processed": self.processed,
            "available": self.available,
            "unavailable": self.unavailable,
            "extraction_failures": self.extraction_failures,
            "segmentation_failures": self.segmentation_failures,
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "mean_inference_ms": round(self.total_inference_ms / count, 3),
            "mean_processing_ms": round(self.total_processing_ms / max(1, self.processed), 3),
            "mean_class_fractions": {
                name: round(value / max(1, self.processed), 6)
                for name, value in sorted(self.class_fraction_sums.items())
            },
        }


@dataclass(slots=True)
class AnalysisResult:
    metadata: VideoMetadata
    recording_id: str
    observations: list[dict[str, Any]]
    detections: list[Detection]
    baseline: BaselineFitResult
    detector_config: DetectorConfig
    threshold_diagnostics: dict[str, Any]
    stats: AnalysisStats
    tide_selection: Any | None = None
    timings: dict[str, float] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    detector_mode: str = "auto"
    resolved_detector_mode: str = "movement_only"
    mode_reason: str = ""
    rolling_config: RollingBaselineConfig | None = None
    rolling_evaluation: RollingEvaluation | None = None


def _fractions(segmentation: Any) -> dict[str, float]:
    statistics = getattr(segmentation, "class_statistics", {}) or {}
    return {
        str(name): round(float(row.get("fraction", 0.0)), 6) for name, row in statistics.items()
    }


def _horizon_y(class_mask: Any) -> float | None:
    """Reference horizon from the model's native class mask.

    This uses the mask the extractor already holds; it is not a second
    segmentation or shoreline implementation. The horizon is the median first
    non-Sky row across columns. Returns ``None`` when no usable Sky boundary
    exists, so the camera-motion gate is reported as unavailable.
    """
    try:
        mask = np.asarray(class_mask)
        if mask.ndim != 2 or mask.size == 0:
            return None
        sky_id = 2  # Beach=0, Ocean=1, Sky=2
        non_sky = mask != sky_id
        has_non_sky = non_sky.any(axis=0)
        if not bool(has_non_sky.any()):
            return None
        first = np.argmax(non_sky, axis=0)
        values = first[has_non_sky]
        if values.size == 0:
            return None
        return float(np.median(values))
    except Exception:  # pragma: no cover - defensive
        return None


def _base_row(
    *,
    recording_id: str,
    camera_id: int | None,
    metadata: VideoMetadata,
    sampled: SampledFrame,
    anchors: list[ClockAnchor],
    timezone: str,
) -> dict[str, Any]:
    derived = derive_timestamp(sampled.elapsed_seconds, anchors, timezone=timezone)
    return {
        "recording_id": recording_id,
        "camera_id": camera_id,
        "frame_index": sampled.frame_index,
        "elapsed_seconds": sampled.elapsed_seconds,
        "capture_timestamp": (
            None if derived.capture_timestamp is None else derived.capture_timestamp.isoformat()
        ),
        "capture_timestamp_utc": (
            None
            if derived.capture_timestamp is None
            else derived.capture_timestamp.astimezone(UTC).isoformat()
        ),
        "timestamp_source": derived.timestamp_source,
        "timestamp_confidence": derived.timestamp_confidence,
        "timestamp_method": derived.method,
        "timestamp_reason": derived.reason,
        "timezone": timezone,
        "frame_path": None,
        "frame_sha256": None,
        "extraction_status": "pending",
        "extraction_reason": None,
        "shoreline_available": False,
        "reason_code": None,
        "shoreline_position_px": None,
        "signed_position_px": None,
        "position_axis": None,
        "position_sign": "seaward_positive",
        "segment_count": 0,
        "segment_measurements": [],
        "horizon_y_px": None,
        "shoreline_segments": [],
        "shoreline_geometry": [],
        "inference_time_ms": None,
        "confidence": None,
        "class_fractions": {},
        "orientation_used": None,
        "valid_columns": 0,
        "processing_time_ms": None,
        # Tide fields are filled by the association callback.
        "tide_data_available": False,
        "tide_station_id": None,
        "tide_station_code": None,
        "tide_timestamp": None,
        "tide_source": "unavailable",
        "tide_height_m": None,
        "tide_unit": None,
        "tide_vertical_datum": None,
        "tide_quality": None,
        "tide_reason": "no tide lookup configured",
    }


def _apply_tide(row: dict[str, Any], association: Any | None) -> None:
    if association is None:
        row["tide_reason"] = "no verified tide association"
        return
    row.update(
        {
            "tide_data_available": association.available,
            "tide_station_id": association.station_id,
            "tide_station_code": association.station_code,
            "tide_timestamp": (
                None if association.timestamp is None else association.timestamp.isoformat()
            ),
            "tide_source": association.source or "unavailable",
            "tide_height_m": association.height,
            "tide_unit": association.unit,
            "tide_vertical_datum": association.vertical_datum,
            "tide_quality": association.quality,
            "tide_reason": association.reason,
        }
    )


@dataclass(slots=True)
class EvaluationOutcome:
    baseline: BaselineFitResult
    detections: list[Detection]
    detector_config: DetectorConfig
    thresholds: dict[str, Any]
    resolved_detector_mode: str
    mode_reason: str
    rolling_config: RollingBaselineConfig | None = None
    rolling_evaluation: RollingEvaluation | None = None


def _rolling_observation(row: dict[str, Any]) -> RollingObservation:
    timestamp = (
        None
        if row.get("capture_timestamp") is None
        else datetime.fromisoformat(row["capture_timestamp"])
    )
    segments = row.get("segment_measurements")
    if isinstance(segments, str):
        try:
            import json

            segments = json.loads(segments)
        except ValueError:
            segments = None
    if segments is not None and not isinstance(segments, list):
        segments = None
    position = row.get("signed_position_px")
    return RollingObservation(
        timestamp=timestamp,
        position_px=None if position is None else float(position),
        shoreline_available=bool(row.get("shoreline_available")),
        confidence=row.get("confidence"),
        frame_index=row.get("frame_index"),
        timestamp_confidence=row.get("timestamp_confidence"),
        reason=row.get("extraction_reason") or row.get("reason_code"),
        segment_measurements=segments,
        horizon_y_px=row.get("horizon_y_px"),
    )


def _rolling_detection(
    row: RollingRowResult,
    *,
    resolved_detector_mode: str,
    mode_reason: str,
) -> Detection:
    from app.services.anomaly.detector import STATE_BY_CLASS
    from app.services.anomaly.scoring import compute_score

    score = compute_score(
        robust_z_value=row.robust_z,
        movement_rate=row.movement_rate_px_s,
        movement_threshold=None,
        persistence=row.persistence_count,
        required_persistence=3,
        quality=row.confidence,
    )
    factors = {
        key: float(value)
        for key, value in {
            "robust_z": row.robust_z,
            "cumulative_drawdown_px": row.cumulative_drawdown_px,
            "rolling_baseline_px": row.rolling_baseline_px,
            "rolling_scale_px": row.rolling_scale_px,
        }.items()
        if isinstance(value, int | float)
    }
    return Detection(
        elapsed_seconds=0.0,
        frame_index=row.frame_index,
        timestamp=row.timestamp,
        classification=row.classification,
        state=STATE_BY_CLASS.get(row.classification, "OBSERVED"),
        reason=row.reason,
        baseline_status=BASELINE_STATUS_ROLLING,
        tide_adjusted=False,
        tide_available=False,
        tide_source="unavailable",
        observed_position_px=row.observed_position_px,
        expected_position_px=row.rolling_baseline_px,
        residual_px=row.residual_px,
        robust_z=row.robust_z,
        movement_rate_px_s=row.movement_rate_px_s,
        persistence_count=row.persistence_count,
        tide_height_m=None,
        quality_status=row.quality_status,
        confidence=row.confidence,
        score=score.score,
        factors=factors,
        detector_mode=DETECTOR_MODE_ROLLING,
        resolved_detector_mode=resolved_detector_mode,
        mode_reason=mode_reason,
        rolling_baseline_px=row.rolling_baseline_px,
        rolling_scale_px=row.rolling_scale_px,
        scale_floored=row.scale_floored,
        cumulative_drawdown_px=row.cumulative_drawdown_px,
        baseline_frozen=row.baseline_frozen,
        in_event=row.in_event,
        event_max_classification=row.event_max_classification,
        onset_time=row.onset_time,
        detection_time=row.detection_time,
        detection_latency_s=row.detection_latency_s,
        event_window_seconds=row.event_window_seconds,
        cooldown_state=row.cooldown_state,
        onset_frame_index=row.onset_frame_index,
        detection_frame_index=row.detection_frame_index,
        segment_evidence=row.segment_evidence,
        config_hash=row.config_hash,
        contributing_factors=row.contributing_factors,
    )


def evaluate_observations_detailed(
    observations: list[dict[str, Any]],
    config: PrototypeConfig,
    *,
    frame_height: float | None = None,
) -> EvaluationOutcome:
    """Fit the tentative tide baseline and dispatch the requested detector.

    The legacy tide-conditioned and movement-only code paths are unchanged.
    ``auto`` selects the tide-independent rolling detector whenever no verified
    tide-conditioned baseline exists.
    """
    validate_detector_mode(config.detector_mode)
    paired = [
        PairedObservation(
            timestamp=datetime.fromisoformat(row["capture_timestamp"]),
            position_px=float(row["signed_position_px"]),
            tide_height_m=float(row["tide_height_m"]),
        )
        for row in observations
        if row["shoreline_available"]
        and row["capture_timestamp"] is not None
        and row["tide_data_available"]
        and row["signed_position_px"] is not None
        and row["tide_height_m"] is not None
    ]
    baseline_config = BaselineFitConfig(
        min_samples=config.min_baseline_samples,
        min_tide_range_m=config.min_tide_range_m,
        min_phase_transitions=config.min_phase_transitions,
        require_both_phases=config.require_both_phases,
    )
    if paired:
        baseline = fit_tide_conditioned_baseline(paired, baseline_config)
    else:
        tide_reason = next(
            (row["tide_reason"] for row in observations if row.get("tide_reason")), None
        )
        baseline = BaselineFitResult(
            available=False,
            status="unavailable",
            reason=(
                "no verified, geographically appropriate tide data was associated with "
                f"any valid observation ({tide_reason or 'tide unavailable'})"
            ),
            sample_count=0,
            min_samples_required=config.min_baseline_samples,
            provisional=True,
        )

    anomaly_observations = [
        AnomalyObservation(
            elapsed_seconds=float(row["elapsed_seconds"]),
            position_px=(
                None if row["signed_position_px"] is None else float(row["signed_position_px"])
            ),
            shoreline_available=bool(row["shoreline_available"]),
            timestamp=(
                None
                if row["capture_timestamp"] is None
                else datetime.fromisoformat(row["capture_timestamp"])
            ),
            frame_index=int(row["frame_index"]),
            reason=row["extraction_reason"] or row["reason_code"],
            confidence=row["confidence"],
            tide_height_m=(None if row["tide_height_m"] is None else float(row["tide_height_m"])),
            tide_available=bool(row["tide_data_available"]),
            tide_source=row["tide_source"],
            quality_status=("ok" if row["shoreline_available"] else "shoreline_unavailable"),
        )
        for row in observations
    ]
    # A missing sampled observation must break the persistence chain.
    max_gap_seconds = max(45.0, config.sample_interval_seconds * 1.5)
    thresholds = derive_movement_thresholds(
        anomaly_observations,
        persistence=config.persistence,
        max_gap_seconds=max_gap_seconds,
    )
    detector_config = DetectorConfig(
        persistence=config.persistence,
        z_threshold=config.z_threshold,
        movement_threshold_px_s=float(thresholds["rapid_threshold_px_s"]),
        drawdown_rate_threshold_px_s=float(thresholds["drawdown_rate_threshold_px_s"]),
        drawdown_distance_px=float(thresholds["drawdown_distance_px"]),
        max_gap_seconds=max_gap_seconds,
        min_confidence=config.min_confidence,
    )

    resolved, mode_reason = resolve_detector_mode(
        config.detector_mode, tide_baseline_available=baseline.available
    )
    if resolved == DETECTOR_MODE_ROLLING:
        height = frame_height if frame_height is not None else config.frame_height
        rolling_config = replace(
            config.rolling,
            frame_height=height,
            nominal_interval_seconds=config.sample_interval_seconds,
            rapid_rate_px_s=(
                config.rolling.rapid_rate_px_s
                if config.rolling.rapid_rate_px_s is not None
                else float(thresholds["rapid_threshold_px_s"])
            ),
        )
        rolling_config.validate()
        evaluation = evaluate_rolling_baseline(
            [_rolling_observation(row) for row in observations], rolling_config
        )
        detections = [
            _rolling_detection(
                row,
                resolved_detector_mode=resolved,
                mode_reason=mode_reason,
            )
            for row in evaluation.rows
        ]
        return EvaluationOutcome(
            baseline=baseline,
            detections=detections,
            detector_config=detector_config,
            thresholds=thresholds,
            resolved_detector_mode=resolved,
            mode_reason=mode_reason,
            rolling_config=rolling_config,
            rolling_evaluation=evaluation,
        )

    detections = AnomalyDetector(detector_config).evaluate_series(
        anomaly_observations, baseline=baseline
    )
    for detection in detections:
        detection.detector_mode = config.detector_mode
        detection.resolved_detector_mode = resolved
        detection.mode_reason = mode_reason
    return EvaluationOutcome(
        baseline=baseline,
        detections=detections,
        detector_config=detector_config,
        thresholds=thresholds,
        resolved_detector_mode=resolved,
        mode_reason=mode_reason,
    )


def evaluate_observations(
    observations: list[dict[str, Any]],
    config: PrototypeConfig,
) -> tuple[BaselineFitResult, list[Detection], DetectorConfig, dict[str, Any]]:
    """Backward-compatible wrapper used by the legacy evaluation callers."""
    outcome = evaluate_observations_detailed(observations, config)
    return (
        outcome.baseline,
        outcome.detections,
        outcome.detector_config,
        outcome.thresholds,
    )


def analyze_video(
    video_path: str | Path,
    config: PrototypeConfig,
    *,
    anchors: list[ClockAnchor],
    tide_lookup: TideLookup | None = None,
    tide_selection: Any | None = None,
    segmenter: Any | None = None,
    resume_from_frame: int | None = None,
    skip_frame_indices: set[int] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> AnalysisResult:
    """Analyse a video archive with the production vision path.

    ``tide_lookup(None)`` must return an unavailable association; it is called
    once per sampled frame with that frame's derived timestamp.
    """
    started = time.perf_counter()
    metadata = probe_video(video_path)
    recording_id = stable_recording_id(video_path, metadata)
    frame_dir = Path(config.frame_dir) if config.frame_dir else None
    if config.save_frames and frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)

    model = segmenter or ShorelineSegmenter(device="cpu")
    stats = AnalysisStats()
    observations: list[dict[str, Any]] = []
    last_progress = time.perf_counter()
    previous_signed_position: float | None = None
    previous_orientation: str | None = None
    previous_position_elapsed: float | None = None

    for sampled in iter_sampled_frames(
        video_path,
        metadata,
        sample_interval_seconds=config.sample_interval_seconds,
        start_elapsed=config.start_elapsed,
        end_elapsed=config.end_elapsed,
        max_samples=config.max_samples,
        resume_from_frame=resume_from_frame,
        skip_frame_indices=skip_frame_indices,
    ):
        frame_started = time.perf_counter()
        stats.sampled += 1
        row = _base_row(
            recording_id=recording_id,
            camera_id=config.camera_id,
            metadata=metadata,
            sampled=sampled,
            anchors=anchors,
            timezone=config.timezone,
        )
        row["monitoring_station_id"] = config.monitoring_station_id

        if sampled.status != "ok" or sampled.frame is None:
            row["extraction_status"] = "frame_unavailable"
            row["extraction_reason"] = sampled.error or sampled.status
            row["reason_code"] = sampled.status
            stats.extraction_failures += 1
            stats.reason_counts[sampled.status] = stats.reason_counts.get(sampled.status, 0) + 1
        else:
            frame = sampled.frame
            if config.hash_frames:
                row["frame_sha256"] = frame_sha256(frame)
            if config.save_frames and frame_dir is not None:
                frame_path = frame_dir / (
                    f"frame_{sampled.frame_index:09d}_t{sampled.elapsed_seconds:08.1f}.png"
                )
                ok = False
                try:
                    ok = bool(
                        cv2.imwrite(
                            str(frame_path),
                            frame,
                            [cv2.IMWRITE_PNG_COMPRESSION, int(config.save_frame_compression)],
                        )
                    )
                except Exception as exc:  # pragma: no cover - disk/permission failure
                    logger.warning("could not write %s: %s", frame_path, exc)
                row["frame_path"] = str(frame_path) if ok else None

            try:
                segmentation = model.predict(frame)
                stats.inference_count += 1
                stats.total_inference_ms += float(getattr(segmentation, "inference_ms", 0.0) or 0.0)
                row["inference_time_ms"] = getattr(segmentation, "inference_ms", None)
                row["confidence"] = round(float(segmentation.confidence), 6)
                fractions = _fractions(segmentation)
                row["class_fractions"] = fractions
                for name, value in fractions.items():
                    stats.class_fraction_sums[name] = (
                        stats.class_fraction_sums.get(name, 0.0) + value
                    )
                class_mask = getattr(segmentation, "class_mask", None)
                if class_mask is None:
                    raise ShorelineExtractionError("segmenter returned no native class mask")
                monitored_mask = apply_roi_to_class_mask(class_mask, config.roi_polygon)
                row["horizon_y_px"] = _horizon_y(monitored_mask)
                result = extract_shoreline_from_classes(
                    monitored_mask, confidence=segmentation.confidence
                )
                row["orientation_used"] = result.orientation_used
                row["valid_columns"] = result.valid_columns
                if not result.shoreline_available:
                    # Nothing to measure; the row records the reason below.
                    row["extraction_status"] = "unavailable"
                    row["reason_code"] = result.reason or "shoreline_unavailable"
                    row["extraction_reason"] = result.reason
                    stats.unavailable += 1
                    stats.reason_counts[row["reason_code"]] = (
                        stats.reason_counts.get(row["reason_code"], 0) + 1
                    )
                else:
                    continuity_ok = (
                        previous_orientation == result.orientation_used
                        and previous_position_elapsed is not None
                        and sampled.elapsed_seconds - previous_position_elapsed <= 300.0
                    )
                    position = measure_shoreline_position(
                        result.segments,
                        result.orientation_used,
                        previous_signed_position=(
                            previous_signed_position if continuity_ok else None
                        ),
                        frame_shape=frame.shape[:2],
                        primary_region_fraction=config.primary_region_fraction,
                    )
                    if position.available:
                        previous_signed_position = position.signed_position_px
                        previous_orientation = result.orientation_used
                        previous_position_elapsed = sampled.elapsed_seconds
                    row["shoreline_segments"] = result.segments
                    row["shoreline_geometry"] = result.points
                    row["shoreline_axis"] = result.shoreline_axis
                    row["interpolated_ranges"] = result.interpolated_ranges
                    row["measured_coverage"] = result.measured_coverage
                    row["segment_count"] = position.segment_count
                    row["segment_measurements"] = position.segment_dicts()
                    row["shoreline_position_px"] = position.position_px
                    row["signed_position_px"] = position.signed_position_px
                    row["position_axis"] = position.position_axis
                    row["position_sign"] = position.position_sign
                    if not position.available:
                        row["extraction_status"] = "unavailable"
                        row["reason_code"] = position.reason or "no_usable_segment"
                        row["extraction_reason"] = position.reason
                        stats.unavailable += 1
                        stats.reason_counts[row["reason_code"]] = (
                            stats.reason_counts.get(row["reason_code"], 0) + 1
                        )
                    else:
                        row["extraction_status"] = "available"
                        row["shoreline_available"] = True
                        stats.available += 1
            except ShorelineExtractionError as exc:
                row["extraction_status"] = "extraction_failed"
                row["extraction_reason"] = str(exc)
                row["reason_code"] = "shoreline_extraction_failed"
                stats.extraction_failures += 1
                stats.reason_counts[row["reason_code"]] = (
                    stats.reason_counts.get(row["reason_code"], 0) + 1
                )
            except Exception as exc:  # segmentation/model failure
                row["extraction_status"] = "segmentation_failed"
                row["extraction_reason"] = str(exc)
                row["reason_code"] = "segmentation_failed"
                stats.segmentation_failures += 1
                stats.reason_counts[row["reason_code"]] = (
                    stats.reason_counts.get(row["reason_code"], 0) + 1
                )
                logger.warning("segmentation failed for frame %s: %s", sampled.frame_index, exc)

        if tide_lookup is not None:
            capture = None
            if row["capture_timestamp"]:
                capture = datetime.fromisoformat(row["capture_timestamp"])
            try:
                _apply_tide(row, tide_lookup(capture))
            except Exception as exc:  # pragma: no cover - defensive
                row["tide_reason"] = f"tide lookup failed: {exc}"

        row["processing_time_ms"] = round((time.perf_counter() - frame_started) * 1000.0, 3)
        stats.processed += 1
        stats.total_processing_ms += row["processing_time_ms"]
        observations.append(row)
        if on_row is not None:
            on_row(row)

        if progress is not None and (
            stats.sampled % 25 == 0 or time.perf_counter() - last_progress > 15
        ):
            progress(
                {
                    "sampled": stats.sampled,
                    "available": stats.available,
                    "unavailable": stats.unavailable,
                    "frame_index": sampled.frame_index,
                }
            )
            last_progress = time.perf_counter()

    # --- baseline + anomaly ------------------------------------------------
    outcome = evaluate_observations_detailed(
        observations, config, frame_height=float(metadata.height) if metadata.height else None
    )
    baseline = outcome.baseline
    detections = outcome.detections
    detector_config = outcome.detector_config
    thresholds = outcome.thresholds

    limitations = [
        "Nighttime footage is out-of-domain for a model trained on daytime Karon imagery; "
        "many masks are poor or unavailable and are reported, never smoothed away.",
        "The recording covers ~4.3 hours, not a full tidal cycle; any fit is provisional.",
        "Single camera, single night, unlabelled night masks: no validated anomaly accuracy.",
    ]
    if outcome.resolved_detector_mode == DETECTOR_MODE_ROLLING:
        limitations.append(
            "Rolling-median mode is tide-independent (not tide-aware): it cannot "
            "establish an absolute expected shoreline or a NORMAL state, and a "
            "drawdown developing over many minutes may fall below the 90 s criterion."
        )

    return AnalysisResult(
        metadata=metadata,
        recording_id=recording_id,
        observations=observations,
        detections=detections,
        baseline=baseline,
        detector_config=detector_config,
        threshold_diagnostics=thresholds,
        stats=stats,
        tide_selection=tide_selection,
        timings={"total_seconds": round(time.perf_counter() - started, 3)},
        limitations=limitations,
        detector_mode=config.detector_mode,
        resolved_detector_mode=outcome.resolved_detector_mode,
        mode_reason=outcome.mode_reason,
        rolling_config=outcome.rolling_config,
        rolling_evaluation=outcome.rolling_evaluation,
    )
