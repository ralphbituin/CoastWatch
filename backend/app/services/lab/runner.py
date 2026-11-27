"""Predictor + anomaly test-lab execution (Phase 7).

Reuses the production ``ShorelineSegmenter``, ``extract_shoreline_from_classes``
and ``measure_shoreline_position``. It does not implement a second model or
shoreline extractor. Batch sources use ``evaluate_rolling_baseline``; live
sources use the same stateful detector.

Persistence is off by default. When the operator sets ``persist=true`` and picks
a camera, rows go through the Phase 7 recording service with
``recording_id = run_id`` and ``frame_index = sample index``.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import numpy as np

from app.core.exceptions import ModelUnavailableError
from app.core.logging import get_logger
from app.models.camera import Camera
from app.services.anomaly.rolling_baseline import (
    DETECTOR_MODE_ROLLING,
    RollingBaselineConfig,
    RollingBaselineDetector,
    RollingRowResult,
)
from app.services.lab.registry import LabRun, write_artifacts
from app.services.recording.service import (
    persist_rolling_rows,
    write_observation,
)
from app.services.video.persistence import persist_analysis
from app.services.video.pipeline import (
    PrototypeConfig,
    _horizon_y,
    analyze_video,
    evaluate_observations_detailed,
)
from app.services.video.sampling import probe_video, stable_recording_id
from app.services.video.timestamps import ClockAnchor, anchor_dicts
from app.services.vision.position import measure_shoreline_position
from app.services.vision.roi import apply_roi_to_class_mask
from app.services.vision.segmentation import InvalidFrameError, ShorelineSegmenter
from app.services.vision.shoreline import (
    ShorelineExtractionError,
    extract_shoreline_from_classes,
)

logger = get_logger(__name__)

OVERLAY_LIMIT = 5
MAX_MANIFEST_BYTES = 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# Timestamp sources
# ---------------------------------------------------------------------------
def _tz(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:  # pragma: no cover - platform data
        raise ValueError(f"unknown timezone {name!r}") from exc


def _parse_timestamp(value: str, timezone: str) -> datetime:
    try:
        moment = datetime.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp {value!r}") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_tz(timezone))
    return moment.astimezone(UTC)


def parse_manifest(text: str) -> dict[str, str]:
    """Parse a filename→observed_at manifest (JSON object/list or CSV)."""
    if len(text.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError("timestamp manifest is too large")
    stripped = text.strip()
    if not stripped:
        raise ValueError("timestamp manifest is empty")
    mapping: dict[str, str] = {}
    if stripped.startswith("{") or stripped.startswith("["):
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    value = value.get("observed_at") or value.get("timestamp")
                mapping[str(key)] = str(value)
        elif isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError("manifest list entries must be objects")
                name = item.get("filename") or item.get("name")
                stamp = item.get("observed_at") or item.get("timestamp")
                if not name or not stamp:
                    raise ValueError("manifest entries need filename and observed_at")
                mapping[str(name)] = str(stamp)
        else:  # pragma: no cover - guarded above
            raise ValueError("unsupported manifest JSON")
    else:
        reader = csv.reader(io.StringIO(stripped))
        for row in reader:
            if not row or row[0].strip().lower() in {"filename", "name", "file"}:
                continue
            if len(row) < 2:
                raise ValueError("manifest CSV rows need filename,observed_at")
            mapping[row[0].strip()] = row[1].strip()
    if not mapping:
        raise ValueError("timestamp manifest contains no entries")
    return mapping


def resolve_image_timestamps(
    filenames: list[str],
    *,
    manifest_text: str | None,
    start_at: str | None,
    interval_seconds: float | None,
    timezone: str,
) -> tuple[list[datetime], str]:
    """Return per-file timestamps and their documented provenance.

    Requires a manifest or start+interval. File modification time is never used.
    """
    if manifest_text:
        mapping = parse_manifest(manifest_text)
        stamps: list[datetime] = []
        for name in filenames:
            direct = mapping.get(name) or mapping.get(Path(name).name)
            if direct is None:
                raise ValueError(f"no timestamp for uploaded file {name!r} in manifest")
            stamps.append(_parse_timestamp(direct, timezone))
        source = "manifest"
    elif start_at and interval_seconds is not None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        start = _parse_timestamp(start_at, timezone)
        stamps = [
            start + timedelta(seconds=index * interval_seconds) for index in range(len(filenames))
        ]
        source = "operator_declared"
    else:
        raise ValueError(
            "batch images require a per-file timestamp manifest or an operator "
            "start time plus interval; file modification time is not used"
        )
    previous: datetime | None = None
    for stamp in stamps:
        if previous is not None and stamp <= previous:
            raise ValueError(
                "image timestamps must be strictly increasing; "
                "reorder the files or fix the manifest"
            )
        previous = stamp
    return stamps, source


# ---------------------------------------------------------------------------
# Shared row building
# ---------------------------------------------------------------------------
def _local_iso(moment: datetime | None, timezone: str) -> str | None:
    if moment is None:
        return None
    return moment.astimezone(_tz(timezone)).isoformat()


def _empty_observation(
    *,
    frame_index: int,
    observed_at: datetime | None,
    elapsed_seconds: float | None,
    timezone: str,
    timestamp_source: str,
    timestamp_confidence: float | None,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "capture_timestamp": None if observed_at is None else observed_at.isoformat(),
        "capture_timestamp_utc": None if observed_at is None else observed_at.isoformat(),
        "observed_at": None if observed_at is None else observed_at.isoformat(),
        "observed_at_local": _local_iso(observed_at, timezone),
        "elapsed_seconds": elapsed_seconds,
        "timestamp_source": timestamp_source,
        "timestamp_confidence": timestamp_confidence,
        "capture_timestamp_source": timestamp_source,
        "capture_timestamp_confidence": timestamp_confidence,
        "timezone": timezone,
        "shoreline_available": False,
        "extraction_status": "unavailable",
        "extraction_reason": None,
        "reason_code": None,
        "confidence": None,
        "segmentation_confidence": None,
        "inference_time_ms": None,
        "class_fractions": {},
        "horizon_y_px": None,
        "shoreline_geometry": None,
        "shoreline_segments": None,
        "shoreline_position_px": None,
        "signed_position_px": None,
        "position_axis": None,
        "position_sign": "seaward_positive",
        "segment_count": 0,
        "segment_measurements": None,
        "frame_height": None if image_shape is None else image_shape[0],
        "frame_width": None if image_shape is None else image_shape[1],
        "tide_data_available": False,
        "tide_height_m": None,
        "tide_source": "unavailable",
        "tide_reason": "no verified tide data is used by the test lab",
        "quality_status": "shoreline_unavailable",
    }


def _apply_prediction(
    row: dict[str, Any],
    *,
    image: np.ndarray,
    segmentation: Any,
    result: Any,
    position: Any,
) -> None:
    row["confidence"] = float(segmentation.confidence)
    row["segmentation_confidence"] = float(segmentation.confidence)
    row["inference_time_ms"] = getattr(segmentation, "inference_ms", None)
    row["class_fractions"] = {
        str(name): float(entry.get("fraction", 0.0))
        for name, entry in (getattr(segmentation, "class_statistics", {}) or {}).items()
    }
    row["shoreline_geometry"] = result.points
    row["shoreline_segments"] = result.segments
    row["shoreline_axis"] = getattr(result, "shoreline_axis", None)
    row["interpolated_ranges"] = getattr(result, "interpolated_ranges", [])
    row["measured_coverage"] = getattr(result, "measured_coverage", None)
    row["horizon_y_px"] = _horizon_y(getattr(segmentation, "class_mask", None))
    if position.available:
        row["shoreline_available"] = True
        row["extraction_status"] = "available"
        row["extraction_reason"] = None
        row["quality_status"] = "ok"
        row["shoreline_position_px"] = position.position_px
        row["signed_position_px"] = position.signed_position_px
        row["position_axis"] = position.position_axis
        row["position_sign"] = position.position_sign
        row["segment_count"] = position.segment_count
        row["segment_measurements"] = position.segment_dicts()
    else:
        row["extraction_status"] = "unavailable"
        row["extraction_reason"] = position.reason or result.reason or "no_usable_segment"
        row["reason_code"] = row["extraction_reason"]


def _merge_rolling(
    observations: list[dict[str, Any]], rolling_rows: list[RollingRowResult]
) -> list[dict[str, Any]]:
    by_frame: dict[int | None, RollingRowResult] = {}
    for rolling_row in rolling_rows:
        by_frame[rolling_row.frame_index] = rolling_row
    merged: list[dict[str, Any]] = []
    for observation in observations:
        row = dict(observation)
        rolling = by_frame.get(observation.get("frame_index"))
        if rolling is not None:
            row.update(
                {
                    "rolling_baseline_px": rolling.rolling_baseline_px,
                    "rolling_scale_px": rolling.rolling_scale_px,
                    "scale_floored": rolling.scale_floored,
                    "residual_px": rolling.residual_px,
                    "robust_z": rolling.robust_z,
                    "cumulative_drawdown_px": rolling.cumulative_drawdown_px,
                    "movement_rate_px_s": rolling.movement_rate_px_s,
                    "classification": rolling.classification,
                    "state": rolling.state,
                    "quality_status": rolling.quality_status or row.get("quality_status"),
                    "in_event": rolling.in_event,
                    "event_max_classification": rolling.event_max_classification,
                    "onset_time": (
                        None if rolling.onset_time is None else rolling.onset_time.isoformat()
                    ),
                    "detection_time": (
                        None
                        if rolling.detection_time is None
                        else rolling.detection_time.isoformat()
                    ),
                    "detection_latency_s": rolling.detection_latency_s,
                    "persistence_count": rolling.persistence_count,
                    "history_count": rolling.history_count,
                    "history_required": rolling.history_required,
                    "camera_motion_gate": rolling.camera_motion_gate,
                    "segment_evidence": rolling.segment_evidence,
                    "reason": rolling.reason,
                }
            )
        merged.append(row)
    return merged


def _rolling_config_from(
    base: RollingBaselineConfig, *, frame_height: float | None, interval: float | None
) -> RollingBaselineConfig:
    from dataclasses import replace

    config = replace(
        base,
        frame_height=frame_height,
        nominal_interval_seconds=interval,
    )
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def _overlay(image: np.ndarray, row: dict[str, Any], *, label: str) -> np.ndarray:
    overlay = image.copy()
    segments = row.get("shoreline_segments") or []
    for segment in segments:
        try:
            points = np.asarray(segment, dtype=np.int32).reshape((-1, 1, 2))
        except Exception:  # pragma: no cover - malformed stored geometry
            continue
        if len(points) < 2:
            continue
        cv2.polylines(overlay, [points], False, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.polylines(overlay, [points], False, (0, 245, 255), 2, cv2.LINE_AA)
    text = f"{label} #{row.get('frame_index')} {row.get('classification', '')}"
    cv2.putText(
        overlay, text[:70], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        overlay, text[:70], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    return overlay


def _write_overlay(run: LabRun, name: str, image: np.ndarray) -> str | None:
    if run.temp_dir is None:
        return None
    path = run.temp_dir / name
    try:
        ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    except Exception:  # pragma: no cover - disk/permission
        return None
    return str(path) if ok else None


def run_images(
    run: LabRun,
    *,
    images: list[tuple[str, bytes]],
    timestamps: list[datetime],
    timestamp_source: str,
    rolling: RollingBaselineConfig,
    interval_seconds: float | None,
    persist: bool,
    camera_id: int | None,
    session_factory: Callable[[], Any] | None = None,
    segmenter_factory: Callable[[], Any] | None = None,
    roi_polygon: dict[str, Any] | None = None,
) -> None:
    run.update(status="running")
    segmenter = (segmenter_factory or ShorelineSegmenter)()
    observations: list[dict[str, Any]] = []
    heights: list[int] = []
    try:
        for index, ((name, payload), observed_at) in enumerate(
            zip(images, timestamps, strict=True)
        ):
            if run.cancel_event.is_set():
                run.update(status="cancelled")
                return
            elapsed = (observed_at - timestamps[0]).total_seconds()
            buffer = np.frombuffer(payload, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            row = _empty_observation(
                frame_index=index,
                observed_at=observed_at,
                elapsed_seconds=elapsed,
                timezone=run.timezone,
                timestamp_source=timestamp_source,
                timestamp_confidence=0.8,
                image_shape=None if image is None else image.shape[:2],
            )
            if image is None:
                row["extraction_status"] = "invalid_image"
                row["extraction_reason"] = "image could not be decoded"
                row["reason_code"] = "invalid_image"
                observations.append(row)
                run.set_progress(processed=index + 1, total=len(images), image=name)
                continue
            heights.append(int(image.shape[0]))
            try:
                segmentation = segmenter.predict(image)
                class_mask = getattr(segmentation, "class_mask", None)
                if class_mask is None:
                    raise ShorelineExtractionError("segmenter returned no native class mask")
                monitored_mask = apply_roi_to_class_mask(class_mask, roi_polygon)
                result = extract_shoreline_from_classes(
                    monitored_mask, confidence=segmentation.confidence
                )
                position = measure_shoreline_position(
                    result.segments,
                    result.orientation_used,
                    frame_shape=image.shape[:2],
                    primary_region_fraction=0.5,
                )
                _apply_prediction(
                    row,
                    image=image,
                    segmentation=segmentation,
                    result=result,
                    position=position,
                )
            except ModelUnavailableError as exc:
                row["extraction_status"] = "segmentation_failed"
                row["extraction_reason"] = str(exc)
                row["reason_code"] = "model_unavailable"
            except InvalidFrameError as exc:
                row["extraction_status"] = "invalid_frame"
                row["extraction_reason"] = str(exc)
                row["reason_code"] = "invalid_frame"
            except ShorelineExtractionError as exc:
                row["extraction_status"] = "extraction_failed"
                row["extraction_reason"] = str(exc)
                row["reason_code"] = "shoreline_extraction_failed"
            except Exception as exc:  # pragma: no cover - segmentation backend
                row["extraction_status"] = "segmentation_failed"
                row["extraction_reason"] = str(exc)
                row["reason_code"] = "segmentation_failed"
            observations.append(row)
            if index < OVERLAY_LIMIT and row["shoreline_available"]:
                overlay = _overlay(image, {**row, "classification": "image"}, label=name)
                artifact = _write_overlay(run, f"overlay_{index:03d}.jpg", overlay)
                if artifact:
                    run.artifacts[f"overlay_{index:03d}.jpg"] = artifact
            run.set_progress(processed=index + 1, total=len(images), image=name)
            run.update(progress=dict(run.progress))

        frame_height = float(statistics.median(heights)) if heights else None
        config = _rolling_config_from(rolling, frame_height=frame_height, interval=interval_seconds)
        prototype = PrototypeConfig(
            sample_interval_seconds=interval_seconds or 30.0,
            timezone=run.timezone,
            detector_mode=DETECTOR_MODE_ROLLING,
            rolling=config,
            frame_height=frame_height,
            primary_region_fraction=0.5,
            camera_id=camera_id,
            monitoring_station_id=None,
        )
        outcome = evaluate_observations_detailed(observations, prototype)
        rolling_rows = (
            outcome.rolling_evaluation.rows if outcome.rolling_evaluation is not None else []
        )
        run.update(rows=_merge_rolling(observations, rolling_rows))
        if outcome.rolling_evaluation is not None:
            run.update(events=[event.as_dict() for event in outcome.rolling_evaluation.events])
        if persist:
            persisted = _persist_batch_rows(
                run,
                observations=observations,
                rolling_rows=rolling_rows,
                camera_id=camera_id,
                session_factory=session_factory,
            )
            run.set_progress(persisted=persisted)
        run.update(status="completed")
    except Exception as exc:
        logger.exception("lab image run %s failed", run.run_id)
        run.errors.append(str(exc))
        run.update(status="failed")
    finally:
        write_artifacts(run, report=_report(run))


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
def run_video(
    run: LabRun,
    *,
    video_path: Path,
    anchors: list[ClockAnchor],
    rolling: RollingBaselineConfig,
    sample_interval_seconds: float,
    start_elapsed: float,
    end_elapsed: float | None,
    max_samples: int,
    persist: bool,
    camera_id: int | None,
    session_factory: Callable[[], Any] | None = None,
    segmenter_factory: Callable[[], Any] | None = None,
    roi_polygon: dict[str, Any] | None = None,
) -> None:
    run.update(status="running")
    try:
        metadata = probe_video(video_path)
        recording_id = stable_recording_id(video_path, metadata)
        frame_dir = (run.temp_dir / "frames") if run.temp_dir is not None else None
        config = PrototypeConfig(
            sample_interval_seconds=sample_interval_seconds,
            start_elapsed=start_elapsed,
            end_elapsed=end_elapsed,
            max_samples=max_samples,
            timezone=run.timezone,
            detector_mode=DETECTOR_MODE_ROLLING,
            rolling=rolling,
            frame_height=float(metadata.height) if metadata.height else None,
            primary_region_fraction=0.5,
            camera_id=camera_id,
            save_frames=frame_dir is not None,
            frame_dir=str(frame_dir) if frame_dir is not None else None,
            save_frame_compression=6,
            hash_frames=False,
            roi_polygon=roi_polygon,
        )

        def progress(payload: dict[str, Any]) -> None:
            run.set_progress(
                processed=payload.get("sampled", 0),
                available=payload.get("available", 0),
                unavailable=payload.get("unavailable", 0),
                frame_index=payload.get("frame_index"),
            )

        result = analyze_video(
            video_path,
            config,
            anchors=anchors,
            tide_lookup=None,
            segmenter=(segmenter_factory or ShorelineSegmenter)(),
            progress=progress,
        )
        rolling_rows = (
            result.rolling_evaluation.rows if result.rolling_evaluation is not None else []
        )
        run.update(rows=_merge_rolling(result.observations, rolling_rows))
        if result.rolling_evaluation is not None:
            run.update(events=[event.as_dict() for event in result.rolling_evaluation.events])
        run.update(
            warnings=list(result.limitations),
            source={
                "kind": "video",
                "recording_id": recording_id,
                "file_name": video_path.name,
                "duration_seconds": metadata.duration_seconds,
                "anchors": anchor_dicts(anchors),
                "timestamp_source": "derived" if anchors else "unavailable",
            },
        )
        # Overlay up to five stored frames using the same segments.
        for row in run.rows:
            overlay_count = len([name for name in run.artifacts if name.startswith("overlay_")])
            if overlay_count >= OVERLAY_LIMIT:
                break
            frame_path = row.get("frame_path")
            if not frame_path or not row.get("shoreline_available"):
                continue
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            name = f"overlay_{row['frame_index']:09d}.jpg"
            artifact = _write_overlay(run, name, _overlay(image, row, label="video"))
            if artifact:
                run.artifacts[name] = artifact
        if persist:
            if camera_id is None:
                raise ValueError("persist=true requires camera_id for video runs")
            if session_factory is None:
                from app.database.session import get_session_factory

                session_factory = get_session_factory()
            with session_factory() as session:
                camera = session.get(Camera, camera_id)
                if camera is None:
                    raise ValueError(f"camera {camera_id} not found")
                summary = persist_analysis(
                    session,
                    result,
                    camera=camera,
                    anchors=anchor_dicts(anchors),
                    timezone=run.timezone,
                    timestamp_source="derived" if anchors else "unavailable",
                )
                session.commit()
            run.set_progress(persisted=summary.get("observations_persisted", 0))
        run.update(status="completed")
    except Exception as exc:
        logger.exception("lab video run %s failed", run.run_id)
        run.errors.append(str(exc))
        run.update(status="failed")
    finally:
        write_artifacts(run, report=_report(run))


# ---------------------------------------------------------------------------
# Live stream
# ---------------------------------------------------------------------------
def run_stream(
    run: LabRun,
    *,
    provider: Any,
    rolling: RollingBaselineConfig,
    sample_interval_seconds: float,
    max_samples: int,
    max_duration_seconds: float,
    persist: bool,
    camera_id: int | None,
    session_factory: Callable[[], Any] | None = None,
    segmenter_factory: Callable[[], Any] | None = None,
    roi_polygon: dict[str, Any] | None = None,
) -> None:
    run.update(status="running")
    observations: list[dict[str, Any]] = []
    detector: RollingBaselineDetector | None = None
    rolling_rows: list[RollingRowResult] = []
    started = time.monotonic()
    next_slot = started
    previous_expired: datetime | None = None
    segmenter = (segmenter_factory or ShorelineSegmenter)()
    try:
        provider.connect()
        index = 0
        while index < max_samples:
            if run.cancel_event.is_set():
                run.update(status="cancelled")
                return
            if time.monotonic() - started >= max_duration_seconds:
                break
            frame = provider.read_frame()
            observed_at = frame.captured_at
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=UTC)
            if previous_expired is not None and observed_at <= previous_expired:
                run.warnings.append(f"frame {index}: non-increasing wall-clock timestamp dropped")
                index += 1
                continue
            previous_expired = observed_at
            row = _empty_observation(
                frame_index=index,
                observed_at=observed_at,
                elapsed_seconds=max(0.0, time.monotonic() - started),
                timezone=run.timezone,
                timestamp_source="stream_wall_clock",
                timestamp_confidence=0.6,
                image_shape=frame.image.shape[:2],
            )
            try:
                segmentation = segmenter.predict(frame.image)
                class_mask = getattr(segmentation, "class_mask", None)
                if class_mask is None:
                    raise ShorelineExtractionError("segmenter returned no native class mask")
                monitored_mask = apply_roi_to_class_mask(class_mask, roi_polygon)
                result = extract_shoreline_from_classes(
                    monitored_mask, confidence=segmentation.confidence
                )
                position = measure_shoreline_position(
                    result.segments,
                    result.orientation_used,
                    frame_shape=frame.image.shape[:2],
                    primary_region_fraction=0.5,
                )
                _apply_prediction(
                    row,
                    image=frame.image,
                    segmentation=segmentation,
                    result=result,
                    position=position,
                )
            except Exception as exc:
                row["extraction_status"] = "unavailable"
                row["extraction_reason"] = str(exc)
                row["reason_code"] = "stream_inference_failed"
            observations.append(row)

            if detector is None:
                detector = RollingBaselineDetector(
                    _rolling_config_from(
                        rolling,
                        frame_height=float(frame.image.shape[0]),
                        interval=sample_interval_seconds,
                    )
                )
            rolling_row = detector.update(_rolling_observation(row))
            rolling_rows.append(rolling_row)
            if index < OVERLAY_LIMIT:
                name = f"overlay_{index:03d}.jpg"
                artifact = _write_overlay(
                    run,
                    name,
                    _overlay(
                        frame.image,
                        {**row, "classification": rolling_row.classification},
                        label="stream",
                    ),
                )
                if artifact:
                    run.artifacts[name] = artifact
            index += 1
            run.set_progress(
                processed=index,
                total=max_samples,
                classification=rolling_row.classification,
            )
            if persist:
                _persist_one_stream_row(
                    run,
                    row=row,
                    rolling_row=rolling_row,
                    camera_id=camera_id,
                    session_factory=session_factory,
                )
            # Monotonic pacing: drop missed slots rather than queue frames.
            next_slot += sample_interval_seconds
            delay = next_slot - time.monotonic()
            if delay > 0.05:
                time.sleep(min(delay, sample_interval_seconds))
            else:
                next_slot = time.monotonic()
        run.update(rows=_merge_rolling(observations, rolling_rows))
        events = []
        if detector is not None:
            events = [event.as_dict() for event in detector.finished_events]
            if detector.current_event is not None:
                events.append(detector.current_event.as_dict())
        run.update(events=events, status="completed")
    except Exception as exc:
        logger.exception("lab stream run %s failed", run.run_id)
        errors = getattr(provider, "last_error", None)
        run.errors.append(str(exc))
        run.warnings.append("DATA_UNRELIABLE: the stream produced no fabricated rows after failure")
        if errors:
            run.warnings.append(str(errors))
        run.update(status="failed")
    finally:
        with contextlib.suppress(Exception):
            provider.disconnect()
        write_artifacts(run, report=_report(run))


def _rolling_observation(row: dict[str, Any]) -> Any:
    from app.services.anomaly.rolling_baseline import RollingObservation

    return RollingObservation(
        timestamp=(
            None if row.get("observed_at") is None else datetime.fromisoformat(row["observed_at"])
        ),
        position_px=row.get("signed_position_px"),
        shoreline_available=bool(row.get("shoreline_available")),
        confidence=row.get("segmentation_confidence"),
        frame_index=row.get("frame_index"),
        timestamp_confidence=row.get("timestamp_confidence"),
        reason=row.get("extraction_reason"),
        segment_measurements=row.get("segment_measurements"),
        horizon_y_px=row.get("horizon_y_px"),
    )


def started_at(run: LabRun) -> datetime:
    return run.created_at


# ---------------------------------------------------------------------------
# Opt-in persistence (same recording service)
# ---------------------------------------------------------------------------
def _persist_batch_rows(
    run: LabRun,
    *,
    observations: list[dict[str, Any]],
    rolling_rows: list[RollingRowResult],
    camera_id: int | None,
    session_factory: Callable[[], Any] | None,
) -> int:
    if camera_id is None:
        raise ValueError("persist=true requires camera_id")
    if session_factory is None:
        from app.database.session import get_session_factory

        session_factory = get_session_factory()
    with session_factory() as session:
        camera = session.get(Camera, camera_id)
        if camera is None:
            raise ValueError(f"camera {camera_id} not found")
        written = 0
        for row in observations:
            observed_at = row.get("observed_at")
            if not observed_at:
                continue
            outcome = write_observation(
                session,
                camera,
                observed_at=datetime.fromisoformat(observed_at),
                recording_id=run.run_id,
                frame_index=row.get("frame_index"),
                elapsed_seconds=row.get("elapsed_seconds"),
                timestamp_source=row.get("timestamp_source"),
                timestamp_confidence=row.get("timestamp_confidence"),
                extraction_status=row.get("extraction_status"),
                extraction_reason=row.get("extraction_reason"),
                shoreline_geometry=row.get("shoreline_geometry"),
                shoreline_segments=row.get("shoreline_segments"),
                segmentation_confidence=row.get("segmentation_confidence"),
                shoreline_position_px=row.get("shoreline_position_px"),
                signed_position_px=row.get("signed_position_px"),
                position_axis=row.get("position_axis"),
                position_sign=row.get("position_sign"),
                segment_count=row.get("segment_count") or 0,
                segment_measurements=row.get("segment_measurements"),
                class_fractions=row.get("class_fractions"),
                inference_time_ms=row.get("inference_time_ms"),
                horizon_y_px=row.get("horizon_y_px"),
            )
            if outcome.status in {"persisted", "updated"}:
                written += 1
        persist_rolling_rows(session, camera, rolling_rows, recording_id=run.run_id)
        session.commit()
    return written


def _persist_one_stream_row(
    run: LabRun,
    *,
    row: dict[str, Any],
    rolling_row: RollingRowResult,
    camera_id: int | None,
    session_factory: Callable[[], Any] | None,
) -> None:
    if camera_id is None:
        raise ValueError("persist=true requires camera_id")
    if session_factory is None:
        from app.database.session import get_session_factory

        session_factory = get_session_factory()
    with session_factory() as session:
        camera = session.get(Camera, camera_id)
        if camera is None:
            raise ValueError(f"camera {camera_id} not found")
        observed_at = row.get("observed_at")
        if not observed_at:
            return
        outcome = write_observation(
            session,
            camera,
            observed_at=datetime.fromisoformat(observed_at),
            recording_id=run.run_id,
            frame_index=row.get("frame_index"),
            elapsed_seconds=row.get("elapsed_seconds"),
            timestamp_source=row.get("timestamp_source"),
            timestamp_confidence=row.get("timestamp_confidence"),
            extraction_status=row.get("extraction_status"),
            extraction_reason=row.get("extraction_reason"),
            shoreline_geometry=row.get("shoreline_geometry"),
            shoreline_segments=row.get("shoreline_segments"),
            segmentation_confidence=row.get("segmentation_confidence"),
            shoreline_position_px=row.get("shoreline_position_px"),
            signed_position_px=row.get("signed_position_px"),
            position_axis=row.get("position_axis"),
            position_sign=row.get("position_sign"),
            segment_count=row.get("segment_count") or 0,
            segment_measurements=row.get("segment_measurements"),
            class_fractions=row.get("class_fractions"),
            inference_time_ms=row.get("inference_time_ms"),
            horizon_y_px=row.get("horizon_y_px"),
        )
        if outcome.status in {"persisted", "updated"}:
            persist_rolling_rows(session, camera, [rolling_row], recording_id=run.run_id)
        session.commit()


def _report(run: LabRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "kind": run.kind,
        "status": run.status,
        "timezone": run.timezone,
        "persist": run.persist,
        "camera_id": run.camera_id,
        "source": run.source,
        "progress": run.progress,
        "row_count": len(run.rows),
        "event_count": len(run.events),
        "errors": run.errors,
        "warnings": run.warnings,
        "detector": {
            "mode": DETECTOR_MODE_ROLLING,
            "note": (
                "Experimental rolling-baseline drawdown detector — not validated "
                "for tsunami detection. No tide data is fabricated."
            ),
        },
    }
