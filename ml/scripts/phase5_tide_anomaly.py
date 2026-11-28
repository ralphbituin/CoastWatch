"""Phase 5/6 tide-aware shoreline anomaly prototype (offline, honest).

Streams the archive, samples one frame every N seconds, runs the **production**
SegFormer ONNX segmenter and class-aware shoreline extractor, derives timestamps
from the verified burned-in clock anchors, associates tide data only when a
geographically appropriate station exists, fits a provisional robust baseline
only when the data supports it, and scores transparent anomaly classifications.

Run with the backend virtualenv (it has the production dependencies)::

    backend\\.venv\\Scripts\\python.exe ml\\scripts\\phase5_tide_anomaly.py

Outputs (default ``ml/reports/phase5_tide_anomaly``):
  video_metadata.json, clock_anchors.json, observations.csv,
  processing_report.json, anomaly_results.csv, extracted_frames/, plots/samples/

Plot generation is separate (``ml/scripts/plot_phase5_results.py``) so this
runner does not require matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import tracemalloc
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rolling_baseline_analysis as rolling_report  # noqa: E402
from app.services.anomaly.detector import (  # noqa: E402
    ANOMALY_CLASSES,
    validate_detector_mode,
)
from app.services.anomaly.rolling_baseline import RollingBaselineConfig  # noqa: E402
from app.services.tides.association import (  # noqa: E402
    TideAssociation,
    select_tide_station,
)
from app.services.video.pipeline import (  # noqa: E402
    PrototypeConfig,
    analyze_video,
    evaluate_observations_detailed,
)
from app.services.video.sampling import probe_video, target_samples  # noqa: E402
from app.services.video.timestamps import ClockAnchor, anchor_dicts  # noqa: E402
from app.services.vision.position import measure_shoreline_position  # noqa: E402
from app.services.vision.segmentation import ShorelineSegmenter  # noqa: E402
from app.services.vision.shoreline import extract_shoreline_from_classes  # noqa: E402

DEFAULT_VIDEO = REPO_ROOT / "data" / "videos" / "webcam_6h.mp4"
DEFAULT_OUTPUT = REPO_ROOT / "ml" / "reports" / "phase5_tide_anomaly"
DEFAULT_TIMEZONE = "Asia/Bangkok"
DEFAULT_LATITUDE = 7.8433
DEFAULT_LONGITUDE = 98.2939

# Verified by manually reading the burned-in CCTV clock in extracted frames.
# OCR was not available/reliable; timestamps for other frames are DERIVED.
DEFAULT_ANCHORS = [
    ClockAnchor(
        elapsed_seconds=5.0,
        timestamp="2026-11-19T21:30:56+07:00",
        confidence=0.95,
        verification="manual visual verification of burned-in clock (frame at t=5 s)",
    ),
    ClockAnchor(
        elapsed_seconds=7200.0,
        timestamp="2026-11-19T23:30:50+07:00",
        confidence=0.95,
        verification="manual visual verification of burned-in clock (frame at t=2 h)",
    ),
    ClockAnchor(
        elapsed_seconds=15500.0,
        timestamp="2026-11-20T01:49:07+07:00",
        confidence=0.90,
        verification="manual visual verification of burned-in clock (frame at t=4h18m20s)",
    ),
]

OBSERVATION_FIELDS = [
    "recording_id",
    "camera_id",
    "monitoring_station_id",
    "frame_index",
    "elapsed_seconds",
    "capture_timestamp",
    "capture_timestamp_utc",
    "timestamp_source",
    "timestamp_confidence",
    "timestamp_method",
    "timestamp_reason",
    "timezone",
    "frame_path",
    "frame_sha256",
    "extraction_status",
    "extraction_reason",
    "shoreline_available",
    "reason_code",
    "shoreline_position_px",
    "signed_position_px",
    "position_axis",
    "position_sign",
    "segment_count",
    "segment_measurements",
    "shoreline_segments",
    "shoreline_geometry",
    "inference_time_ms",
    "confidence",
    "class_fractions",
    "orientation_used",
    "valid_columns",
    "processing_time_ms",
    "tide_data_available",
    "tide_station_id",
    "tide_station_code",
    "tide_timestamp",
    "tide_source",
    "tide_height_m",
    "tide_unit",
    "tide_vertical_datum",
    "tide_quality",
    "tide_reason",
    "horizon_y_px",
]

JSON_FIELDS = {
    "segment_measurements",
    "shoreline_segments",
    "shoreline_geometry",
    "class_fractions",
}
BOOL_FIELDS = {"shoreline_available", "tide_data_available"}
INT_FIELDS = {
    "camera_id",
    "monitoring_station_id",
    "frame_index",
    "segment_count",
    "valid_columns",
    "tide_station_id",
}
FLOAT_FIELDS = {
    "elapsed_seconds",
    "timestamp_confidence",
    "shoreline_position_px",
    "signed_position_px",
    "inference_time_ms",
    "confidence",
    "processing_time_ms",
    "tide_height_m",
    "horizon_y_px",
}


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | dict):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parse_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key in OBSERVATION_FIELDS:
        raw = (row.get(key) or "").strip()
        if raw == "":
            parsed[key] = None
        elif key in JSON_FIELDS:
            try:
                parsed[key] = json.loads(raw)
            except json.JSONDecodeError:
                parsed[key] = None
        elif key in BOOL_FIELDS:
            parsed[key] = raw.lower() in {"true", "1", "yes"}
        elif key in INT_FIELDS:
            try:
                parsed[key] = int(float(raw))
            except ValueError:
                parsed[key] = None
        elif key in FLOAT_FIELDS:
            try:
                parsed[key] = float(raw)
            except ValueError:
                parsed[key] = None
        else:
            parsed[key] = raw
    return parsed


def _load_observations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_parse_row(row) for row in reader if row.get("frame_index")]


def _write_observations(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["frame_index"])):
            writer.writerow({key: _csv_value(row.get(key)) for key in OBSERVATION_FIELDS})


def _append_row(handle: Any, writer: csv.DictWriter, row: dict[str, Any]) -> None:
    writer.writerow({key: _csv_value(row.get(key)) for key in OBSERVATION_FIELDS})
    handle.flush()
    os.fsync(handle.fileno())


def _load_anchors(path: Path, timezone: str) -> list[ClockAnchor]:
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            anchors = [
                ClockAnchor(
                    elapsed_seconds=float(item["elapsed_seconds"]),
                    timestamp=str(item["timestamp"]),
                    source=str(item.get("source", "visible_clock")),
                    confidence=float(item.get("confidence", 0.9)),
                    verification=item.get("verification"),
                )
                for item in payload.get("anchors", [])
            ]
            if anchors:
                return anchors
        except (OSError, ValueError, KeyError, TypeError):
            pass
    path.write_text(
        json.dumps(
            {
                "timezone": timezone,
                "note": (
                    "Container creation time is absent. These anchors were read manually "
                    "from the burned-in CCTV clock in extracted frames; timestamps for other "
                    "frames are derived (piecewise linear) and are not container-verified."
                ),
                "anchors": [
                    {
                        "elapsed_seconds": anchor.elapsed_seconds,
                        "timestamp": anchor.timestamp,
                        "source": anchor.source,
                        "confidence": anchor.confidence,
                        "verification": anchor.verification,
                    }
                    for anchor in DEFAULT_ANCHORS
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return list(DEFAULT_ANCHORS)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_memory_mb() -> tuple[float | None, str]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        value = getattr(info, "peak_wset", info.rss)
        return round(value / (1024 * 1024), 1), "psutil peak working set"
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                process, ctypes.byref(counters), counters.cb
            )
            if ok:
                return (
                    round(counters.PeakWorkingSetSize / (1024 * 1024), 1),
                    "Windows peak working set",
                )
        except Exception:
            pass
    _current, peak = tracemalloc.get_traced_memory()
    return round(peak / (1024 * 1024), 1), "Python tracemalloc peak (native allocations excluded)"


def _tide_lookup_factory(args, session):
    if session is None:
        reason = (
            "no verified Thai tide source is available in this repository/database; "
            "Philippine NAMRIA/PHIVOLCS stations are not valid for Karon Beach"
        )

        def lookup(_moment):
            return TideAssociation(available=False, source="unavailable", reason=reason)

        return lookup, None

    selection = select_tide_station(
        session,
        args.latitude,
        args.longitude,
        country_iso=args.country,
        timezone=args.timezone,
        max_distance_km=args.max_station_distance_km,
    )
    if selection.station is None:
        reason = selection.reason

        def lookup(_moment):
            return TideAssociation(
                available=False,
                source="unavailable",
                candidate_count=selection.candidate_count,
                rejected_stations=[item.as_dict() for item in selection.rejected],
                reason=reason,
            )

        return lookup, selection

    reason = (
        "tide prediction generation was removed from the system; "
        "no tide association is available for this run"
    )

    def lookup(_moment):
        return TideAssociation(
            available=False,
            source="unavailable",
            candidate_count=selection.candidate_count,
            rejected_stations=[item.as_dict() for item in selection.rejected],
            reason=reason,
        )

    return lookup, selection


def _draw_sample(frame, class_mask, segments, label: str):
    colors = {0: (137, 181, 214), 1: (190, 112, 32), 2: (235, 206, 135)}  # BGR
    overlay = frame.copy()
    for class_id, color in colors.items():
        overlay[class_mask == class_id] = color
    blended = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    for segment in segments:
        points = []
        for x, y in segment:
            points.append([int(round(x)), int(round(y))])
        if len(points) >= 2:
            cv2.polylines(
                blended,
                [np.array(points, dtype="int32")],
                False,
                (0, 255, 255),
                4,
            )
    cv2.putText(
        blended,
        label,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 0),
        6,
        cv2.LINE_AA,
    )
    cv2.putText(
        blended,
        label,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return blended


def _sample_contact_sheet(
    output_path: Path,
    rows: list[dict[str, Any]],
    segmenter: ShorelineSegmenter,
    count: int = 6,
) -> list[str]:
    candidates = [
        row for row in rows if row.get("frame_path") and Path(row["frame_path"]).is_file()
    ]
    if not candidates:
        return []
    if len(candidates) > count:
        indices = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)]
        chosen = [candidates[index] for index in indices]
    else:
        chosen = candidates
    output_path.mkdir(parents=True, exist_ok=True)
    panels = []
    written = []
    for row in chosen:
        frame = cv2.imread(row["frame_path"])
        if frame is None:
            continue
        try:
            segmentation = segmenter.predict(frame)
            result = extract_shoreline_from_classes(
                segmentation.class_mask, confidence=segmentation.confidence
            )
            position = measure_shoreline_position(result.segments, result.orientation_used)
            label = (
                f"t={row['capture_timestamp']} frame={row['frame_index']} "
                f"{'shore' if position.available else 'NO SHORE'}"
            )
            annotated = _draw_sample(frame, segmentation.class_mask, result.segments, label)
        except Exception:
            annotated = frame.copy()
            cv2.putText(
                annotated,
                f"frame {row['frame_index']} segmentation failed",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        path = output_path / f"sample_frame_{int(row['frame_index']):09d}.jpg"
        cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written.append(str(path))
        panels.append(cv2.resize(annotated, (960, 540), interpolation=cv2.INTER_AREA))
    if not panels:
        return written
    while len(panels) % 2:
        panels.append(panels[-1])
    grid_rows = []
    for index in range(0, len(panels), 2):
        grid_rows.append(cv2.hconcat(panels[index : index + 2]))
    sheet = cv2.vconcat(grid_rows)
    cv2.imwrite(str(output_path / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return written


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _write_dict_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _write_legacy_anomaly_csv(
    path: Path,
    detections,
    *,
    recording_id: str | None,
    camera_id: int | None,
) -> None:
    fields = [
        "recording_id",
        "camera_id",
        "frame_index",
        "capture_timestamp",
        "classification",
        "state",
        "baseline_status",
        "tide_adjusted",
        "observed_position_px",
        "expected_position_px",
        "residual_px",
        "robust_z",
        "movement_rate_px_s",
        "persistence_count",
        "tide_data_available",
        "tide_source",
        "tide_height_m",
        "quality_status",
        "confidence",
        "score",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as anomaly_handle:
        writer = csv.DictWriter(anomaly_handle, fieldnames=fields)
        writer.writeheader()
        for detection in detections:
            payload = detection.as_dict()
            payload["capture_timestamp"] = payload.get("timestamp")
            payload["recording_id"] = recording_id
            payload["camera_id"] = camera_id
            writer.writerow({key: _csv_value(payload.get(key)) for key in fields})


def _infer_frame_height(output_dir: Path, explicit: float | None) -> float | None:
    if explicit is not None:
        return float(explicit)
    metadata_path = output_dir / "video_metadata.json"
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            height = payload.get("height")
            return None if height is None else float(height)
        except (OSError, ValueError, TypeError):
            return None
    return None


def _write_rolling_outputs(
    rows: list[dict[str, Any]],
    outcome,
    config: RollingBaselineConfig,
    output_dir: Path,
    *,
    observations_sha256: str,
    recording_id: str | None,
    camera_id: int | None,
    skip_supplementary: bool,
) -> dict[str, Any]:
    evaluation = outcome.rolling_evaluation
    assert evaluation is not None
    rolling_report.write_anomaly_csv(
        output_dir / "anomaly_results.csv",
        rows,
        evaluation,
        recording_id=recording_id,
        camera_id=camera_id,
    )
    counts = Counter(row.classification for row in evaluation.rows)
    observations = rolling_report.rows_to_rolling_observations(rows)

    sensitivity: list[dict[str, Any]] = []
    bias: list[dict[str, Any]] = []
    injection: dict[str, Any] | None = None
    plot_ok = rolling_report.plot_rolling(
        evaluation, output_dir / "plots" / "rolling_baseline_drawdown.png"
    )
    if not skip_supplementary:
        sensitivity = rolling_report.sensitivity_grid(observations, config)
        _write_dict_csv(
            output_dir / "sensitivity.csv",
            sensitivity,
            [
                "window_seconds",
                "z_threshold",
                "drawdown_floor_px",
                "persistence",
                "event_count",
                "rows_in_event",
                "max_robust_z",
                "note",
                "error",
            ],
        )
        bias = rolling_report.bias_diagnostic(observations, config)
        _write_dict_csv(
            output_dir / "bias_diagnostic.csv",
            bias,
            [
                "drift_px_per_min",
                "event_count",
                "median_z",
                "max_z",
                "z_shift_vs_zero_drift",
                "theil_sen_slope_px_per_min",
                "note",
            ],
        )
        injection = rolling_report.injection_test(observations, config, output_dir=output_dir)
        _write_dict_csv(
            output_dir / "injection" / "injection_results.csv",
            injection["cells"],
            [
                "amplitude_px",
                "ramp_seconds",
                "attempts",
                "detections",
                "detection_rate",
                "median_latency_s",
                "false_event_count_unmodified",
                "label",
            ],
        )
        (output_dir / "injection" / "summary.json").write_text(
            json.dumps(injection, indent=2), encoding="utf-8"
        )

    zs = [row.robust_z for row in evaluation.rows if row.robust_z is not None]
    strongest = None
    if zs:
        index = max(range(len(zs)), key=lambda i: zs[i])
        strongest_rows = [row for row in evaluation.rows if row.robust_z is not None]
        strongest_row = strongest_rows[index]
        strongest = {
            "timestamp": (
                None if strongest_row.timestamp is None else strongest_row.timestamp.isoformat()
            ),
            "robust_z": round(zs[index], 6),
            "observed_position_px": strongest_row.observed_position_px,
            "rolling_baseline_px": strongest_row.rolling_baseline_px,
            "residual_px": strongest_row.residual_px,
            "classification": strongest_row.classification,
        }
    return {
        "detector_mode": "rolling_baseline",
        "requested_mode": outcome.resolved_detector_mode,
        "resolved_detector_mode": outcome.resolved_detector_mode,
        "mode_reason": outcome.mode_reason,
        "baseline_status": "rolling_unverified",
        "config": config.as_dict(),
        "config_hash": evaluation.config_hash,
        "observations_sha256": observations_sha256,
        "input_observations": len(rows),
        "evaluated_rows": len(evaluation.rows),
        "dropped_duplicate_count": evaluation.dropped_duplicate_count,
        "nominal_interval_seconds": evaluation.nominal_interval_seconds,
        "gate_counts": evaluation.gate_counts,
        "classification_counts": dict(sorted(counts.items())),
        "event_count": len(evaluation.events),
        "events": [event.as_dict() for event in evaluation.events],
        "strongest_candidate": strongest,
        "plot": str(output_dir / "plots" / "rolling_baseline_drawdown.png"),
        "plot_generated_in_runner": plot_ok,
        "plot_note": (
            None
            if plot_ok
            else "matplotlib is not installed in the runner venv; run "
            "ml/scripts/plot_rolling_baseline.py with the ML venv"
        ),
        "sensitivity_cells": len(sensitivity),
        "bias_rows": len(bias),
        "injection": injection,
    }


def _run_from_observations(args, output_dir: Path, rolling_config) -> int:
    source = Path(args.from_observations).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"[phase6] observations not found: {source}")
    required = {
        "frame_index",
        "capture_timestamp",
        "shoreline_available",
        "signed_position_px",
        "confidence",
    }
    with source.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    missing = sorted(required - set(header))
    if missing:
        raise SystemExit(f"[phase6] observations CSV missing columns: {missing}")

    rows = _load_observations(source)
    if not rows:
        raise SystemExit(f"[phase6] no usable observation rows in {source}")
    observations_sha = _sha256_file(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "observations.csv"
    if target.resolve() != source:
        _write_observations(target, rows)
    height = _infer_frame_height(output_dir, args.frame_height)

    config = PrototypeConfig(
        sample_interval_seconds=args.sample_every,
        persistence=args.persistence,
        z_threshold=args.z_threshold,
        detector_mode=args.detector_mode,
        rolling=rolling_config,
        frame_height=height,
    )
    outcome = evaluate_observations_detailed(rows, config, frame_height=height)
    recording_id = next((row.get("recording_id") for row in rows if row.get("recording_id")), None)
    camera_id = next(
        (row.get("camera_id") for row in rows if row.get("camera_id") is not None), None
    )
    if outcome.rolling_evaluation is None:
        _write_legacy_anomaly_csv(
            output_dir / "anomaly_results.csv",
            outcome.detections,
            recording_id=recording_id,
            camera_id=camera_id,
        )
        detector_report = {
            "detector_mode": args.detector_mode,
            "resolved_detector_mode": outcome.resolved_detector_mode,
            "mode_reason": outcome.mode_reason,
            "observations_sha256": observations_sha,
            "classification_counts": dict(
                sorted(Counter(d.classification for d in outcome.detections).items())
            ),
        }
    else:
        detector_report = _write_rolling_outputs(
            rows,
            outcome,
            outcome.rolling_config or rolling_config,
            output_dir,
            observations_sha256=observations_sha,
            recording_id=recording_id,
            camera_id=camera_id,
            skip_supplementary=args.skip_supplementary,
        )

    video_metadata = None
    metadata_path = output_dir / "video_metadata.json"
    if metadata_path.is_file():
        try:
            video_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            video_metadata = None
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "mode": "from_observations_no_inference",
        "observations": {
            "path": str(source),
            "sha256": observations_sha,
            "rows": len(rows),
            "sorted_by": "capture_timestamp",
            "duplicates_dropped": detector_report.get("dropped_duplicate_count", 0),
        },
        "video": video_metadata,
        "detector": detector_report,
        "baseline": outcome.baseline.as_dict(),
        "limitations": [
            "Experimental rolling-baseline detector: not validated for tsunami detection.",
            "Tide-independent: no absolute expected shoreline, tide-adjusted residual "
            "or NORMAL state.",
            "Slow drawdowns developing over many minutes may fall below the 90 s criterion.",
            "No verified Thai tide source is associated with this footage.",
        ],
    }
    (output_dir / "processing_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[phase6] {len(rows)} rows, resolved={outcome.resolved_detector_mode}, "
        f"events={detector_report.get('event_count', 0)}, "
        f"classes={detector_report.get('classification_counts')}",
        flush=True,
    )
    print(f"[phase6] wrote {output_dir}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-every", type=float, default=30.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--no-file-hash", action="store_true")
    parser.add_argument("--no-sample-sheet", action="store_true")
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--station-id", type=int, default=None)
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--country", default="TH")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--max-station-distance-km", type=float, default=120.0)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--z-threshold", type=float, default=3.5)
    parser.add_argument("--min-baseline-samples", type=int, default=30)
    parser.add_argument("--min-tide-range-m", type=float, default=0.20)
    parser.add_argument(
        "--detector-mode",
        default="auto",
        choices=["auto", "rolling_baseline", "tide_conditioned", "movement_only"],
        help="auto resolves to rolling_baseline when no verified tide baseline exists.",
    )
    parser.add_argument(
        "--from-observations",
        default=None,
        help="Re-evaluate a saved observations.csv instead of decoding the video.",
    )
    parser.add_argument("--rolling-window-seconds", type=float, default=1200.0)
    parser.add_argument("--rolling-k-seconds", type=float, default=90.0)
    parser.add_argument("--drawdown-floor-frac", type=float, default=150.0 / 1080.0)
    parser.add_argument("--scale-floor-frac", type=float, default=0.004)
    parser.add_argument("--max-gap-seconds", type=float, default=45.0)
    parser.add_argument("--cooldown-seconds", type=float, default=600.0)
    parser.add_argument(
        "--freeze-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the rolling baseline during candidate/event (default: freeze).",
    )
    parser.add_argument("--confidence-floor", type=float, default=0.80)
    parser.add_argument("--stale-run-length", type=int, default=10)
    parser.add_argument(
        "--frame-height",
        type=float,
        default=None,
        help="Frame height for pixel floors; inferred from video_metadata.json when absent.",
    )
    parser.add_argument(
        "--skip-supplementary",
        action="store_true",
        help="Skip sensitivity/bias/injection reporting (not the main result).",
    )
    parser.add_argument(
        "--database",
        action="store_true",
        help="Attempt DB-backed tide association/persistence (requires PostgreSQL).",
    )
    args = parser.parse_args(argv)
    try:
        validate_detector_mode(args.detector_mode)
        rolling_config = RollingBaselineConfig(
            window_seconds=args.rolling_window_seconds,
            k_seconds=args.rolling_k_seconds,
            z_threshold=args.z_threshold,
            drawdown_floor_frac=args.drawdown_floor_frac,
            scale_floor_frac=args.scale_floor_frac,
            persistence=args.persistence,
            max_gap_seconds=args.max_gap_seconds,
            cooldown_seconds=args.cooldown_seconds,
            persistent_seconds=600.0,
            confidence_floor=args.confidence_floor,
            stale_run_length=args.stale_run_length,
            freeze_baseline=args.freeze_baseline,
            frame_height=args.frame_height,
            nominal_interval_seconds=args.sample_every,
        )
        rolling_config.validate()
    except ValueError as exc:
        parser.error(str(exc))

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        parser.error(f"video not found: {video_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_observations:
        return _run_from_observations(args, output_dir, rolling_config)
    frame_dir = output_dir / "extracted_frames"
    observations_path = output_dir / "observations.csv"
    anchors_path = output_dir / "clock_anchors.json"
    report_path = output_dir / "processing_report.json"
    anomaly_path = output_dir / "anomaly_results.csv"

    print(f"[phase5] video: {video_path}", flush=True)
    started = time.perf_counter()
    tracemalloc.start()
    metadata = probe_video(video_path)
    file_sha256 = None if args.no_file_hash else _sha256_file(video_path)
    anchors = _load_anchors(anchors_path, args.timezone)
    expected_samples = target_samples(
        metadata,
        sample_interval_seconds=args.sample_every,
        start_elapsed=args.start,
        end_elapsed=args.end,
        max_samples=args.max_samples,
    )
    (output_dir / "video_metadata.json").write_text(
        json.dumps(
            {
                **metadata.as_dict(),
                "sha256": file_sha256,
                "timezone": args.timezone,
                "timestamp_source": "derived",
                "clock_anchors": [
                    {
                        "elapsed_seconds": anchor.elapsed_seconds,
                        "timestamp": anchor.timestamp,
                        "source": anchor.source,
                        "confidence": anchor.confidence,
                        "verification": anchor.verification,
                    }
                    for anchor in anchors
                ],
                "expected_samples": len(expected_samples),
                "sample_interval_seconds": args.sample_every,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[phase5] duration={metadata.duration_text} fps={metadata.fps:.3f} "
        f"frames={metadata.frame_count} samples={len(expected_samples)}",
        flush=True,
    )

    existing = _load_observations(observations_path) if args.resume else []
    if not args.resume and observations_path.exists():
        observations_path.unlink()
    processed_indices = {
        int(row["frame_index"]) for row in existing if row.get("frame_index") is not None
    }
    resume_from = (max(processed_indices) + 1) if processed_indices else None
    if existing:
        print(
            f"[phase5] resume: {len(existing)} rows already processed; "
            f"seeking to frame {resume_from}",
            flush=True,
        )

    # Optional DB-backed tide association only.
    session = None
    db_error = None
    if args.database:
        try:
            from app.database.session import get_session_factory

            session = get_session_factory()()
        except Exception as exc:
            db_error = str(exc)
            print(f"[phase5] database unavailable: {exc}", flush=True)

    tide_lookup, tide_selection = _tide_lookup_factory(args, session)

    config = PrototypeConfig(
        sample_interval_seconds=args.sample_every,
        start_elapsed=args.start,
        end_elapsed=args.end,
        max_samples=args.max_samples,
        timezone=args.timezone,
        country_iso=args.country,
        latitude=args.latitude,
        longitude=args.longitude,
        max_station_distance_km=args.max_station_distance_km,
        camera_id=args.camera_id,
        monitoring_station_id=args.station_id,
        min_baseline_samples=args.min_baseline_samples,
        min_tide_range_m=args.min_tide_range_m,
        persistence=args.persistence,
        z_threshold=args.z_threshold,
        save_frames=not args.no_frames,
        detector_mode=args.detector_mode,
        rolling=rolling_config,
        frame_height=float(metadata.height) if metadata.height else args.frame_height,
        frame_dir=str(frame_dir),
    )

    segmenter = ShorelineSegmenter()
    needs_header = not observations_path.exists() or observations_path.stat().st_size == 0
    handle = observations_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS, extrasaction="ignore")
    if needs_header:
        writer.writeheader()
        handle.flush()

    def on_row(row: dict[str, Any]) -> None:
        _append_row(handle, writer, row)

    def on_progress(info: dict[str, Any]) -> None:
        elapsed = time.perf_counter() - started
        print(
            f"[phase5] sampled={info['sampled']} frame={info['frame_index']} "
            f"shore={info['available']} unavailable={info['unavailable']} "
            f"elapsed={elapsed/60:.1f}min",
            flush=True,
        )

    try:
        result = analyze_video(
            video_path,
            config,
            anchors=anchors,
            tide_lookup=tide_lookup,
            tide_selection=tide_selection,
            segmenter=segmenter,
            resume_from_frame=resume_from,
            skip_frame_indices=processed_indices,
            progress=on_progress,
            on_row=on_row,
        )
    finally:
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()

    merged = existing + result.observations
    # De-duplicate by frame index (the last completed row wins).
    by_frame = {int(row["frame_index"]): row for row in merged}
    rows = [by_frame[index] for index in sorted(by_frame)]
    _write_observations(observations_path, rows)

    outcome = evaluate_observations_detailed(
        rows,
        config,
        frame_height=float(metadata.height) if metadata.height else args.frame_height,
    )
    baseline = outcome.baseline
    detections = outcome.detections
    detector_config = outcome.detector_config
    thresholds = outcome.thresholds
    rolling_report_data: dict[str, Any] | None = None
    observations_sha256 = _sha256_file(observations_path)
    if outcome.rolling_evaluation is not None:
        rolling_report_data = _write_rolling_outputs(
            rows,
            outcome,
            outcome.rolling_config or rolling_config,
            output_dir,
            observations_sha256=observations_sha256,
            recording_id=result.recording_id,
            camera_id=args.camera_id,
            skip_supplementary=args.skip_supplementary,
        )
    else:
        _write_legacy_anomaly_csv(
            anomaly_path,
            detections,
            recording_id=result.recording_id,
            camera_id=args.camera_id,
        )

    # Optional sample visualisation (re-runs inference for ~6 frames).
    sample_files: list[str] = []
    if not args.no_sample_sheet:
        try:
            sample_files = _sample_contact_sheet(output_dir / "plots" / "samples", rows, segmenter)
        except Exception as exc:
            print(f"[phase5] sample sheet skipped: {exc}", flush=True)

    # Optional persistence through the existing API/DB architecture.
    persistence: dict[str, Any] | None = None
    if session is not None and args.camera_id is not None:
        try:
            from app.models.camera import Camera
            from app.services.video.persistence import persist_analysis
            from sqlalchemy.orm import joinedload

            camera = session.get(Camera, args.camera_id, options=[joinedload(Camera.station)])
            if camera is None:
                persistence = {"error": f"camera not found: {args.camera_id}"}
            else:
                result.observations = rows
                result.detections = detections
                result.baseline = baseline
                result.detector_config = detector_config
                result.threshold_diagnostics = thresholds
                result.tide_selection = tide_selection
                result.resolved_detector_mode = outcome.resolved_detector_mode
                result.mode_reason = outcome.mode_reason
                result.rolling_evaluation = outcome.rolling_evaluation
                result.rolling_config = outcome.rolling_config
                persistence = persist_analysis(
                    session,
                    result,
                    camera=camera,
                    anchors=anchor_dicts(anchors),
                    timezone=args.timezone,
                    timestamp_source="derived",
                    file_sha256=file_sha256,
                )
                session.commit()
        except Exception as exc:
            if session is not None:
                session.rollback()
            persistence = {"error": str(exc)}

    classification_counts = _counts(item.classification for item in detections)
    state_counts = _counts(item.state for item in detections)
    availability_counts = _counts(
        "available" if row.get("shoreline_available") else (row.get("reason_code") or "unavailable")
        for row in rows
    )
    tide_available_count = sum(1 for row in rows if row.get("tide_data_available"))
    tide_reason = next((row.get("tide_reason") for row in rows if row.get("tide_reason")), None)
    stats = result.stats
    # Recompute stats over the merged series (a resumed run only extracted new rows).
    stats.processed = len(rows)
    stats.available = sum(1 for row in rows if row.get("shoreline_available"))
    stats.unavailable = sum(
        1
        for row in rows
        if not row.get("shoreline_available") and row.get("extraction_status") == "unavailable"
    )
    stats.reason_counts = {
        str(key): int(value)
        for key, value in Counter(
            row.get("reason_code")
            for row in rows
            if not row.get("shoreline_available") and row.get("reason_code")
        ).items()
    }
    inference_values = [
        float(row["inference_time_ms"]) for row in rows if row.get("inference_time_ms") is not None
    ]
    processing_values = [
        float(row["processing_time_ms"])
        for row in rows
        if row.get("processing_time_ms") is not None
    ]
    peak_memory_mb, peak_memory_source = _peak_memory_mb()
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "video": {
            **metadata.as_dict(),
            "sha256": file_sha256,
            "expected_samples": len(expected_samples),
            "processed_samples": len(rows),
            "sample_interval_seconds": args.sample_every,
            "timezone": args.timezone,
        },
        "timestamps": {
            "container_creation_time": metadata.creation_time,
            "source": "derived",
            "method": (
                "piecewise linear interpolation between manually verified " "visible-clock anchors"
            ),
            "timezone": args.timezone,
            "anchors": [
                {
                    "elapsed_seconds": anchor.elapsed_seconds,
                    "timestamp": anchor.timestamp,
                    "source": anchor.source,
                    "confidence": anchor.confidence,
                    "verification": anchor.verification,
                }
                for anchor in anchors
            ],
        },
        "segmentation": {
            "architecture": "SegFormer MiT-B2 ONNX",
            "class_ids": {"Beach": 0, "Ocean": 1, "Sky": 2},
            "native_resolution": [metadata.width, metadata.height],
            "available": stats.available,
            "unavailable": stats.unavailable,
            "extraction_failures": stats.extraction_failures,
            "segmentation_failures": stats.segmentation_failures,
            "availability_counts": availability_counts,
            "reason_counts": stats.reason_counts,
            "mean_inference_ms": (
                round(sum(inference_values) / len(inference_values), 3)
                if inference_values
                else None
            ),
            "max_inference_ms": round(max(inference_values), 3) if inference_values else None,
            "mean_processing_ms": (
                round(sum(processing_values) / len(processing_values), 3)
                if processing_values
                else None
            ),
            "mean_confidence": (
                round(
                    sum(float(row["confidence"]) for row in rows if row.get("confidence"))
                    / max(1, sum(1 for row in rows if row.get("confidence"))),
                    6,
                )
            ),
        },
        "tide": {
            "data_available": tide_available_count > 0,
            "observations_with_tide": tide_available_count,
            "reason": tide_reason,
            "country_iso": args.country,
            "timezone": args.timezone,
            "site_latitude": args.latitude,
            "site_longitude": args.longitude,
            "max_station_distance_km": args.max_station_distance_km,
            "selection": (
                None
                if tide_selection is None
                else {
                    "station": (
                        None
                        if tide_selection.station is None
                        else {
                            "id": getattr(tide_selection.station, "id", None),
                            "code": tide_selection.station.code,
                            "name": tide_selection.station.name,
                            "distance_km": tide_selection.distance_km,
                        }
                    ),
                    "candidate_count": tide_selection.candidate_count,
                    "rejected": [item.as_dict() for item in tide_selection.rejected],
                    "reason": tide_selection.reason,
                }
            ),
            "database_configured": session is not None,
            "database_error": db_error,
        },
        "baseline": baseline.as_dict(),
        "thresholds": {
            **thresholds,
            "config": {
                "persistence": detector_config.persistence,
                "z_threshold": detector_config.z_threshold,
                "movement_threshold_px_s": detector_config.movement_threshold_px_s,
                "drawdown_rate_threshold_px_s": detector_config.drawdown_rate_threshold_px_s,
                "drawdown_distance_px": detector_config.drawdown_distance_px,
                "max_gap_seconds": detector_config.max_gap_seconds,
            },
        },
        "anomalies": {
            "classifications": classification_counts,
            "states": state_counts,
            "anomaly_events": sum(
                count for key, count in classification_counts.items() if key in ANOMALY_CLASSES
            ),
            "baseline_mode": (
                "rolling_baseline"
                if outcome.rolling_evaluation is not None
                else ("tide_conditioned" if baseline.available else "movement_only")
            ),
            "persistence_required": detector_config.persistence,
        },
        "detector": {
            "requested_mode": config.detector_mode,
            "resolved_detector_mode": outcome.resolved_detector_mode,
            "mode_reason": outcome.mode_reason,
            "observations_sha256": observations_sha256,
            "rolling_baseline": rolling_report_data,
        },
        "performance": {
            "total_seconds": round(time.perf_counter() - started, 3),
            "total_minutes": round((time.perf_counter() - started) / 60, 2),
            "peak_memory_mb": peak_memory_mb,
            "peak_memory_source": peak_memory_source,
            "streaming": (
                "cv2.VideoCapture sequential grab/retrieve; only sampled frames are retained"
            ),
        },
        "persistence": persistence,
        "sample_artifacts": sample_files,
        "limitations": result.limitations,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    tracemalloc.stop()
    if session is not None:
        session.close()

    print(
        f"[phase5] done: {len(rows)} rows, shore available {stats.available}, "
        f"baseline={baseline.status}, classifications={classification_counts}",
        flush=True,
    )
    print(f"[phase5] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
