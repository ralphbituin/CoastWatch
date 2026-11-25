"""Offline reporting for the tide-independent rolling-median detector.

All routines here re-use the production rolling module. They never touch the
database, the model, the class IDs or the saved main results. Synthetic
injection output is labelled as synthetic everywhere.

These helpers are called by ``ml/scripts/phase5_tide_anomaly.py`` when
``--detector-mode rolling_baseline`` (or ``auto`` without a tide baseline) is
selected.
"""

from __future__ import annotations

import csv
import itertools
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from app.services.anomaly.rolling_baseline import (
    EVENT_CLASSES,
    RollingBaselineConfig,
    RollingEvaluation,
    RollingObservation,
    evaluate_rolling_baseline,
)

ANOMALY_FIELDS = [
    "recording_id",
    "camera_id",
    "frame_index",
    "capture_timestamp",
    "detector_mode",
    "classification",
    "state",
    "baseline_status",
    "tide_adjusted",
    "observed_position_px",
    "rolling_baseline_px",
    "rolling_scale_px",
    "scale_floored",
    "residual_px",
    "robust_z",
    "cumulative_drawdown_px",
    "movement_rate_px_s",
    "persistence_count",
    "expected_position_px",
    "tide_data_available",
    "tide_source",
    "tide_height_m",
    "quality_status",
    "confidence",
    "reason",
    "onset_time",
    "detection_time",
    "detection_latency_s",
    "event_window_seconds",
    "event_max_classification",
    "cooldown_state",
    "baseline_frozen",
    "config_hash",
]

INJECTION_AMPLITUDES = (100.0, 150.0, 200.0, 300.0, 500.0)
INJECTION_DURATIONS = (60.0, 120.0, 300.0, 600.0, 1200.0)
SENSITIVITY_WINDOWS = (600.0, 1200.0, 1800.0)
SENSITIVITY_Z = (3.0, 3.5, 4.0)
SENSITIVITY_FLOORS = (100.0, 150.0, 200.0)
SENSITIVITY_PERSISTENCE = (2, 3, 4)
BIAS_DRIFTS_PX_PER_MIN = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)
INJECTION_SEED = 20261125


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def rows_to_rolling_observations(
    rows: list[dict[str, Any]],
) -> list[RollingObservation]:
    out: list[RollingObservation] = []
    for row in rows:
        segments = row.get("segment_measurements")
        if segments is not None and not isinstance(segments, list):
            segments = None
        position = row.get("signed_position_px")
        out.append(
            RollingObservation(
                timestamp=_parse_timestamp(row.get("capture_timestamp")),
                position_px=(
                    None
                    if position is None or position == ""
                    else float(position)
                ),
                shoreline_available=bool(row.get("shoreline_available")),
                confidence=(
                    None if row.get("confidence") is None else float(row["confidence"])
                ),
                frame_index=row.get("frame_index"),
                timestamp_confidence=(
                    None
                    if row.get("timestamp_confidence") is None
                    else float(row["timestamp_confidence"])
                ),
                reason=row.get("extraction_reason") or row.get("reason_code"),
                segment_measurements=segments,
                horizon_y_px=row.get("horizon_y_px"),
            )
        )
    return out


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict | list):
        import json

        return json.dumps(value, separators=(",", ":"), default=str)
    return str(value)


def write_anomaly_csv(
    path: Path,
    rows: list[dict[str, Any]],
    evaluation: RollingEvaluation,
    *,
    recording_id: str | None,
    camera_id: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANOMALY_FIELDS)
        writer.writeheader()
        for result in evaluation.rows:
            writer.writerow(
                {
                    "recording_id": recording_id,
                    "camera_id": camera_id,
                    "frame_index": result.frame_index,
                    "capture_timestamp": (
                        None
                        if result.timestamp is None
                        else result.timestamp.isoformat()
                    ),
                    "detector_mode": result.detector_mode,
                    "classification": result.classification,
                    "state": (
                        "ANOMALY_DETECTED"
                        if result.classification in EVENT_CLASSES
                        else (
                            "DATA_UNRELIABLE"
                            if result.classification == "DATA_UNRELIABLE"
                            else "OBSERVED"
                        )
                    ),
                    "baseline_status": result.baseline_status,
                    "tide_adjusted": False,
                    "observed_position_px": result.observed_position_px,
                    "rolling_baseline_px": result.rolling_baseline_px,
                    "rolling_scale_px": result.rolling_scale_px,
                    "scale_floored": result.scale_floored,
                    "residual_px": result.residual_px,
                    "robust_z": result.robust_z,
                    "cumulative_drawdown_px": result.cumulative_drawdown_px,
                    "movement_rate_px_s": result.movement_rate_px_s,
                    "persistence_count": result.persistence_count,
                    "expected_position_px": result.rolling_baseline_px,
                    "tide_data_available": False,
                    "tide_source": "unavailable",
                    "tide_height_m": None,
                    "quality_status": result.quality_status,
                    "confidence": result.confidence,
                    "reason": result.reason,
                    "onset_time": (
                        None if result.onset_time is None else result.onset_time.isoformat()
                    ),
                    "detection_time": (
                        None
                        if result.detection_time is None
                        else result.detection_time.isoformat()
                    ),
                    "detection_latency_s": result.detection_latency_s,
                    "event_window_seconds": result.event_window_seconds,
                    "event_max_classification": result.event_max_classification,
                    "cooldown_state": result.cooldown_state,
                    "baseline_frozen": result.baseline_frozen,
                    "config_hash": result.config_hash,
                }
            )


def sensitivity_grid(
    observations: list[RollingObservation], config: RollingBaselineConfig
) -> list[dict[str, Any]]:
    """Reported for transparency, not used to choose defaults."""
    results: list[dict[str, Any]] = []
    height = config.frame_height
    for window, z_value, floor_px, persistence in itertools.product(
        SENSITIVITY_WINDOWS,
        SENSITIVITY_Z,
        SENSITIVITY_FLOORS,
        SENSITIVITY_PERSISTENCE,
    ):
        floor_frac = (
            floor_px / height
            if height
            else config.drawdown_floor_frac
        )
        candidate = replace(
            config,
            window_seconds=window,
            z_threshold=z_value,
            drawdown_floor_frac=floor_frac,
            persistence=persistence,
        )
        try:
            evaluation = evaluate_rolling_baseline(observations, candidate)
        except ValueError as exc:  # pragma: no cover - defensive
            results.append(
                {
                    "window_seconds": window,
                    "z_threshold": z_value,
                    "drawdown_floor_px": floor_px,
                    "persistence": persistence,
                    "event_count": None,
                    "max_robust_z": None,
                    "error": str(exc),
                }
            )
            continue
        zs = [row.robust_z for row in evaluation.rows if row.robust_z is not None]
        results.append(
            {
                "window_seconds": window,
                "z_threshold": z_value,
                "drawdown_floor_px": floor_px,
                "persistence": persistence,
                "event_count": len(evaluation.events),
                "rows_in_event": sum(1 for row in evaluation.rows if row.in_event),
                "max_robust_z": round(max(zs), 6) if zs else None,
                "note": "grid reported for transparency; not used to choose defaults",
            }
        )
    return results


def _slope_px_per_minute(observations: list[RollingObservation]) -> float | None:
    pairs = [
        (item.timestamp.timestamp(), float(item.position_px))
        for item in observations
        if item.timestamp is not None
        and item.position_px is not None
        and item.timestamp.tzinfo is not None
    ]
    if len(pairs) < 3:
        return None
    pairs.sort()
    slopes: list[float] = []
    # Deterministic subsample keeps pairwise Theil-Sen cheap while covering the
    # whole recording.
    step = max(1, len(pairs) // 60)
    sample = pairs[::step]
    for left, right in itertools.combinations(sample, 2):
        dt = right[0] - left[0]
        if dt > 0:
            slopes.append((right[1] - left[1]) / dt * 60.0)
    if not slopes:
        return None
    return float(np.median(slopes))


def bias_diagnostic(
    observations: list[RollingObservation], config: RollingBaselineConfig
) -> list[dict[str, Any]]:
    """Inject linear tide-like drift with no anomaly and report the z shift."""
    valid = [
        item
        for item in observations
        if item.timestamp is not None and item.position_px is not None
    ]
    if not valid:
        return []
    start = min(item.timestamp for item in valid if item.timestamp is not None)
    baseline_eval = evaluate_rolling_baseline(observations, config)
    baseline_z = [
        row.robust_z for row in baseline_eval.rows if row.robust_z is not None
    ]
    baseline_median_z = float(np.median(baseline_z)) if baseline_z else None

    slope = _slope_px_per_minute(observations)
    rows: list[dict[str, Any]] = []
    for drift in BIAS_DRIFTS_PX_PER_MIN:
        drifted: list[RollingObservation] = []
        for item in observations:
            if item.timestamp is None or item.position_px is None:
                drifted.append(item)
                continue
            minutes = (item.timestamp - start).total_seconds() / 60.0
            drifted.append(
                replace(
                    item,
                    position_px=float(item.position_px) + drift * minutes,
                )
            )
        evaluation = evaluate_rolling_baseline(drifted, config)
        zs = [row.robust_z for row in evaluation.rows if row.robust_z is not None]
        rows.append(
            {
                "drift_px_per_min": drift,
                "event_count": len(evaluation.events),
                "median_z": round(float(np.median(zs)), 6) if zs else None,
                "max_z": round(float(np.max(zs)), 6) if zs else None,
                "z_shift_vs_zero_drift": (
                    None
                    if not zs or baseline_median_z is None
                    else round(float(np.median(zs)) - baseline_median_z, 6)
                ),
                "theil_sen_slope_px_per_min": (
                    None if slope is None else round(slope, 6)
                ),
                "note": (
                    "Injected linear drift, no synthetic anomaly. A drift is "
                    "partly absorbed by the rolling median (lag), so the z shift "
                    "is not a tide correction."
                ),
            }
        )

    if slope is not None:
        detrended: list[RollingObservation] = []
        for item in observations:
            if item.timestamp is None or item.position_px is None:
                detrended.append(item)
                continue
            minutes = (item.timestamp - start).total_seconds() / 60.0
            detrended.append(
                replace(item, position_px=float(item.position_px) - slope * minutes)
            )
        eval_detrended = evaluate_rolling_baseline(detrended, config)
        rows.append(
            {
                "drift_px_per_min": None,
                "event_count": len(eval_detrended.events),
                "median_z": None,
                "max_z": (
                    round(
                        max(
                            (
                                row.robust_z
                                for row in eval_detrended.rows
                                if row.robust_z is not None
                            ),
                            default=float("nan"),
                        ),
                        6,
                    )
                ),
                "z_shift_vs_zero_drift": None,
                "theil_sen_slope_px_per_min": round(slope, 6),
                "note": "Offline Theil-Sen trend removal only; not exposed in the API/DB.",
            }
        )
    return rows


def _inject_ramp(
    observations: list[RollingObservation],
    *,
    onset_index: int,
    amplitude_px: float,
    duration_seconds: float,
) -> list[RollingObservation]:
    dt = _median_interval(observations)
    ramp_samples = max(1, int(round(duration_seconds / dt)))
    out = list(observations)
    for offset in range(0, len(out) - onset_index):
        index = onset_index + offset
        item = out[index]
        if item.position_px is None:
            continue
        added = amplitude_px * min(1.0, (offset + 1) / ramp_samples)
        out[index] = replace(item, position_px=float(item.position_px) + added)
    return out


def _median_interval(observations: list[RollingObservation]) -> float:
    stamps = [
        item.timestamp.timestamp()
        for item in observations
        if item.timestamp is not None
    ]
    stamps.sort()
    diffs = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
    return float(np.median(diffs)) if diffs else 30.0


def injection_test(
    observations: list[RollingObservation],
    config: RollingBaselineConfig,
    *,
    output_dir: Path,
    seed: int = INJECTION_SEED,
    attempts_per_cell: int = 20,
) -> dict[str, Any]:
    """Synthetic ramp-then-hold injection. Never touches the DB or main rows."""
    rng = random.Random(seed)
    dt = _median_interval(observations)
    false_evaluation = evaluate_rolling_baseline(observations, config)
    false_events = len(false_evaluation.events)

    valid_indices = [
        index
        for index, item in enumerate(observations)
        if item.position_px is not None and item.timestamp is not None
    ]
    cells: list[dict[str, Any]] = []
    for amplitude in INJECTION_AMPLITUDES:
        for duration in INJECTION_DURATIONS:
            ramp_samples = max(1, int(round(duration / dt)))
            usable = [
                index
                for index in valid_indices
                if index + ramp_samples + 10 < len(observations)
            ]
            if not usable:
                continue
            detections = 0
            latencies: list[float] = []
            for _ in range(attempts_per_cell):
                onset_index = rng.choice(usable)
                injected = _inject_ramp(
                    observations,
                    onset_index=onset_index,
                    amplitude_px=amplitude,
                    duration_seconds=duration,
                )
                evaluation = evaluate_rolling_baseline(injected, config)
                onset_time = observations[onset_index].timestamp
                if onset_time is None:
                    continue
                window_end = onset_time.timestamp() + duration + 2.0 * dt
                matches = [
                    event
                    for event in evaluation.events
                    if event.onset_time.timestamp() >= onset_time.timestamp() - 1e-6
                    and event.onset_time.timestamp() <= window_end
                ]
                if matches:
                    detections += 1
                    latencies.append(
                        matches[0].detection_time.timestamp()
                        - onset_time.timestamp()
                    )
            cells.append(
                {
                    "amplitude_px": amplitude,
                    "ramp_seconds": duration,
                    "attempts": attempts_per_cell,
                    "detections": detections,
                    "detection_rate": round(detections / attempts_per_cell, 6),
                    "median_latency_s": (
                        round(float(np.median(latencies)), 3) if latencies else None
                    ),
                    "false_event_count_unmodified": false_events,
                    "label": "synthetic injection - not real tsunami performance",
                }
            )
    _plot_injection(cells, output_dir / "plots" / "injection_detection_rate.png")
    return {
        "seed": seed,
        "attempts_per_cell": attempts_per_cell,
        "false_event_count_unmodified": false_events,
        "cells": cells,
        "label": "synthetic injection - not real tsunami performance",
    }


def _plot_injection(cells: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return
    if not cells:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    amplitudes = sorted({cell["amplitude_px"] for cell in cells})
    for amplitude in amplitudes:
        subset = sorted(
            [cell for cell in cells if cell["amplitude_px"] == amplitude],
            key=lambda cell: cell["ramp_seconds"],
        )
        axis.plot(
            [cell["ramp_seconds"] for cell in subset],
            [cell["detection_rate"] for cell in subset],
            marker="o",
            label=f"A={amplitude:.0f} px",
        )
    axis.set_xlabel("ramp duration T (s)")
    axis.set_ylabel("detection rate")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title(
        "Synthetic injection — not real tsunami performance\n"
        "Seaward ramp-then-hold into real shoreline series"
    )
    figure.text(
        0.5,
        0.01,
        "synthetic injection — not real tsunami performance",
        ha="center",
        fontsize=8,
        color="#7f8c8d",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path)
    plt.close(figure)


def plot_rolling(
    evaluation: RollingEvaluation, path: Path
) -> bool:
    """Observed vs rolling median, residual/z, and event markers."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return False
    rows = [
        row
        for row in evaluation.rows
        if row.timestamp is not None and row.observed_position_px is not None
    ]
    if not rows:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [row.timestamp for row in rows]
    observed = [row.observed_position_px for row in rows]
    baseline = [row.rolling_baseline_px for row in rows]
    residual = [row.residual_px for row in rows]
    z_values = [row.robust_z for row in rows]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(13, 7), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1.5, 1]}
    )
    top.plot(
        times,
        observed,
        color="#2c3e50",
        linewidth=1.2,
        marker=".",
        markersize=3,
        label="observed (px, + = seaward)",
    )
    top.plot(
        times,
        baseline,
        color="#0f766e",
        linewidth=1.6,
        linestyle="--",
        label="Rolling baseline (unverified; no tide data)",
    )
    for event in evaluation.events:
        top.axvline(event.onset_time, color="#c0392b", linewidth=1.0, alpha=0.8)
        top.scatter(
            [event.detection_time],
            [event.frozen_baseline_px],
            color="#c0392b",
            edgecolor="black",
            zorder=5,
            s=36,
            label="event (onset/detection)",
        )
        if event.end_time is not None:
            top.axvspan(event.onset_time, event.end_time, color="#c0392b", alpha=0.08)
    top.set_ylabel("shoreline position (px)")
    top.set_title(
        "Tide-independent rolling-median drawdown detector\n"
        "Rolling baseline is self-referential and unverified; it is not a tide model"
    )
    handles, labels = top.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    top.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    top.grid(alpha=0.25)

    bottom.plot(times, residual, color="#c0392b", linewidth=1.2, label="residual (px)")
    bottom.plot(times, z_values, color="#2563eb", linewidth=1.0, label="robust z", alpha=0.8)
    bottom.axhline(evaluation.config.z_threshold, color="#2563eb", linestyle=":", linewidth=1)
    bottom.axhline(0.0, color="#94a3b8", linewidth=1)
    bottom.set_ylabel("residual / z")
    bottom.grid(alpha=0.25)
    bottom.legend(fontsize=8, loc="best")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    figure.autofmt_xdate()
    figure.text(
        0.5,
        0.01,
        "Experimental rolling-baseline detector — not validated for tsunami detection.",
        ha="center",
        fontsize=8,
        color="#7f8c8d",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(path)
    plt.close(figure)
    return True
