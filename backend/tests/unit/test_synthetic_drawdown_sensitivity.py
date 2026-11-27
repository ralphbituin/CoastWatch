"""Synthetic seaward-ramp sensitivity check for the rolling detector.

This is a measurement test, not a calibration: thresholds are unchanged. It
prints a small table of first-candidate/first-event latency so the first
sensitivity result does not require waiting for a real event.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from app.services.anomaly.rolling_baseline import (
    CLASS_PENDING,
    EVENT_CLASSES,
    RollingBaselineConfig,
    RollingBaselineDetector,
    RollingObservation,
)

STEP_SECONDS = 30.0
WARMUP_SAMPLES = 40
BASE_PX = 800.0
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _config() -> RollingBaselineConfig:
    return RollingBaselineConfig(
        window_seconds=1200.0,
        k_seconds=90.0,
        min_history_fraction=0.75,
        z_threshold=3.5,
        sigma_k_multiplier=5.0,
        drawdown_floor_frac=150.0 / 1080.0,
        scale_floor_frac=0.004,
        persistence=3,
        max_gap_seconds=45.0,
        exit_z=2.0,
        exit_count=2,
        max_hold_seconds=3600.0,
        cooldown_seconds=600.0,
        confidence_floor=0.80,
        frame_height=1080.0,
        nominal_interval_seconds=STEP_SECONDS,
    )


def _noise(index: int) -> float:
    return 0.8 if index % 2 == 0 else -0.4


def _observations(values: list[float]) -> list[RollingObservation]:
    return [
        RollingObservation(
            timestamp=T0 + timedelta(seconds=STEP_SECONDS * index),
            position_px=value,
            confidence=0.95,
            frame_index=index,
            timestamp_confidence=1.0,
        )
        for index, value in enumerate(values)
    ]


def _ramp_series(ramp_px: float, *, ramp_seconds: float = 90.0) -> list[float]:
    """Warm-up, then a linear seaward ramp, then a 5-minute hold."""
    values = [BASE_PX + _noise(index) for index in range(WARMUP_SAMPLES)]
    steps = max(1, int(round(ramp_seconds / STEP_SECONDS)))
    for step in range(steps + 1):
        values.append(BASE_PX + ramp_px * step / steps + _noise(len(values)))
    hold_samples = int(300.0 / STEP_SECONDS)
    values.extend(BASE_PX + ramp_px + _noise(len(values)) for _ in range(hold_samples))
    return values


def _tidal_drift_series(samples: int = 120) -> list[float]:
    """A 12.4-hour sinusoid at a few centimetres of pixel excursion."""
    period_seconds = 12.4 * 3600.0
    return [
        BASE_PX
        + 40.0 * math.sin(2.0 * math.pi * STEP_SECONDS * index / period_seconds)
        + _noise(index)
        for index in range(samples)
    ]


def _first(rows, predicate) -> int | None:
    for index, row in enumerate(rows):
        if predicate(row):
            return index
    return None


def _measure(case: str, values: list[float]) -> dict[str, object]:
    detector = RollingBaselineDetector(_config())
    rows = [detector.update(item) for item in _observations(values)]
    first_candidate = _first(rows, lambda row: row.classification == CLASS_PENDING)
    first_event = _first(rows, lambda row: row.classification in EVENT_CLASSES)
    event_class = None if first_event is None else rows[first_event].classification
    if first_event is not None:
        reached = "EVENT"
    elif first_candidate is not None:
        reached = "CANDIDATE"
    else:
        reached = "NONE"
    ramp_start = WARMUP_SAMPLES
    return {
        "case": case,
        "reached": reached,
        "candidate_latency": (None if first_candidate is None else first_candidate - ramp_start),
        "event_latency": None if first_event is None else first_event - ramp_start,
        "event_class": event_class,
    }


def test_synthetic_seaward_ramp_sensitivity(capsys) -> None:
    cases = [
        _measure("80 px / 90 s", _ramp_series(80.0)),
        _measure("150 px / 90 s", _ramp_series(150.0)),
        _measure("200 px / 90 s", _ramp_series(200.0)),
        _measure("300 px / 90 s", _ramp_series(300.0)),
        _measure("negative control -50 px / 90 s", _ramp_series(-50.0)),
        _measure("tidal drift 12.4 h sinusoid", _tidal_drift_series()),
    ]

    with capsys.disabled():
        print("\nSynthetic seaward ramp sensitivity (floor 150 px at 1080 px):")
        print(f"{'case':34} {'reached':10} {'candidate@':>10} {'event@':>8} " f"{'event class':24}")
        for row in cases:
            print(
                f"{row['case']:34} {row['reached']:10} "
                f"{str(row['candidate_latency']):>10} "
                f"{str(row['event_latency']):>8} "
                f"{str(row['event_class']):24}"
            )

    # Controls must stay quiet; the ramp cases are reported, not forced.
    by_case = {row["case"]: row for row in cases}
    assert by_case["negative control -50 px / 90 s"]["reached"] == "NONE"
    assert by_case["tidal drift 12.4 h sinusoid"]["reached"] == "NONE"
    assert any(row["reached"] == "EVENT" for row in cases)
