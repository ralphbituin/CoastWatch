"""Pure replay helpers for DEMO MODE.

The detector is the production implementation from
``app.services.anomaly.rolling_baseline``. Nothing in this module reimplements
its baseline, robust z, movement gate, persistence or state machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from app.services.anomaly.rolling_baseline import (
    CLASS_DATA_UNRELIABLE,
    EVENT_CLASSES,
    STATE_CANDIDATE,
    STATE_EVENT,
    STATE_MONITORING,
    RollingBaselineConfig,
    RollingBaselineDetector,
    RollingObservation,
)

EVENT_WINDOW_PAD_SECONDS = 300.0
TAIL_SECONDS = 300.0
DEFAULT_HOLD_SECONDS = 300.0
MAX_RAMP_SECONDS = 600.0
Z_THRESHOLD = 3.5


@dataclass(slots=True)
class Sample:
    """One recorded shoreline sample, read-only."""

    index: int
    time: datetime
    p: float | None
    conf: float | None
    coverage: float | None
    frame_index: int | None
    session: str | None
    available: bool
    ts_conf: float | None
    reason: str | None
    horizon: float | None
    recorded_class: str
    recorded_state: str
    recorded_quality: str | None
    recorded_reason: str | None
    recorded_baseline: float | None
    recorded_z: float | None
    recorded_ck: float | None
    recorded_sigma_k: float | None
    recorded_persistence: int | None

    @property
    def unreliable(self) -> bool:
        return self.recorded_class == CLASS_DATA_UNRELIABLE


@dataclass(slots=True)
class Segment:
    """Contiguous replayable span of one recording (one session/frame size)."""

    start: int
    end: int  # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start


def make_observation(sample: Sample, position: float | None) -> RollingObservation:
    """Mirror the observation construction used by the offline injection test."""
    return RollingObservation(
        timestamp=sample.time,
        position_px=position,
        shoreline_available=bool(sample.available and position is not None),
        confidence=sample.conf,
        frame_index=sample.frame_index,
        timestamp_confidence=sample.ts_conf,
        reason=sample.reason,
        segment_measurements=None,
        horizon_y_px=sample.horizon,
    )


def effective_gate(sigma_k_px: float | None, config: RollingBaselineConfig) -> float | None:
    if sigma_k_px is None:
        return None
    return max(5.0 * float(sigma_k_px), float(config.drawdown_floor_px))


def injection_factor(
    seconds_since_start: float,
    ramp_seconds: float,
    hold_seconds: float,
    shape: str,
) -> float:
    """Piecewise-linear excursion factor, identical to the offline test.

    ``retreat_return``: 0 -> 1 over R, hold 1 for H, 1 -> 0 over R.
    ``persistent``: 0 -> 1 over R, then hold 1 for the rest of the replay.
    """
    if ramp_seconds <= 0:
        raise ValueError("ramp_seconds must be positive")
    t = float(seconds_since_start)
    if t <= 0:
        return 0.0
    if t < ramp_seconds:
        return t / ramp_seconds
    if shape == "persistent":
        return 1.0
    if shape != "retreat_return":
        raise ValueError(f"unknown shape {shape!r}")
    if t <= ramp_seconds + hold_seconds:
        return 1.0
    descent_end = 2.0 * ramp_seconds + hold_seconds
    if t >= descent_end:
        return 0.0
    return (descent_end - t) / ramp_seconds


def classify_failure(z_hits: int, gate_hits: int, both_hits: int, persistence_max: int) -> str:
    """Explain a miss from the detector's own counters (offline-test wording)."""
    if z_hits == 0:
        return "z_not_reached"
    if gate_hits == 0:
        return "movement_gate_not_reached"
    if both_hits == 0:
        return "z_and_gate_not_simultaneous"
    if persistence_max < 3:
        return "persistence_not_reached"
    return "no_event_in_window"


FAILURE_TEXT = {
    "z_not_reached": "Robust z never reached 3.5 during the injection window.",
    "movement_gate_not_reached": (
        "Cumulative drawdown never reached the recording's effective movement gate."
    ),
    "z_and_gate_not_simultaneous": (
        "Robust z and the effective movement gate were never met on the same sample."
    ),
    "persistence_not_reached": (
        "The detector saw a candidate but it did not persist for the required samples."
    ),
    "no_event_in_window": "No detector event was emitted during the injection window.",
}


def find_segment(segments: list[Segment], index: int) -> Segment:
    for segment in segments:
        if segment.start <= index < segment.end:
            return segment
    raise ValueError("sample index is outside every replayable segment")


def detector_after_prefix(
    samples: list[Sample],
    segments: list[Segment],
    config: RollingBaselineConfig,
    start_index: int,
) -> RollingBaselineDetector:
    """Fresh detector state seeded only from the unmodified same-segment history."""
    segment = find_segment(segments, start_index)
    detector = RollingBaselineDetector(config)
    for index in range(segment.start, start_index):
        sample = samples[index]
        detector.update(make_observation(sample, sample.p))
    return detector


def snapshot_after_prefix(
    samples: list[Sample],
    segments: list[Segment],
    config: RollingBaselineConfig,
    start_index: int,
) -> dict[str, Any]:
    return detector_after_prefix(samples, segments, config, start_index).export_state()


def replay_unmodified(
    samples: list[Sample],
    segments: list[Segment],
    config: RollingBaselineConfig,
) -> list[Any]:
    """Replay the unmodified series, resetting the detector at each segment."""
    rows: list[Any] = [None] * len(samples)
    for segment in segments:
        detector = RollingBaselineDetector(config)
        for index in range(segment.start, segment.end):
            sample = samples[index]
            rows[index] = detector.update(make_observation(sample, sample.p))
    return rows


def _row_payload(
    row: Any, sample: Sample, injected: float | None, gate: float | None
) -> dict[str, Any]:
    return {
        "time": row.timestamp,
        "real": sample.p,
        "injected": injected,
        "baseline": row.rolling_baseline_px,
        "z": row.robust_z,
        "ck": row.cumulative_drawdown_px,
        "gate": gate,
        "sigma_k": row.sigma_k_px,
        "persistence": row.persistence_count,
        "classification": row.classification,
        "state": row.state,
        "onset_time": row.onset_time,
        "detection_time": row.detection_time,
    }


def replay_injection(
    *,
    samples: list[Sample],
    segments: list[Segment],
    config: RollingBaselineConfig,
    start_index: int,
    amplitude_px: float,
    ramp_s: float,
    hold_s: float,
    shape: str,
    sign: int,
    snapshot: dict[str, Any] | None = None,
    injected_min_px: float | None = None,
    injected_max_px: float | None = None,
) -> dict[str, Any]:
    """One independent replay with a synthetic position excursion."""
    segment = find_segment(segments, start_index)
    if snapshot is None:
        snapshot = snapshot_after_prefix(samples, segments, config, start_index)
    detector = RollingBaselineDetector.from_state(snapshot, config)
    t0 = samples[start_index].time
    window_end = t0 + timedelta(seconds=2.0 * ramp_s + hold_s + EVENT_WINDOW_PAD_SECONDS)
    replay_end = window_end + timedelta(seconds=TAIL_SECONDS)

    series: list[dict[str, Any]] = []
    min_injected = math.inf
    max_injected = -math.inf
    in_bounds = True
    detection: dict[str, Any] | None = None
    first_gate: dict[str, Any] | None = None
    false_events = 0
    previous_event = False

    for index in range(start_index, segment.end):
        sample = samples[index]
        if sample.time > replay_end:
            break
        second = (sample.time - t0).total_seconds()
        factor = injection_factor(second, ramp_s, hold_s, shape)
        injected = None if sample.p is None else float(sample.p) + sign * amplitude_px * factor
        if injected is not None:
            min_injected = min(min_injected, injected)
            max_injected = max(max_injected, injected)
            if (
                injected_min_px is not None
                and injected_max_px is not None
                and not (injected_min_px <= injected <= injected_max_px)
            ):
                in_bounds = False
        row = detector.update(make_observation(sample, injected))
        gate = effective_gate(row.sigma_k_px, config)
        is_event = row.classification in EVENT_CLASSES
        if sample.time > window_end and is_event and not previous_event:
            false_events += 1
        previous_event = is_event
        item = _row_payload(row, sample, injected, gate)
        series.append(item)
        in_window = t0 <= sample.time <= window_end
        if (
            in_window
            and row.robust_z is not None
            and row.cumulative_drawdown_px is not None
            and gate is not None
            and float(row.robust_z) >= Z_THRESHOLD
            and float(row.cumulative_drawdown_px) >= float(gate)
        ) and first_gate is None:
            first_gate = item
        if in_window and detection is None and is_event:
            detection = item

    window_rows = [item for item in series if t0 <= item["time"] <= window_end]
    z_hits = sum(
        1 for item in window_rows if item["z"] is not None and float(item["z"]) >= Z_THRESHOLD
    )
    gate_hits = sum(
        1
        for item in window_rows
        if item["ck"] is not None
        and item["gate"] is not None
        and float(item["ck"]) >= float(item["gate"])
    )
    both_hits = sum(
        1
        for item in window_rows
        if item["z"] is not None
        and item["ck"] is not None
        and item["gate"] is not None
        and float(item["z"]) >= Z_THRESHOLD
        and float(item["ck"]) >= float(item["gate"])
    )
    persistence_max = max((int(item["persistence"] or 0) for item in window_rows), default=0)
    peak_z = max(
        (float(item["z"]) for item in window_rows if item["z"] is not None),
        default=float("nan"),
    )
    peak_ck = max(
        (float(item["ck"]) for item in window_rows if item["ck"] is not None),
        default=float("nan"),
    )
    detected = detection is not None
    detection_time = None
    latency_t0 = None
    latency_gate = None
    classification = "NONE"
    state_reached = STATE_MONITORING
    if detected and detection is not None:
        detection_time = detection["detection_time"] or detection["time"]
        latency_t0 = (detection_time - t0).total_seconds()
        classification = detection["classification"]
        if first_gate is not None:
            latency_gate = (detection_time - first_gate["time"]).total_seconds()
        state_reached = STATE_EVENT
    else:
        if any(item["classification"] == "PENDING_PERSISTENCE" for item in window_rows):
            state_reached = "PENDING_PERSISTENCE"
        elif any(item["state"] == STATE_CANDIDATE for item in window_rows):
            state_reached = STATE_CANDIDATE
    failure = "" if detected else classify_failure(z_hits, gate_hits, both_hits, persistence_max)

    return {
        "t0": t0,
        "start_index": start_index,
        "in_bounds": in_bounds,
        "min_injected_px": None if min_injected is math.inf else min_injected,
        "max_injected_px": None if max_injected == -math.inf else max_injected,
        "detected": detected,
        "detection_time": detection_time,
        "latency_from_t0_s": latency_t0,
        "latency_from_gate_s": latency_gate,
        "classification": classification,
        "state_reached": state_reached,
        "peak_z": peak_z,
        "peak_ck_px": peak_ck,
        "z_hit_samples": int(z_hits),
        "gate_hit_samples": int(gate_hits),
        "both_hit_samples": int(both_hits),
        "persistence_max": int(persistence_max),
        "failure_mode": failure,
        "failure_text": "" if detected else FAILURE_TEXT.get(failure, ""),
        "false_events": int(false_events),
        "series": series,
    }
