"""Tide-independent rolling-median drawdown detector (experimental).

This module is deliberately **not tide-aware**. It never uses, invents or
associates a tide height. It answers a narrower question:

    Is the waterline substantially farther seaward than its own recent local
    median, and did it get there quickly and persistently?

The local expectation is a causal rolling median of recent valid shoreline
positions. The robust scale is a MAD-derived sigma with a documented floor.
A candidate requires a large *and* rapid seaward residual; an event requires
several consecutive candidates.

Sign convention follows the repository: **positive = seaward** (the waterline
moved away from land / the water drew back). Class names are unchanged:
``RAPID_SHORELINE_RETREAT`` therefore means rapid seaward movement here.

This is a monitoring prototype, not a validated tsunami detector. The official
national warning authority for a deployment is configuration, never hardcoded
in this module.

Design notes
------------
* The stateful :class:`RollingBaselineDetector` is the single implementation.
  :func:`evaluate_rolling_baseline` is a thin batch wrapper around it, so batch
  and streaming results are identical by construction.
* The baseline/scoring statistics are **frozen** when a candidate starts.
  Candidate and event samples are quarantined and do not adapt the baseline.
* Invalid samples break the persistence chain and are never evidence of normal
  conditions.
* No outbound alert/notification hook exists here or is called by callers
  introduced in this task.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from app.services.anomaly.scoring import robust_scale
from app.services.anomaly.velocity import movement_rate as movement_rate_px_s

DETECTOR_MODE_ROLLING = "rolling_baseline"
BASELINE_STATUS_ROLLING = "rolling_unverified"

STATE_MONITORING = "MONITORING"
STATE_CANDIDATE = "CANDIDATE"
STATE_EVENT = "EVENT"
STATE_COOLDOWN = "COOLDOWN"

CLASS_DATA_UNRELIABLE = "DATA_UNRELIABLE"
CLASS_EXPLORATORY_ROLLING = "EXPLORATORY_ROLLING_BASELINE"
CLASS_PENDING = "PENDING_PERSISTENCE"
CLASS_POSSIBLE_DRAWDOWN = "POSSIBLE_DRAWDOWN"
CLASS_RAPID_RETREAT = "RAPID_SHORELINE_RETREAT"
CLASS_PERSISTENT_DRAWDOWN = "PERSISTENT_DRAWDOWN"

EVENT_CLASSES = {
    CLASS_POSSIBLE_DRAWDOWN,
    CLASS_RAPID_RETREAT,
    CLASS_PERSISTENT_DRAWDOWN,
}

SEVERITY = {
    CLASS_POSSIBLE_DRAWDOWN: 1,
    CLASS_RAPID_RETREAT: 2,
    CLASS_PERSISTENT_DRAWDOWN: 3,
}

_MAD_TO_SIGMA = 1.4826


def _utc_key(value: datetime | None) -> float:
    if value is None:
        return float("-inf")
    if value.tzinfo is None:
        return float("-inf")
    return value.timestamp()


@dataclass(slots=True)
class RollingBaselineConfig:
    """Time-based configuration. Sample counts are derived from the data."""

    window_seconds: float = 1200.0
    min_history_fraction: float = 0.75
    k_seconds: float = 90.0
    z_threshold: float = 3.5
    sigma_k_multiplier: float = 5.0
    # Fraction of the frame height. 150/1080 is the documented default floor.
    drawdown_floor_frac: float = 150.0 / 1080.0
    scale_floor_frac: float = 0.004
    persistence: int = 3
    max_gap_seconds: float = 45.0
    exit_z: float = 2.0
    exit_count: int = 2
    max_hold_seconds: float = 3600.0
    cooldown_seconds: float = 600.0
    persistent_seconds: float = 600.0
    confidence_floor: float = 0.80
    stale_run_length: int = 10
    require_segment_agreement: bool = False
    freeze_baseline: bool = True
    # Only used by live processing when a horizon measurement is available.
    camera_motion_shift_px: float = 20.0
    camera_motion_window: int = 5
    # A derived stored timestamp confidence below this is treated as unreliable.
    min_timestamp_confidence: float = 0.5
    # Optional frame height; when absent the documented pixel fallbacks apply.
    frame_height: float | None = None
    # Nominal sampling interval. When absent the detector estimates it.
    nominal_interval_seconds: float | None = None
    # Optional threshold for the existing rapid-rate predicate. When absent a
    # transparent floor/k threshold is used (see ``rapid_threshold_px_s``).
    rapid_rate_px_s: float | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` for impossible/incoherent configuration."""
        problems: list[str] = []
        for name in (
            "window_seconds",
            "k_seconds",
            "z_threshold",
            "sigma_k_multiplier",
            "drawdown_floor_frac",
            "scale_floor_frac",
            "max_gap_seconds",
            "max_hold_seconds",
            "persistent_seconds",
            "stale_run_length",
            "camera_motion_window",
            "min_timestamp_confidence",
        ):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                problems.append(f"{name} must be a finite number")
            elif float(value) <= 0:
                problems.append(f"{name} must be positive")
        if self.exit_z <= 0:
            problems.append("exit_z must be positive")
        if self.exit_count < 1:
            problems.append("exit_count must be >= 1")
        if self.persistence < 2:
            problems.append("persistence must be >= 2")
        if self.window_seconds < 3.0 * self.k_seconds:
            problems.append("window_seconds must be >= 3 * k_seconds")
        if not 0.0 < self.min_history_fraction <= 1.0:
            problems.append("min_history_fraction must be in (0, 1]")
        if not 0.0 <= self.confidence_floor <= 1.0:
            problems.append("confidence_floor must be in [0, 1]")
        if self.cooldown_seconds < 0:
            problems.append("cooldown_seconds must be >= 0")
        if self.frame_height is not None and self.frame_height <= 0:
            problems.append("frame_height must be positive when provided")
        if self.nominal_interval_seconds is not None and self.nominal_interval_seconds <= 0:
            problems.append("nominal_interval_seconds must be positive")
        if self.rapid_rate_px_s is not None and self.rapid_rate_px_s <= 0:
            problems.append("rapid_rate_px_s must be positive when provided")
        if problems:
            raise ValueError("invalid rolling baseline config: " + "; ".join(problems))

    # --- derived values -------------------------------------------------
    @property
    def drawdown_floor_px(self) -> float:
        if self.frame_height is not None:
            return float(self.drawdown_floor_frac) * float(self.frame_height)
        return 150.0

    @property
    def scale_floor_px(self) -> float:
        if self.frame_height is not None:
            return float(self.scale_floor_frac) * float(self.frame_height)
        return 4.0

    @property
    def drawdown_floor_fallback(self) -> bool:
        return self.frame_height is None

    @property
    def scale_floor_fallback(self) -> bool:
        return self.frame_height is None

    @property
    def rapid_threshold_px_s(self) -> float | None:
        """Threshold for the imported legacy rapid predicate, when configured."""
        return None if self.rapid_rate_px_s is None else float(self.rapid_rate_px_s)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "drawdown_floor_px": round(self.drawdown_floor_px, 6),
                "scale_floor_px": round(self.scale_floor_px, 6),
                "drawdown_floor_fallback": self.drawdown_floor_fallback,
                "scale_floor_fallback": self.scale_floor_fallback,
                "rapid_threshold_px_s": (
                    None
                    if self.rapid_threshold_px_s is None
                    else round(self.rapid_threshold_px_s, 6)
                ),
            }
        )
        return payload

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class RollingObservation:
    """One shoreline sample as seen by the rolling detector."""

    timestamp: datetime | None
    position_px: float | None
    shoreline_available: bool = True
    confidence: float | None = None
    frame_index: int | None = None
    timestamp_confidence: float | None = None
    reason: str | None = None
    # The existing ``measure_shoreline_position`` result, not a re-extraction.
    segment_measurements: list[dict[str, Any]] | None = None
    dominant_segment_index: int | None = None
    horizon_y_px: float | None = None


@dataclass(slots=True)
class RollingRowResult:
    timestamp: datetime | None
    frame_index: int | None
    classification: str
    state: str
    reason: str
    reason_code: str | None
    detector_mode: str = DETECTOR_MODE_ROLLING
    baseline_status: str = BASELINE_STATUS_ROLLING
    tide_adjusted: bool = False
    tide_data_available: bool = False
    tide_source: str = "unavailable"
    observed_position_px: float | None = None
    rolling_baseline_px: float | None = None
    rolling_scale_px: float | None = None
    scale_floored: bool = False
    residual_px: float | None = None
    robust_z: float | None = None
    cumulative_drawdown_px: float | None = None
    movement_rate_px_s: float | None = None
    persistence_count: int = 0
    history_count: int = 0
    history_required: int = 0
    quality_status: str | None = None
    confidence: float | None = None
    baseline_frozen: bool = False
    in_event: bool = False
    event_max_classification: str | None = None
    onset_time: datetime | None = None
    detection_time: datetime | None = None
    detection_latency_s: float | None = None
    event_window_seconds: float | None = None
    cooldown_state: str = "inactive"
    onset_frame_index: int | None = None
    detection_frame_index: int | None = None
    segment_evidence: list[dict[str, Any]] | None = None
    sigma_k_px: float | None = None
    sigma_k_samples: int = 0
    config_hash: str = ""
    camera_motion_gate: str = "unavailable"
    contributing_factors: dict[str, Any] = field(default_factory=dict)

    @property
    def is_event(self) -> bool:
        return self.classification in EVENT_CLASSES

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = None if self.timestamp is None else self.timestamp.isoformat()
        payload["onset_time"] = None if self.onset_time is None else self.onset_time.isoformat()
        payload["detection_time"] = (
            None if self.detection_time is None else self.detection_time.isoformat()
        )
        return payload


@dataclass(slots=True)
class DrawdownEvent:
    onset_time: datetime
    detection_time: datetime
    onset_frame_index: int | None
    detection_frame_index: int | None
    detection_latency_s: float
    persistence: int
    max_classification: str = CLASS_POSSIBLE_DRAWDOWN
    end_time: datetime | None = None
    end_reason: str | None = None
    max_robust_z: float | None = None
    max_residual_px: float | None = None
    peak_cumulative_drawdown_px: float | None = None
    frozen_baseline_px: float | None = None
    frozen_scale_px: float | None = None
    scale_floored: bool = False
    config_hash: str = ""
    sample_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "onset_time": self.onset_time.isoformat(),
            "detection_time": self.detection_time.isoformat(),
            "onset_frame_index": self.onset_frame_index,
            "detection_frame_index": self.detection_frame_index,
            "detection_latency_s": round(self.detection_latency_s, 6),
            "persistence": self.persistence,
            "max_classification": self.max_classification,
            "end_time": None if self.end_time is None else self.end_time.isoformat(),
            "end_reason": self.end_reason,
            "max_robust_z": self.max_robust_z,
            "max_residual_px": self.max_residual_px,
            "peak_cumulative_drawdown_px": self.peak_cumulative_drawdown_px,
            "frozen_baseline_px": self.frozen_baseline_px,
            "frozen_scale_px": self.frozen_scale_px,
            "scale_floored": self.scale_floored,
            "config_hash": self.config_hash,
            "sample_count": self.sample_count,
        }


@dataclass(slots=True)
class RollingEvaluation:
    rows: list[RollingRowResult]
    events: list[DrawdownEvent]
    dropped_duplicate_count: int
    gate_counts: dict[str, int]
    config: RollingBaselineConfig
    config_hash: str
    nominal_interval_seconds: float | None
    input_count: int


@dataclass(slots=True)
class _Frozen:
    baseline_px: float
    scale_px: float
    sigma_k_px: float
    scale_floored: bool
    sigma_k_samples: int
    frozen_at: datetime


@dataclass(slots=True)
class _PendingSample:
    timestamp: datetime
    position_px: float
    frame_index: int | None
    segment_value: float | None = None


class RollingBaselineDetector:
    """Stateful, causal rolling-median drawdown detector.

    Feed observations in timestamp order with :meth:`update`. The returned row
    is the verdict for that observation. History only ever contains valid,
    non-quarantined samples with ``t < t_i``.
    """

    def __init__(self, config: RollingBaselineConfig | None = None) -> None:
        self.config = config or RollingBaselineConfig()
        self.config.validate()
        self._history: list[tuple[datetime, float]] = []
        self._valid: list[tuple[datetime, float]] = []
        self._segment_history: list[tuple[datetime, float]] = []
        self._segment_valid: list[tuple[datetime, float]] = []
        self._pending: list[_PendingSample] = []
        self._frozen: _Frozen | None = None
        self._event: DrawdownEvent | None = None
        self._cooldown_until: datetime | None = None
        self._state = STATE_MONITORING
        self._last_seen_timestamp: datetime | None = None
        self._last_valid_timestamp: datetime | None = None
        self._stale_position: float | None = None
        self._stale_run = 0
        self._exit_run = 0
        self._strong_since: datetime | None = None
        self._strong_run = 0
        self._invalid_run_start: datetime | None = None
        self._after_event = False
        self._pending_reset_reason: str | None = None
        self._estimated_dt: float | None = self.config.nominal_interval_seconds
        self._horizon: deque[tuple[datetime, float]] = deque(
            maxlen=max(1, int(self.config.camera_motion_window))
        )
        self._gate_counts: Counter[str] = Counter()
        self._candidate_count = 0
        self._rows = 0
        self._closed_events: list[DrawdownEvent] = []
        self._segment_value_current: float | None = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def current_event(self) -> DrawdownEvent | None:
        return self._event

    @property
    def finished_events(self) -> list[DrawdownEvent]:
        return list(self._closed_events)

    @property
    def nominal_interval_seconds(self) -> float | None:
        return self._estimated_dt

    @property
    def gate_counts(self) -> dict[str, int]:
        return dict(self._gate_counts)

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------
    def update(self, observation: RollingObservation) -> RollingRowResult:
        self._rows += 1
        timestamp = observation.timestamp

        # Timestamp gates. A duplicate/backwards timestamp cannot be placed
        # causally, so it is unreliable and breaks persistence.
        if timestamp is None:
            return self._invalid(observation, "timestamp_missing", timestamp=None)
        if timestamp.tzinfo is None:
            return self._invalid(observation, "timestamp_not_timezone_aware", timestamp)
        if self._last_seen_timestamp is not None and timestamp <= self._last_seen_timestamp:
            code = (
                "duplicate_timestamp"
                if timestamp == self._last_seen_timestamp
                else "timestamp_not_increasing"
            )
            return self._invalid(observation, code, timestamp)

        # A frame-size change invalidates every prior pixel measurement. The
        # caller resets the state through ``reset_for_resolution_change``; the
        # next sample is reported unreliable and does not enter history.
        if self._pending_reset_reason is not None:
            code = self._pending_reset_reason
            self._pending_reset_reason = None
            self._last_seen_timestamp = timestamp
            return self._invalid(observation, code, timestamp)

        # Stale-feed detection uses the raw position, not residual.
        position = observation.position_px
        if (
            position is not None
            and math.isfinite(float(position))
            and self._stale_position is not None
            and float(position) == self._stale_position
        ):
            self._stale_run += 1
        elif position is not None and math.isfinite(float(position)):
            self._stale_position = float(position)
            self._stale_run = 1

        # Camera-motion gate. No horizon measurement -> explicitly unavailable.
        motion_reason = self._camera_motion_state(observation)
        if motion_reason is not None:
            self._last_seen_timestamp = timestamp
            self._reset_after_camera_motion(timestamp)
            return self._invalid(observation, "camera_motion", timestamp)

        gate_reason = self._gate_failure(observation)
        if gate_reason is not None:
            self._last_seen_timestamp = timestamp
            return self._invalid(observation, gate_reason, timestamp)

        assert position is not None and timestamp is not None
        self._last_seen_timestamp = timestamp
        self._invalid_run_start = None
        self._segment_value_current = self._segment_value(observation)
        self._update_dt_estimate(timestamp)

        # A valid sample after a gap is still a valid sample, but an ongoing
        # event ends when the gap exceeds max_gap.
        if self._state == STATE_EVENT and self._event is not None:
            gap_break = (
                self._last_valid_timestamp is not None
                and (timestamp - self._last_valid_timestamp).total_seconds()
                > self.config.max_gap_seconds
            )
            if gap_break:
                self._close_event(timestamp, "EVENT_GAP")
                self._enter_cooldown(timestamp)
        elif self._state == STATE_COOLDOWN and self._cooldown_expired(timestamp):
            self._state = STATE_MONITORING
            self._cooldown_until = None

        if self._state == STATE_MONITORING:
            result = self._monitoring_row(observation, timestamp, float(position))
        elif self._state == STATE_CANDIDATE:
            result = self._candidate_row(observation, timestamp, float(position))
        elif self._state == STATE_EVENT:
            result = self._event_row(observation, timestamp, float(position))
        else:  # COOLDOWN
            result = self._cooldown_row(observation, timestamp, float(position))

        self._last_valid_timestamp = timestamp
        self._valid.append((timestamp, float(position)))
        if self._segment_value_current is not None:
            self._segment_valid.append((timestamp, self._segment_value_current))
            self._prune_valid(timestamp)
        self._prune_valid(timestamp)
        return result

    def warm_start(self, history: list[RollingObservation]) -> list[RollingRowResult]:
        """Replay stored observations so a restart is not blind.

        The detector reconstructs exactly the state it would have had while
        consuming the same observations live. Returned rows are the replay
        results (callers normally discard them).
        """
        return [self.update(item) for item in history]

    def reset_for_resolution_change(self, timestamp: datetime | None = None) -> None:
        """Reset causal state after the decoded frame size changed.

        Reuses the camera-motion reset path. The next sample is reported as
        ``DATA_UNRELIABLE (resolution_change)`` and contributes no evidence.
        """
        self._reset_after_camera_motion(timestamp or datetime.now(UTC))
        self._pending_reset_reason = "resolution_change"

    # ------------------------------------------------------------------
    # Public state export/import
    # ------------------------------------------------------------------
    def export_state(self) -> dict[str, Any]:
        return {
            "config_hash": self.config.config_hash,
            "state": self._state,
            "history": [{"timestamp": t.isoformat(), "position_px": p} for t, p in self._history],
            "valid": [{"timestamp": t.isoformat(), "position_px": p} for t, p in self._valid],
            "segment_history": [
                {"timestamp": t.isoformat(), "position_px": p} for t, p in self._segment_history
            ],
            "segment_valid": [
                {"timestamp": t.isoformat(), "position_px": p} for t, p in self._segment_valid
            ],
            "pending": [
                {
                    "timestamp": item.timestamp.isoformat(),
                    "position_px": item.position_px,
                    "frame_index": item.frame_index,
                    "segment_value": item.segment_value,
                }
                for item in self._pending
            ],
            "frozen": None if self._frozen is None else asdict(self._frozen),
            "event": None if self._event is None else self._event.as_dict(),
            "closed_events": [event.as_dict() for event in self._closed_events],
            "cooldown_until": (
                None if self._cooldown_until is None else self._cooldown_until.isoformat()
            ),
            "last_seen_timestamp": (
                None if self._last_seen_timestamp is None else self._last_seen_timestamp.isoformat()
            ),
            "last_valid_timestamp": (
                None
                if self._last_valid_timestamp is None
                else self._last_valid_timestamp.isoformat()
            ),
            "stale_position": self._stale_position,
            "stale_run": self._stale_run,
            "exit_run": self._exit_run,
            "strong_since": (
                None if self._strong_since is None else self._strong_since.isoformat()
            ),
            "strong_run": self._strong_run,
            "after_event": self._after_event,
            "pending_reset_reason": self._pending_reset_reason,
            "estimated_dt": self._estimated_dt,
            "gate_counts": dict(self._gate_counts),
        }

    @classmethod
    def from_state(
        cls, state: dict[str, Any], config: RollingBaselineConfig | None = None
    ) -> RollingBaselineDetector:
        detector = cls(config)
        if state.get("config_hash") not in {None, detector.config.config_hash}:
            raise ValueError("rolling baseline state/config hash mismatch")
        detector._state = str(state.get("state", STATE_MONITORING))
        detector._history = [
            (datetime.fromisoformat(item["timestamp"]), float(item["position_px"]))
            for item in state.get("history", [])
        ]
        detector._valid = [
            (datetime.fromisoformat(item["timestamp"]), float(item["position_px"]))
            for item in state.get("valid", [])
        ]
        detector._segment_history = [
            (datetime.fromisoformat(item["timestamp"]), float(item["position_px"]))
            for item in state.get("segment_history", [])
        ]
        detector._segment_valid = [
            (datetime.fromisoformat(item["timestamp"]), float(item["position_px"]))
            for item in state.get("segment_valid", [])
        ]
        detector._pending = [
            _PendingSample(
                timestamp=datetime.fromisoformat(item["timestamp"]),
                position_px=float(item["position_px"]),
                frame_index=item.get("frame_index"),
                segment_value=item.get("segment_value"),
            )
            for item in state.get("pending", [])
        ]
        frozen = state.get("frozen")
        detector._frozen = (
            None
            if not frozen
            else _Frozen(
                baseline_px=float(frozen["baseline_px"]),
                scale_px=float(frozen["scale_px"]),
                sigma_k_px=float(frozen["sigma_k_px"]),
                scale_floored=bool(frozen["scale_floored"]),
                sigma_k_samples=int(frozen["sigma_k_samples"]),
                frozen_at=datetime.fromisoformat(frozen["frozen_at"]),
            )
        )
        event = state.get("event")
        if event:
            detector._event = DrawdownEvent(
                onset_time=datetime.fromisoformat(event["onset_time"]),
                detection_time=datetime.fromisoformat(event["detection_time"]),
                onset_frame_index=event.get("onset_frame_index"),
                detection_frame_index=event.get("detection_frame_index"),
                detection_latency_s=float(event["detection_latency_s"]),
                persistence=int(event["persistence"]),
                max_classification=str(event.get("max_classification", CLASS_POSSIBLE_DRAWDOWN)),
                end_time=(
                    None
                    if event.get("end_time") is None
                    else datetime.fromisoformat(event["end_time"])
                ),
                end_reason=event.get("end_reason"),
                max_robust_z=event.get("max_robust_z"),
                max_residual_px=event.get("max_residual_px"),
                peak_cumulative_drawdown_px=event.get("peak_cumulative_drawdown_px"),
                frozen_baseline_px=event.get("frozen_baseline_px"),
                frozen_scale_px=event.get("frozen_scale_px"),
                scale_floored=bool(event.get("scale_floored", False)),
                config_hash=event.get("config_hash", ""),
                sample_count=int(event.get("sample_count", 0)),
            )
        detector._closed_events = [
            DrawdownEvent(
                onset_time=datetime.fromisoformat(item["onset_time"]),
                detection_time=datetime.fromisoformat(item["detection_time"]),
                onset_frame_index=item.get("onset_frame_index"),
                detection_frame_index=item.get("detection_frame_index"),
                detection_latency_s=float(item["detection_latency_s"]),
                persistence=int(item["persistence"]),
                max_classification=str(item.get("max_classification", CLASS_POSSIBLE_DRAWDOWN)),
                end_time=(
                    None
                    if item.get("end_time") is None
                    else datetime.fromisoformat(item["end_time"])
                ),
                end_reason=item.get("end_reason"),
                max_robust_z=item.get("max_robust_z"),
                max_residual_px=item.get("max_residual_px"),
                peak_cumulative_drawdown_px=item.get("peak_cumulative_drawdown_px"),
                frozen_baseline_px=item.get("frozen_baseline_px"),
                frozen_scale_px=item.get("frozen_scale_px"),
                scale_floored=bool(item.get("scale_floored", False)),
                config_hash=item.get("config_hash", ""),
                sample_count=int(item.get("sample_count", 0)),
            )
            for item in state.get("closed_events", [])
        ]
        detector._cooldown_until = (
            None
            if state.get("cooldown_until") is None
            else datetime.fromisoformat(state["cooldown_until"])
        )
        detector._last_seen_timestamp = (
            None
            if state.get("last_seen_timestamp") is None
            else datetime.fromisoformat(state["last_seen_timestamp"])
        )
        detector._last_valid_timestamp = (
            None
            if state.get("last_valid_timestamp") is None
            else datetime.fromisoformat(state["last_valid_timestamp"])
        )
        detector._stale_position = state.get("stale_position")
        detector._stale_run = int(state.get("stale_run", 0))
        detector._exit_run = int(state.get("exit_run", 0))
        detector._strong_since = (
            None
            if state.get("strong_since") is None
            else datetime.fromisoformat(state["strong_since"])
        )
        detector._strong_run = int(state.get("strong_run", 0))
        detector._after_event = bool(state.get("after_event", False))
        detector._pending_reset_reason = state.get("pending_reset_reason")
        detector._estimated_dt = state.get("estimated_dt")
        detector._gate_counts = Counter(state.get("gate_counts", {}))
        return detector

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    def _gate_failure(self, observation: RollingObservation) -> str | None:
        if not observation.shoreline_available:
            return "shoreline_unavailable"
        position = observation.position_px
        if position is None or not math.isfinite(float(position)):
            return "position_missing"
        if observation.segment_measurements is not None:
            if not observation.segment_measurements:
                return "dominant_segment_missing"
            if observation.dominant_segment_index is not None and not any(
                int(item.get("index", -1)) == int(observation.dominant_segment_index)
                for item in observation.segment_measurements
                if isinstance(item, dict)
            ):
                return "dominant_segment_missing"
        if observation.confidence is None:
            return "confidence_missing"
        if float(observation.confidence) < self.config.confidence_floor:
            return "low_confidence"
        if (
            observation.timestamp_confidence is not None
            and float(observation.timestamp_confidence) < self.config.min_timestamp_confidence
        ):
            return "low_timestamp_confidence"
        if self._stale_run >= self.config.stale_run_length:
            return "stale_feed"
        return None

    def _camera_motion_state(self, observation: RollingObservation) -> str | None:
        """Return a motion reason, or ``None`` when the gate is unavailable/ok."""
        horizon = observation.horizon_y_px
        if horizon is None or not math.isfinite(float(horizon)):
            return None
        timestamp = observation.timestamp
        if timestamp is None:
            return None
        if len(self._horizon) >= 2:
            values = [value for _, value in self._horizon]
            median = float(np.median(values))
            if abs(float(horizon) - median) > self.config.camera_motion_shift_px:
                return "camera_motion"
        self._horizon.append((timestamp, float(horizon)))
        return None

    # ------------------------------------------------------------------
    # Score primitives
    # ------------------------------------------------------------------
    def _nominal_dt(self) -> float:
        if self._estimated_dt is not None and self._estimated_dt > 0:
            return float(self._estimated_dt)
        return 30.0

    def _update_dt_estimate(self, timestamp: datetime) -> None:
        if self.config.nominal_interval_seconds is not None:
            self._estimated_dt = float(self.config.nominal_interval_seconds)
            return
        if self._last_valid_timestamp is None:
            return
        delta = (timestamp - self._last_valid_timestamp).total_seconds()
        if 0 < delta <= 10.0 * self._nominal_dt():
            recent = [
                (b - a).total_seconds()
                for (a, _), (b, _) in zip(self._valid[-20:-1], self._valid[-19:], strict=False)
                if 0 < (b - a).total_seconds() <= 10.0 * self._nominal_dt()
            ]
            recent.append(delta)
            self._estimated_dt = float(np.median(recent))

    def _required_history(self) -> int:
        return max(
            1,
            int(
                math.ceil(
                    self.config.min_history_fraction
                    * self.config.window_seconds
                    / self._nominal_dt()
                )
            ),
        )

    def _prune_history(self, timestamp: datetime) -> None:
        cutoff = timestamp.timestamp() - self.config.window_seconds
        self._history = [(t, p) for t, p in self._history if t.timestamp() >= cutoff]
        if self.config.require_segment_agreement:
            self._segment_history = [
                (t, p) for t, p in self._segment_history if t.timestamp() >= cutoff
            ]

    def _prune_valid(self, timestamp: datetime) -> None:
        # Keep enough valid history for the k-second lag and the k-pair scale.
        cutoff = timestamp.timestamp() - max(
            self.config.window_seconds, 2.0 * self.config.k_seconds
        )
        self._valid = [(t, p) for t, p in self._valid if t.timestamp() >= cutoff]

    def _nearest_valid(
        self, target: datetime, tolerance_seconds: float
    ) -> tuple[datetime, float] | None:
        if not self._valid:
            return None
        keys = [t.timestamp() for t, _ in self._valid]
        index = bisect_left(keys, target.timestamp())
        best: tuple[datetime, float] | None = None
        best_delta = float("inf")
        for candidate in (index - 1, index):
            if 0 <= candidate < len(self._valid):
                moment, value = self._valid[candidate]
                delta = abs((moment - target).total_seconds())
                if delta <= tolerance_seconds and delta < best_delta:
                    best = (moment, value)
                    best_delta = delta
        return best

    def _cumulative(
        self, timestamp: datetime, position: float
    ) -> tuple[float | None, float | None, float | None]:
        dt = self._nominal_dt()
        target = timestamp - timedelta(seconds=self.config.k_seconds)
        nearest = self._nearest_valid(target, 0.5 * dt)
        if nearest is None:
            return None, None, None
        moment, previous = nearest
        ck = float(position) - float(previous)
        rate = movement_rate_px_s(
            float(position), float(previous), (timestamp - moment).total_seconds()
        )
        return ck, rate, (timestamp - moment).total_seconds()

    def _sigma_k(self) -> tuple[float, int]:
        dt = self._nominal_dt()
        diffs: list[float] = []
        if not self._history:
            return self.config.scale_floor_px, 0
        keys = [t.timestamp() for t, _ in self._history]
        for timestamp, position in self._history:
            target = timestamp - timedelta(seconds=self.config.k_seconds)
            nearest = self._nearest_with_keys(self._history, keys, target, 0.5 * dt)
            if nearest is not None:
                diffs.append(float(position) - float(nearest[1]))
        if not diffs:
            return self.config.scale_floor_px, 0
        scale = robust_scale(diffs)
        value = scale.scale if scale.scale is not None else 0.0
        return max(value, self.config.scale_floor_px), len(diffs)

    @staticmethod
    def _nearest_with_keys(
        series: list[tuple[datetime, float]],
        keys: list[float],
        target: datetime,
        tolerance: float,
    ) -> tuple[datetime, float] | None:
        index = bisect_left(keys, target.timestamp())
        best: tuple[datetime, float] | None = None
        best_delta = float("inf")
        for candidate in (index - 1, index):
            if 0 <= candidate < len(series):
                moment, value = series[candidate]
                delta = abs((moment - target).total_seconds())
                if delta <= tolerance and delta < best_delta:
                    best = (moment, value)
                    best_delta = delta
        return best

    def _segment_value(self, observation: RollingObservation) -> float | None:
        """Median signed position across substantial segments (optional gate)."""
        if not self.config.require_segment_agreement:
            return None
        segments = observation.segment_measurements or []
        values: list[float] = []
        counts: list[int] = []
        for item in segments:
            if not isinstance(item, dict):
                continue
            raw = item.get("signed_median")
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            count = item.get("region_point_count") or item.get("point_count") or 0
            values.append(value)
            counts.append(int(count))
        if not values:
            return None
        maximum = max(counts)
        minimum = max(2, int(round(0.25 * maximum)))
        substantial = [
            value for value, count in zip(values, counts, strict=False) if count >= minimum
        ]
        return float(np.median(substantial or values))

    def _segment_score(self, timestamp: datetime) -> tuple[bool, dict[str, Any]]:
        """Second-series z/floor test across substantial segments."""
        if not self.config.require_segment_agreement:
            return True, {"required": False}
        value = self._segment_value_current
        if value is None:
            return False, {"available": False}
        cutoff = timestamp.timestamp() - self.config.window_seconds
        self._segment_history = [
            (t, p) for t, p in self._segment_history if t.timestamp() >= cutoff
        ]
        required = self._required_history()
        if len(self._segment_history) < required:
            return False, {
                "available": True,
                "history_count": len(self._segment_history),
                "history_required": required,
            }
        scale_stats = robust_scale([p for _, p in self._segment_history])
        assert scale_stats.median is not None
        baseline = float(scale_stats.median)
        raw_scale = scale_stats.scale if scale_stats.scale is not None else 0.0
        scale = max(float(raw_scale), self.config.scale_floor_px)
        residual = value - baseline
        z_value = residual / scale if scale > 0 else None
        dt = self._nominal_dt()
        target = timestamp - timedelta(seconds=self.config.k_seconds)
        nearest = self._nearest_in_series(self._segment_valid, target, 0.5 * dt)
        ck = None if nearest is None else value - nearest[1]
        candidate = bool(
            z_value is not None
            and ck is not None
            and z_value >= self.config.z_threshold
            and ck
            >= max(
                self.config.sigma_k_multiplier * scale,
                self.config.drawdown_floor_px,
            )
        )
        return candidate, {
            "available": True,
            "value_px": round(value, 6),
            "baseline_px": round(baseline, 6),
            "scale_px": round(scale, 6),
            "residual_px": round(residual, 6),
            "robust_z": round(z_value, 6) if z_value is not None else None,
            "cumulative_drawdown_px": round(ck, 6) if ck is not None else None,
            "candidate": candidate,
            "history_count": len(self._segment_history),
            "history_required": required,
        }

    @staticmethod
    def _nearest_in_series(
        series: list[tuple[datetime, float]],
        target: datetime,
        tolerance: float,
    ) -> tuple[datetime, float] | None:
        if not series:
            return None
        keys = [t.timestamp() for t, _ in series]
        return RollingBaselineDetector._nearest_with_keys(series, keys, target, tolerance)

    def _score(self, timestamp: datetime, position: float, *, frozen: bool) -> dict[str, Any]:
        self._prune_history(timestamp)
        required = self._required_history()
        history_count = len(self._history)

        if frozen and self._frozen is not None:
            baseline = self._frozen.baseline_px
            scale = self._frozen.scale_px
            scale_floored = self._frozen.scale_floored
            sigma_k = self._frozen.sigma_k_px
            sigma_samples = self._frozen.sigma_k_samples
        else:
            if history_count >= required and self._history:
                scale_stats = robust_scale([p for _, p in self._history])
                assert scale_stats.median is not None
                baseline = float(scale_stats.median)
                raw_scale = float(scale_stats.scale) if scale_stats.scale is not None else 0.0
                scale_floored = raw_scale < self.config.scale_floor_px
                scale = max(raw_scale, self.config.scale_floor_px)
            else:
                baseline = None
                scale = None
                scale_floored = False
            sigma_k, sigma_samples = self._sigma_k() if baseline is not None else (0.0, 0)

        residual = None if baseline is None else float(position) - float(baseline)
        z = None if residual is None or scale is None or scale <= 0 else residual / float(scale)
        ck, rate, ck_span = self._cumulative(timestamp, float(position))
        candidate = bool(
            baseline is not None
            and z is not None
            and ck is not None
            and z >= self.config.z_threshold
            and ck
            >= max(
                self.config.sigma_k_multiplier * sigma_k,
                self.config.drawdown_floor_px,
            )
        )
        segment_candidate, segment_evidence = self._segment_score(timestamp)
        if self.config.require_segment_agreement:
            candidate = bool(candidate and segment_candidate)
        return {
            "baseline": baseline,
            "scale": scale,
            "scale_floored": scale_floored,
            "residual": residual,
            "z": z,
            "ck": ck,
            "rate": rate,
            "ck_span": ck_span,
            "sigma_k": sigma_k if baseline is not None else None,
            "sigma_samples": sigma_samples,
            "candidate": candidate,
            "segment_candidate": segment_candidate,
            "segment_evidence": segment_evidence,
            "history_count": history_count,
            "history_required": required,
        }

    # ------------------------------------------------------------------
    # State rows
    # ------------------------------------------------------------------
    def _monitoring_row(
        self, observation: RollingObservation, timestamp: datetime, position: float
    ) -> RollingRowResult:
        scores = self._score(timestamp, position, frozen=False)
        if scores["baseline"] is None:
            code = (
                "insufficient_history_after_event" if self._after_event else "insufficient_history"
            )
            self._add_to_history(timestamp, position)
            return self._make_result(
                observation,
                timestamp,
                CLASS_DATA_UNRELIABLE,
                STATE_MONITORING,
                f"valid observations in the rolling window {scores['history_count']}"
                f"/{scores['history_required']} after sorting duplicates/gates; "
                "baseline cannot be computed",
                code,
                scores,
                quality_status=code,
            )
        if scores["ck"] is None:
            self._record_gate("insufficient_cumulative_history")
            self._add_to_history(timestamp, position)
            return self._make_result(
                observation,
                timestamp,
                CLASS_DATA_UNRELIABLE,
                STATE_MONITORING,
                "no valid sample near t - k_seconds; cumulative drawdown is unavailable",
                "insufficient_cumulative_history",
                scores,
                quality_status="insufficient_cumulative_history",
            )

        if scores["candidate"]:
            self._frozen = self._freeze(scores, timestamp)
            self._pending = [
                _PendingSample(
                    timestamp=timestamp,
                    position_px=position,
                    frame_index=observation.frame_index,
                    segment_value=self._segment_value_current,
                )
            ]
            self._candidate_count = 1
            self._state = STATE_CANDIDATE
            threshold_px = max(
                self.config.sigma_k_multiplier * scores["sigma_k"],
                self.config.drawdown_floor_px,
            )
            return self._make_result(
                observation,
                timestamp,
                CLASS_PENDING,
                STATE_CANDIDATE,
                f"candidate 1/{self.config.persistence}: z={scores['z']:.2f} >= "
                f"{self.config.z_threshold:.2f} and cumulative drawdown "
                f"{scores['ck']:.1f} px >= {threshold_px:.1f} px",
                "candidate",
                scores,
                quality_status="candidate",
                frozen=True,
            )

        self._after_event = False
        self._add_to_history(timestamp, position)
        return self._make_result(
            observation,
            timestamp,
            CLASS_EXPLORATORY_ROLLING,
            STATE_MONITORING,
            "valid observation within the rolling median; no persistent rapid seaward drawdown",
            None,
            scores,
            quality_status="ok",
        )

    def _candidate_row(
        self, observation: RollingObservation, timestamp: datetime, position: float
    ) -> RollingRowResult:
        scores = self._score(timestamp, position, frozen=bool(self.config.freeze_baseline))
        gap = (
            float("inf")
            if self._last_valid_timestamp is None
            else (timestamp - self._last_valid_timestamp).total_seconds()
        )
        if scores["candidate"] and gap <= self.config.max_gap_seconds:
            self._candidate_count += 1
            self._pending.append(
                _PendingSample(
                    timestamp,
                    position,
                    observation.frame_index,
                    self._segment_value_current,
                )
            )
            if self._candidate_count >= self.config.persistence:
                event = self._start_event(timestamp, observation.frame_index, scores)
                self._state = STATE_EVENT
                return self._event_result(observation, timestamp, position, scores, event)
            return self._make_result(
                observation,
                timestamp,
                CLASS_PENDING,
                STATE_CANDIDATE,
                f"candidate {self._candidate_count}/{self.config.persistence}: "
                f"z={scores['z']:.2f}, cumulative drawdown {scores['ck']:.1f} px",
                "candidate",
                scores,
                quality_status="candidate",
                frozen=True,
            )

        # Failed candidate run: unfreeze, return pending samples to history.
        self._return_pending_to_history()
        self._frozen = None
        self._state = STATE_MONITORING
        self._after_event = False
        self._add_to_history(timestamp, position)
        return self._make_result(
            observation,
            timestamp,
            CLASS_EXPLORATORY_ROLLING,
            STATE_MONITORING,
            f"candidate run failed ({self._candidate_count}/{self.config.persistence}); "
            "baseline unfrozen and pending samples returned to history",
            "candidate_failed",
            scores,
            quality_status="candidate_failed",
        )

    def _event_row(
        self, observation: RollingObservation, timestamp: datetime, position: float
    ) -> RollingRowResult:
        assert self._event is not None
        scores = self._score(timestamp, position, frozen=bool(self.config.freeze_baseline))
        event = self._event
        event.sample_count += 1

        # Max-hold expiry: release the freeze without inventing an event end.
        if (timestamp - event.onset_time).total_seconds() >= self.config.max_hold_seconds:
            self._close_event(timestamp, "EVENT_EXPIRED")
            self._enter_cooldown(timestamp)
            self._add_to_history(timestamp, position)
            return self._make_result(
                observation,
                timestamp,
                CLASS_EXPLORATORY_ROLLING,
                STATE_COOLDOWN,
                "EVENT_EXPIRED: maximum hold time reached; frozen baseline released",
                "EVENT_EXPIRED",
                scores,
                quality_status="event_expired",
                event=event,
                cooldown=True,
            )

        # Exit rule: consecutive valid samples with z < exit_z.
        z = scores["z"]
        if z is not None and z < self.config.exit_z:
            self._exit_run += 1
        else:
            self._exit_run = 0
        if self._exit_run >= self.config.exit_count:
            self._close_event(timestamp, "event_exit")
            self._enter_cooldown(timestamp)
            self._add_to_history(timestamp, position)
            return self._make_result(
                observation,
                timestamp,
                CLASS_EXPLORATORY_ROLLING,
                STATE_COOLDOWN,
                f"event ended after {self._exit_run} consecutive valid samples with "
                f"z < {self.config.exit_z:.2f}",
                "event_exit",
                scores,
                quality_status="event_exit",
                event=event,
                cooldown=True,
            )

        classification = self._event_classification(timestamp, scores)
        if SEVERITY[classification] > SEVERITY.get(event.max_classification, 0):
            event.max_classification = classification
        if scores["z"] is not None:
            event.max_robust_z = (
                scores["z"] if event.max_robust_z is None else max(event.max_robust_z, scores["z"])
            )
        if scores["residual"] is not None:
            event.max_residual_px = (
                scores["residual"]
                if event.max_residual_px is None
                else max(event.max_residual_px, scores["residual"])
            )
        if scores["ck"] is not None:
            event.peak_cumulative_drawdown_px = (
                scores["ck"]
                if event.peak_cumulative_drawdown_px is None
                else max(event.peak_cumulative_drawdown_px, scores["ck"])
            )
        return self._event_result(observation, timestamp, position, scores, event, classification)

    def _cooldown_row(
        self, observation: RollingObservation, timestamp: datetime, position: float
    ) -> RollingRowResult:
        if self._cooldown_expired(timestamp):
            self._state = STATE_MONITORING
            self._cooldown_until = None
            return self._monitoring_row(observation, timestamp, position)
        scores = self._score(timestamp, position, frozen=False)
        if scores["baseline"] is None:
            self._add_to_history(timestamp, position)
            code = (
                "insufficient_history_after_event" if self._after_event else "insufficient_history"
            )
            return self._make_result(
                observation,
                timestamp,
                CLASS_DATA_UNRELIABLE,
                STATE_COOLDOWN,
                f"valid observations in the rolling window {scores['history_count']}"
                f"/{scores['history_required']} after cooldown/quarantine; "
                "baseline cannot be computed",
                code,
                scores,
                quality_status=code,
                cooldown=True,
            )
        self._add_to_history(timestamp, position)
        return self._make_result(
            observation,
            timestamp,
            CLASS_EXPLORATORY_ROLLING,
            STATE_COOLDOWN,
            "cooldown active after a previous event; candidate suppression active",
            "cooldown",
            scores,
            quality_status="cooldown",
            cooldown=True,
        )

    def _event_classification(self, timestamp: datetime, scores: dict[str, Any]) -> str:
        strong = bool(
            scores["residual"] is not None
            and scores["z"] is not None
            and scores["residual"] >= 2.0 * self.config.drawdown_floor_px
            and scores["z"] >= self.config.z_threshold
        )
        if strong:
            if self._strong_since is None:
                self._strong_since = timestamp
            self._strong_run += 1
        else:
            self._strong_since = None
            self._strong_run = 0
        if (
            strong
            and self._strong_since is not None
            and (timestamp - self._strong_since).total_seconds() >= self.config.persistent_seconds
        ):
            return CLASS_PERSISTENT_DRAWDOWN
        rate = scores["rate"]
        from app.services.anomaly.detector import is_rapid_retreat

        if is_rapid_retreat(rate, self.config.rapid_threshold_px_s):
            return CLASS_RAPID_RETREAT
        return CLASS_POSSIBLE_DRAWDOWN

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------
    def _freeze(self, scores: dict[str, Any], timestamp: datetime) -> _Frozen:
        assert scores["baseline"] is not None and scores["scale"] is not None
        return _Frozen(
            baseline_px=float(scores["baseline"]),
            scale_px=float(scores["scale"]),
            sigma_k_px=float(scores["sigma_k"] or 0.0),
            scale_floored=bool(scores["scale_floored"]),
            sigma_k_samples=int(scores["sigma_samples"]),
            frozen_at=timestamp,
        )

    def _start_event(
        self, timestamp: datetime, frame_index: int | None, scores: dict[str, Any]
    ) -> DrawdownEvent:
        first = self._pending[0]
        frozen = self._frozen
        assert frozen is not None
        event = DrawdownEvent(
            onset_time=first.timestamp,
            detection_time=timestamp,
            onset_frame_index=first.frame_index,
            detection_frame_index=frame_index,
            detection_latency_s=(timestamp - first.timestamp).total_seconds(),
            persistence=self._candidate_count,
            max_classification=CLASS_POSSIBLE_DRAWDOWN,
            frozen_baseline_px=frozen.baseline_px,
            frozen_scale_px=frozen.scale_px,
            scale_floored=frozen.scale_floored,
            config_hash=self.config.config_hash,
            max_robust_z=scores["z"],
            max_residual_px=scores["residual"],
            peak_cumulative_drawdown_px=scores["ck"],
        )
        self._event = event
        self._frozen = frozen
        self._pending = []
        self._exit_run = 0
        self._strong_since = None
        self._strong_run = 0
        return event

    def _close_event(self, timestamp: datetime, reason: str) -> None:
        if self._event is None:
            return
        event = self._event
        event.end_time = timestamp
        event.end_reason = reason
        if event not in self._closed_events:
            self._closed_events.append(event)
        self._event = None
        self._frozen = None
        self._pending = []
        self._exit_run = 0
        self._strong_since = None
        self._strong_run = 0
        self._after_event = True
        self._state = STATE_COOLDOWN

    def _enter_cooldown(self, timestamp: datetime) -> None:
        self._cooldown_until = timestamp + timedelta(seconds=self.config.cooldown_seconds)

    def _cooldown_expired(self, timestamp: datetime) -> bool:
        return self._cooldown_until is not None and timestamp >= self._cooldown_until

    def _return_pending_to_history(self) -> None:
        for item in self._pending:
            self._history.append((item.timestamp, item.position_px))
            if item.segment_value is not None:
                self._segment_history.append((item.timestamp, item.segment_value))
        self._history.sort(key=lambda pair: pair[0])
        self._segment_history.sort(key=lambda pair: pair[0])
        self._pending = []
        self._candidate_count = 0

    def _add_to_history(self, timestamp: datetime, position: float) -> None:
        self._history.append((timestamp, position))
        self._history.sort(key=lambda pair: pair[0])
        if self._segment_value_current is not None:
            self._segment_history.append((timestamp, self._segment_value_current))
            self._segment_history.sort(key=lambda pair: pair[0])
        self._prune_history(timestamp)

    def _reset_after_camera_motion(self, timestamp: datetime) -> None:
        self._history = []
        self._valid = []
        self._segment_history = []
        self._segment_valid = []
        self._segment_value_current = None
        self._pending = []
        self._frozen = None
        self._exit_run = 0
        self._strong_since = None
        self._strong_run = 0
        self._candidate_count = 0
        self._after_event = False
        had_event = self._event is not None
        if had_event:
            self._close_event(timestamp, "camera_motion")
            self._enter_cooldown(timestamp)
        else:
            self._state = STATE_MONITORING
            self._cooldown_until = None
        self._horizon.clear()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _record_gate(self, code: str) -> None:
        self._gate_counts[code] += 1

    def _invalid(
        self,
        observation: RollingObservation,
        code: str,
        timestamp: datetime | None,
    ) -> RollingRowResult:
        self._record_gate(code)
        event = self._event
        if self._state == STATE_CANDIDATE:
            self._return_pending_to_history()
            self._frozen = None
            self._state = STATE_MONITORING
            self._candidate_count = 0
        elif self._state == STATE_EVENT and self._event is not None:
            if timestamp is not None:
                if self._invalid_run_start is None:
                    self._invalid_run_start = timestamp
                if (
                    timestamp - self._invalid_run_start
                ).total_seconds() > self.config.max_gap_seconds:
                    self._close_event(timestamp, "EVENT_GAP")
                    self._enter_cooldown(timestamp)
            else:
                last_seen = self._last_valid_timestamp or self._event.onset_time
                self._close_event(last_seen, "EVENT_GAP")
                self._enter_cooldown(last_seen)
        return RollingRowResult(
            timestamp=timestamp,
            frame_index=observation.frame_index,
            classification=CLASS_DATA_UNRELIABLE,
            state=(
                self._state if self._state in {STATE_COOLDOWN, STATE_EVENT} else STATE_MONITORING
            ),
            reason=f"DATA_UNRELIABLE ({code}); sample cannot contribute evidence or normality",
            reason_code=code,
            observed_position_px=(
                None
                if observation.position_px is None
                or not math.isfinite(float(observation.position_px))
                else float(observation.position_px)
            ),
            persistence_count=self._candidate_count,
            confidence=observation.confidence,
            baseline_frozen=self._frozen is not None,
            in_event=event is not None and self._state == STATE_EVENT,
            event_max_classification=None if event is None else event.max_classification,
            onset_time=None if event is None else event.onset_time,
            detection_time=None if event is None else event.detection_time,
            detection_latency_s=None if event is None else event.detection_latency_s,
            onset_frame_index=None if event is None else event.onset_frame_index,
            detection_frame_index=None if event is None else event.detection_frame_index,
            cooldown_state="active" if self._state == STATE_COOLDOWN else "inactive",
            segment_evidence=observation.segment_measurements,
            quality_status=code,
            config_hash=self.config.config_hash,
            camera_motion_gate="ok" if observation.horizon_y_px is not None else "unavailable",
            contributing_factors={
                "detector_mode": DETECTOR_MODE_ROLLING,
                "baseline_status": BASELINE_STATUS_ROLLING,
                "tide_adjusted": False,
                "tide_data_available": False,
                "baseline_frozen": self._frozen is not None,
                "scale_floored": self._frozen.scale_floored if self._frozen else False,
                "camera_motion": (
                    "detected"
                    if code == "camera_motion"
                    else ("ok" if observation.horizon_y_px is not None else "unavailable")
                ),
                "config_hash": self.config.config_hash,
            },
        )

    def _make_result(
        self,
        observation: RollingObservation,
        timestamp: datetime,
        classification: str,
        state: str,
        reason: str,
        code: str | None,
        scores: dict[str, Any],
        *,
        quality_status: str,
        frozen: bool = False,
        event: DrawdownEvent | None = None,
        cooldown: bool = False,
    ) -> RollingRowResult:
        if code is not None:
            self._record_gate(code)
        event_obj = event if event is not None else self._event
        in_event = classification in EVENT_CLASSES
        return RollingRowResult(
            timestamp=timestamp,
            frame_index=observation.frame_index,
            classification=classification,
            state=state,
            reason=reason,
            reason_code=code,
            observed_position_px=(
                float(observation.position_px) if observation.position_px is not None else None
            ),
            rolling_baseline_px=scores.get("baseline"),
            rolling_scale_px=scores.get("scale"),
            scale_floored=bool(scores.get("scale_floored", False)),
            residual_px=scores.get("residual"),
            robust_z=scores.get("z"),
            cumulative_drawdown_px=scores.get("ck"),
            movement_rate_px_s=scores.get("rate"),
            persistence_count=self._candidate_count,
            history_count=int(scores.get("history_count", 0)),
            history_required=int(scores.get("history_required", 0)),
            quality_status=quality_status,
            confidence=observation.confidence,
            baseline_frozen=frozen or self._frozen is not None,
            in_event=in_event,
            event_max_classification=None if event_obj is None else event_obj.max_classification,
            onset_time=None if event_obj is None else event_obj.onset_time,
            detection_time=None if event_obj is None else event_obj.detection_time,
            detection_latency_s=(None if event_obj is None else event_obj.detection_latency_s),
            event_window_seconds=(
                None if event_obj is None else (timestamp - event_obj.onset_time).total_seconds()
            ),
            cooldown_state="active" if (cooldown or state == STATE_COOLDOWN) else "inactive",
            onset_frame_index=None if event_obj is None else event_obj.onset_frame_index,
            detection_frame_index=(None if event_obj is None else event_obj.detection_frame_index),
            segment_evidence=observation.segment_measurements,
            sigma_k_px=scores.get("sigma_k"),
            sigma_k_samples=int(scores.get("sigma_samples", 0)),
            config_hash=self.config.config_hash,
            camera_motion_gate=("ok" if observation.horizon_y_px is not None else "unavailable"),
            contributing_factors={
                "detector_mode": DETECTOR_MODE_ROLLING,
                "baseline_status": BASELINE_STATUS_ROLLING,
                "tide_adjusted": False,
                "tide_data_available": False,
                "rolling_baseline_px": scores.get("baseline"),
                "rolling_scale_px": scores.get("scale"),
                "scale_floored": bool(scores.get("scale_floored", False)),
                "cumulative_drawdown_px": scores.get("ck"),
                "baseline_frozen": frozen or self._frozen is not None,
                "onset_time": (None if event_obj is None else event_obj.onset_time.isoformat()),
                "detection_time": (
                    None if event_obj is None else event_obj.detection_time.isoformat()
                ),
                "detection_latency_s": (
                    None if event_obj is None else event_obj.detection_latency_s
                ),
                "event_window_seconds": (
                    None
                    if event_obj is None
                    else (timestamp - event_obj.onset_time).total_seconds()
                ),
                "cooldown_state": "active" if (cooldown or state == STATE_COOLDOWN) else "inactive",
                "camera_motion": ("ok" if observation.horizon_y_px is not None else "unavailable"),
                "segment_evidence": observation.segment_measurements,
                "segment_agreement": scores.get("segment_evidence"),
                "sigma_k_px": scores.get("sigma_k"),
                "sigma_k_samples": scores.get("sigma_samples", 0),
                "config_hash": self.config.config_hash,
                "units": "pixels; positive shoreline position is seaward",
            },
        )

    def _event_result(
        self,
        observation: RollingObservation,
        timestamp: datetime,
        position: float,
        scores: dict[str, Any],
        event: DrawdownEvent,
        classification: str | None = None,
    ) -> RollingRowResult:
        cls = classification or self._event_classification(timestamp, scores)
        if SEVERITY[cls] > SEVERITY.get(event.max_classification, 0):
            event.max_classification = cls
        self._state = STATE_EVENT
        return self._make_result(
            observation,
            timestamp,
            cls,
            STATE_EVENT,
            f"inside drawdown event; onset {event.onset_time.isoformat()}, "
            f"detection {event.detection_time.isoformat()}, "
            f"{cls} (onset reclassified from {CLASS_POSSIBLE_DRAWDOWN})",
            None,
            scores,
            quality_status="event",
            frozen=bool(self.config.freeze_baseline),
            event=event,
        )


def evaluate_rolling_baseline(
    observations: list[RollingObservation],
    config: RollingBaselineConfig | None = None,
) -> RollingEvaluation:
    """Batch wrapper around :class:`RollingBaselineDetector`.

    Rows are sorted by timestamp. Exact duplicate timestamps are dropped and
    counted (they cannot be placed causally and would double-count a frame).
    The detector is the same object used for live streaming; no second math.
    """
    cfg = config or RollingBaselineConfig()
    cfg.validate()
    ordered = sorted(
        observations,
        key=lambda item: (
            item.timestamp is None or item.timestamp.tzinfo is None,
            _utc_key(item.timestamp),
        ),
    )
    detector = RollingBaselineDetector(cfg)
    rows: list[RollingRowResult] = []
    dropped = 0
    seen: set[datetime] = set()
    for observation in ordered:
        if observation.timestamp is not None and observation.timestamp.tzinfo is not None:
            if observation.timestamp in seen:
                dropped += 1
                continue
            seen.add(observation.timestamp)
        rows.append(detector.update(observation))

    events = list(detector.finished_events) + (
        [detector.current_event] if detector.current_event is not None else []
    )
    return RollingEvaluation(
        rows=rows,
        events=events,
        dropped_duplicate_count=dropped,
        gate_counts=dict(sorted(detector.gate_counts.items())),
        config=cfg,
        config_hash=cfg.config_hash,
        nominal_interval_seconds=detector.nominal_interval_seconds,
        input_count=len(observations),
    )
