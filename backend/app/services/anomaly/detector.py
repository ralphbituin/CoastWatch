"""Transparent anomaly detection and system-state machine (Phase 6 prototype).

Two operating modes are explicit:

``tide_conditioned``
    A provisional tide-conditioned baseline exists and the current observation
    has a verified, geographically appropriate tide height. Position residuals
    are scored with a robust Z. ``NORMAL`` is only possible in this mode.

``movement_only``
    No verified tide data/baseline exists. Only shoreline movement rate and
    persistence are evaluated. Valid observations are labelled
    ``EXPLORATORY_MOVEMENT_ONLY`` — never ``NORMAL`` — because without a
    baseline "normal" cannot be established.

Persistence is required: one frame cannot create an anomaly. Missing
observations break the chain and are reported as ``DATA_UNRELIABLE``; they never
count as normal or as evidence.

Sign convention: position is **seaward-positive**. A positive movement rate
means the waterline moved seaward (drawdown/retreat of the water edge); a
negative rate means landward movement. ``RAPID_SHORELINE_RETREAT`` therefore
requires a positive rate.

This is a monitoring prototype, not a validated tsunami detector.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from app.services.anomaly.scoring import compute_score, robust_z

CLASS_NORMAL = "NORMAL"
CLASS_UNUSUAL = "UNUSUAL_SHORELINE_POSITION"
CLASS_RAPID_RETREAT = "RAPID_SHORELINE_RETREAT"
CLASS_PERSISTENT_DRAWDOWN = "PERSISTENT_DRAWDOWN"
CLASS_DATA_UNRELIABLE = "DATA_UNRELIABLE"
CLASS_EXPLORATORY = "EXPLORATORY_MOVEMENT_ONLY"
CLASS_PENDING = "PENDING_PERSISTENCE"
# Tide-independent rolling-median modes (Phase 6 rev 2).
CLASS_EXPLORATORY_ROLLING = "EXPLORATORY_ROLLING_BASELINE"
CLASS_POSSIBLE_DRAWDOWN = "POSSIBLE_DRAWDOWN"

ANOMALY_CLASSES = {
    CLASS_UNUSUAL,
    CLASS_RAPID_RETREAT,
    CLASS_PERSISTENT_DRAWDOWN,
    CLASS_POSSIBLE_DRAWDOWN,
}

STATE_BY_CLASS = {
    CLASS_NORMAL: "NORMAL",
    CLASS_UNUSUAL: "ANOMALY_DETECTED",
    CLASS_RAPID_RETREAT: "ANOMALY_DETECTED",
    CLASS_PERSISTENT_DRAWDOWN: "ANOMALY_DETECTED",
    CLASS_POSSIBLE_DRAWDOWN: "ANOMALY_DETECTED",
    CLASS_DATA_UNRELIABLE: "DATA_UNRELIABLE",
    CLASS_EXPLORATORY: "OBSERVED",
    CLASS_EXPLORATORY_ROLLING: "OBSERVED",
    CLASS_PENDING: "OBSERVED",
}

# Operational detector modes. ``auto`` is resolved per run.
DETECTOR_MODE_AUTO = "auto"
DETECTOR_MODE_ROLLING = "rolling_baseline"
DETECTOR_MODE_TIDE = "tide_conditioned"
DETECTOR_MODE_MOVEMENT = "movement_only"
DETECTOR_MODES = {
    DETECTOR_MODE_AUTO,
    DETECTOR_MODE_ROLLING,
    DETECTOR_MODE_TIDE,
    DETECTOR_MODE_MOVEMENT,
}


@dataclass(slots=True)
class DetectorConfig:
    persistence: int = 3
    z_threshold: float = 3.5
    # Seaward-positive px/s. ``None`` means movement anomalies are disabled.
    movement_threshold_px_s: float | None = None
    drawdown_rate_threshold_px_s: float | None = None
    drawdown_distance_px: float | None = None
    # Larger gaps break the movement chain (sampling is 30 s by default).
    max_gap_seconds: float = 90.0
    min_confidence: float | None = None


@dataclass(slots=True)
class AnomalyObservation:
    elapsed_seconds: float
    position_px: float | None
    shoreline_available: bool = True
    timestamp: datetime | None = None
    frame_index: int | None = None
    reason: str | None = None
    confidence: float | None = None
    tide_height_m: float | None = None
    tide_available: bool = False
    tide_source: str | None = None
    quality_status: str | None = None


@dataclass(slots=True)
class Detection:
    elapsed_seconds: float
    frame_index: int | None
    timestamp: datetime | None
    classification: str
    state: str
    reason: str
    baseline_status: str
    tide_adjusted: bool
    tide_available: bool
    tide_source: str | None
    observed_position_px: float | None
    expected_position_px: float | None = None
    residual_px: float | None = None
    robust_z: float | None = None
    movement_rate_px_s: float | None = None
    persistence_count: int = 0
    tide_height_m: float | None = None
    quality_status: str | None = None
    confidence: float | None = None
    score: float | None = None
    factors: dict[str, float] = field(default_factory=dict)
    # --- Tide-independent rolling mode (all optional; legacy rows unaffected)
    detector_mode: str | None = None
    resolved_detector_mode: str | None = None
    mode_reason: str | None = None
    rolling_baseline_px: float | None = None
    rolling_scale_px: float | None = None
    scale_floored: bool | None = None
    cumulative_drawdown_px: float | None = None
    baseline_frozen: bool | None = None
    in_event: bool | None = None
    event_max_classification: str | None = None
    onset_time: datetime | None = None
    detection_time: datetime | None = None
    detection_latency_s: float | None = None
    event_window_seconds: float | None = None
    cooldown_state: str | None = None
    onset_frame_index: int | None = None
    detection_frame_index: int | None = None
    segment_evidence: list[dict[str, Any]] | None = None
    config_hash: str | None = None
    contributing_factors: dict[str, Any] | None = None

    @property
    def is_anomaly(self) -> bool:
        return self.classification in ANOMALY_CLASSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "elapsed_seconds": self.elapsed_seconds,
            "timestamp": None if self.timestamp is None else self.timestamp.isoformat(),
            "classification": self.classification,
            "state": self.state,
            "reason": self.reason,
            "baseline_status": self.baseline_status,
            "tide_adjusted": self.tide_adjusted,
            "tide_data_available": self.tide_available,
            "tide_source": self.tide_source,
            "observed_position_px": self.observed_position_px,
            "expected_position_px": self.expected_position_px,
            "residual_px": self.residual_px,
            "robust_z": self.robust_z,
            "movement_rate_px_s": self.movement_rate_px_s,
            "persistence_count": self.persistence_count,
            "tide_height_m": self.tide_height_m,
            "quality_status": self.quality_status,
            "confidence": self.confidence,
            "score": self.score,
            "factors": self.factors,
            "detector_mode": self.detector_mode,
            "resolved_detector_mode": self.resolved_detector_mode,
            "mode_reason": self.mode_reason,
            "rolling_baseline_px": self.rolling_baseline_px,
            "rolling_scale_px": self.rolling_scale_px,
            "scale_floored": self.scale_floored,
            "cumulative_drawdown_px": self.cumulative_drawdown_px,
            "baseline_frozen": self.baseline_frozen,
            "in_event": self.in_event,
            "event_max_classification": self.event_max_classification,
            "onset_time": None if self.onset_time is None else self.onset_time.isoformat(),
            "detection_time": (
                None if self.detection_time is None else self.detection_time.isoformat()
            ),
            "detection_latency_s": self.detection_latency_s,
            "event_window_seconds": self.event_window_seconds,
            "cooldown_state": self.cooldown_state,
            "onset_frame_index": self.onset_frame_index,
            "detection_frame_index": self.detection_frame_index,
            "segment_evidence": self.segment_evidence,
            "config_hash": self.config_hash,
            "contributing_factors": self.contributing_factors,
        }


def is_rapid_retreat(movement_rate: float | None, threshold_px_s: float | None) -> bool:
    """The existing ``RAPID_SHORELINE_RETREAT`` rate predicate.

    Seaward-positive px/s. Extracted verbatim from the historical inline check
    so the rolling detector can import it rather than copy the comparison.
    """
    return bool(
        movement_rate is not None and threshold_px_s is not None and movement_rate >= threshold_px_s
    )


def validate_detector_mode(mode: str | None) -> str:
    value = (mode or DETECTOR_MODE_AUTO).strip().lower()
    if value not in DETECTOR_MODES:
        raise ValueError(f"detector_mode must be one of {sorted(DETECTOR_MODES)}; got {mode!r}")
    return value


def resolve_detector_mode(
    requested: str | None, *, tide_baseline_available: bool
) -> tuple[str, str]:
    """Resolve ``auto`` and explain the decision.

    ``tide_conditioned`` is selected only when a verified active
    tide-conditioned baseline is available. Otherwise ``auto`` resolves to the
    tide-independent rolling baseline. An explicit mode is honoured (the
    legacy tide-conditioned path keeps its existing behaviour when its
    baseline is unavailable, but never emits ``NORMAL`` in that case).
    """
    mode = validate_detector_mode(requested)
    if mode == DETECTOR_MODE_AUTO:
        if tide_baseline_available:
            return (
                DETECTOR_MODE_TIDE,
                "auto: a verified active tide-conditioned baseline is available",
            )
        return (
            DETECTOR_MODE_ROLLING,
            "auto: no verified active tide-conditioned baseline; using the "
            "tide-independent rolling-median drawdown detector",
        )
    if mode == DETECTOR_MODE_ROLLING:
        return (
            DETECTOR_MODE_ROLLING,
            "explicit rolling_baseline: tide-independent by construction; "
            "no tide data is used or required",
        )
    if mode == DETECTOR_MODE_TIDE:
        reason = (
            "explicit tide_conditioned: verified baseline available"
            if tide_baseline_available
            else "explicit tide_conditioned: no verified baseline; the legacy path "
            "reports DATA_UNRELIABLE/EXPLORATORY and never NORMAL"
        )
        return DETECTOR_MODE_TIDE, reason
    return (
        DETECTOR_MODE_MOVEMENT,
        "explicit movement_only (legacy): no baseline is used",
    )


def derive_movement_thresholds(
    observations: Sequence[AnomalyObservation],
    *,
    persistence: int = 3,
    percentile: float = 99.0,
    floor_px_s: float = 0.5,
    max_gap_seconds: float = 90.0,
) -> dict[str, float | int | str]:
    """Derive movement thresholds from the observed rate distribution.

    This is deliberately data-driven and documented: the rapid threshold is the
    given percentile of absolute seaward-positive rates, with a small floor.
    The drawdown rate uses the median plus three robust sigma; the cumulative
    distance threshold scales with the rapid threshold and persistence.
    """
    rates: list[float] = []
    previous: AnomalyObservation | None = None
    for observation in observations:
        if not observation.shoreline_available or observation.position_px is None:
            previous = None
            continue
        if previous is not None and previous.position_px is not None:
            dt = observation.elapsed_seconds - previous.elapsed_seconds
            if 0 < dt <= max_gap_seconds:
                rates.append((float(observation.position_px) - float(previous.position_px)) / dt)
        previous = observation
    if not rates:
        return {
            "rapid_threshold_px_s": floor_px_s,
            "drawdown_rate_threshold_px_s": floor_px_s / 2.0,
            "drawdown_distance_px": 10.0,
            "sample_count": 0,
            "method": "floor only (no valid movement pairs)",
        }
    array = np.asarray(rates, dtype=np.float64)
    absolute = np.abs(array)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    robust_sigma = 1.4826 * mad
    rapid = max(floor_px_s, float(np.percentile(absolute, percentile)))
    drawdown_rate = max(floor_px_s / 2.0, median + 3.0 * robust_sigma)
    # Require a cumulative move over the persistence window; 60% of the rapid
    # rate times the window is a conservative starting point.
    window_seconds = 30.0 * max(1, int(persistence))
    drawdown_distance = max(10.0, rapid * window_seconds * 0.6)
    return {
        "rapid_threshold_px_s": round(rapid, 6),
        "drawdown_rate_threshold_px_s": round(drawdown_rate, 6),
        "drawdown_distance_px": round(drawdown_distance, 3),
        "sample_count": len(rates),
        "median_rate_px_s": round(median, 6),
        "robust_sigma_px_s": round(robust_sigma, 6),
        "percentile": percentile,
        "method": (
            "rapid=max(floor, percentile(abs(rate),p)); "
            "drawdown_rate=max(floor/2, median+3*1.4826*MAD); "
            "distance=max(10, rapid*persistence_window*0.6)"
        ),
    }


def _baseline_fields(
    baseline: Any,
) -> tuple[bool, float | None, float | None, float | None, float | None, str]:
    """Return available, intercept, slope, residual median, scale, status."""
    if baseline is None:
        return False, None, None, None, None, "unavailable"
    if hasattr(baseline, "available"):
        available = bool(baseline.available)
        intercept = getattr(baseline, "intercept_px", None)
        slope = getattr(baseline, "slope_px_per_m", None)
        median = getattr(baseline, "residual_median_px", None)
        scale = getattr(baseline, "residual_scale_px", None)
        status = getattr(baseline, "status", None) or (
            "provisional" if available else "unavailable"
        )
        return available, intercept, slope, median, scale, str(status)
    if isinstance(baseline, dict):
        available = bool(baseline.get("available"))
        return (
            available,
            baseline.get("intercept_px"),
            baseline.get("slope_px_per_m"),
            baseline.get("residual_median_px"),
            baseline.get("residual_scale_px"),
            str(baseline.get("status") or ("provisional" if available else "unavailable")),
        )
    return False, None, None, None, None, "unavailable"


def _evaluate(
    observations: Sequence[AnomalyObservation],
    baseline: Any,
    config: DetectorConfig,
) -> list[Detection]:
    available, intercept, slope, residual_median, residual_scale, baseline_status = (
        _baseline_fields(baseline)
    )
    detections: list[Detection] = []
    previous: AnomalyObservation | None = None
    unusual_run = 0
    rapid_run = 0
    drawdown_run = 0
    drawdown_distance = 0.0

    for observation in observations:
        position = observation.position_px
        movement_rate = None
        if (
            previous is not None
            and position is not None
            and previous.position_px is not None
            and observation.shoreline_available
            and previous.shoreline_available
        ):
            dt = observation.elapsed_seconds - previous.elapsed_seconds
            if 0 < dt <= config.max_gap_seconds:
                movement_rate = (float(position) - float(previous.position_px)) / dt

        # A missing/invalid frame is never normal and never evidence.
        if not observation.shoreline_available or position is None:
            unusual_run = rapid_run = drawdown_run = 0
            drawdown_distance = 0.0
            previous = None
            detections.append(
                Detection(
                    elapsed_seconds=observation.elapsed_seconds,
                    frame_index=observation.frame_index,
                    timestamp=observation.timestamp,
                    classification=CLASS_DATA_UNRELIABLE,
                    state="DATA_UNRELIABLE",
                    reason=observation.reason or "shoreline unavailable",
                    baseline_status=baseline_status,
                    tide_adjusted=False,
                    tide_available=observation.tide_available,
                    tide_source=observation.tide_source,
                    observed_position_px=None,
                    persistence_count=0,
                    tide_height_m=observation.tide_height_m,
                    quality_status=observation.quality_status or "data_unavailable",
                    confidence=observation.confidence,
                )
            )
            continue

        if (
            config.min_confidence is not None
            and observation.confidence is not None
            and observation.confidence < config.min_confidence
        ):
            unusual_run = rapid_run = drawdown_run = 0
            drawdown_distance = 0.0
            previous = None
            detections.append(
                Detection(
                    elapsed_seconds=observation.elapsed_seconds,
                    frame_index=observation.frame_index,
                    timestamp=observation.timestamp,
                    classification=CLASS_DATA_UNRELIABLE,
                    state="DATA_UNRELIABLE",
                    reason=(
                        f"segmentation confidence {observation.confidence:.3f} below "
                        f"{config.min_confidence:.3f}"
                    ),
                    baseline_status=baseline_status,
                    tide_adjusted=False,
                    tide_available=observation.tide_available,
                    tide_source=observation.tide_source,
                    observed_position_px=position,
                    movement_rate_px_s=movement_rate,
                    tide_height_m=observation.tide_height_m,
                    quality_status="low_confidence",
                    confidence=observation.confidence,
                )
            )
            continue

        expected = None
        residual = None
        z_value = None
        baseline_ready = (
            available
            and intercept is not None
            and slope is not None
            and residual_median is not None
            and residual_scale is not None
        )
        if baseline_ready:
            assert intercept is not None
            assert slope is not None
            assert residual_median is not None
            assert residual_scale is not None
            if not observation.tide_available or observation.tide_height_m is None:
                unusual_run = 0
                previous = observation
                detections.append(
                    Detection(
                        elapsed_seconds=observation.elapsed_seconds,
                        frame_index=observation.frame_index,
                        timestamp=observation.timestamp,
                        classification=CLASS_DATA_UNRELIABLE,
                        state="DATA_UNRELIABLE",
                        reason="tide unavailable for a tide-conditioned baseline",
                        baseline_status=baseline_status,
                        tide_adjusted=False,
                        tide_available=False,
                        tide_source=observation.tide_source,
                        observed_position_px=position,
                        movement_rate_px_s=movement_rate,
                        tide_height_m=observation.tide_height_m,
                        quality_status="tide_unavailable",
                        confidence=observation.confidence,
                    )
                )
                continue
            expected = float(intercept) + float(slope) * float(observation.tide_height_m)
            residual = float(position) - expected
            z_value = robust_z(residual, residual_median, residual_scale / 1.4826)

        unusual_candidate = bool(z_value is not None and abs(z_value) >= config.z_threshold)
        unusual_run = unusual_run + 1 if unusual_candidate else 0

        rapid_candidate = is_rapid_retreat(movement_rate, config.movement_threshold_px_s)
        rapid_run = rapid_run + 1 if rapid_candidate else 0

        drawdown_rate_threshold = (
            config.drawdown_rate_threshold_px_s
            if config.drawdown_rate_threshold_px_s is not None
            else (
                None
                if config.movement_threshold_px_s is None
                else config.movement_threshold_px_s * 0.5
            )
        )
        drawdown_candidate = bool(
            movement_rate is not None
            and drawdown_rate_threshold is not None
            and movement_rate > 0
            and movement_rate >= drawdown_rate_threshold
        )
        if drawdown_candidate:
            assert movement_rate is not None
            drawdown_run += 1
            previous_elapsed = (
                previous.elapsed_seconds if previous is not None else observation.elapsed_seconds
            )
            drawdown_distance += max(0.0, float(movement_rate)) * max(
                0.0, observation.elapsed_seconds - previous_elapsed
            )
        else:
            drawdown_run = 0
            drawdown_distance = 0.0

        classification: str
        reason: str
        persistence_count: int
        if rapid_run >= config.persistence:
            classification = CLASS_RAPID_RETREAT
            persistence_count = rapid_run
            reason = (
                f"seaward movement {movement_rate:.4f} px/s >= "
                f"{config.movement_threshold_px_s:.4f} px/s for {rapid_run} "
                "consecutive valid observations"
            )
        elif drawdown_run >= config.persistence and (
            config.drawdown_distance_px is None or drawdown_distance >= config.drawdown_distance_px
        ):
            classification = CLASS_PERSISTENT_DRAWDOWN
            persistence_count = drawdown_run
            reason = (
                f"persistent seaward drawdown over {drawdown_run} valid observations "
                f"({drawdown_distance:.2f} px cumulative)"
            )
        elif unusual_run >= config.persistence:
            classification = CLASS_UNUSUAL
            persistence_count = unusual_run
            reason = (
                f"robust Z {z_value:.2f} >= {config.z_threshold:.2f} for "
                f"{unusual_run} consecutive valid observations"
            )
        elif unusual_candidate or rapid_candidate or drawdown_candidate:
            classification = CLASS_PENDING
            persistence_count = max(unusual_run, rapid_run, drawdown_run)
            reason = (
                f"transient deviation below persistence requirement "
                f"({persistence_count}/{config.persistence})"
            )
        elif baseline_ready:
            classification = CLASS_NORMAL
            persistence_count = 0
            reason = (
                "within provisional tide-conditioned baseline and no persistent " "movement anomaly"
            )
        elif available:
            # A baseline row exists but cannot produce a finite robust score
            # (e.g. zero residual MAD). Never call that NORMAL.
            classification = CLASS_DATA_UNRELIABLE
            persistence_count = 0
            reason = (
                "baseline residual scale is zero or unusable; tide-adjusted "
                "scoring is not available"
            )
        else:
            classification = CLASS_EXPLORATORY
            persistence_count = 0
            reason = (
                "movement-only exploratory mode: no verified tide-conditioned baseline; "
                "this is not a NORMAL classification"
            )

        score = compute_score(
            robust_z_value=z_value,
            movement_rate=movement_rate,
            movement_threshold=config.movement_threshold_px_s,
            persistence=persistence_count,
            required_persistence=config.persistence,
            quality=observation.confidence,
        )
        detections.append(
            Detection(
                elapsed_seconds=observation.elapsed_seconds,
                frame_index=observation.frame_index,
                timestamp=observation.timestamp,
                classification=classification,
                state=STATE_BY_CLASS.get(classification, "OBSERVED"),
                reason=reason,
                baseline_status=baseline_status,
                tide_adjusted=bool(baseline_ready and observation.tide_available),
                tide_available=observation.tide_available,
                tide_source=observation.tide_source,
                observed_position_px=round(float(position), 6),
                expected_position_px=None if expected is None else round(expected, 6),
                residual_px=None if residual is None else round(residual, 6),
                robust_z=None if z_value is None else round(z_value, 6),
                movement_rate_px_s=(
                    None if movement_rate is None else round(float(movement_rate), 6)
                ),
                persistence_count=persistence_count,
                tide_height_m=observation.tide_height_m,
                quality_status=(
                    "ok"
                    if baseline_ready
                    else ("baseline_unusable" if available else "exploratory_no_baseline")
                ),
                confidence=observation.confidence,
                score=score.score,
                factors=score.factors,
            )
        )
        previous = observation

    return detections


def tide_is_usable(observations: Sequence[AnomalyObservation]) -> bool:
    """True when at least one valid observation carries a verified tide height."""
    return any(
        observation.shoreline_available
        and observation.position_px is not None
        and observation.tide_available
        and observation.tide_height_m is not None
        for observation in observations
    )


class AnomalyDetector:
    """Series/streaming-compatible detector facade."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def evaluate_series(
        self,
        observations: Iterable[AnomalyObservation],
        baseline: Any = None,
    ) -> list[Detection]:
        return _evaluate(list(observations), baseline, self.config)

    def evaluate(
        self,
        observations: Iterable[AnomalyObservation] | AnomalyObservation,
        baseline: Any = None,
    ) -> list[Detection]:
        if isinstance(observations, AnomalyObservation):
            observations = [observations]
        return self.evaluate_series(observations, baseline)
