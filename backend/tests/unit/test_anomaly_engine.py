"""Transparent anomaly engine: persistence, missing data, zero MAD, movement sign."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.anomaly.detector import (
    AnomalyDetector,
    AnomalyObservation,
    DetectorConfig,
)
from app.services.anomaly.scoring import robust_scale, robust_z
from app.services.baseline.expected_shoreline import BaselineFitResult

START = datetime(2026, 11, 19, 21, 30, tzinfo=UTC)


def _observation(
    index: int,
    position: float | None,
    *,
    tide: float = 0.0,
    tide_available: bool = True,
    available: bool = True,
    reason: str | None = None,
) -> AnomalyObservation:
    return AnomalyObservation(
        elapsed_seconds=30.0 * index,
        frame_index=index,
        timestamp=START + timedelta(seconds=30 * index),
        position_px=position,
        shoreline_available=available,
        reason=reason,
        confidence=0.9,
        tide_height_m=tide,
        tide_available=tide_available,
        tide_source="TEST_CSV" if tide_available else "unavailable",
    )


def _baseline(**overrides) -> BaselineFitResult:
    values = {
        "available": True,
        "status": "provisional",
        "intercept_px": 0.0,
        "slope_px_per_m": 1.0,
        "residual_median_px": 0.0,
        "residual_scale_px": 1.0,
        "residual_scale_reliable": True,
        "sample_count": 40,
    }
    values.update(overrides)
    return BaselineFitResult(**values)


def test_zero_mad_is_not_scored() -> None:
    scale = robust_scale([5.0, 5.0, 5.0])
    assert scale.mad == 0.0
    assert scale.reliable is False
    assert robust_z(6.0, scale.median, scale.mad) is None


def test_single_deviation_does_not_create_an_anomaly() -> None:
    detector = AnomalyDetector(DetectorConfig(persistence=3, z_threshold=3.5))
    observations = [
        _observation(0, 0.0),
        _observation(1, 0.0),
        _observation(2, 8.0),
        _observation(3, 0.0),
    ]
    detections = detector.evaluate_series(observations, _baseline())
    assert detections[2].classification == "PENDING_PERSISTENCE"
    assert detections[2].persistence_count == 1
    assert detections[3].classification == "NORMAL"


def test_three_consecutive_deviations_create_unusual_position() -> None:
    detector = AnomalyDetector(DetectorConfig(persistence=3, z_threshold=3.5))
    observations = [
        _observation(0, 0.0),
        _observation(1, 0.0),
        _observation(2, 8.0),
        _observation(3, 8.0),
        _observation(4, 8.0),
    ]
    detections = detector.evaluate_series(observations, _baseline())
    assert [item.classification for item in detections[2:]] == [
        "PENDING_PERSISTENCE",
        "PENDING_PERSISTENCE",
        "UNUSUAL_SHORELINE_POSITION",
    ]
    assert detections[4].persistence_count == 3
    assert detections[4].state == "ANOMALY_DETECTED"
    assert detections[4].tide_adjusted is True


def test_missing_observation_is_unreliable_and_breaks_the_chain() -> None:
    detector = AnomalyDetector(DetectorConfig(persistence=3, z_threshold=3.5))
    observations = [
        _observation(0, 0.0),
        _observation(1, 8.0),
        _observation(2, None, available=False, reason="no Ocean/Beach contact"),
        _observation(3, 8.0),
        _observation(4, 8.0),
        _observation(5, 8.0),
    ]
    detections = detector.evaluate_series(observations, _baseline())
    assert detections[2].classification == "DATA_UNRELIABLE"
    assert detections[2].state == "DATA_UNRELIABLE"
    assert detections[2].observed_position_px is None
    # The chain restarted after the gap: three new deviations are required.
    assert detections[3].classification == "PENDING_PERSISTENCE"
    assert detections[5].classification == "UNUSUAL_SHORELINE_POSITION"


def test_missing_tide_with_baseline_is_never_normal() -> None:
    detector = AnomalyDetector(DetectorConfig(persistence=3, z_threshold=3.5))
    observations = [
        _observation(0, 0.0, tide_available=False),
        _observation(1, 0.0, tide_available=False),
    ]
    detections = detector.evaluate_series(observations, _baseline())
    assert all(item.classification == "DATA_UNRELIABLE" for item in detections)
    assert all(item.tide_adjusted is False for item in detections)


def test_unusable_baseline_scale_is_unreliable_not_normal() -> None:
    detector = AnomalyDetector(DetectorConfig(persistence=3, z_threshold=3.5))
    detections = detector.evaluate_series(
        [_observation(0, 10.0)], _baseline(residual_scale_px=None)
    )
    assert detections[0].classification == "DATA_UNRELIABLE"
    assert "residual scale" in detections[0].reason


def test_movement_only_mode_never_emits_normal() -> None:
    detector = AnomalyDetector(
        DetectorConfig(
            persistence=3,
            movement_threshold_px_s=1.0,
            drawdown_rate_threshold_px_s=0.5,
            drawdown_distance_px=1000.0,
        )
    )
    observations = [_observation(i, position) for i, position in enumerate([0.0, 1.0, 2.0, 3.0])]
    detections = detector.evaluate_series(observations, baseline=None)
    assert all(item.classification != "NORMAL" for item in detections)
    assert all(item.baseline_status == "unavailable" for item in detections)
    assert detections[0].classification == "EXPLORATORY_MOVEMENT_ONLY"


def test_rapid_seaward_retreat_requires_positive_rate_and_persistence() -> None:
    detector = AnomalyDetector(
        DetectorConfig(
            persistence=3,
            movement_threshold_px_s=1.0,
            drawdown_rate_threshold_px_s=0.5,
            drawdown_distance_px=1000.0,
        )
    )
    # +60 px per 30 s = +2 px/s: positive is seaward.
    positions = [0.0, 60.0, 120.0, 180.0]
    detections = detector.evaluate_series(
        [_observation(i, position) for i, position in enumerate(positions)],
        baseline=None,
    )
    assert detections[1].classification == "PENDING_PERSISTENCE"
    assert detections[3].classification == "RAPID_SHORELINE_RETREAT"
    assert detections[3].movement_rate_px_s == pytest.approx(2.0)
    assert detections[3].state == "ANOMALY_DETECTED"
    assert "seaward" in detections[3].reason


def test_landward_movement_does_not_trigger_retreat() -> None:
    detector = AnomalyDetector(
        DetectorConfig(
            persistence=3,
            movement_threshold_px_s=1.0,
            drawdown_rate_threshold_px_s=0.5,
            drawdown_distance_px=1000.0,
        )
    )
    positions = [180.0, 120.0, 60.0, 0.0]
    detections = detector.evaluate_series(
        [_observation(i, position) for i, position in enumerate(positions)],
        baseline=None,
    )
    assert all(item.classification != "RAPID_SHORELINE_RETREAT" for item in detections)
