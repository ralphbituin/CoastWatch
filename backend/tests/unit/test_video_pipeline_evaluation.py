"""Offline prototype evaluation must never invent a tide baseline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.services.video.pipeline import PrototypeConfig, evaluate_observations

START = datetime(2026, 11, 19, 21, 30, tzinfo=UTC)


def _row(index: int, position: float | None, *, tide: bool = False) -> dict:
    return {
        "frame_index": index,
        "elapsed_seconds": 30.0 * index,
        "capture_timestamp": (START + timedelta(seconds=30 * index)).isoformat(),
        "shoreline_available": position is not None,
        "signed_position_px": position,
        "confidence": 0.9,
        "extraction_reason": None if position is not None else "no Ocean/Beach contact",
        "reason_code": None if position is not None else "no contact",
        "tide_data_available": tide,
        "tide_height_m": 1.0 if tide else None,
        "tide_source": "TEST" if tide else "unavailable",
        "tide_reason": None if tide else "no verified Thai tide source",
    }


def test_no_tide_defaults_to_rolling_and_never_normal() -> None:
    rows = [_row(index, 800.0 + index) for index in range(8)]
    baseline, detections, config, thresholds = evaluate_observations(
        rows, PrototypeConfig(persistence=3)
    )
    assert baseline.available is False
    assert "no verified" in (baseline.reason or "")
    assert all(detection.classification != "NORMAL" for detection in detections)
    assert all(detection.resolved_detector_mode == "rolling_baseline" for detection in detections)
    assert all(detection.baseline_status == "rolling_unverified" for detection in detections)
    assert all(detection.tide_adjusted is False for detection in detections)
    assert config.movement_threshold_px_s is not None
    assert thresholds["sample_count"] == 7


def test_legacy_movement_only_mode_remains_available() -> None:
    rows = [_row(index, 800.0 + index) for index in range(8)]
    _, detections, _, _ = evaluate_observations(
        rows, PrototypeConfig(persistence=3, detector_mode="movement_only")
    )
    assert all(detection.resolved_detector_mode == "movement_only" for detection in detections)
    assert all(detection.baseline_status == "unavailable" for detection in detections)


def test_missing_shoreline_is_unreliable_and_breaks_movement_chain() -> None:
    rows = [
        _row(0, 800.0),
        _row(1, 800.0),
        _row(2, None),
        _row(3, 800.0),
    ]
    _, detections, _, _ = evaluate_observations(
        rows, PrototypeConfig(persistence=3, detector_mode="movement_only")
    )
    assert detections[2].classification == "DATA_UNRELIABLE"
    assert detections[2].observed_position_px is None
    assert detections[3].movement_rate_px_s is None


def test_tide_rows_without_a_fit_still_do_not_produce_normal() -> None:
    rows = [_row(index, 800.0 + index, tide=True) for index in range(8)]
    baseline, detections, _, _ = evaluate_observations(
        rows, PrototypeConfig(persistence=3, min_baseline_samples=30)
    )
    assert baseline.available is False
    assert "insufficient paired" in (baseline.reason or "")
    assert all(detection.classification != "NORMAL" for detection in detections)
