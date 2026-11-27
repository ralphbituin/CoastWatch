"""Tide-independent rolling-median drawdown detector contract tests."""

from __future__ import annotations

import csv
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db
from app.main import app
from app.services.anomaly.rolling_baseline import (
    CLASS_DATA_UNRELIABLE,
    CLASS_EXPLORATORY_ROLLING,
    CLASS_PENDING,
    CLASS_POSSIBLE_DRAWDOWN,
    RollingBaselineConfig,
    RollingBaselineDetector,
    RollingObservation,
    evaluate_rolling_baseline,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
STEP = 30.0
BASE = 1000.0


def obs(
    index: int,
    position: float | None,
    *,
    confidence: float | None = 0.95,
    timestamp_confidence: float | None = 1.0,
    available: bool = True,
    horizon: float | None = None,
    timestamp: datetime | None = None,
    segments: list[dict] | None = None,
) -> RollingObservation:
    return RollingObservation(
        timestamp=timestamp or (T0 + timedelta(seconds=STEP * index)),
        position_px=position,
        shoreline_available=available,
        confidence=confidence,
        frame_index=index,
        timestamp_confidence=timestamp_confidence,
        horizon_y_px=horizon,
        segment_measurements=segments,
    )


def cfg(**overrides) -> RollingBaselineConfig:
    values = {
        "window_seconds": 240.0,
        "k_seconds": 60.0,
        "min_history_fraction": 0.75,
        "z_threshold": 3.5,
        "sigma_k_multiplier": 5.0,
        "drawdown_floor_frac": 150.0 / 1080.0,
        "scale_floor_frac": 0.004,
        "persistence": 3,
        "max_gap_seconds": 45.0,
        "exit_z": 2.0,
        "exit_count": 2,
        "max_hold_seconds": 3600.0,
        "cooldown_seconds": 0.0,
        "persistent_seconds": 600.0,
        "confidence_floor": 0.80,
        "stale_run_length": 10,
        "frame_height": 1080.0,
        "nominal_interval_seconds": STEP,
    }
    values.update(overrides)
    return RollingBaselineConfig(**values)


def noisy(count: int, *, value: float = BASE, amplitude: float = 1.0) -> list[float]:
    return [value + (index % 2) * amplitude for index in range(count)]


def with_jumps(count: int) -> list[float]:
    """Noisy baseline followed by three consecutive +200 px seaward jumps."""
    values = noisy(count)
    last = values[-1]
    values.extend([last + 200.0, last + 400.0, last + 600.0])
    return values


def series(values: list[float], **kwargs) -> list[RollingObservation]:
    return [obs(index, value, **kwargs) for index, value in enumerate(values)]


def classifications(evaluation) -> list[str]:
    return [row.classification for row in evaluation.rows]


# 1. Causality ---------------------------------------------------------------
def test_changing_future_samples_never_changes_earlier_outputs() -> None:
    values = noisy(60)
    for index, value in enumerate(values):
        values[index] = value + (index % 5) * 0.4
    changed = list(values)
    for index in range(40, 60):
        changed[index] = values[index] + 300.0
    first = evaluate_rolling_baseline(series(values), cfg())
    second = evaluate_rolling_baseline(series(changed), cfg())
    for left, right in zip(first.rows[:40], second.rows[:40], strict=True):
        assert left.classification == right.classification
        assert left.robust_z == right.robust_z
        assert left.rolling_baseline_px == right.rolling_baseline_px


# 2. Window excludes the current sample -------------------------------------
def test_window_excludes_current_sample() -> None:
    values = noisy(20)
    values[10] = BASE + 500.0
    evaluation = evaluate_rolling_baseline(series(values), cfg())
    row = evaluation.rows[10]
    # 240 s at 30 s intervals: timestamps >= t_i - 240 are indices 2..9.
    history = values[2:10]
    expected = sum(sorted(history)[3:5]) / 2.0
    assert row.rolling_baseline_px == pytest.approx(expected)
    assert row.observed_position_px == values[10]


# 3. Zero MAD -> floor, no infinity, jump detected ---------------------------
def test_zero_mad_uses_scale_floor_and_still_detects_a_jump() -> None:
    values = [BASE] * 6 + [BASE + 200.0]
    evaluation = evaluate_rolling_baseline(series(values), cfg())
    row = evaluation.rows[6]
    assert row.rolling_baseline_px == pytest.approx(BASE)
    assert row.rolling_scale_px == pytest.approx(0.004 * 1080.0)
    assert row.scale_floored is True
    assert row.robust_z is not None
    assert math.isfinite(row.robust_z)
    assert row.classification == CLASS_PENDING


# 4. Stale feed ---------------------------------------------------------------
def test_stale_feed_is_unreliable() -> None:
    evaluation = evaluate_rolling_baseline(series([BASE] * 12), cfg())
    assert evaluation.rows[9].classification == CLASS_DATA_UNRELIABLE
    assert evaluation.rows[9].reason_code == "stale_feed"


# 5. One-frame jump does not create an event --------------------------------
def test_single_frame_jump_does_not_create_an_event() -> None:
    values = noisy(30)
    values[20] = values[20] + 200.0
    evaluation = evaluate_rolling_baseline(series(values), cfg())
    assert evaluation.rows[20].classification == CLASS_PENDING
    assert evaluation.events == []


# 6. Three consecutive jumps create POSSIBLE_DRAWDOWN -----------------------
def test_three_consecutive_jumps_create_possible_drawdown() -> None:
    values = with_jumps(20)
    evaluation = evaluate_rolling_baseline(series(values), cfg())
    assert evaluation.rows[20].classification == CLASS_PENDING
    assert evaluation.rows[22].classification == CLASS_POSSIBLE_DRAWDOWN
    assert evaluation.rows[22].state == "EVENT"
    event = evaluation.events[0]
    assert event.onset_frame_index == 20
    assert event.detection_frame_index == 22
    assert event.detection_latency_s == pytest.approx(60.0)
    assert event.max_classification == CLASS_POSSIBLE_DRAWDOWN


# 7. Failed candidate returns pending samples to history ---------------------
def test_failed_candidate_unfreezes_and_returns_samples_to_history() -> None:
    values = noisy(24)
    values[20] = values[20] + 200.0
    values[21] = values[20]
    values[22] = values[20]  # held for two samples, so Ck drops back to ~0
    evaluation = evaluate_rolling_baseline(series(values), cfg())
    assert evaluation.events == []
    assert evaluation.rows[22].classification == CLASS_EXPLORATORY_ROLLING
    assert evaluation.rows[22].reason_code == "candidate_failed"
    detector = RollingBaselineDetector(cfg())
    rows = [detector.update(item) for item in series(values)]
    state = detector.export_state()
    assert any(item["position_px"] == pytest.approx(values[20]) for item in state["history"])
    assert rows[22].classification == CLASS_EXPLORATORY_ROLLING


# 8. Invalid samples break persistence --------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        {"available": False},
        {"confidence": 0.10},
        {"timestamp_confidence": 0.10},
    ],
)
def test_invalid_sample_breaks_persistence(bad: dict) -> None:
    values = with_jumps(16)
    observations = series(values)
    observations[17] = obs(17, values[17], **bad)
    evaluation = evaluate_rolling_baseline(observations, cfg())
    assert evaluation.rows[17].classification == CLASS_DATA_UNRELIABLE
    assert evaluation.events == []


def test_gap_breaks_persistence() -> None:
    values = with_jumps(16)
    observations = series(values)
    observations[17] = obs(17, values[17], timestamp=T0 + timedelta(seconds=STEP * 18))
    evaluation = evaluate_rolling_baseline(observations, cfg())
    assert evaluation.events == []


# 9. Freeze, exit, max hold --------------------------------------------------
def test_baseline_stays_frozen_and_event_exits() -> None:
    values = with_jumps(20)
    values.extend([BASE + (index % 2) for index in range(12)])
    evaluation = evaluate_rolling_baseline(series(values), cfg())
    event_rows = [row for row in evaluation.rows if row.in_event]
    assert len(event_rows) >= 1
    baselines = {row.rolling_baseline_px for row in event_rows}
    assert len(baselines) == 1
    assert evaluation.rows[22].baseline_frozen is True
    event = evaluation.events[0]
    assert event.end_reason == "event_exit"
    assert event.end_time is not None
    assert evaluation.rows[-1].classification == CLASS_EXPLORATORY_ROLLING


def test_max_hold_expiry_releases_freeze() -> None:
    values = with_jumps(20)
    values.extend([BASE + 600.0 + (index % 2) for index in range(6)])
    evaluation = evaluate_rolling_baseline(series(values), cfg(max_hold_seconds=60.0))
    expiry = next(row for row in evaluation.rows if row.reason_code == "EVENT_EXPIRED")
    assert expiry.quality_status == "event_expired"
    assert evaluation.events[0].end_reason == "EVENT_EXPIRED"


# 10. Quarantine and post-event insufficient history ------------------------
def test_event_samples_are_quarantined_and_post_event_rows_recover() -> None:
    values = noisy(10)
    values.extend([BASE + 200.0, BASE + 400.0, BASE + 600.0])
    values.extend([BASE + 600.0 + (index % 2) for index in range(17)])
    values.extend([BASE, BASE, BASE, BASE, BASE, BASE])
    evaluation = evaluate_rolling_baseline(series(values), cfg())
    assert evaluation.events
    post = [
        row
        for row in evaluation.rows
        if row.timestamp is not None
        and row.timestamp >= evaluation.events[0].end_time
        and row.classification == CLASS_DATA_UNRELIABLE
    ]
    if post:
        assert post[0].reason_code in {
            "insufficient_history_after_event",
            "stale_feed",
        }
    # Event samples are never history once the event is running.
    detector = RollingBaselineDetector(cfg())
    for item in series(values):
        detector.update(item)
    state = detector.export_state()
    assert all(abs(item["position_px"] - BASE) < 10.0 for item in state["history"])


# 11. Cooldown prevents immediate duplicate events --------------------------
def test_cooldown_prevents_immediate_duplicate_event() -> None:
    values = with_jumps(20)
    values.extend([BASE, BASE, BASE, BASE])
    values.extend([BASE + 200.0, BASE + 400.0, BASE + 600.0])
    evaluation = evaluate_rolling_baseline(series(values), cfg(cooldown_seconds=600.0))
    assert len(evaluation.events) == 1
    assert any(row.cooldown_state == "active" for row in evaluation.rows)


# 12. Never NORMAL, never tide-adjusted -------------------------------------
def test_rolling_mode_never_emits_normal_or_tide_adjustment() -> None:
    evaluation = evaluate_rolling_baseline(series(noisy(60)), cfg())
    assert "NORMAL" not in classifications(evaluation)
    assert all(row.tide_adjusted is False for row in evaluation.rows)
    assert all(row.tide_data_available is False for row in evaluation.rows)
    assert all(row.tide_source == "unavailable" for row in evaluation.rows)


# 13. Ck sign ----------------------------------------------------------------
def test_cumulative_drawdown_is_positive_seaward() -> None:
    values = with_jumps(20)
    evaluation = evaluate_rolling_baseline(series(values), cfg())
    assert evaluation.events
    assert evaluation.rows[22].cumulative_drawdown_px is not None
    assert evaluation.rows[22].cumulative_drawdown_px > 0
    # Landward residual is diagnostic only and does not create an event.
    landward = noisy(30)
    landward[20] = landward[20] - 400.0
    result = evaluate_rolling_baseline(series(landward), cfg())
    assert result.events == []


# 14. Offset invariance ------------------------------------------------------
def test_constant_offset_does_not_change_classifications() -> None:
    values = with_jumps(20)
    first = evaluate_rolling_baseline(series(values), cfg())
    second = evaluate_rolling_baseline(series([value + 5000.0 for value in values]), cfg())
    assert classifications(first) == classifications(second)


# 15. Segment agreement (optional) ------------------------------------------
def _segments(*values: float) -> list[dict]:
    return [
        {"index": index, "signed_median": value, "point_count": 100}
        for index, value in enumerate(values)
    ]


def test_require_segment_agreement_accepts_agreeing_series() -> None:
    values = noisy(20)
    observations = [
        obs(index, value, segments=_segments(value, value + 1.0))
        for index, value in enumerate(values)
    ]
    observations.extend(
        obs(
            20 + offset,
            values[-1] + 200.0 * (offset + 1),
            segments=_segments(
                values[-1] + 200.0 * (offset + 1),
                values[-1] + 200.0 * (offset + 1) + 1.0,
            ),
        )
        for offset in range(3)
    )
    evaluation = evaluate_rolling_baseline(observations, cfg(require_segment_agreement=True))
    assert evaluation.rows[22].classification == CLASS_POSSIBLE_DRAWDOWN
    assert evaluation.events


def test_require_segment_agreement_rejects_disagreeing_branch() -> None:
    values = noisy(20)
    observations = [
        obs(index, value, segments=_segments(value, value + 1.0))
        for index, value in enumerate(values)
    ]
    # The primary position jumps, but the substantial segment medians do not.
    observations.extend(
        obs(
            20 + offset,
            values[-1] + 200.0 * (offset + 1),
            segments=_segments(values[-1], values[-1] + 1.0),
        )
        for offset in range(3)
    )
    evaluation = evaluate_rolling_baseline(observations, cfg(require_segment_agreement=True))
    assert evaluation.events == []
    assert evaluation.rows[22].classification in {
        CLASS_PENDING,
        CLASS_EXPLORATORY_ROLLING,
    }


# 16. Camera motion gate -----------------------------------------------------
def test_camera_motion_gate_resets_history() -> None:
    observations = [obs(index, BASE + (index % 2), horizon=100.0) for index in range(15)]
    observations[10] = obs(10, BASE, horizon=160.0)
    evaluation = evaluate_rolling_baseline(observations, cfg(camera_motion_shift_px=20.0))
    assert evaluation.rows[10].classification == CLASS_DATA_UNRELIABLE
    assert evaluation.rows[10].reason_code == "camera_motion"
    assert evaluation.rows[11].reason_code in {
        "insufficient_history",
        "insufficient_history_after_event",
    }
    assert evaluation.rows[0].camera_motion_gate == "ok"
    without_horizon = evaluate_rolling_baseline([obs(0, BASE)], cfg())
    assert without_horizon.rows[0].camera_motion_gate == "unavailable"
    with_horizon = evaluate_rolling_baseline(observations[:5], cfg())
    assert with_horizon.rows[-1].camera_motion_gate == "ok"


# 16b. Frame-height floor and resolution changes ------------------------------
def test_540px_frame_gets_75px_drawdown_floor() -> None:
    config = cfg(frame_height=540.0)
    assert config.drawdown_floor_px == pytest.approx(75.0)
    assert config.drawdown_floor_px == pytest.approx(0.1388888888888889 * 540.0)
    assert config.scale_floor_px == pytest.approx(0.004 * 540.0)


def test_resolution_change_resets_and_marks_next_row() -> None:
    detector = RollingBaselineDetector(cfg(frame_height=540.0))
    warm = [detector.update(item) for item in series(noisy(20))]
    assert warm[-1].classification == CLASS_EXPLORATORY_ROLLING

    detector.reset_for_resolution_change(T0 + timedelta(seconds=STEP * 20))
    row = detector.update(obs(20, BASE + 200.0))
    assert row.classification == CLASS_DATA_UNRELIABLE
    assert row.reason_code == "resolution_change"
    assert row.reason is not None and row.reason.startswith("DATA_UNRELIABLE")
    # The reset sample is not evidence: the next sample rebuilds history.
    assert detector.update(obs(21, BASE + 200.0)).classification == CLASS_DATA_UNRELIABLE
    assert detector.update(obs(22, BASE + 200.0)).reason_code == "insufficient_history"


# 16. Streaming == batch, warm start ----------------------------------------
def test_streaming_and_batch_are_identical() -> None:
    values = noisy(50)
    values[25] = values[25] + 200.0
    values[26] = values[26] + 400.0
    values[27] = values[27] + 600.0
    observations = series(values)
    batch = evaluate_rolling_baseline(observations, cfg())
    detector = RollingBaselineDetector(cfg())
    streamed = [detector.update(item) for item in observations]
    assert classifications(batch) == [row.classification for row in streamed]
    assert [row.robust_z for row in batch.rows] == [row.robust_z for row in streamed]


def test_warm_start_reproduces_uninterrupted_result() -> None:
    values = noisy(50)
    values[25] = values[25] + 200.0
    values[26] = values[26] + 400.0
    values[27] = values[27] + 600.0
    observations = series(values)
    uninterrupted = RollingBaselineDetector(cfg())
    expected = [uninterrupted.update(item) for item in observations]

    warm = RollingBaselineDetector(cfg())
    warm.warm_start(observations[:25])
    actual = [warm.update(item) for item in observations[25:]]
    assert [row.classification for row in actual] == [row.classification for row in expected[25:]]
    assert [row.robust_z for row in actual] == [row.robust_z for row in expected[25:]]


def test_export_import_state_resumes_identically() -> None:
    values = noisy(40)
    first = RollingBaselineDetector(cfg())
    warm_rows = [first.update(item) for item in series(values)]
    state = first.export_state()
    restored = RollingBaselineDetector.from_state(state, cfg())
    observations = series(values)
    for index in range(len(observations), len(observations) + 10):
        observations.append(obs(index, BASE + (index % 2)))
    expected = [first.update(item) for item in observations[len(values) :]]
    actual = [restored.update(item) for item in observations[len(values) :]]
    assert [row.classification for row in actual] == [row.classification for row in expected]
    assert warm_rows


# 18. Config validation ------------------------------------------------------
@pytest.mark.parametrize(
    "overrides",
    [
        {"window_seconds": 100.0, "k_seconds": 90.0},
        {"persistence": 1},
        {"z_threshold": 0.0},
        {"min_history_fraction": 0.0},
        {"confidence_floor": 1.5},
    ],
)
def test_invalid_config_raises(overrides: dict) -> None:
    with pytest.raises(ValueError):
        RollingBaselineConfig(**overrides).validate()


def test_api_rejects_invalid_rolling_config_with_422() -> None:
    app.dependency_overrides[get_db] = lambda: None
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/videos/process",
                json={
                    "recording_id": "does-not-matter",
                    "detector_mode": "rolling_baseline",
                    "rolling": {"persistence": 1},
                },
            )
    finally:
        app.dependency_overrides.pop(get_db, None)
    assert response.status_code == 422


# 19. Tide-drift bias --------------------------------------------------------
def test_linear_drift_without_anomaly_does_not_create_events() -> None:
    base = [BASE + (index % 3) * 6.0 for index in range(80)]
    positive = [value + 0.5 * index for index, value in enumerate(base)]  # 30 px/min seaward
    negative = [value - 0.5 * index for index, value in enumerate(base)]  # 30 px/min landward
    up = evaluate_rolling_baseline(series(positive), cfg())
    down = evaluate_rolling_baseline(series(negative), cfg())
    up_z = [row.robust_z for row in up.rows if row.robust_z is not None]
    down_z = [row.robust_z for row in down.rows if row.robust_z is not None]
    assert up.events == []
    assert down.events == []
    assert statistics_median(up_z) > statistics_median(down_z)


def statistics_median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


# Actual-footage regression --------------------------------------------------
def test_saved_517_observations_regression() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "ml"
        / "reports"
        / "phase5_tide_anomaly"
        / "observations.csv"
    )
    if not path.is_file():
        pytest.skip("saved observations.csv is not present")
    observations: list[RollingObservation] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            observations.append(
                RollingObservation(
                    timestamp=datetime.fromisoformat(row["capture_timestamp"]),
                    position_px=(
                        None if not row["signed_position_px"] else float(row["signed_position_px"])
                    ),
                    shoreline_available=row["shoreline_available"].lower() == "true",
                    confidence=(None if not row["confidence"] else float(row["confidence"])),
                    frame_index=int(row["frame_index"]),
                    timestamp_confidence=(
                        None
                        if not row["timestamp_confidence"]
                        else float(row["timestamp_confidence"])
                    ),
                )
            )
    evaluation = evaluate_rolling_baseline(
        observations,
        RollingBaselineConfig(frame_height=1080.0, nominal_interval_seconds=30.0),
    )
    assert len(evaluation.rows) == 517
    assert evaluation.dropped_duplicate_count == 0
    assert evaluation.events == []
    assert evaluation.gate_counts == {"insufficient_history": 30}
    zs = [(row.robust_z, row.timestamp) for row in evaluation.rows if row.robust_z is not None]
    strongest = max(zs, key=lambda item: item[0])
    assert strongest[0] == pytest.approx(3.12, abs=0.05)
    assert strongest[1].isoformat().startswith("2026-11-19T22:23:20")
    assert all(row.classification != "NORMAL" for row in evaluation.rows)
    assert all(row.tide_adjusted is False for row in evaluation.rows)
    assert all(row.tide_data_available is False for row in evaluation.rows)
