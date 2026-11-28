"""Tests for DEMO MODE: real detector replay, read-only registry, isolation."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.config import REPO_ROOT, settings
from app.demo import service as demo_service
from app.demo.registry import (
    DemoRegistry,
    Recording,
    Sample,
    Segment,
    _build_segments,
    _valid_starts,
)
from app.demo.replay import replay_injection
from app.demo.service import (
    DemoCapacityError,
    DemoValidationError,
    SimulationManager,
    validate_simulation,
)
from app.main import app
from app.services.anomaly.rolling_baseline import RollingBaselineConfig

RUN = REPO_ROOT / "ml" / "reports" / "live_eval" / "kohsamui_4h_20261128T014103Z"
RID = RUN.name


def _client() -> TestClient:
    # No context manager: do not run the application lifespan (no live worker).
    return TestClient(app)


def _copy_recording(tmp_path: Path, *, name: str = "copied_run") -> Path:
    target = tmp_path / name
    (target / "tables").mkdir(parents=True)
    (target / "injection" / "tables").mkdir(parents=True)
    for table in ("observations.csv", "anomalies.csv"):
        shutil.copy2(RUN / "tables" / table, target / "tables" / table)
    shutil.copy2(
        RUN / "injection" / "tables" / "config_used.json",
        target / "injection" / "tables" / "config_used.json",
    )
    shutil.copy2(RUN / "run_state.json", target / "run_state.json")
    return target


def _synthetic_sample(
    index: int,
    moment: datetime,
    position: float | None,
    *,
    session: str = "s1",
    recorded_class: str = "EXPLORATORY_ROLLING_BASELINE",
) -> Sample:
    return Sample(
        index=index,
        time=moment,
        p=position,
        conf=0.95,
        coverage=0.8,
        frame_index=index,
        session=session,
        available=position is not None,
        ts_conf=0.9,
        reason=None,
        horizon=240.0,
        recorded_class=recorded_class,
        recorded_state="MONITORING",
        recorded_quality="ok",
        recorded_reason=None,
        recorded_baseline=position,
        recorded_z=0.0,
        recorded_ck=0.0,
        recorded_sigma_k=20.0,
        recorded_persistence=0,
    )


def _synthetic_recording(tmp_path: Path, height: float) -> Recording:
    start = datetime(2026, 11, 28, 1, 40, tzinfo=UTC)
    samples = [
        _synthetic_sample(i, start + timedelta(seconds=10 * i), 400.0 + (i % 5)) for i in range(600)
    ]
    config = RollingBaselineConfig(frame_height=height, nominal_interval_seconds=10.0)
    recording = Recording(
        id=f"synthetic_{int(height)}",
        run_dir=tmp_path,
        label="synthetic",
        frame_height=height,
        frame_width=1280.0,
        nominal_interval_s=10.0,
        config=config,
        config_hash=config.config_hash,
        status="replayable",
        samples=samples,
        segments=[Segment(0, len(samples))],
        valid_starts=[200, 250, 300, 350],
    )
    return recording


def test_flag_off_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "demo_injection_enabled", False)
    response = _client().get("/api/v1/demo/injection/recordings")
    assert response.status_code == 404


def test_recordings_discovered_and_profiled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "demo_injection_enabled", True)
    response = _client().get("/api/v1/demo/injection/recordings")
    assert response.status_code == 200
    recordings = {item["id"]: item for item in response.json()["recordings"]}
    assert RID in recordings
    koh = recordings[RID]
    assert koh["status"] == "replayable"
    assert koh["parity"]["pass"] is True
    assert koh["frame_height_px"] == 720.0
    assert koh["valid_start_count"] >= 10
    assert koh["gate_median_px"] is not None

    dropcam = recordings["dropcam_7h_20261127T155725Z"]
    assert dropcam["status"] == "not_replayable"
    assert "config" in dropcam["reason"]


def test_config_mismatch_is_not_replayable(tmp_path: Path) -> None:
    target = _copy_recording(tmp_path, name="bad_config")
    config_path = target / "injection" / "tables" / "config_used.json"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            '"frame_height": 720.0', '"frame_height": 721.0'
        ),
        encoding="utf-8",
    )
    registry = DemoRegistry(tmp_path)
    recording = registry.get("bad_config")
    assert recording.status == "not_replayable"
    assert "does not match" in (recording.reason or "")


def test_failed_parity_is_not_replayable(tmp_path: Path) -> None:
    target = _copy_recording(tmp_path, name="bad_parity")
    anomalies_path = target / "tables" / "anomalies.csv"
    frame = pd.read_csv(anomalies_path)
    frame.loc[frame.index >= 500, "rolling_baseline_px"] = (
        pd.to_numeric(frame.loc[frame.index >= 500, "rolling_baseline_px"], errors="coerce") + 10
    )
    frame.to_csv(anomalies_path, index=False)
    registry = DemoRegistry(tmp_path)
    recording = registry.get("bad_parity")
    assert recording.status == "not_replayable"
    assert recording.reason == "replay does not match the recorded run"


def test_path_traversal_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "demo_injection_enabled", True)
    client = _client()
    for bad in ("..", "..%2F..%2Fetc", "a/b", "a\\b"):
        response = client.get(f"/api/v1/demo/injection/recordings/{bad}/presets")
        assert response.status_code == 404
    registry = DemoRegistry(RUN.parent)
    with pytest.raises(KeyError):
        registry.get("..")


def test_simulation_does_not_touch_worker_or_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("live worker must not be started by DEMO MODE")

    import app.services.live.capture as live_capture

    monkeypatch.setattr(live_capture, "get_live_manager", forbidden)
    before = {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in (RUN / "tables").iterdir()
    }
    registry = DemoRegistry(RUN.parent)
    recording = registry.get(RID)
    manager = SimulationManager()
    body = {
        "amplitude_pct": 40,
        "ramp_s": 60,
        "hold_s": 300,
        "shape": "retreat_return",
        "sign": 1,
        "speed": 60,
        "start": "random",
        "seed": 20261128,
    }
    simulation = manager.create(recording, body)
    assert simulation.id.startswith("SIM-")
    assert manager.get(simulation.id) is not None
    manager.delete(simulation.id)
    after = {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in (RUN / "tables").iterdir()
    }
    assert before == after


def test_segments_break_on_session_and_gap() -> None:
    start = datetime(2026, 11, 28, 1, 0, tzinfo=UTC)
    samples = []
    for index in range(20):
        session = "s1" if index < 10 else "s2"
        samples.append(
            _synthetic_sample(index, start + timedelta(seconds=10 * index), 400.0, session=session)
        )
    config = RollingBaselineConfig(frame_height=720.0, nominal_interval_seconds=10.0)
    segments = _build_segments(samples, config, {})
    assert [(segment.start, segment.end) for segment in segments] == [(0, 10), (10, 20)]

    gap_samples = [
        _synthetic_sample(0, start, 400.0),
        _synthetic_sample(1, start + timedelta(seconds=10), 400.0),
        _synthetic_sample(2, start + timedelta(seconds=200), 400.0),
    ]
    gap_segments = _build_segments(gap_samples, config, {})
    assert [(segment.start, segment.end) for segment in gap_segments] == [(0, 2), (2, 3)]


def test_start_selection_excludes_warmup_and_unreliable() -> None:
    start = datetime(2026, 11, 28, 1, 0, tzinfo=UTC)
    samples = [_synthetic_sample(i, start + timedelta(seconds=10 * i), 400.0) for i in range(400)]
    samples[40] = _synthetic_sample(
        40,
        start + timedelta(seconds=400),
        400.0,
        recorded_class="DATA_UNRELIABLE",
    )
    config = RollingBaselineConfig(
        window_seconds=300.0,
        k_seconds=90.0,
        frame_height=720.0,
        nominal_interval_seconds=10.0,
    )
    valid = set(_valid_starts(samples, [Segment(0, 400)], config))
    assert 10 not in valid  # inside warm-up
    assert 50 not in valid  # 300 s history includes the unreliable sample at 400 s
    assert 80 in valid
    assert 90 in valid


def test_amplitude_scales_with_frame_height(tmp_path: Path) -> None:
    body = {
        "amplitude_pct": 10,
        "ramp_s": 60,
        "hold_s": 300,
        "shape": "retreat_return",
        "sign": 1,
        "speed": 30,
        "start": "random",
        "seed": 1,
    }
    rec720 = _synthetic_recording(tmp_path, 720.0)
    rec1080 = _synthetic_recording(tmp_path, 1080.0)
    params720, _, _ = validate_simulation(rec720, body)
    params1080, _, _ = validate_simulation(rec1080, body)
    assert params720["amplitude_px"] == pytest.approx(72.0)
    assert params1080["amplitude_px"] == pytest.approx(108.0)


def test_kohsamui_replay_matches_offline_results() -> None:
    registry = DemoRegistry(RUN.parent)
    recording = registry.get(RID)
    results = pd.read_csv(RUN / "injection" / "tables" / "injection_results.csv")
    sample_by_iso = {
        sample.time.isoformat(): index for index, sample in enumerate(recording.samples)
    }
    checked = 0
    for row in results.itertuples(index=False):
        if row.arm != "primary" or row.shape != "retreat_return" or row.sign != 1:
            continue
        if (row.amplitude_px, row.ramp_s) not in ((300.0, 60.0), (80.0, 60.0)):
            continue
        start_index = sample_by_iso[row.t0]
        output = replay_injection(
            samples=recording.samples,
            segments=recording.segments,
            config=recording.config,
            start_index=start_index,
            amplitude_px=float(row.amplitude_px),
            ramp_s=float(row.ramp_s),
            hold_s=float(row.hold_s),
            shape=str(row.shape),
            sign=int(row.sign),
            snapshot=recording.snapshot(start_index),
            injected_min_px=recording.injected_min_px,
            injected_max_px=recording.injected_max_px,
        )
        assert output["detected"] is bool(row.detected)
        if row.detected and output["detected"]:
            interval = float(recording.nominal_interval_s or 10.0)
            assert (
                abs(float(output["latency_from_t0_s"]) - float(row.latency_from_t0_s)) <= interval
            )
        checked += 1
        if checked >= 6:
            break
    assert checked >= 1


def test_replay_never_crosses_a_segment_boundary(tmp_path: Path) -> None:
    recording = _synthetic_recording(tmp_path, 720.0)
    recording.segments = [Segment(0, 300), Segment(300, 600)]
    recording.valid_starts = [200, 350]
    output = replay_injection(
        samples=recording.samples,
        segments=recording.segments,
        config=recording.config,
        start_index=200,
        amplitude_px=300.0,
        ramp_s=600.0,
        hold_s=300.0,
        shape="retreat_return",
        sign=1,
        snapshot=recording.snapshot(200),
    )
    last_time = output["series"][-1]["time"]
    assert last_time <= recording.samples[299].time
    assert all(item["time"] < recording.samples[300].time for item in output["series"])


def test_concurrency_cap_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo_service, "MAX_CONCURRENT_SIMULATIONS", 1)
    registry = DemoRegistry(RUN.parent)
    recording = registry.get(RID)
    manager = SimulationManager()
    body = {
        "amplitude_pct": 40,
        "ramp_s": 60,
        "hold_s": 300,
        "shape": "retreat_return",
        "sign": 1,
        "speed": 60,
        "start": "random",
        "seed": 1,
    }
    first = manager.create(recording, body)
    with pytest.raises(DemoCapacityError):
        manager.create(recording, body)
    assert manager.stop(first.id) is not None
    assert manager.delete(first.id) is True
    assert manager.get(first.id) is None
    assert manager.active_count() == 0
    second = manager.create(recording, body)
    manager.delete(second.id)


def test_invalid_parameters_rejected() -> None:
    registry = DemoRegistry(RUN.parent)
    recording = registry.get(RID)
    base = {
        "amplitude_pct": 40,
        "ramp_s": 60,
        "hold_s": 300,
        "shape": "retreat_return",
        "sign": 1,
        "speed": 30,
        "start": "random",
    }
    for bad in (
        {"amplitude_pct": 4},
        {"amplitude_pct": 61},
        {"ramp_s": 1},
        {"shape": "nope"},
        {"sign": 0},
        {"speed": 2},
    ):
        with pytest.raises(DemoValidationError):
            validate_simulation(recording, {**base, **bad})
