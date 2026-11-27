"""Phase 7 test-lab tests (bounded, no real model/network/DB)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db
from app.main import app
from app.schemas.lab import LabStreamRunRequest
from app.services.anomaly.rolling_baseline import RollingBaselineConfig
from app.services.lab import runner
from app.services.lab.registry import LabRegistry, LabRun

T0 = datetime(2026, 11, 25, 6, 0, tzinfo=UTC)


class FakeSegmentation:
    def __init__(self) -> None:
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[:12, :] = 2
        mask[12:38, :] = 1
        mask[38:, :] = 0
        self.class_mask = mask
        self.confidence = 0.93
        self.inference_ms = 3.5
        self.class_statistics = {
            "Beach": {"fraction": 0.3},
            "Ocean": {"fraction": 0.5},
            "Sky": {"fraction": 0.2},
        }


class FakeSegmenter:
    def predict(self, _image) -> FakeSegmentation:
        return FakeSegmentation()


def _png() -> bytes:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def _config() -> RollingBaselineConfig:
    return RollingBaselineConfig(
        window_seconds=240.0,
        k_seconds=60.0,
        frame_height=60.0,
        nominal_interval_seconds=60.0,
    )


def test_manifest_and_start_interval_timestamp_sources() -> None:
    stamps, source = runner.resolve_image_timestamps(
        ["a.jpg"],
        manifest_text='{"a.jpg": "2026-11-25T06:00:00+07:00"}',
        start_at=None,
        interval_seconds=None,
        timezone="Asia/Bangkok",
    )
    assert source == "manifest"
    assert stamps[0] == datetime(2026, 11, 24, 23, 0, tzinfo=UTC)

    stamps, source = runner.resolve_image_timestamps(
        ["a.jpg", "b.jpg"],
        manifest_text=None,
        start_at="2026-11-25T06:00:00+07:00",
        interval_seconds=60,
        timezone="Asia/Bangkok",
    )
    assert source == "operator_declared"
    assert stamps[1] - stamps[0] == timedelta(seconds=60)


def test_images_without_timestamp_source_are_rejected() -> None:
    with pytest.raises(ValueError, match="timestamp manifest"):
        runner.resolve_image_timestamps(
            ["a.jpg"],
            manifest_text=None,
            start_at=None,
            interval_seconds=None,
            timezone="Asia/Bangkok",
        )


def test_non_increasing_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        runner.resolve_image_timestamps(
            ["a.jpg", "b.jpg"],
            manifest_text=('{"a.jpg": "2026-11-25T06:00:00Z", "b.jpg": "2026-11-25T05:59:00Z"}'),
            start_at=None,
            interval_seconds=None,
            timezone="Asia/Bangkok",
        )


def test_lab_images_do_not_touch_the_database_by_default(tmp_path) -> None:
    run = LabRun(run_id="lab-test", kind="images", timezone="Asia/Bangkok", temp_dir=tmp_path)
    image = _png()

    def forbidden_session_factory():
        raise AssertionError("persist=false must not open a database session")

    runner.run_images(
        run,
        images=[("a.png", image), ("b.png", image)],
        timestamps=[T0, T0 + timedelta(seconds=60)],
        timestamp_source="manifest",
        rolling=_config(),
        interval_seconds=60,
        persist=False,
        camera_id=None,
        session_factory=forbidden_session_factory,
        segmenter_factory=FakeSegmenter,
    )
    assert run.status == "completed", run.errors
    assert len(run.rows) == 2
    assert run.rows[0]["observed_at"] == T0.isoformat()
    assert run.rows[0]["observed_at_local"].startswith("2026-11-25T13:00:00")
    assert run.rows[0]["timestamp_source"] == "manifest"
    assert run.rows[0]["signed_position_px"] is not None
    assert "classification" in run.rows[0]
    assert "rows.csv" in run.artifacts
    assert "report.json" in run.artifacts


def test_lab_run_registry_enforces_concurrency_and_cleans_temp(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_settings = SimpleNamespace(
        data_dir=str(tmp_path),
        lab_max_concurrent_runs=1,
        lab_run_ttl_seconds=3600.0,
    )
    monkeypatch.setattr("app.services.lab.registry.get_settings", lambda: fake_settings)
    registry = LabRegistry()
    run = registry.create("images", timezone="Asia/Bangkok", persist=False, camera_id=None)
    assert run.temp_dir is not None and run.temp_dir.is_dir()
    with pytest.raises(ValueError, match="too many active"):
        registry.create("images", timezone="Asia/Bangkok", persist=False, camera_id=None)
    assert registry.delete(run.run_id) is True
    assert not run.temp_dir.exists()


def test_lab_registry_expires_finished_runs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_settings = SimpleNamespace(
        data_dir=str(tmp_path),
        lab_max_concurrent_runs=2,
        lab_run_ttl_seconds=1.0,
    )
    monkeypatch.setattr("app.services.lab.registry.get_settings", lambda: fake_settings)
    registry = LabRegistry()
    run = registry.create("images", timezone="Asia/Bangkok", persist=False, camera_id=None)
    run.update(status="completed")
    run.updated_at = datetime.now(UTC) - timedelta(seconds=10)
    assert registry.cleanup_expired() == 1
    assert registry.get(run.run_id) is None


def test_stream_request_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        LabStreamRunRequest()
    with pytest.raises(ValueError):
        LabStreamRunRequest(camera_id=1, youtube_url="https://youtu.be/abc123")
    request = LabStreamRunRequest(youtube_url="https://youtu.be/abc123")
    assert request.persist is False


def test_image_route_returns_422_without_timestamps(client: TestClient) -> None:
    def fake_db():
        yield None

    app.dependency_overrides[get_db] = fake_db
    try:
        response = client.post(
            "/api/v1/lab/runs/images",
            files=[("images", ("a.png", _png(), "image/png"))],
            data={"timezone": "Asia/Bangkok"},
        )
        assert response.status_code == 422, response.text
        assert "timestamp" in response.json()["error"]["message"].lower()
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_camera_create_requires_youtube_url_for_youtube_type(client: TestClient) -> None:
    base = {
        "code": "YT-1",
        "name": "YouTube camera",
        "monitoring_station_id": 1,
        "camera_type": "youtube_live",
    }
    response = client.post("/api/v1/cameras", json=base)
    assert response.status_code == 422
    response = client.post(
        "/api/v1/cameras",
        json={
            **base,
            "camera_type": "rtsp",
            "rtsp_url": "rtsp://cam.local/stream",
            "youtube_url": "https://youtu.be/abc123",
        },
    )
    assert response.status_code == 422
    response = client.post(
        "/api/v1/cameras",
        json={**base, "youtube_url": "https://localhost/watch?v=abc123"},
    )
    assert response.status_code == 422


def test_new_lab_and_stream_routes_are_registered() -> None:
    paths = {route.path for route in app.routes}
    for expected in (
        "/api/v1/lab/runs/images",
        "/api/v1/lab/runs/video",
        "/api/v1/lab/runs/stream",
        "/api/v1/lab/runs/{run_id}",
        "/api/v1/lab/runs/{run_id}/rows",
        "/api/v1/lab/runs/{run_id}/artifacts/{name}",
        "/api/v1/streams",
        "/api/v1/streams/{session_id}/observations",
        "/api/v1/cameras/{camera_id}/stream/start",
        "/api/v1/cameras/{camera_id}/stream/stop",
    ):
        assert expected in paths, expected
