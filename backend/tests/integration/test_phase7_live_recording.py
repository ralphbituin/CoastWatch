"""Phase 7 integration tests (PostgreSQL + TimescaleDB, opt-in).

These are skipped unless ``COASTWATCH_INTEGRATION=1``. They never open a real
YouTube stream: the live worker gets an injected fake provider/segmenter.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from app.database.session import get_engine, get_session_factory
from app.models.anomaly import AnomalyEvent
from app.models.shoreline import ShorelineObservation
from app.models.stream import StreamSession
from app.services.camera.base import (
    CameraMetadata,
    CameraProvider,
    CameraStatus,
    Frame,
)
from app.services.lab import runner
from app.services.lab.registry import LabRun
from app.services.live.capture import LiveCaptureWorker

pytestmark = pytest.mark.skipif(
    os.getenv("COASTWATCH_INTEGRATION") != "1",
    reason="set COASTWATCH_INTEGRATION=1 (and run a DB) to enable",
)

API = "/api/v1"


@pytest.fixture(scope="module", autouse=True)
def _cleanup() -> None:
    yield
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM cameras WHERE code LIKE 'TEST-P7-%'"))
        conn.execute(text("DELETE FROM monitoring_stations WHERE code LIKE 'TEST-P7-%'"))


@pytest.fixture()
def station_id(client: TestClient) -> int:
    suffix = uuid.uuid4().hex[:8]
    response = client.post(
        f"{API}/stations",
        json={
            "code": f"TEST-P7-{suffix}",
            "name": f"Phase 7 station {suffix}",
            "country_iso": "PH",
            "latitude": 14.5,
            "longitude": 120.9,
        },
    )
    assert response.status_code == 201, response.text
    return int(response.json()["id"])


def _create_youtube_camera(client: TestClient, station_id: int) -> dict:
    suffix = uuid.uuid4().hex[:8]
    response = client.post(
        f"{API}/cameras",
        json={
            "code": f"TEST-P7-{suffix}",
            "name": f"YouTube test camera {suffix}",
            "monitoring_station_id": station_id,
            "camera_type": "youtube_live",
            "youtube_url": "https://youtu.be/abcdefghijk",
            "resolution_height": 60,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["youtube_url_redacted"] == "https://youtu.be/abcdefghijk"
    return body


class _FakeProvider(CameraProvider):
    def __init__(self) -> None:
        self.base = datetime.now(UTC)
        self.reads = 0

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def read_frame(self) -> Frame:
        self.reads += 1
        return Frame(
            image=np.zeros((60, 80, 3), dtype=np.uint8),
            captured_at=self.base + timedelta(seconds=30 * self.reads),
            metadata=CameraMetadata(camera_id=0, camera_type="youtube_live"),
            sequence=self.reads,
        )

    def get_status(self) -> CameraStatus:
        return CameraStatus(True, True, "ok")

    def get_metadata(self) -> CameraMetadata:
        return CameraMetadata(camera_id=0, camera_type="youtube_live")


class _Segmentation:
    def __init__(self) -> None:
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[:12, :] = 2
        mask[12:38, :] = 1
        mask[38:, :] = 0
        self.class_mask = mask
        self.confidence = 0.94
        self.inference_ms = 2.0
        self.class_statistics = {
            "Beach": {"fraction": 0.3},
            "Ocean": {"fraction": 0.5},
            "Sky": {"fraction": 0.2},
        }


class _Segmenter:
    def predict(self, _image) -> _Segmentation:
        return _Segmentation()


def test_phase7_schema_columns_and_stream_table() -> None:
    inspector = __import__("sqlalchemy").inspect(get_engine())
    observation_columns = {
        column["name"] for column in inspector.get_columns("shoreline_observations")
    }
    assert {"elapsed_seconds", "stream_session_id", "ingest_latency_ms"} <= observation_columns
    camera_columns = {column["name"] for column in inspector.get_columns("cameras")}
    assert {"youtube_url", "stream_quality", "stream_format"} <= camera_columns
    assert "stream_sessions" in inspector.get_table_names()


def test_live_worker_persists_observation_session_and_rolling_row(
    client: TestClient, station_id: int
) -> None:
    camera = _create_youtube_camera(client, station_id)
    worker = LiveCaptureWorker(
        camera["id"],
        session_factory=get_session_factory(),
        provider_factory=lambda _camera: _FakeProvider(),
        segmenter_factory=_Segmenter,
        sample_interval_seconds=30.0,
    )
    provider, context = worker._initialise()  # noqa: SLF001 - integration contract
    worker._sample_once(context)  # noqa: SLF001
    worker._sample_once(context)  # noqa: SLF001
    provider.disconnect()
    worker._finish_session()  # noqa: SLF001

    session = get_session_factory()()
    try:
        recording_id = context.status.recording_id
        assert recording_id
        observations = session.scalars(
            select(ShorelineObservation)
            .where(ShorelineObservation.recording_id == recording_id)
            .order_by(ShorelineObservation.frame_index)
        ).all()
        assert len(observations) == 2
        assert observations[0].observed_at < observations[1].observed_at
        assert observations[0].stream_session_id == recording_id
        assert observations[0].capture_timestamp_source == "stream_wall_clock"
        assert observations[0].ingest_latency_ms is not None
        assert observations[0].signed_position_px is not None

        stream = session.scalar(
            select(StreamSession).where(StreamSession.session_id == recording_id)
        )
        assert stream is not None
        assert stream.frames_persisted == 2
        assert stream.status == "stopped"

        events = session.scalars(
            select(AnomalyEvent).where(AnomalyEvent.recording_id == recording_id)
        ).all()
        assert len(events) == 2
        assert all(event.detector_mode == "rolling_baseline" for event in events)
        assert all(event.classification != "NORMAL" for event in events)
    finally:
        session.close()


def test_lab_persist_opt_in_is_idempotent(client: TestClient, station_id: int, tmp_path) -> None:
    camera = _create_youtube_camera(client, station_id)
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    payload = buffer.tobytes()
    run = LabRun(
        run_id=f"lab-test-{uuid.uuid4().hex[:8]}",
        kind="images",
        timezone="Asia/Bangkok",
        temp_dir=tmp_path,
        persist=True,
        camera_id=camera["id"],
    )
    timestamps = [datetime.now(UTC), datetime.now(UTC) + timedelta(seconds=60)]
    arguments = {
        "images": [("a.png", payload), ("b.png", payload)],
        "timestamps": timestamps,
        "timestamp_source": "manifest",
        "rolling": runner.RollingBaselineConfig(
            window_seconds=240.0,
            k_seconds=60.0,
            frame_height=60.0,
            nominal_interval_seconds=60.0,
        ),
        "interval_seconds": 60,
        "persist": True,
        "camera_id": camera["id"],
        "session_factory": get_session_factory(),
        "segmenter_factory": _Segmenter,
    }
    runner.run_images(run, **arguments)
    assert run.status == "completed", run.errors
    session = get_session_factory()()
    try:
        count = session.scalar(
            select(func.count())
            .select_from(ShorelineObservation)
            .where(ShorelineObservation.recording_id == run.run_id)
        )
        assert count == 2
    finally:
        session.close()

    # Reprocessing the same run must update in place, not duplicate rows.
    runner.run_images(run, **arguments)
    session = get_session_factory()()
    try:
        count = session.scalar(
            select(func.count())
            .select_from(ShorelineObservation)
            .where(ShorelineObservation.recording_id == run.run_id)
        )
        assert count == 2
    finally:
        session.close()


def test_camera_status_reports_live_fields(client: TestClient, station_id: int) -> None:
    camera = _create_youtube_camera(client, station_id)
    status = client.get(f"{API}/cameras/{camera['id']}/status")
    assert status.status_code == 200
    body = status.json()
    assert body["stream_state"] in {"stopped", "starting", "running", "reconnecting", "failed"}
    assert body["frames_persisted"] is not None
    assert "experimental_notice" in body
