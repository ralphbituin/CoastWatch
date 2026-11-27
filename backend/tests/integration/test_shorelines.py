"""Integration tests for the Phase 4 vision module.

Requires a running PostgreSQL + TimescaleDB with migrations applied and
``COASTWATCH_INTEGRATION=1``. Waterlines are exercised via **operator-labelled
masks** (synthetic, clearly labelled as such), a three-class ONNX fixture, and
the synthetic video-file virtual camera. Fixture predictions prove wiring, not
segmentation accuracy.
"""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    os.getenv("COASTWATCH_INTEGRATION") != "1",
    reason="set COASTWATCH_INTEGRATION=1 (and run a DB) to enable",
)

API = "/api/v1"
CREATED_FILES: list[Path] = []


@pytest.fixture(scope="module", autouse=True)
def _cleanup_synthetic_rows():
    yield
    from sqlalchemy import text

    from app.database.session import get_engine
    from app.services.camera import manager as camera_manager

    # Release file handles (cv2.VideoCapture) before deleting test videos.
    camera_manager.drop_all_providers()
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM cameras WHERE code LIKE 'TEST-CAM-%'"))
        conn.execute(text("DELETE FROM monitoring_stations WHERE code LIKE 'TEST-CAM-%'"))
    for path in CREATED_FILES:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            print(f"warning: could not delete {path} (still locked)")


def _station_id(client: TestClient) -> int:
    suffix = uuid.uuid4().hex[:8]
    response = client.post(
        f"{API}/stations",
        json={
            "code": f"TEST-CAM-{suffix}",
            "name": f"Phase 4 test station {suffix}",
            "country_iso": "PH",
            "latitude": 14.5995,
            "longitude": 120.9842,
        },
    )
    assert response.status_code == 201, response.text
    return int(response.json()["id"])


def _create_camera(client: TestClient, station_id: int, **overrides) -> dict:
    suffix = uuid.uuid4().hex[:8]
    payload = {
        "code": f"TEST-CAM-{suffix}",
        "name": f"Phase 4 test camera {suffix}",
        "monitoring_station_id": station_id,
        "camera_type": "upload",
    }
    payload.update(overrides)
    response = client.post(f"{API}/cameras", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _tiny_mp4_bytes(width: int = 64, height: int = 48, frames: int = 6) -> bytes:
    path = Path(os.environ.get("TEMP", "/tmp")) / f"coastwatch_test_{uuid.uuid4().hex}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    assert writer.isOpened(), "could not create synthetic mp4 for upload"
    for i in range(frames):
        frame = np.full((height, width, 3), 40 + i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    data = path.read_bytes()
    path.unlink()
    return data


def _synthetic_water_mask_png(width: int = 120, height: int = 80, sea_top: int = 45) -> bytes:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[sea_top:, :] = 255
    ok, buffer = cv2.imencode(".png", mask)
    assert ok
    return buffer.tobytes()


def _write_video_into_data_dir(name: str) -> Path:
    from app.core.config import get_settings

    videos = Path(get_settings().data_dir) / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    path = videos / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened(), "could not create synthetic mp4 in DATA_DIR/videos"
    for i in range(8):
        writer.write(np.full((48, 64, 3), 30 + i * 20, dtype=np.uint8))
    writer.release()
    CREATED_FILES.append(path)
    return path


def test_video_virtual_camera_end_to_end(client: TestClient) -> None:
    station_id = _station_id(client)
    video_name = f"testclip_{uuid.uuid4().hex[:8]}.mp4"
    _write_video_into_data_dir(video_name)

    camera = _create_camera(
        client,
        station_id,
        camera_type="video",
        video_path=video_name,
        fps=10.0,
    )

    frame = client.get(f"{API}/cameras/{camera['id']}/frame")
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"

    test_conn = client.post(f"{API}/cameras/{camera['id']}/test-connection")
    assert test_conn.status_code == 200
    body = test_conn.json()
    assert body["ok"] is True
    assert body["frame_width"] == 64
    assert body["frame_height"] == 48
    assert body["quality"] is not None

    status_resp = client.get(f"{API}/cameras/{camera['id']}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["connected"] is True


def test_video_path_traversal_rejected(client: TestClient) -> None:
    station_id = _station_id(client)
    response = client.post(
        f"{API}/cameras",
        json={
            "code": f"TEST-CAM-{uuid.uuid4().hex[:8]}",
            "name": "traversal attempt",
            "monitoring_station_id": station_id,
            "camera_type": "video",
            "video_path": "../../outside.mp4",
        },
    )
    assert response.status_code == 422


def test_upload_video_links_camera(client: TestClient) -> None:
    station_id = _station_id(client)
    camera = _create_camera(client, station_id, camera_type="video", video_path="placeholder.mp4")

    upload = client.post(
        f"{API}/cameras/{camera['id']}/upload-video",
        files={"file": ("coastal_clip.mp4", io.BytesIO(_tiny_mp4_bytes()), "video/mp4")},
    )
    assert upload.status_code == 200, upload.text

    detail = client.get(f"{API}/cameras/{camera['id']}")
    stored = detail.json()["video_path"]
    assert stored and stored.endswith("coastal_clip.mp4")
    from app.core.config import get_settings

    stored_path = Path(get_settings().data_dir) / "videos" / stored
    CREATED_FILES.append(stored_path)
    assert stored_path.is_file()


def test_upload_video_rejected_for_non_video_camera(client: TestClient) -> None:
    station_id = _station_id(client)
    camera = _create_camera(client, station_id, camera_type="upload")
    response = client.post(
        f"{API}/cameras/{camera['id']}/upload-video",
        files={"file": ("clip.mp4", io.BytesIO(_tiny_mp4_bytes()), "video/mp4")},
    )
    assert response.status_code == 422


def test_observe_without_model_reports_unreliable(client: TestClient) -> None:
    station_id = _station_id(client)
    camera = _create_camera(client, station_id, camera_type="upload")
    response = client.post(
        f"{API}/shorelines/observe",
        json={"station_id": station_id, "camera_id": camera["id"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "DATA_UNRELIABLE"
    assert body["actual_shoreline_m"] is None
    assert body["details"]["reason"] in {
        "segmentation_model_unavailable",
        "frame_unavailable",
    }


def test_operator_mask_observation_end_to_end(client: TestClient) -> None:
    station_id = _station_id(client)
    camera = _create_camera(client, station_id, camera_type="upload")

    # Identity homography so world == pixel coordinates; seaward axis +y.
    calibration = client.post(
        f"{API}/cameras/{camera['id']}/calibrations",
        json={
            "homography_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "meters_per_pixel": 0.05,
            "notes": "synthetic identity calibration (test)",
        },
    )
    assert calibration.status_code == 201, calibration.text

    mask = _synthetic_water_mask_png(width=120, height=80, sea_top=45)
    upload = client.post(
        f"{API}/shorelines/observations",
        data={
            "camera_id": str(camera["id"]),
            "seaward_axis": "0.0,1.0",
            "observed_at": "2026-11-25T06:00:00+00:00",
        },
        files={"file": ("mask.png", io.BytesIO(mask), "image/png")},
    )
    assert upload.status_code == 201, upload.text
    body = upload.json()
    assert len(body["shoreline_geometry"]) == 120
    ys = [p[1] for p in body["shoreline_geometry"]]
    assert sum(ys) / len(ys) == pytest.approx(45.0, abs=1.5)
    assert body["calibrated_position_m"] == pytest.approx(45.0, abs=1.5)
    assert body["segmentation_confidence"] == 1.0
    # The binary operator path also exposes one ordered segment, and the
    # legacy flat geometry is exactly its concatenation.
    assert len(body["shoreline_segments"]) == 1
    assert body["shoreline_segments"][0] == body["shoreline_geometry"]

    latest = client.get(f"{API}/shorelines/observations/latest?camera_id={camera['id']}")
    assert latest.status_code == 200
    assert latest.json()["camera_id"] == camera["id"]

    listing = client.get(f"{API}/shorelines/observations?camera_id={camera['id']}")
    assert listing.status_code == 200
    assert any(o["camera_id"] == camera["id"] for o in listing.json())

    calibrations = client.get(f"{API}/shorelines/calibrations")
    assert calibrations.status_code == 200
    assert any(c["camera_id"] == camera["id"] for c in calibrations.json())


def test_baselines_report_explicit_availability(client: TestClient) -> None:
    response = client.get(f"{API}/shorelines/baselines")
    assert response.status_code == 200
    body = response.json()
    # Either a provisional baseline exists, or the endpoint says why it does
    # not. It must never return an empty "normal" model.
    assert "available" in body
    assert body["status"] in {"provisional", "available", "unavailable"}
    if not body["available"]:
        assert body["reason"]


def test_module_status_reflects_artifact_presence(client: TestClient) -> None:
    response = client.get(f"{API}/shorelines")
    assert response.status_code == 200
    body = response.json()
    from app.services.vision.segmentation import find_model_artifact, model_status

    # Honest in both directions: availability is derived from the artifact on
    # disk, never hard-coded.
    assert body["segmentation_model_available"] is (find_model_artifact() is not None)
    assert "POST /shorelines/observe" in body["implemented_endpoints"]
    status = model_status()
    if status.get("config_valid"):
        assert body["architecture"] == "SegFormer"
        assert body["encoder"] == "MiT-B2"
        assert body["class_ids"] == {"Beach": 0, "Ocean": 1, "Sky": 2}
        assert body["water_classes"] == [1]
        assert body["checkpoint_sha256"] == status["checkpoint_sha256"]
        assert body["limitations"]


def _jpeg_bytes(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return buffer.tobytes()


def test_observe_with_artifact_end_to_end(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trained-model path replaces the operator-mask fallback.

    A three-class fixture ONNX model is used (Sky above, Ocean, Beach below); it
    is not a trained network, so only the contract is asserted, not accuracy.
    """
    from app.services.vision import segmentation
    from tests.onnx_fixture import frame_bgr, write_artifact

    artifact_root = tmp_path / "models"
    write_artifact(artifact_root / "shoreline-segmentation")
    monkeypatch.setattr(
        segmentation,
        "get_settings",
        lambda: type("Settings", (), {"model_path": str(artifact_root)})(),
    )
    segmentation._load_session.cache_clear()

    station_id = _station_id(client)
    camera = _create_camera(client, station_id, camera_type="upload")
    calibration = client.post(
        f"{API}/cameras/{camera['id']}/calibrations",
        json={
            "homography_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "meters_per_pixel": 0.05,
            "notes": "synthetic identity calibration (model test)",
        },
    )
    assert calibration.status_code == 201, calibration.text

    upload = client.post(
        f"{API}/cameras/{camera['id']}/upload-frame",
        files={"file": ("frame.jpg", io.BytesIO(_jpeg_bytes(frame_bgr(120, 80))), "image/jpeg")},
    )
    assert upload.status_code == 200, upload.text

    response = client.post(
        f"{API}/shorelines/observe",
        json={"station_id": station_id, "camera_id": camera["id"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "OBSERVED", body
    assert body["details"]["reason"] == "observation_persisted"
    # Border margins are now excluded, so the exact count can be slightly below
    # the frame width; the extracted line must still span the frame.
    assert 0 < body["details"]["points"] <= 120
    assert body["details"]["segments"] >= 1
    assert body["details"]["orientation_used"] in {
        "land_below",
        "land_above",
        "land_left",
        "land_right",
    }

    latest = client.get(f"{API}/shorelines/observations/latest?camera_id={camera['id']}")
    assert latest.status_code == 200
    stored = latest.json()
    # A model confidence is a probability and therefore < 1.0, unlike the
    # operator-mask path which is treated as ground truth (exactly 1.0).
    assert stored["segmentation_confidence"] is not None
    assert 0.0 < stored["segmentation_confidence"] < 1.0
    assert 0 < len(stored["shoreline_geometry"]) <= 120
    assert stored["shoreline_segments"]
    flattened = [point for segment in stored["shoreline_segments"] for point in segment]
    assert flattened == stored["shoreline_geometry"]
    assert stored["calibration_id"] is not None

    # The uploaded still keeps its captured_at. Phase 7 drops the exact
    # duplicate timestamp instead of inventing a nudged capture time, and the
    # one-off cycle still reports that the row is available.
    second = client.post(
        f"{API}/shorelines/observe",
        json={"station_id": station_id, "camera_id": camera["id"]},
    )
    assert second.status_code == 200, second.text
    assert second.json()["state"] == "OBSERVED", second.json()
    rows = client.get(f"{API}/shorelines/observations?camera_id={camera['id']}").json()
    assert len(rows) == 1


def test_observe_with_broken_artifact_reports_unreliable(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.vision import segmentation
    from tests.onnx_fixture import frame_bgr, write_artifact

    artifact = write_artifact(tmp_path / "models" / "shoreline-segmentation")
    (artifact / "config.json").unlink()
    monkeypatch.setattr(
        segmentation,
        "get_settings",
        lambda: type("Settings", (), {"model_path": str(tmp_path / "models")})(),
    )
    segmentation._load_session.cache_clear()

    station_id = _station_id(client)
    camera = _create_camera(client, station_id, camera_type="upload")
    client.post(
        f"{API}/cameras/{camera['id']}/upload-frame",
        files={"file": ("frame.jpg", io.BytesIO(_jpeg_bytes(frame_bgr(80, 60))), "image/jpeg")},
    )
    response = client.post(
        f"{API}/shorelines/observe",
        json={"station_id": station_id, "camera_id": camera["id"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "DATA_UNRELIABLE"
    assert body["details"]["reason"] == "segmentation_model_unavailable"
    assert "config.json" in body["details"]["message"]
