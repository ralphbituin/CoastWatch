"""Integration tests for the camera module (Phase 3).

Requires a running PostgreSQL + TimescaleDB with migrations applied and
``COASTWATCH_INTEGRATION=1``. Hardware-dependent tests (RTSP/USB) are
additionally gated on ``COASTWATCH_RTSP_TEST_URL`` / ``COASTWATCH_USB_TEST_DEVICE``
and are skipped by default — never on a machine without a real device.
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

from app.core.config import get_settings

pytestmark = pytest.mark.skipif(
    os.getenv("COASTWATCH_INTEGRATION") != "1",
    reason="set COASTWATCH_INTEGRATION=1 (and run a DB) to enable",
)

API = "/api/v1"


@pytest.fixture(scope="module", autouse=True)
def _cleanup_synthetic_rows():
    """Remove TEST-CAM-* rows after the module runs so the dev DB stays tidy."""
    yield
    from sqlalchemy import text

    from app.database.session import get_engine

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM cameras WHERE code LIKE 'TEST-CAM-%'"))
        conn.execute(text("DELETE FROM monitoring_stations WHERE code LIKE 'TEST-CAM-%'"))


def _png_bytes(seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


@pytest.fixture()
def station_id(client: TestClient) -> int:
    suffix = uuid.uuid4().hex[:8]
    response = client.post(
        f"{API}/stations",
        json={
            "code": f"TEST-CAM-{suffix}",
            "name": f"Camera test station {suffix}",
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
        "name": f"Test camera {suffix}",
        "monitoring_station_id": station_id,
        "camera_type": "upload",
    }
    payload.update(overrides)
    response = client.post(f"{API}/cameras", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_camera_crud_and_status(client: TestClient, station_id: int) -> None:
    camera = _create_camera(client, station_id)

    listing = client.get(f"{API}/cameras")
    assert listing.status_code == 200
    assert any(c["id"] == camera["id"] for c in listing.json())

    detail = client.get(f"{API}/cameras/{camera['id']}")
    assert detail.status_code == 200
    assert detail.json()["station_code"].startswith("TEST-CAM-")
    assert detail.json()["rtsp_url_redacted"] is None

    status_resp = client.get(f"{API}/cameras/{camera['id']}/status")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["camera_id"] == camera["id"]
    assert body["enabled"] is True


def test_camera_create_requires_station(client: TestClient) -> None:
    response = client.post(
        f"{API}/cameras",
        json={
            "code": f"TEST-CAM-{uuid.uuid4().hex[:8]}",
            "name": "orphan",
            "monitoring_station_id": 99999999,
            "camera_type": "upload",
        },
    )
    assert response.status_code == 404


def test_rtsp_camera_requires_url(client: TestClient, station_id: int) -> None:
    response = client.post(
        f"{API}/cameras",
        json={
            "code": f"TEST-CAM-{uuid.uuid4().hex[:8]}",
            "name": "missing url",
            "monitoring_station_id": station_id,
            "camera_type": "rtsp",
        },
    )
    assert response.status_code == 422


def test_upload_frame_roundtrip(client: TestClient, station_id: int) -> None:
    camera = _create_camera(client, station_id)
    cam_id = camera["id"]

    # No uploads yet -> frame endpoint reports data_unavailable (never a blank frame).
    missing = client.get(f"{API}/cameras/{cam_id}/frame")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "data_unavailable"

    upload = client.post(
        f"{API}/cameras/{cam_id}/upload-frame",
        files={"file": ("shot.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert upload.status_code == 200, upload.text

    frame = client.get(f"{API}/cameras/{cam_id}/frame")
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"
    assert int(frame.headers["x-frame-sequence"]) >= 1
    assert "x-quality-score" in frame.headers
    assert len(frame.content) > 0

    status_resp = client.get(f"{API}/cameras/{cam_id}/status")
    assert status_resp.status_code == 200

    # Clean up the stored upload so re-runs stay tidy.
    settings = get_settings()
    upload_dir = Path(settings.data_dir) / "uploads" / "cameras" / str(cam_id)
    for leftover in upload_dir.glob("*"):
        leftover.unlink()


def test_upload_rejected_for_non_upload_camera(client: TestClient, station_id: int) -> None:
    camera = _create_camera(client, station_id, camera_type="usb", device_index=0)
    response = client.post(
        f"{API}/cameras/{camera['id']}/upload-frame",
        files={"file": ("shot.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert response.status_code == 422


def test_upload_rejects_non_image_payload(client: TestClient, station_id: int) -> None:
    camera = _create_camera(client, station_id)
    response = client.post(
        f"{API}/cameras/{camera['id']}/upload-frame",
        files={"file": ("notes.png", io.BytesIO(b"not an image at all"), "image/png")},
    )
    assert response.status_code == 422


def test_calibration_profile_lifecycle(client: TestClient, station_id: int) -> None:
    camera = _create_camera(client, station_id)
    cam_id = camera["id"]

    created = client.post(
        f"{API}/cameras/{cam_id}/calibrations",
        json={
            "reference_points": {"points": [{"pixel": [0, 0], "world_m": [0.0, 0.0]}]},
            "homography_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "meters_per_pixel": 0.05,
            "notes": "synthetic test calibration",
        },
    )
    assert created.status_code == 201, created.text
    calibration = created.json()
    assert calibration["version"] == 1
    assert calibration["is_active"] is True

    listing = client.get(f"{API}/cameras/{cam_id}/calibrations")
    assert listing.status_code == 200
    assert len(listing.json()) == 1

    detail = client.get(f"{API}/cameras/{cam_id}")
    assert detail.json()["active_calibration_id"] == calibration["id"]


@pytest.mark.skipif(
    not os.getenv("COASTWATCH_RTSP_TEST_URL"),
    reason="set COASTWATCH_RTSP_TEST_URL to run against a real RTSP stream",
)
def test_rtsp_test_connection_hardware(client: TestClient, station_id: int) -> None:
    camera = _create_camera(
        client,
        station_id,
        camera_type="rtsp",
        rtsp_url=os.environ["COASTWATCH_RTSP_TEST_URL"],
    )
    response = client.post(f"{API}/cameras/{camera['id']}/test-connection")
    assert response.status_code == 200
    assert response.json()["ok"] is True


@pytest.mark.skipif(
    os.getenv("COASTWATCH_USB_TEST_DEVICE") is None,
    reason="set COASTWATCH_USB_TEST_DEVICE=<index> to run against a real USB camera",
)
def test_usb_test_connection_hardware(client: TestClient, station_id: int) -> None:
    camera = _create_camera(
        client,
        station_id,
        camera_type="usb",
        device_index=int(os.environ["COASTWATCH_USB_TEST_DEVICE"]),
    )
    response = client.post(f"{API}/cameras/{camera['id']}/test-connection")
    assert response.status_code == 200
    assert response.json()["ok"] is True
