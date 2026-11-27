"""Integration tests for camera/station management, stream deletion and the
event-to-waterline link.

Requires PostgreSQL + TimescaleDB with migrations applied and
``COASTWATCH_INTEGRATION=1``.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    os.getenv("COASTWATCH_INTEGRATION") != "1",
    reason="set COASTWATCH_INTEGRATION=1 (and run a DB) to enable",
)

API = "/api/v1"


def _mask_png() -> bytes:
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[45:, :] = 255
    ok, buffer = cv2.imencode(".png", mask)
    assert ok
    return buffer.tobytes()


def _create_station(client: TestClient, suffix: str) -> int:
    response = client.post(
        f"{API}/stations",
        json={
            "code": f"TEST-UI-{suffix}",
            "name": f"Management API test station {suffix}",
            "country_iso": "PH",
            "latitude": 14.5,
            "longitude": 120.9,
        },
    )
    assert response.status_code == 201, response.text
    return int(response.json()["id"])


def _create_camera(client: TestClient, station_id: int, suffix: str) -> int:
    response = client.post(
        f"{API}/cameras",
        json={
            "code": f"TEST-UI-CAM-{suffix}",
            "name": f"Management API test camera {suffix}",
            "monitoring_station_id": station_id,
            "camera_type": "upload",
        },
    )
    assert response.status_code == 201, response.text
    return int(response.json()["id"])


def test_camera_and_station_edit_delete(client: TestClient) -> None:
    suffix = uuid.uuid4().hex[:8]
    station_id = _create_station(client, suffix)
    camera_id = _create_camera(client, station_id, suffix)

    updated_station = client.patch(
        f"{API}/stations/{station_id}",
        json={"name": f"Edited station {suffix}", "timezone": "GMT"},
    )
    assert updated_station.status_code == 200, updated_station.text
    assert updated_station.json()["timezone"] == "GMT"

    updated_camera = client.patch(
        f"{API}/cameras/{camera_id}",
        json={"name": f"Edited camera {suffix}", "enabled": True},
    )
    assert updated_camera.status_code == 200, updated_camera.text
    assert "Edited camera" in updated_camera.json()["name"]

    assert client.delete(f"{API}/cameras/{camera_id}").status_code == 200
    assert client.get(f"{API}/cameras/{camera_id}").status_code == 404
    assert client.delete(f"{API}/stations/{station_id}").status_code == 200
    assert client.get(f"{API}/stations/{station_id}").status_code == 404


def test_calibration_monitoring_outline_is_versioned(client: TestClient) -> None:
    suffix = uuid.uuid4().hex[:8]
    station_id = _create_station(client, suffix)
    camera_id = _create_camera(client, station_id, suffix)
    polygon = {
        "coordinate_space": "normalized",
        "points": [[0.1, 0.2], [0.7, 0.2], [0.7, 0.8], [0.1, 0.8]],
        "frame_width": 1280,
        "frame_height": 720,
    }

    first = client.post(
        f"{API}/cameras/{camera_id}/calibrations",
        json={
            "meters_per_pixel": 0.05,
            "reference_points": {"points": []},
            "notes": "baseline",
            "roi_polygon": polygon,
        },
    )
    assert first.status_code == 201, first.text
    first_body = first.json()
    assert first_body["version"] == 1
    assert first_body["roi_polygon"]["points"] == polygon["points"]
    assert first_body["is_active"] is True

    second = client.post(
        f"{API}/cameras/{camera_id}/calibrations/{first_body['id']}/roi",
        json={
            "roi_polygon": {
                "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.2, 0.9]],
            },
            "notes": "tightened",
        },
    )
    assert second.status_code == 201, second.text
    second_body = second.json()
    assert second_body["version"] == 2
    assert second_body["meters_per_pixel"] == 0.05
    assert second_body["reference_points"] == {"points": []}
    assert "baseline" in second_body["notes"]
    assert "tightened" in second_body["notes"]
    assert second_body["is_active"] is True

    profiles = client.get(f"{API}/cameras/{camera_id}/calibrations")
    assert profiles.status_code == 200
    by_version = {row["version"]: row for row in profiles.json()}
    assert by_version[1]["is_active"] is False
    assert by_version[2]["is_active"] is True

    cleared = client.post(
        f"{API}/cameras/{camera_id}/calibrations/{second_body['id']}/roi",
        json={"roi_polygon": None},
    )
    assert cleared.status_code == 201, cleared.text
    assert cleared.json()["roi_polygon"] is None

    invalid = client.post(
        f"{API}/cameras/{camera_id}/calibrations/{second_body['id']}/roi",
        json={
            "roi_polygon": {
                "points": [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]],
            }
        },
    )
    assert invalid.status_code == 422, invalid.text

    missing = client.post(
        f"{API}/cameras/{camera_id}/calibrations/999999/roi",
        json={"roi_polygon": None},
    )
    assert missing.status_code == 404

    assert client.delete(f"{API}/cameras/{camera_id}").status_code == 200
    assert client.delete(f"{API}/stations/{station_id}").status_code == 200


def test_event_links_to_stored_waterline_and_stream_delete(client: TestClient) -> None:
    from app.database.session import get_session_factory
    from app.models.anomaly import AnomalyEvent
    from app.models.stream import StreamSession

    suffix = uuid.uuid4().hex[:8]
    station_id = _create_station(client, suffix)
    camera_id = _create_camera(client, station_id, suffix)
    observed_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    observation = client.post(
        f"{API}/shorelines/observations",
        data={"camera_id": str(camera_id), "observed_at": observed_at},
        files={"file": ("mask.png", _mask_png(), "image/png")},
    )
    assert observation.status_code == 201, observation.text

    detail = client.get(
        f"{API}/shorelines/observations/detail",
        params={"camera_id": camera_id, "observed_at": observed_at},
    )
    assert detail.status_code == 200, detail.text
    assert detail.json()["camera_id"] == camera_id

    session_id = f"test-ui-{suffix}"
    factory = get_session_factory()
    with factory() as session:
        event = AnomalyEvent(
            detected_at=datetime.fromisoformat(observed_at),
            monitoring_station_id=station_id,
            camera_id=camera_id,
            state="ANOMALY_DETECTED",
            classification="RAPID_SHORELINE_RETREAT",
            acknowledged=False,
            reason="integration test event",
        )
        session.add(event)
        session.add(
            StreamSession(
                session_id=session_id,
                recording_id=session_id,
                camera_id=camera_id,
                started_at=datetime.fromisoformat(observed_at),
                status="stopped",
            )
        )
        session.commit()
        event_id = event.id

    linked = client.get(f"{API}/anomalies/{event_id}/observation")
    assert linked.status_code == 200, linked.text
    assert linked.json()["camera_id"] == camera_id

    filtered = client.get(f"{API}/anomalies", params={"station_id": station_id})
    assert filtered.status_code == 200, filtered.text
    assert filtered.json()["total"] >= 1

    assert client.delete(f"{API}/streams/{session_id}").status_code == 200
    assert client.get(f"{API}/streams/{session_id}").status_code == 404

    assert client.delete(f"{API}/cameras/{camera_id}").status_code == 200
    assert client.delete(f"{API}/stations/{station_id}").status_code == 200
