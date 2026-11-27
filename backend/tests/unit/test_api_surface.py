"""API surface tests that do not require a database."""

from __future__ import annotations

from fastapi.testclient import TestClient

API = "/api/v1"


def test_root_metadata(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"]
    assert body["api"] == API


def test_health_ok(client: TestClient) -> None:
    response = client.get(f"{API}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["timestamp"]


def test_request_id_header_present(client: TestClient) -> None:
    response = client.get(f"{API}/health")
    assert "X-Request-ID" in response.headers


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    response = client.get(f"{API}/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_error"


def test_phase5_phase6_routes_registered() -> None:
    """Anomalies/baselines/videos are implemented; do not regress to 501 stubs."""
    from app.main import app

    paths = {route.path for route in app.routes}
    for expected in (
        f"{API}/anomalies",
        f"{API}/anomalies/{{event_id}}",
        f"{API}/anomalies/current",
        f"{API}/shorelines/baselines",
        f"{API}/videos",
        f"{API}/videos/process",
        f"{API}/videos/{{recording_id}}/observations",
    ):
        assert expected in paths, expected


def test_openapi_schema_includes_tide_routes(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for expected in (
        "/api/v1/health",
        "/api/v1/tides/stations",
        "/api/v1/stations",
        "/api/v1/countries",
        "/api/v1/cameras",
        "/api/v1/cameras/{camera_id}/status",
        "/api/v1/cameras/{camera_id}/test-connection",
        "/api/v1/cameras/{camera_id}/frame",
        "/api/v1/cameras/{camera_id}/upload-video",
        "/api/v1/cameras/{camera_id}/calibrations",
        "/api/v1/cameras/{camera_id}/calibrations/{calibration_id}/roi",
        "/api/v1/shorelines",
        "/api/v1/shorelines/observations",
        "/api/v1/shorelines/observations/latest",
        "/api/v1/shorelines/observe",
        "/api/v1/shorelines/calibrations",
    ):
        assert expected in paths, expected


def test_openapi_schema_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/v1/health" in response.json()["paths"]
