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


def test_phase_stub_modules_report_phase(client: TestClient) -> None:
    stubs = (("shorelines", 4), ("anomalies", 6), ("cameras", 3))
    for module, phase in stubs:
        response = client.get(f"{API}/{module}")
        assert response.status_code == 501, module
        body = response.json()
        assert body["module"] == module
        assert body["phase"] == phase
        assert body["implemented"] is False


def test_openapi_schema_includes_tide_routes(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for expected in (
        "/api/v1/health",
        "/api/v1/tides/stations",
        "/api/v1/tides/at",
        "/api/v1/tides/current",
        "/api/v1/tides/chart",
        "/api/v1/tides/imports",
        "/api/v1/stations",
        "/api/v1/countries",
    ):
        assert expected in paths, expected


def test_openapi_schema_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/v1/health" in response.json()["paths"]
