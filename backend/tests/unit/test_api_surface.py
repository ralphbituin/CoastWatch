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
    stubs = (("cameras", 3), ("tides", 2), ("stations", 2), ("shorelines", 4), ("anomalies", 6))
    for module, phase in stubs:
        response = client.get(f"{API}/{module}")
        assert response.status_code == 501, module
        body = response.json()
        assert body["module"] == module
        assert body["phase"] == phase
        assert body["implemented"] is False


def test_openapi_schema_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/v1/health" in response.json()["paths"]
