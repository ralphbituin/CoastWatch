"""Integration test for the database health endpoint.

Skipped automatically unless ``COASTWATCH_INTEGRATION=1`` is set, so the default
CI unit job stays independent of external services.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    os.getenv("COASTWATCH_INTEGRATION") != "1",
    reason="set COASTWATCH_INTEGRATION=1 (and run a DB) to enable",
)


def test_health_db_reports_extensions(client: TestClient) -> None:
    response = client.get("/api/v1/health/db")
    assert response.status_code == 200
    body = response.json()
    assert body["reachable"] is True
    assert body["server_version"]
    assert body["timescaledb_version"], "TimescaleDB extension must be installed"
    assert body["postgis_version"], "PostGIS extension must be installed"
