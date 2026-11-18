"""Integration test for the tide prediction engine (Phase 2).

Requires a running PostgreSQL + TimescaleDB with migrations applied
(``alembic upgrade head``). Skips otherwise. All data used here is
**synthetic** — never real NAMRIA observations.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    os.getenv("COASTWATCH_INTEGRATION") != "1",
    reason="set COASTWATCH_INTEGRATION=1 (and run a DB) to enable",
)

SYN_STATION = "TEST-SYNTH-001"
CSV_FILENAME = "synthetic_tide_manila.csv"


def synthetic_csv_bytes() -> bytes:
    rows = ["timestamp,tide_height,event_type"]
    t0 = datetime(2026, 11, 1)
    for hour in range(0, 24 * 60 + 1):  # 60 days of hourly samples (synthetic)
        height = 1.1 + 0.7 * math.sin(2 * math.pi * hour / 12.4)
        stamp = (t0 + timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M")
        rows.append(f"{stamp},{height:.4f},")
    return "\n".join(rows).encode()


SYNTHETIC_CSV = synthetic_csv_bytes()


@pytest.fixture(scope="module")
def tide_station(client: TestClient) -> str:
    """Ensure the synthetic test station exists (insert-only, idempotent)."""
    payload = {
        "code": SYN_STATION,
        "name": "Synthetic Test Station (Manila)",
        "country_iso": "PH",
        "latitude": 14.5,
        "longitude": 120.9,
        "timezone": "Asia/Manila",
        "data_source": "NAMRIA_CSV",
        "vertical_datum": "MLWS",
        "measurement_unit": "meters",
    }
    response = client.post("/api/v1/tides/stations", json=payload)
    assert response.status_code in (201, 409), response.text
    return SYN_STATION


@pytest.fixture(scope="module")
def imported_dataset(client: TestClient, tide_station: str) -> dict:
    """Import the synthetic dataset (idempotent)."""
    response = client.post(
        "/api/v1/tides/imports",
        files={"file": (CSV_FILENAME, SYNTHETIC_CSV, "text/csv")},
        data={
            "station_code": tide_station,
            "name": "Synthetic tide for Phase 2 integration tests",
            "timezone": "Asia/Manila",
            "unit": "meters",
            "vertical_datum": "MLWS",
            "description": "Synthetic 12.4 h periodic tide — NOT real observations",
            "source": "SYNTHETIC_TEST_FIXTURE",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    # Re-runs hit the insert-only duplicate path, which is fine.
    assert body["status"] in ("committed", "partial", "duplicate")
    return body


def test_list_tide_stations(client: TestClient, tide_station: str) -> None:
    response = client.get("/api/v1/tides/stations")
    assert response.status_code == 200
    codes = [s["code"] for s in response.json()]
    assert tide_station in codes


def test_station_detail_has_coverage(
    client: TestClient, tide_station: str, imported_dataset: dict
) -> None:
    response = client.get(f"/api/v1/tides/stations/{tide_station}")
    assert response.status_code == 200
    body = response.json()
    assert body["timezone"] == "Asia/Manila"
    assert body["vertical_datum"] == "MLWS"
    assert body["data_source"] == "NAMRIA_CSV"
    if imported_dataset["status"] != "duplicate":
        assert body["prediction_start"] is not None
        assert body["prediction_end"] is not None


def test_preview_import_no_db_writes(client: TestClient, tide_station: str) -> None:
    response = client.post(
        "/api/v1/tides/imports/preview",
        files={"file": (CSV_FILENAME, SYNTHETIC_CSV, "text/csv")},
        data={"timezone": "Asia/Manila"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["valid_records"] == 24 * 60 + 1
    assert body["rejected_records"] == 0
    assert body["columns_detected"][:2] == ["timestamp", "tide_height"]


def test_predictions_at_exact_sample(
    client: TestClient, tide_station: str, imported_dataset: dict
) -> None:
    response = client.get(
        "/api/v1/tides/at",
        params={"station_code": tide_station, "timestamp": "2026-11-05 03:00"},
    )
    assert response.status_code == 200
    body = response.json()
    expected = 1.1 + 0.7 * math.sin(2 * math.pi * 99 / 12.4)
    assert body["predicted_height_m"] == pytest.approx(expected, abs=0.01)
    assert body["quality"] in ("ok", "interpolated")
    assert body["direction"] in ("rising", "falling", "slack_high", "slack_low", "unknown")
    assert body["vertical_datum"] == "MLWS"
    assert body["source"] in ("NAMRIA_CSV", "SYNTHETIC_TEST_FIXTURE")
    assert body["timezone"] == "Asia/Manila"
    assert "local_timestamp" in body


def test_predictions_outside_coverage_return_404(client: TestClient, tide_station: str) -> None:
    response = client.get(
        "/api/v1/tides/at",
        params={"station_code": tide_station, "timestamp": "2030-01-01 00:00"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "data_unavailable"


def test_high_low_waters_alternate(client: TestClient, tide_station: str) -> None:
    response = client.get(
        "/api/v1/tides/high-low-waters",
        params={
            "station_code": tide_station,
            "start": "2026-11-01T00:00:00Z",
            "end": "2026-11-03T00:00:00Z",
        },
    )
    assert response.status_code == 200
    events = response.json()
    assert len(events) >= 4
    types = [e["type"] for e in events]
    for first, second in zip(types, types[1:], strict=False):
        assert first != second
    assert {"high", "low"}.issubset(set(types))


def test_chart_has_points_and_events(client: TestClient, tide_station: str) -> None:
    response = client.get(
        "/api/v1/tides/chart",
        params={
            "station_code": tide_station,
            "start": "2026-11-01T00:00:00Z",
            "end": "2026-11-02T00:00:00Z",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 25  # inclusive hourly range
    assert len(body["points"]) == 25
    assert len(body["events"]) >= 1
    assert body["sufficient_for_7d"] is True


def test_current_time_prediction(client: TestClient, tide_station: str) -> None:
    response = client.get("/api/v1/tides/current", params={"station_code": tide_station})
    assert response.status_code == 200
    body = response.json()
    assert body["direction"] in ("rising", "falling", "slack_high", "slack_low", "unknown")
    assert body["interpolation_method"] == "pchip"
    assert body["previous_event"] is not None
    assert body["next_event"] is not None


def test_import_history_records_import(
    client: TestClient, tide_station: str, imported_dataset: dict
) -> None:
    response = client.get("/api/v1/tides/imports", params={"station_code": tide_station})
    assert response.status_code == 200
    imports = response.json()["imports"]
    assert any(i["dataset_id"] == imported_dataset.get("dataset_id") for i in imports)


def test_duplicate_file_does_not_overwrite(client: TestClient, tide_station: str) -> None:
    response = client.post(
        "/api/v1/tides/imports",
        files={"file": (CSV_FILENAME, SYNTHETIC_CSV, "text/csv")},
        data={
            "station_code": tide_station,
            "name": "should not be inserted",
            "timezone": "Asia/Manila",
            "unit": "meters",
            "vertical_datum": "MLWS",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "duplicate"
    assert body["imported_records"] == 0


def test_create_station_with_unknown_country_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/tides/stations",
        json={
            "code": "TEST-BADCOUNTRY-01",
            "name": "bad country",
            "country_iso": "ZZ",
            "latitude": 1.0,
            "longitude": 1.0,
        },
    )
    assert response.status_code == 422


def test_import_invalid_timezone_returns_422(client: TestClient, tide_station: str) -> None:
    response = client.post(
        "/api/v1/tides/imports",
        files={"file": ("bad.csv", SYNTHETIC_CSV, "text/csv")},
        data={"station_code": tide_station, "name": "x", "timezone": "Not/AZone"},
    )
    assert response.status_code == 422


def test_import_missing_required_column_returns_422(client: TestClient, tide_station: str) -> None:
    response = client.post(
        "/api/v1/tides/imports",
        files={"file": ("bad.csv", b"when,height\n2026-11-19 00:00,0.5\n", "text/csv")},
        data={"station_code": tide_station, "name": "x"},
    )
    assert response.status_code == 422


def test_monitoring_station_crud_and_link(client: TestClient) -> None:
    code = "TEST-STATION-001"
    response = client.post(
        "/api/v1/stations",
        json={
            "code": code,
            "name": "Synthetic Monitoring Station",
            "country_iso": "PH",
            "latitude": 14.6,
            "longitude": 121.0,
            "timezone": "Asia/Manila",
        },
    )
    assert response.status_code in (201, 409)
    listing = client.get("/api/v1/stations")
    assert listing.status_code == 200
    assert any(s["code"] == code for s in listing.json())

    station_id = next(s["id"] for s in listing.json() if s["code"] == code)
    tide_id = next(
        s["id"] for s in client.get("/api/v1/tides/stations").json() if s["code"] == SYN_STATION
    )
    link = client.post(f"/api/v1/stations/{station_id}/tide-stations/{tide_id}")
    assert link.status_code == 200
    assert link.json()["monitoring_station_id"] == station_id

    unlink = client.delete(f"/api/v1/stations/{station_id}/tide-stations/{tide_id}")
    assert unlink.status_code == 200

    countries = client.get("/api/v1/countries")
    assert countries.status_code == 200
    assert any(c["iso_code"] == "PH" for c in countries.json())
