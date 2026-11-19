# API Documentation

Base URL: `/api/v1` (configurable via `API_V1_PREFIX`).
Interactive docs: `/docs` (Swagger UI) and `/redoc`.

Every error response uses one envelope:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "details": {}
  }
}
```

Every response carries an `X-Request-ID` header (echoed from the request if
provided).

---

## Health

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness; does not touch the database |
| GET | `/health/db` | Readiness; PostgreSQL / TimescaleDB / PostGIS versions (200 reachable / 503 not) |

### `GET /api/v1/health/db`
```json
{
  "status": "ok",
  "database": "postgresql",
  "reachable": true,
  "latency_ms": 15.77,
  "server_version": "16.15",
  "timescaledb_version": "2.32.0",
  "postgis_version": "3.7.0",
  "error": null
}
```

---

## Realtime

### `WebSocket /api/v1/ws`
Phase 1 plumbing: the server greets with `{"type":"welcome",...}`; `ping` is
answered with `{"type":"pong",...}`; anything else is echoed. The dashboard
connects on every page.

---

## Countries (seeded reference data)

| Method | Path | Description |
|---|---|---|
| GET | `/countries` | List supported countries (PH, plus SEA/other reference list) |

---

## Stations (monitoring stations)

| Method | Path | Description |
|---|---|---|
| GET | `/stations` | List monitoring stations |
| POST | `/stations` | Create a monitoring station (409 on duplicate code) |
| GET | `/stations/{station_id}` | Monitoring station detail |
| GET | `/stations/{station_id}/tide-stations` | Tide stations associated with the station |
| POST | `/stations/{station_id}/tide-stations/{tide_station_id}` | Associate a tide station (idempotent re-link) |
| DELETE | `/stations/{station_id}/tide-stations/{tide_station_id}` | Disassociate (idempotent) |

### `POST /api/v1/stations`
```json
{
  "code": "PH-MNL-001",
  "name": "Manila Bay Monitoring Point",
  "country_iso": "PH",
  "region": "NCR",
  "latitude": 14.5895,
  "longitude": 120.9842,
  "timezone": "Asia/Manila",
  "vertical_datum": "MLWS",
  "description": "synthetic example — replace with a real site"
}
```
`latitude` ∈ [−90, 90], `longitude` ∈ [−180, 180], timezone is validated as IANA.

---

## Tide stations

| Method | Path | Description |
|---|---|---|
| GET | `/tides/stations` | List tide stations (includes `prediction_start`/`prediction_end` coverage) |
| POST | `/tides/stations` | Create a tide station |
| GET | `/tides/stations/{code}` | Tide station detail |

### `POST /api/v1/tides/stations`
```json
{
  "code": "PH-NAMRIA-001",
  "name": "Manila Bay (synthetic example)",
  "country_iso": "PH",
  "latitude": 14.5895,
  "longitude": 120.9842,
  "timezone": "Asia/Manila",
  "data_source": "NAMRIA_CSV",
  "vertical_datum": "MLWS",
  "measurement_unit": "meters"
}
```
`measurement_unit` ∈ `meters | feet | centimeters`. Different stations' datums and units are stored per-station and are never blended or defaulted between stations.

---

## Predictions & phases

| Method | Path | Description |
|---|---|---|
| GET | `/tides/at` | SELECTED TIME: predicted tide at a chosen timestamp |
| GET | `/tides/current` | CURRENT TIME: predicted tide at the current timestamp |
| GET | `/tides/predictions?station_code=&start=&end=` | Raw points for a range |
| GET | `/tides/high-low-waters?station_code=&start=&end=` | High/low water events in a range |
| GET | `/tides/chart?station_code=&start=&end=&method=` | Chart points + events + coverage (24/48 h, 7 d) |

Query parameters: `station_code`, `timestamp` (ISO-8601; naive input is interpreted in the station's IANA time zone), `method` (`pchip` default, or `linear`), `start`/`end` (ISO-8601).

### `GET /api/v1/tides/current?station_code=PH-NAMRIA-001`
```json
{
  "station_code": "PH-NAMRIA-001",
  "station_name": "Manila Bay (synthetic example)",
  "requested_timestamp": "2026-11-18T18:50:11Z",
  "local_timestamp": "2026-11-19T02:50:11+08:00",
  "server_time": "2026-11-18T18:50:11Z",
  "predicted_height_m": 1.3897,
  "unit": "meters",
  "vertical_datum": "MLWS",
  "direction": "rising",
  "quality": "interpolated",
  "quality_note": "Interpolated between samples (not an independently validated forecast).",
  "source": "NAMRIA_CSV",
  "dataset_id": 1,
  "timezone": "Asia/Manila",
  "interpolation_method": "pchip",
  "previous_event": {"timestamp": "2026-11-18T15:00:00Z", "height": 0.4009, "type": "low", "unit": "meters", "datum": "MLWS"},
  "next_event": {"timestamp": "2026-11-18T21:00:00Z", "height": 1.7991, "type": "high", "unit": "meters", "datum": "MLWS"}
}
```
`direction` ∈ `rising | falling | slack_high | slack_low | unknown`. `quality` ∈ `ok | interpolated | estimated | unknown | unavailable`.

Requests **outside the supported coverage** return `404 data_unavailable` with the `coverage_start`/`coverage_end` in the error details. The system never extrapolates silently.

### `GET /api/v1/tides/chart?station_code=PH-NAMRIA-001&start=...&end=...`
```json
{
  "station_code": "PH-NAMRIA-001",
  "start": "2026-11-18T18:50:11Z",
  "end": "2026-11-20T18:50:11Z",
  "method": "pchip",
  "count": 49,
  "coverage_start": "2026-10-31T16:00:00Z",
  "coverage_end": "2026-12-30T16:00:00Z",
  "sufficient_for_7d": true,
  "points": [
    {"timestamp": "2026-11-18T19:00:00Z", "height": 1.40, "unit": "meters", "datum": "MLWS", "quality_status": "ok"}
  ],
  "events": [
    {"timestamp": "2026-11-18T21:00:00Z", "height": 1.80, "type": "high", "unit": "meters", "datum": "MLWS"}
  ]
}
```
`sufficient_for_7d` is `true` only when the dataset covers at least 7 days.

---

## CSV import

| Method | Path | Description |
|---|---|---|
| POST | `/tides/imports/preview` | Validate a CSV and return a preview; **nothing written to the database** |
| POST | `/tides/imports` | Import a CSV; insert-only, rejected rows retained with reasons |
| GET | `/tides/imports?station_code=&limit=&offset=` | Import history (latest first) |
| GET | `/tides/imports/{import_id}` | Import detail, including rejected rows and reasons |

### Requests (`multipart/form-data`)
Fields: `file` (required), `station_code` (commit only), `name` (commit only), `timezone` (IANA, default `Asia/Manila`), `unit` (`meters | feet | centimeters`), `vertical_datum` (free text, per station/dataset), `description`, `source` (default `NAMRIA_CSV`), `column_mapping` (JSON object mapping logical fields → columns, e.g. `{"timestamp":"tide_time","height":"height_m","event_type":"event_type"}`).

Default mapping: `timestamp` → `timestamp`, `height` → `tide_height`, `station` → `station`, `event_type` → `event_type`.

Validation per row: missing timestamp, invalid date, duplicate (first wins), missing height, non-numeric height, out-of-range height, mixed naive/aware timestamps (naive rows rejected), unknown IANA time zone (422), and chronologically out-of-order rows (sorted on import, counted in the preview). Heights accept decimal comma or dot; quoted fields such as `"0,75"` parse as 0.75.

### `POST /api/v1/tides/imports/preview`
```json
{
  "station_code": null,
  "timezone": "Asia/Manila",
  "file_hash": "9f2c…",
  "columns_detected": ["timestamp", "tide_height", "event_type"],
  "column_mapping": {"timestamp": "timestamp", "height": "tide_height", "station": "station", "event_type": "event_type"},
  "total_records": 1441,
  "valid_records": 1441,
  "rejected_records": 0,
  "rejection_summary": {"by_reason": {}, "out_of_order": 0, "warnings": []},
  "rejection_samples": [],
  "coverage_start": "2026-10-31T16:00:00Z",
  "coverage_end": "2026-12-30T16:00:00Z",
  "message": "1441 valid records, 0 rejected, 0 out of order."
}
```

### `POST /api/v1/tides/imports`
Returns `201` with the same payload plus `dataset_id`, `import_id`, `status`
(`committed | partial | rejected | duplicate`), and `imported_records`.
Re-uploading a file whose SHA-256 matches an existing dataset at that station is
a no-op (`status = duplicate`, nothing overwritten). Rejected rows are stored on
the import record as `rejection_samples` (bounded to 100 entries) with reasons,
and counts in `rejection_summary`.

Every inserted row is stamped with the **station and dataset** it came from
(`tide_predictions.tide_station_id`, `tide_predictions.tide_dataset_id`).

---

## Cameras (Phase 3 — implemented)

Cameras are registered per monitoring station. Three transports are supported:
`rtsp` (network streams, auto-reconnect + stream-health tracking), `usb`
(local devices via OpenCV `VideoCapture`) and `upload` (operators drop still
frames into the platform — useful where no live feed exists). Frames are
acquired on demand or consumed by the pipeline in later phases; every frame is
quality-assessed (black / frozen / blurry / obstructed / timestamp
inconsistent) before it leaves the service.

RTSP URLs may carry credentials; they are stored but **never returned in full** —
API responses expose `rtsp_url_redacted` (scheme + host only).

| Method | Path | Description |
|---|---|---|
| GET | `/cameras` | List cameras (joined with station code/name; RTSP URLs redacted) |
| POST | `/cameras` | Register a camera (`201`). `rtsp` requires `rtsp_url`; `usb` requires `device_index`; `upload` takes neither |
| GET | `/cameras/{id}` | Camera detail |
| GET | `/cameras/{id}/status` | Health snapshot (connected/healthy/message, frames captured, last frame time). Reports this process's live connection, or `not connected` if none — it never opens hardware on its own |
| POST | `/cameras/{id}/test-connection` | Short-lived connection test: opens the stream, reads one frame, disconnects. Returns `ok`, `latency_ms`, frame size/timestamp and a quality report (`score` 0–1, `flags`) |
| GET | `/cameras/{id}/frame` | Capture and return the latest frame as `image/jpeg`. Response headers carry `X-Frame-Sequence`, `X-Frame-Captured-At`, `X-Quality-Score`, `X-Quality-Flags`. Returns `404 data_unavailable` when no frame can be acquired — never a blank or fabricated frame |
| POST | `/cameras/{id}/upload-frame` | `upload`-type cameras only. Multipart `file` (JPEG/PNG, ≤ 10 MB); the payload is decoded and validated before being stored under `DATA_DIR/uploads/cameras/{id}/` |
| GET | `/cameras/{id}/calibrations` | List calibration profiles (newest first) |
| POST | `/cameras/{id}/calibrations` | Create the next calibration version (`201`). Optional `reference_points`, `homography_matrix` (3×3), `meters_per_pixel`, `valid_from`/`valid_to`, `notes`; `set_active` (default `true`) demotes other profiles and points `cameras.active_calibration_id` at the new one |

**Camera fields:** `code` (unique), `name`, `monitoring_station_id`, `camera_type`,
`resolution_width` / `resolution_height` / `fps` (per camera — never assumed
identical), `latitude` / `longitude` (+ PostGIS `location`), `orientation_deg`,
`monitoring_region` (GeoJSON-like JSONB), `enabled`.

**Calibration profiles** are versioned per camera (`unique(camera_id, version)`)
and never shared between cameras. `is_active` marks the current profile;
`cameras.active_calibration_id` points at it.

---

## Module endpoints (phase stubs)

The following routes return **HTTP 501** with an honest payload naming the phase
that implements them (versioned API surface is stable throughout):
| Method | Path | Phase |
|---|---|---|
| GET | `/shorelines` | 4 (vision waterline) |
| GET | `/anomalies` | 6 (anomaly engine) |

---

## Conventions

- Timestamps are stored in **UTC** internally and converted to the station's local time for presentation (`local_timestamp` in phase responses).
- Distances are metres; **positive is seaward**.
- Predicted tide heights and measured sea levels are always separate fields.
- Missing values are `null`; they are never replaced with a default.
- **Interpolation is not a validated physical forecast.**
