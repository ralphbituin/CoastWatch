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

### `GET /api/v1/health`
Liveness probe; does not touch the database.

```json
{
  "status": "ok",
  "service": "CoastWatch",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-01-01T00:00:00Z"
}
```

### `GET /api/v1/health/db`
Readiness probe. Returns `200` when reachable, `503` otherwise.

```json
{
  "status": "ok",
  "database": "postgresql",
  "reachable": true,
  "latency_ms": 1.42,
  "server_version": "16.4",
  "timescaledb_version": "2.17.2",
  "postgis_version": "3.4.2",
  "error": null
}
```

---

## Realtime

### `WebSocket /api/v1/ws`
Phase 1 provides the connection plumbing.

- On connect the server sends `{"type": "welcome", "timestamp": "..."}`.
- Sending `ping` returns `{"type": "pong", "timestamp": "..."}`.
- Any other text is echoed as `{"type": "echo", "data": "..."}`.

From Phase 7 the server pushes shoreline observations, tide updates and anomaly
events on this channel.

---

## Module endpoints (phase stubs)

The following routes exist so the API surface and frontend route map are
stable, but return **HTTP 501** until their phase lands:

| Path | Phase | Purpose |
|---|---|---|
| `GET /api/v1/stations` | 2 | Monitoring stations, tide stations, countries |
| `GET /api/v1/tides` | 2 | Tide stations, datasets, CSV import, predictions |
| `GET /api/v1/cameras` | 3 | Camera connection, status, frames |
| `GET /api/v1/shorelines` | 4 | Observations, calibrations, baselines |
| `GET /api/v1/anomalies` | 6 | Events, current state |

Stub payload:

```json
{
  "module": "cameras",
  "phase": 3,
  "implemented": false,
  "message": "The 'cameras' module is scheduled for Phase 3 and is not implemented yet.",
  "planned_endpoints": ["GET /cameras", "POST /cameras", "..."]
}
```

Each module's `planned_endpoints` lists the concrete routes that will be added.

---

## Conventions

- Timestamps are stored in **UTC** internally and converted to the station's
  local time (IANA) for presentation.
- Distances are metres; **positive is seaward**.
- Predicted tide heights and measured sea levels are always separate fields.
- Missing values are `null`; they are never replaced with a default.
