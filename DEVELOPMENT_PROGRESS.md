# Development Progress

CoastWatch is developed **one phase at a time**. Each phase is implemented,
tested, documented and verified before the next begins. This file is the
running log.


---

## Phase status

| Phase | Title | Status |
|---|---|---|
| 1 | Project foundation (backend, frontend, DB, Docker, CI, docs) | [x] complete |
| 2 | Tide prediction engine | [x] complete |
| 3 | Camera integration | [ ] not started |
| 4 | Shoreline computer vision | [ ] not started |
| 5 | Tide-aware shoreline baseline | [ ] not started |
| 6 | Anomaly detection engine | [ ] not started |
| 7 | Monitoring dashboard | [ ] not started |
| 8 | Testing and validation | [ ] not started |
| 9 | Edge deployment | [ ] not started |

---

## Phase 1 — Project Foundation

**Goal:** a complete, reproducible development environment.

### Implemented
- FastAPI backend with a versioned API under `/api/v1`.
- `GET /api/v1/health` (liveness) and `GET /api/v1/health/db` (readiness,
  reports PostgreSQL / TimescaleDB / PostGIS versions).
- `WebSocket /api/v1/ws` with connect/ping-pong plumbing and a broadcast helper.
- Modular architecture: `api/`, `core/`, `database/`, `models/`, `schemas/`,
  `services/`, and `main.py`.
- Structured logging (JSON or console) + request-logging middleware with
  `X-Request-ID` propagation.
- Consistent error envelope for all failures.
- `app/api/deps.py` for DB sessions and shared dependencies.
- SQLAlchemy 2.0 models for all 13 required entities, with TimescaleDB
  hypertables and PostGIS geography columns.
- Initial Alembic migration (extensions + tables + hypertables).
- Next.js 14 (App Router, TypeScript, Tailwind) dashboard with all six module
  routes mirroring the backend: `dashboard`, `stations`, `tides`, `cameras`,
  `shoreline`, `anomalies`.
- Docker Compose: `timescale/timescaledb-ha:pg16` + backend + frontend, each
  with healthchecks; dependents wait for `service_healthy`.
- `.env.example` + `pydantic-settings`; no secrets in code.
- GitHub Actions CI: backend `ruff check` + `ruff format --check` + `mypy` +
  `pytest` (with a real TimescaleDB service and applied migrations), and
  frontend `lint` + `build`.
- Docs: `ARCHITECTURE.md` (data flow + module boundaries), `DATA_QUALITY.md`,
  `API_DOCUMENTATION.md`, `MODEL_DOCUMENTATION.md`, `DATASET_DOCUMENTATION.md`.
- `.gitkeep` files for every intentionally-empty tracked directory.

### Verification (run 2026-11-19 on the primary dev machine)
- `ruff check .` → ✅ All checks passed
- `ruff format --check .` → ✅ 72 files formatted
- `mypy app` → ✅ no issues found in 57 source files
- `pytest` → ✅ 20 passed, 3 skipped (integration gated by default)
- `COASTWATCH_INTEGRATION=1 pytest` → ✅ 23 passed (all against the live TimescaleDB service)
- `alembic upgrade head` (from empty DB) → ✅ applied; `alembic downgrade base` → ✅ clean rollback
- `alembic check` → ✅ No new upgrade operations detected
- Hypertables confirmed present: `tide_predictions`, `shoreline_observations`,
  `system_health_logs`, `anomaly_events`, `environmental_observations`
- `docker compose up --build` → ✅ db (healthy), backend (healthy), frontend (healthy)
- Live probes → ✅ `/api/v1/health` ok; `/api/v1/health/db` reports TimescaleDB 2.32.0 + PostGIS 3.7.0;
  frontend `/dashboard` HTTP 200; WebSocket connect + `ping→pong` handshake ok
- `npm run lint` → ✅ no ESLint warnings/errors; `npm run build` → ✅ success
- CI (`.github/workflows/ci.yml`) mirrors these steps exactly (backend: ruff check,
  ruff format --check, mypy, `alembic upgrade head`, pytest with TimescaleDB service;
  frontend: `npm ci`, lint, build). The workflow has not yet been executed on GitHub —
  it will run on the next push to `main`.

### Known issues / limitations
- Module routes for Phases 2–6 return HTTP 501 by design (no fake behaviour).
- `torch`/ONNX Runtime are intentionally not installed in Phase 1 (see
  `requirements-ml.txt`).
- Only automated tests for the foundation are present; domain tests arrive with
  their phases.

### Remaining tasks (next phases)
- Phase 2: tide providers, CSV import, interpolation, phase classification,
  tide dashboard.

---

## Phase 2 — Tide Prediction Engine

**Goal:** a complete tide prediction system before anomaly detection. Any
timestamp within the imported dataset coverage can be predicted.

### Implemented
- **Typed `TideProvider` interface** (`app/services/tides/base.py`) with value
  objects carrying datum/unit/quality on every height.
- **`NAMRIATideProvider`** (`app/services/tides/namria.py`) — list stations,
  CSV import (preview + commit), range predictions, high/low water detection,
  PCHIP/linear interpolation, rising/falling/slack classification, previous/next
  events; full coverage lookup; insert-only semantics (existing
  `(station, timestamp)` predictions are rejected as `existing_record`, never
  overwritten).
- **CSV import** (`csv_import.py`) — configurable column mapping (timestamp,
  height, optional station, optional tidal event type); validates missing/invalid
  timestamps, duplicates, missing/non-numeric/out-of-range heights,
  time-zone consistency (naive vs aware vs mixed, varying UTC offsets reported),
  unit metadata, vertical-datum metadata, IANA names; chronological ordering
  validated and sorted on import (reported). UTF-8-BOM tolerated; decimal comma
  supported; coverage reported; **rejected rows retained with reasons**
  (`dataset_imports.rejection_samples`, bounded to 100) — nothing is discarded
  silently.
- **Import history** — every import (even duplicates and empty imports) writes
  `dataset_imports`; file hash (`sha256`) recorded; same-file re-upload is a
  no-op (`status=duplicate`).
- **Station management APIs** — countries seeded (PH + 11 others, migration
  0002); create/list/detail monitoring stations and tide stations; link/unlink
  tide stations from monitoring stations; datums/units/timezones display.
- **Tide-phase APIs** — `/tides/current`, `/tides/at` (SELECTED TIME),
  `/tides/predictions`, `/tides/high-low-waters`, `/tides/chart` (24 h / 48 h / 7 d
  with high/low markers, coverage and 7-day sufficiency flag).
- **Dashboard** — Tides page now fully interactive: station selector, CURRENT
  TIME and SELECTED TIME cards (height, phase, events, quality/datum/source),
  range + method toggles, high/low event markers and a live chart; Import panel
  with CSV upload → preview → commit; import history list; Stations page:
  countries, create monitoring/tide stations, link/unlink associations.
- **Quality handling** — out-of-coverage timestamps return `404
  data_unavailable` with coverage details; interpolation is explicitly labelled
  "not an independently validated forecast"; exact samples report `ok`,
  interpolation reports `interpolated`.

### New files
`app/schemas/{stations,tides}.py`, `app/api/routes/{tides,stations,countries}.py`,
`app/services/tides/{csv_import,interpolation,tide_phase,namria}.py`,
`alembic/versions/7bcefb290c3d_add_rejection_samples_to_dataset_imports.py`,
`tests/integration/test_tide_engine.py`, `tests/unit/test_interpolation.py`,
`tests/unit/test_tide_phase.py`, `tests/unit/test_csv_import.py`,
`frontend/lib/{tides,stations}.ts`, `frontend/features/tides/{TidesView,TideChart,TidalSnapshot,ImportPanel}.tsx`,
`frontend/features/stations/StationsView.tsx`. Modified:
`app/models/tide.py` (`rejection_samples`), `app/core/{config,exceptions}.py`,
`app/api/deps.py`, `app/schemas/common.py`, `app/services/tides/base.py`,
api docs, ARCHITECTURE.md, DATASET_DOCUMENTATION.md, CI unchanged.

### Tests executed
`ruff check`, `ruff format --check`, `mypy app`, `pytest` (unit),
`COASTWATCH_INTEGRATION=1 pytest` (against live docker DB), live HTTP probes of
the new routes, `npm run lint`, `npm run build`, docker compose rebuild
(backend+frontend), `alembic upgrade head` (0002).

### Test results
- Lint/format/mypy ✅ 69 tests (52 unit + 17 integration) all passing.
- End-to-end: create station → preview CSV → commit (1441 rows) →
  `/tides/at` matches expected sine values exactly → out-of-coverage → 404
  `data_unavailable` → chart 25/24h points → high/low events alternate →
  re-import returns `duplicate` with zero overwrites.

### Known issues / notes
- `/tides/*` predictions interpolate between imported samples; this is **not** a
  validated tidal forecast. PCHIP is preferred; `linear` available via `method=`.
- The local dev database contains a clearly-labelled **synthetic** demonstration
  dataset (`TEST-SYNTH-001`) used to exercise the pipeline; replace it with an
  official NAMRIA CSV before any real-world claim (see DATASET_DOCUMENTATION.md).
- Committing is insert-only; there is no "delete dataset" endpoint by design.

### Remaining tasks
- Phase 3 (Camera integration) requires explicit approval.

---

## Reporting format (each phase)

1. Implemented features
2. Created and modified files
3. Tests executed
4. Test results
5. Known issues
6. Remaining tasks
7. Instructions to run the application
