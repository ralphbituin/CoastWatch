# Development Progress

CoastWatch is developed **one phase at a time**. Each phase is implemented,
tested, documented and verified before the next begins. This file is the
running log.

Legend: ✅ complete · 🔄 in progress · ⏳ not started

---

## Phase status

| Phase | Title | Status |
|---|---|---|
| 1 | Project foundation (backend, frontend, DB, Docker, CI, docs) | ✅ complete |
| 2 | Tide prediction engine | ⏳ not started |
| 3 | Camera integration | ⏳ not started |
| 4 | Shoreline computer vision | ⏳ not started |
| 5 | Tide-aware shoreline baseline | ⏳ not started |
| 6 | Anomaly detection engine | ⏳ not started |
| 7 | Monitoring dashboard | ⏳ not started |
| 8 | Testing and validation | ⏳ not started |
| 9 | Edge deployment | ⏳ not started |

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

⏳ Not started. Requires explicit approval of Phase 1.

---

## Reporting format (each phase)

1. Implemented features
2. Created and modified files
3. Tests executed
4. Test results
5. Known issues
6. Remaining tasks
7. Instructions to run the application
