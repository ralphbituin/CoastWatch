# CoastWatch

CoastWatch is a **tide-aware shoreline monitoring and decision-support system**
for coastal communities in the Philippines, designed to extend to Southeast
Asia. It turns fixed coastal cameras into a continuous shoreline record, relates
waterline movement to tide predictions and recent shoreline behaviour, and
surfaces unusual seaward recession as a **monitoring signal** for human review.

CoastWatch is **not** a tsunami warning service and does not confirm a tsunami.
Official PHIVOLCS advisories remain the authoritative source for tsunami
information. Anomalies produced here are research-grade observations, not
detections with validated accuracy.

## Current status

| Capability | State |
|---|---|
| Foundation: FastAPI, Next.js dashboard, TimescaleDB/PostGIS, Docker, CI | Implemented |
| Tide engine: NAMRIA CSV import, interpolation, phase, high/low water, charts | Implemented |
| Camera ingestion: RTSP, USB, operator stills, video files, YouTube Live | Implemented; hardware and Live paths are environment-gated |
| Shoreline vision: SegFormer MiT-B2 ONNX, Ocean/Beach waterline, ordered segments | Implemented |
| Per-camera homography calibration and camera-movement check | Implemented |
| Tide-aware baseline (Theil–Sen) | Implemented; refuses to fit without verified, geographically appropriate tide data |
| Rolling-baseline drawdown detector | Implemented; **tide-independent and unverified** |
| Archive recordings, bounded video processing, anomaly events | Implemented prototype |
| Live capture sessions and stream recording | Implemented; autostart is opt-in |
| Test Lab: batch images, video, and live stream modes | Implemented |
| Validated tsunami detection, accuracy claims, or alerts | **Not claimed / not implemented** |
| Edge deployment (e.g. Jetson Orin Nano) | Not started |

The current dataset is deliberately small and honest about its limits: 531
image/mask pairs across 11 sites (mostly automatic SAM3 labels, with whole-site
holdouts) plus a 4.3-hour night recording of one camera. There is no verified
tide source for that footage, so its anomalies are reported as
`EXPLORATORY_ROLLING_BASELINE` — never `NORMAL` and never a tsunami.

## Project Development

CoastWatch is built one phase at a time, each verified before the next. The
original plan: import tide data, observe the waterline, learn the normal
tide–shoreline relationship, and flag drawdowns the tide cannot explain is now
implemented end to end, but the final inference step remains a prototype:

- **Foundation and data.** A versioned FastAPI backend, a Next.js dashboard, and
  a PostgreSQL/TimescaleDB + PostGIS schema with Alembic migrations and Docker
  Compose. Tide predictions, shoreline observations, anomaly events and stream
  sessions are stored as time-series.
- **Tide and cameras.** NAMRIA-format tide datasets can be previewed, validated
  and imported; predictions, interpolation, phase classification and high/low
  water events are exposed through the API. Cameras can be registered as RTSP,
  USB, uploaded stills, video files, or public YouTube Live streams, with frame
  quality checks, calibration profiles and live capture sessions.
- **Vision.** A SegFormer MiT-B2 segmenter (Beach / Ocean / Sky) runs through
  ONNX Runtime, and a class-aware extractor traces the Ocean/Beach boundary into
  ordered shoreline segments. Operator-labelled masks remain a fallback and are
  reported honestly as such.
- **Baseline and anomaly detection.** A tide-conditioned Theil–Sen baseline is
  implemented but will not fit without paired tide data. The delivered detector
  is a causal 20-minute rolling-median baseline with robust-z, cumulative
  drawdown and multi-frame persistence; it is explicitly self-referential, and
  that mode never emits `NORMAL`.
- **Delivery surfaces.** A test lab for batch images, videos and live streams, a
  recordings archive, session-level APIs, and dashboard pages for stations,
  tides, cameras, shoreline, anomalies, recordings and lab runs.

Near-term work is **validation**, not more features: verified tide stations and
datums, labelled day/night footage across sites, calibration in metres,
cross-camera generalisation, and review of thresholds against non-tsunami
events. Edge deployment remains future work. The architecture, models and
thresholds may change as that evidence arrives.

## Quick Start

### Prerequisites
Docker + Docker Compose, Python 3.11+, Node.js 20+.

### Full stack (Docker Compose)
```bash
cp .env.example .env          # adjust if needed
docker compose up --build     # db (TimescaleDB+PostGIS), backend, frontend
```
- Backend API + docs: http://localhost:8000/docs · health: http://localhost:8000/api/v1/health
- Dashboard: http://localhost:3000/dashboard
- Test Lab: http://localhost:3000/lab · Recordings: http://localhost:3000/recordings

The backend container waits for the database, applies Alembic migrations, then
starts the API. Every service has a healthcheck. The compose file mounts
`./ml/models` at `/app/models` and `./data` at `/data`.

Live capture is **opt-in** (`LIVE_CAPTURE_AUTOSTART=false` by default) so tests
and development imports never open a real stream or device. Use
`POST /api/v1/cameras/{id}/stream/start` (or the Test Lab) to start it explicitly.

### Backend (local development)
```bash
cd backend
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements-dev.txt
alembic upgrade head                               # needs a running database
uvicorn app.main:app --reload --port 8000
```

### Frontend (local development)
```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
npm run build      # production build
```

### Tests
```bash
# backend: unit + API tests (integration tests are gated)
cd backend && pytest
# with a running database:
#   $env:COASTWATCH_INTEGRATION=1   (PowerShell)  then  pytest

# segmentation pipeline and ML scripts
ml/.venv/Scripts/python.exe -m pytest ml/tests

# frontend
cd frontend && npm run typecheck
npm run test:shoreline     # geometry/segment rendering tests
npm run build
```

## Segmentation model (SegFormer MiT-B2)

The active shoreline segmenter is a three-class **SegFormer MiT-B2** ONNX
artifact (`Beach 0 / Ocean 1 / Sky 2`, 512×512, ImageNet normalization). It is
provisioned, not committed:

```powershell
# verify/export the active fine-tuned checkpoint and run the 20-frame parity gate
ml/.venv/Scripts/python.exe -m ml.src.inference.export_onnx `
  ml/checkpoints/segformer_b2_multisite/best.pt --segformer `
  --allow-different-checkpoint --version 2.1.0 `
  --config ml/configs/segformer_b2_multisite.yaml `
  --metrics ml/outputs/segformer_b2_multisite/evaluation/evaluation_report.json `
  --dataset-summary ml/datasets/coastwatch_segformer_multisite/dataset_summary.json

# the backend reads MODEL_PATH/shoreline-segmentation/{model.onnx,config.json}
# default MODEL_PATH = ./ml/models ; Docker mount = /app/models
```

Model checkpoint: `ml/checkpoints/segformer_b2_multisite/best.pt` (SHA-256
`F598031A1A5891C64EDA8F8D58BF56E8398528BDF457C9336D5741594C0C8A20`, epoch 8,
artifact version 2.1.0). The baseline v2.0.0 artifact is archived under
`ml/models/archive/segformer_b2_baseline_v2.0.0/` for rollback.

The binary water mask is Ocean only; the shoreline is the Ocean/Beach boundary
and never the Sky/Ocean horizon. If no artifact is present, observations fall
back to operator-labelled masks and are reported honestly as `DATA_UNRELIABLE`;
no placeholder prediction is shown. See
[`docs/MODEL_DOCUMENTATION.md`](docs/MODEL_DOCUMENTATION.md) and
[`ml/COASTWATCH_INTEGRATION.md`](ml/COASTWATCH_INTEGRATION.md).

## Documentation
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components, data flow, module boundaries
- [`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md) — how unreliable data is represented
- [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md) — API reference
- [`docs/MODEL_DOCUMENTATION.md`](docs/MODEL_DOCUMENTATION.md) — segmentation/anomaly models
- [`docs/DATASET_DOCUMENTATION.md`](docs/DATASET_DOCUMENTATION.md) — data sources and formats
- [`docs/PHASE7_LIVE_TIMESTAMPING.md`](docs/PHASE7_LIVE_TIMESTAMPING.md) — timestamps, live capture, YouTube, Test Lab
- [`DEVELOPMENT_PROGRESS.md`](DEVELOPMENT_PROGRESS.md) — phase-by-phase status, verification and limitations

## Repository layout

```
backend/    FastAPI app, Alembic migrations, tests, ops scripts
frontend/   Next.js 14 dashboard (App Router, TypeScript, Tailwind, Recharts)
data/       tide datasets, videos, uploads, lab-run artifacts (gitignored)
ml/         datasets, configs, manifests, models, src/ pipeline, reports/
docs/       architecture, API, model, dataset and phase documentation
```

## Development phases

| # | Phase | Status |
|---|---|---|
| 1 | Project foundation (backend, frontend, DB, Docker, CI, docs) | Complete |
| 2 | Tide prediction engine | Complete |
| 3 | Camera integration | Complete |
| 4 | Shoreline computer vision and calibration | Complete |
| 4b | Trained segmentation model wired in | Complete |
| 4c | SegFormer MiT-B2 three-class migration | Complete |
| 4d | Multi-site fine-tune deployment (artifact v2.1.0) | Complete |
| 5 | Tide-aware shoreline baseline | Prototype — needs verified tide data |
| 6 | Rolling-baseline anomaly detection | Prototype — tide-independent and unverified |
| 7 | Timestamped recording, YouTube Live, Test Lab | Experimental |
| 8 | Testing and validation | In progress |
| 9 | Edge deployment | Not started |

See [`DEVELOPMENT_PROGRESS.md`](DEVELOPMENT_PROGRESS.md) for results, known
issues and remaining tasks per phase.

## Technology stack

| Component | Technology |
|---|---|
| Backend | Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2 |
| Computer vision | OpenCV, NumPy, PyTorch (training) |
| Segmentation model | SegFormer MiT-B2 (active); U-Net/DeepLabV3+ (legacy) |
| Model inference | ONNX Runtime (FP32, CUDA with CPU fallback) |
| Data processing | NumPy, Pandas, SciPy, scikit-learn |
| Database | PostgreSQL, TimescaleDB, PostGIS |
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Recharts, Lucide |
| Communication | REST API, WebSocket |
| Live sources | RTSP, USB, video files, public YouTube Live via `yt-dlp` + FFMPEG |
| Deployment | Docker, Docker Compose |
| Optional edge device | NVIDIA Jetson Orin Nano |

## Data Sources

| Data | Source / Method |
|---|---|
| Tide Predictions and Observations | NAMRIA (Philippine stations); imported CSV datasets |
| Tide Dataset Import | Manually provided CSV files with column mapping and validation |
| Camera Footage | Existing CCTV via RTSP, USB cameras, uploaded stills and video files |
| Live Public Streams | Public/authorized YouTube Live URLs (operator-supplied, legal review required) |
| Shoreline Observations | Extracted from camera footage with the active segmentation model |
| Tsunami Information and Advisories | PHIVOLCS — reference only; no automated ingestion |

## Important Notes

- Tide predictions, actual sea-level observations, and camera-derived shoreline
  measurements are kept as separate data.
- Tide data retains its station, timestamp, time zone, units, and vertical datum.
- Missing or unreliable data is reflected in the system's data-quality status;
  nothing is silently fabricated or imputed to look normal.
- The delivered anomaly detector is **tide-independent**. It cannot produce an
  expected shoreline, a tide-adjusted residual, or a `NORMAL` classification.
- An anomaly does not confirm a tsunami. Official PHIVOLCS advisories remain the
  authoritative source for tsunami warnings.
- YouTube Live provides wall-clock receipt times, not true per-frame capture
  timestamps. Timestamp provenance is stored and displayed with every row.
- No outbound alerts or notifications are implemented; anomalies are reviewed in
  the dashboard and API.
- The stack and development direction may be adjusted as evidence and datasets
  improve.

## Team Members

| Name | Username |
|---|---|
| Bituin, Ralph Ivan | @ralphbituin |
| Arago, Mary Jobine | TBD |
| Carubio, Ma. Bridget Faye | TBD |
| Jamias, Lyzette | TBD |
| Juanites, Rein Lander | TBD |
| Mendoza, Arriane Mae | TBD |
| Perez, Samantha Isabel | TBD |
| Reyes, Kent Rowen | TBD |
| Sanchez, Trixie Feigh | TBD |
