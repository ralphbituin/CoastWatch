# CoastWatch

A tide-aware shoreline monitoring system that uses computer vision and tide data to observe shoreline movement and identify unusual coastal water recession. The project aims to support early anomaly detection, particularly for possible tsunami-related drawdown events. It is intended as a monitoring and decision-support tool, not a replacement for official tsunami warning systems.

## Project Development

CoastWatch will initially focus on integrating tide data and camera feeds from selected coastal areas in the Philippines, with the possibility of expanding to other parts of Southeast Asia. The initial implementation will use existing CCTV cameras through RTSP and USB cameras for testing. Tide information will come from available NAMRIA resources and manually imported CSV datasets.

The system will process camera footage using computer vision to identify the waterline and measure shoreline movement. These observations will be compared with predicted tide levels for the corresponding timestamps. By learning the expected shoreline position under normal tidal conditions, the system can identify unusual recession patterns and calculate indicators such as shoreline displacement, retreat velocity, and anomaly scores.

The planned development will cover the backend and database foundation, tide-data processing, camera integration, shoreline detection, tide-aware baseline modeling, anomaly detection, and a dashboard for monitoring and historical analysis. Testing and validation will follow as the system develops. These stages serve as an initial direction rather than a fixed implementation plan. The architecture, models, and features may change based on available datasets, testing results, and project requirements.

## Quick Start

### Prerequisites
Docker + Docker Compose, Python 3.11+, Node.js 20+.

### Full stack (Docker Compose)
```bash
cp .env.example .env          # adjust if needed
docker compose up --build     # db (TimescaleDB+PostGIS), backend, frontend
```
- Backend API + docs: http://localhost:8000/docs · health: http://localhost:8000/api/v1/health
- Frontend dashboard: http://localhost:3000/dashboard

The backend container waits for the database, applies Alembic migrations, then
starts the API. Every service has a healthcheck.

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
cd backend && pytest                       # unit tests (integration tests skip unless enabled)
# with a running database:
#   $env:COASTWATCH_INTEGRATION=1   (PowerShell)  then  pytest
```

### Documentation
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components, data flow, module boundaries
- [`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md) — how unreliable data is represented
- [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md) — API reference
- [`docs/MODEL_DOCUMENTATION.md`](docs/MODEL_DOCUMENTATION.md) — segmentation/anomaly models
- [`docs/DATASET_DOCUMENTATION.md`](docs/DATASET_DOCUMENTATION.md) — data sources and formats
- [`DEVELOPMENT_PROGRESS.md`](DEVELOPMENT_PROGRESS.md) — phase-by-phase status

## Repository layout

```
backend/    FastAPI app, Alembic migrations, tests, ops scripts
frontend/   Next.js 14 dashboard (App Router, TypeScript, Tailwind)
data/       tide datasets, sample images/videos, processed artifacts
ml/         datasets, models, training, evaluation
docs/       architecture, API, model and dataset documentation
```

## Development phases

Development proceeds one phase at a time, each verified before the next:

1. Project foundation ← **current**
2. Tide prediction engine
3. Camera integration
4. Shoreline computer vision
5. Tide-aware shoreline baseline
6. Anomaly detection engine
7. Monitoring dashboard
8. Testing and validation
9. Edge deployment

See [`DEVELOPMENT_PROGRESS.md`](DEVELOPMENT_PROGRESS.md) for status.

## Planned Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI, Pydantic |
| Computer Vision | OpenCV, PyTorch |
| Segmentation Models | U-Net, DeepLabV3+ |
| Model Inference | ONNX Runtime |
| Data Processing | NumPy, Pandas, SciPy |
| Anomaly Detection | Scikit-learn |
| Database | PostgreSQL, TimescaleDB, PostGIS |
| Frontend | Next.js, TypeScript |
| UI Styling | Tailwind CSS, Lucide React |
| Charts | Recharts or Plotly |
| Communication | REST API, WebSockets |
| Deployment | Docker, Docker Compose |
| Optional Edge Device | NVIDIA Jetson Orin Nano |

## Data Sources

| Data | Source / Method |
|---|---|
| Tide Predictions and Observations | NAMRIA |
| Tide Dataset Import | Manually provided CSV files |
| Camera Footage | Existing CCTV via RTSP and USB cameras |
| Shoreline Observations | Extracted from camera footage |
| Tsunami Information and Advisories | PHIVOLCS |

## Important Notes

- Tide predictions, actual sea-level observations, and camera-derived shoreline measurements will be treated as separate data.
- Tide data must retain its station, timestamp, time zone, units, and vertical datum where available.
- Missing or unreliable data must be reflected in the system's data-quality status.
- An anomaly does not confirm a tsunami. Official PHIVOLCS advisories remain the authoritative source for tsunami warnings.
- The proposed stack and development direction may be adjusted as the project progresses.

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