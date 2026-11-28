# DEMO MODE - drawdown injection on recorded runs

**Every screen, chart, event and saved result in this page is a simulation:
SIMULATION - not real data. It is a position-level injection replay on recorded
noise, not a real event, not a tsunami validation, and not a real-world
sensitivity or false-alarm claim.**

## Enable (default is off)

Backend (repository `.env` or shell):

```powershell
$env:DEMO_INJECTION_ENABLED = "true"
```

Frontend:

```powershell
# frontend/.env.local
NEXT_PUBLIC_DEMO_INJECTION_ENABLED=true
```

With the flag off, every `/api/v1/demo/injection/*` endpoint returns 404 and the
sidebar link is hidden.

## Run

```powershell
# terminal 1 - backend (from the repository root)
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = "backend"
python -m uvicorn app.main:app --port 8000

# terminal 2 - frontend
cd frontend
npm run dev
```

### Docker deployment

The compose file exposes the demo flag and mounts `ml/reports` read-only so
the container can read recorded runs:

```powershell
$env:DEMO_INJECTION_ENABLED = "true"
$env:NEXT_PUBLIC_DEMO_INJECTION_ENABLED = "true"
$env:LIVE_CAPTURE_AUTOSTART = "false"   # demo needs no camera worker
docker compose up --build -d
```

Open <http://localhost:3000/demo/injection>. The camera worker, stream and
previews are not used. The demo only reads the CSV tables under
`ml/reports/live_eval/<run>/tables/`; it never writes to them, the database or
any live table. It works with no internet and with the camera worker stopped.

Optional labels/notes: `ml/reports/live_eval/demo/recordings.json`, keyed by run
directory name:

```json
{
  "kohsamui_4h_20261128T014103Z": {
    "label": "Koh Samui 4 h",
    "notes": "Daytime scene, 10 s samples, 720p."
  }
}
```

## 3-minute talk track

1. Pick **Koh Samui**. Point at the profile card: real noise, effective gate,
   parity pass, valid starts.
2. **No injection (real series)**: no synthetic excursion; show that any event
   would belong to the recorded series.
3. **Slow retreat**: 600 s ramp; state that the causal baseline absorbs it and
   the run is missed.
4. **Below the gate**: half the median gate; missed because cumulative drawdown
   never reaches the effective gate.
5. **Borderline**: about the median gate; the rate is not 100%, so do not call
   it guaranteed.
6. **Fast retreat**: p95 gate, 60 s ramp; caught; latency is shown in samples
   and seconds.
7. **Landward move**: negative sign; ignored (no seaward event).
8. Open **Compare recordings** with the same amplitude and ramp. The bars show
   each recording's own noise and effective gate. Say plainly that recordings
   are not combined and that one run is not a real-world rate.

## Fallback

- **Replay saved demo** loads `frontend/public/demo/injection-saved.json`
  (pre-computed from the saved Koh Samui offline tables). It starts no detector,
  needs no backend computation and carries the same banner.
- Backup screenshots (needs Playwright once, while online):

```powershell
cd frontend
npm install --no-save playwright
npx playwright install chromium
node scripts/demo_screenshots.mjs
```

Output goes to `frontend/demo-screenshots/`. The script follows the same preset
order and the comparison panel.

## Claim boundary

- Position-level injection on recorded noise only.
- Segmentation behaviour during a real retreat (wet sand, scene change) is not
  tested.
- Each result applies to that recording's scene and conditions only.
- Recordings are never combined into one rate.
- Not a tsunami validation and no real-world sensitivity or false-alarm claim.
- The detector is the production rolling-baseline implementation; its code,
  thresholds, floors, config values, model weights, polygons and camera records
  are unchanged.
