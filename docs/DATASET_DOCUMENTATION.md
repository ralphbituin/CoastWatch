# Dataset Documentation

> What data CoastWatch uses, where it comes from, how it is stored, and how
> mock/synthetic data is kept clearly separate from real data.

## 1. Data families

| Family | Source | Storage | Notes |
|---|---|---|---|
| Predicted tide heights | NAMRIA (manual CSV import); CSV from official tide tables | `tide_datasets`, `tide_predictions` | Insert-only; datums kept per station |
| Measured sea level | Historical station observations (when available) | separate from predictions | Never merged with predictions |
| Shoreline observations | Extracted from camera footage | `shoreline_observations` | Include confidence + frame quality |
| Environmental context | Wave/wind/rain/visibility feeds (optional) | `environmental_observations` | Context for the baseline model |
| Camera media | RTSP / USB / uploaded files | filesystem under `data/` | Sample media in `data/sample_*` |
| Official advisories | PHIVOLCS | recorded as `OFFICIAL_TSUNAMI_ADVISORY` events | Authoritative; never inferred by the system |

## 2. Directory conventions

```
data/
├── tide_datasets/   # operator-provided tide CSVs (gitignored)
├── sample_videos/   # small sample clips for offline evaluation
├── sample_images/   # small sample frames
└── processed/       # derived/processed artifacts

ml/
├── datasets/        # training/eval datasets (gitignored, .gitkeep kept)
├── models/          # trained artifacts (gitignored, .gitkeep kept)
├── training/        # training scripts
└── evaluation/      # evaluation scripts
```

Bulk data and model binaries are **gitignored**; every intentionally-empty
tracked directory keeps a `.gitkeep` so a fresh clone works.

## 3. Tide data format (target of the Phase 2 importer)

CSV import supports configurable column mapping for:

- **timestamp** (required)
- **tide height** (required)
- **station identifier** (optional if the station is fixed for the import)
- **tidal event type** (optional: high / low)

Validation covers: missing/invalid timestamps, duplicates, missing or
non-numeric heights, time-zone and unit consistency, vertical-datum metadata,
and chronological ordering. A preview is shown before commit; rejects are
retained with reasons in `dataset_imports`.

> NAMRIA does not expose a documented public API that CoastWatch depends on.
> If a dataset arrives as PDF, it must go through an explicit parser/conversion
> and validation step — it is never silently treated as CSV.

## 4. Mock and synthetic data policy

- **Always labelled.** Mock/synthetic data lives under clearly named fixtures
  (e.g. `tests/.../synthetic_*`) or is marked in filenames, metadata and the UI.
- The Phase 1 tide chart is explicitly labelled *synthetic placeholder*.
- No synthetic record may populate `tide_predictions` as if it were an imported
  real dataset; synthetic data is for tests and demos only.
- CoastWatch never fabricates tide observations.

## 5. Provenance fields

Every imported dataset records: station, source, unit, vertical datum, coverage
start/end, record count, source filename and a **file hash**. Every import event
records the column mapping, counts, rejection summary and timestamp
(`dataset_imports`). This makes each number traceable to a file.

## 6. Time and units

- All timestamps are stored in **UTC** and converted to the station's IANA time
  zone for display (default `Asia/Manila`).
- Heights are stored with their unit and vertical datum; conversion between
  datums is explicit and never implicit.
