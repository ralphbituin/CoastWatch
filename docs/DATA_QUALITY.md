# Data Quality

> How CoastWatch represents, tracks and surfaces missing, unreliable or
> mismatched data. The governing rule: **unavailable data is never interpreted
> as normal conditions.**

CoastWatch combines four data families that fail in different ways:

1. Predicted tide data
2. Measured sea level (used for validation, kept separate from predictions)
3. Camera-derived shoreline observations
4. Environmental context (waves, wind, rain, visibility)

Each family carries its own quality signal, and the pipeline degrades to an
explicit `DATA_UNRELIABLE` state rather than pretending everything is fine.

---

## 1. Quality vocabulary

| Status | Meaning |
|---|---|
| `ok` | Value comes directly from a validated source record. |
| `interpolated` | Value was interpolated between source samples (see §2). |
| `estimated` | Approximate value with a stated uncertainty. |
| `unknown` | Present but unverified; treat with caution. |
| `unavailable` | No value exists for the requested timestamp / station. |

`unknown` and `unavailable` are **distinct**: `unknown` means "we have a value
we can't vouch for"; `unavailable` means "there is no value at all".

---

## 2. Tide data

### Missing or incomplete coverage
- Every prediction request is bounded by the dataset's `coverage_start` /
  `coverage_end` (`tide_datasets`) and the station's `prediction_start` /
  `prediction_end` (`tide_stations`).
- Requests **outside** the available range return an explicit `unavailable`
  result. The interpolation layer never extrapolates silently.
- Gaps inside a dataset are reported as coverage gaps; interpolation across a
  gap is only performed when the gap is below a configurable threshold, and the
  result is labelled `interpolated`.

### Interpolation vs. forecast
- Linear and PCHIP interpolation produce an **interpolation** quality, not a
  physical forecast. This distinction is carried in `tide_predictions.quality_status`
  and shown in the UI.

### Datum and unit mismatches
- Every tide station stores its own `vertical_datum` and `measurement_unit`.
- Heights are never compared or blended across stations with different datums.
- The system never silently falls back to an unrelated tide station. If the
  requested station has no data, the answer is `unavailable`.

### Import integrity
- Every import writes a `dataset_imports` row capturing the source file, file
  hash, column mapping, record counts and a rejection summary.
- Rejected rows (invalid dates, duplicates, missing/invalid heights, time-zone
  or unit inconsistencies) are retained with reasons — never silently dropped.
- Imported datasets are **insert-only**; a dataset is never overwritten in
  place. Re-importing the same file hash is flagged, not merged silently.

---

## 3. Camera and computer-vision data

### Frame quality flags
Frames are assessed before segmentation. Flags include:

| Flag | Trigger |
|---|---|
| `black_frame` | Near-zero mean intensity |
| `frozen_frame` | Identical to previous frame(s) beyond a tolerance |
| `blurry` | Low Laplacian variance (focus measure) |
| `obstructed` | Large uniform region covering the ROI |
| `timestamp_inconsistent` | Frame time diverges from expected cadence |

`shoreline_observations.frame_quality_score` and `frame_quality_flags` record
these per observation.

### Segmentation confidence
- Each observation stores `segmentation_confidence`.
- Low confidence or a failed calibration yields a NULL `calibrated_position_m`
  rather than a fabricated meter value.

### Calibration validity
- Calibration is versioned per camera (`camera_calibrations`).
- Suspected camera movement invalidates the active calibration; observations
  taken under an invalid calibration are flagged and excluded from baseline
  fitting.

---

## 4. How quality reaches the anomaly engine

The anomaly engine must not treat bad inputs as normal. Inputs it checks:

- Tide availability and quality for the timestamp.
- Shoreline observation presence and quality.
- Segmentation confidence above a configured floor.
- A valid, current calibration.

If any required input fails, the system state becomes `DATA_UNRELIABLE`
(see `app/models/anomaly.py`), the event is persisted with `contributing_factors`
explaining why, and **no anomaly score is emitted as if it were valid**.

---

## 5. Surfacing quality in the product

- `docs/API_DOCUMENTATION.md` documents the `quality_status` fields on tide and
  observation responses.
- The dashboard's **Data reliability** indicator reflects the current state.
- `/api/v1/health/db` reports whether TimescaleDB and PostGIS are actually
  present, so environment problems are visible, not silent.
- Import history is visible so operators can see exactly what was accepted and
  what was rejected, and why.

---

## 6. Explicit non-goals / disclaimers

- Synthetic and mock data are always labelled as such (fixtures, filenames,
  docs, and UI).
- CoastWatch does not fabricate tide observations.
- CoastWatch does not claim tsunami-warning capability. An anomaly is a
  monitoring signal, not a confirmed tsunami. Official advisories (PHIVOLCS)
  remain authoritative and are recorded as `OFFICIAL_TSUNAMI_ADVISORY`.
