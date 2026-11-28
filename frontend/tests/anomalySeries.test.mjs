/**
 * Pure series tests for the tide/anomaly chart.
 *
 * A missing observation must break the line; a time gap larger than the
 * sampling interval must also break it. Tide and position have independent
 * bounds so they can use separate axes.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { transpileModule } from "typescript";

const sourcePath = fileURLToPath(
  new URL("../features/anomalies/anomalySeries.ts", import.meta.url),
);
const source = readFileSync(sourcePath, "utf8");
const compiled = transpileModule(source, {
  compilerOptions: { module: 1, target: 7 },
}).outputText;
const module = { exports: {} };
new Function("exports", "module", compiled)(module.exports, module);
const {
  toAnomalyPoints,
  splitRuns,
  finiteBounds,
  scaleToHeight,
  classificationCounts,
  buildAnomalyChartData,
  anomalyStateSegments,
  anomalyStateColor,
} = module.exports;

function event(overrides) {
  return {
    id: 1,
    detected_at: "2026-11-19T14:30:00+00:00",
    monitoring_station_id: 1,
    camera_id: 1,
    recording_id: "rec-1",
    frame_index: 0,
    state: "OBSERVED",
    classification: "EXPLORATORY_MOVEMENT_ONLY",
    anomaly_score: null,
    baseline_status: "unavailable",
    tide_adjusted: false,
    tide_data_available: false,
    tide_source: "unavailable",
    tide_height_m: null,
    observed_position_px: 800,
    expected_position_px: null,
    residual_px: null,
    robust_z: null,
    movement_rate_px_s: null,
    persistence_count: 0,
    quality_status: "exploratory_no_baseline",
    reason: "no baseline",
    explanation: null,
    contributing_factors: null,
    acknowledged: false,
    created_at: "2026-11-19T14:30:00+00:00",
    ...overrides,
  };
}

test("events are sorted and nulls preserved", () => {
  const points = toAnomalyPoints([
    event({ id: 2, detected_at: "2026-11-19T14:31:00+00:00", observed_position_px: null }),
    event({ id: 1 }),
  ]);
  assert.equal(points[0].id, 1);
  assert.equal(points[1].id, 2);
  assert.equal(points[1].observed, null);
  assert.equal(points[0].observed, 800);
});

test("missing values split runs instead of joining neighbours", () => {
  const points = toAnomalyPoints([
    event({ id: 1, detected_at: "2026-11-19T14:30:00+00:00", observed_position_px: 800 }),
    event({ id: 2, detected_at: "2026-11-19T14:30:30+00:00", observed_position_px: null }),
    event({ id: 3, detected_at: "2026-11-19T14:31:00+00:00", observed_position_px: 810 }),
  ]);
  const runs = splitRuns(points, (point) => point.observed, 90_000);
  assert.equal(runs.length, 2);
  assert.deepEqual(runs[0].map((point) => point.id), [1]);
  assert.deepEqual(runs[1].map((point) => point.id), [3]);
});

test("a time gap splits runs", () => {
  const points = toAnomalyPoints([
    event({ id: 1, detected_at: "2026-11-19T14:30:00+00:00" }),
    event({ id: 2, detected_at: "2026-11-19T14:35:00+00:00" }),
  ]);
  const runs = splitRuns(points, (point) => point.observed, 90_000);
  assert.equal(runs.length, 2);
});

test("tide can be summarised independently of position", () => {
  const points = toAnomalyPoints([
    event({ id: 1, tide_data_available: true, tide_height_m: 0.4 }),
    event({ id: 2, tide_data_available: false, tide_height_m: 99 }),
  ]);
  const tideBounds = finiteBounds(points.map((point) => point.tide));
  assert.deepEqual(tideBounds, { min: -0.6, max: 1.4 });
  assert.equal(points[1].tide, null);
});

test("scaleToHeight puts larger values higher", () => {
  const high = scaleToHeight(10, { min: 0, max: 10 }, 100, 10);
  const low = scaleToHeight(0, { min: 0, max: 10 }, 100, 10);
  assert.ok(high < low);
});

test("classification counts are transparent", () => {
  const points = toAnomalyPoints([
    event({ id: 1, classification: "DATA_UNRELIABLE" }),
    event({ id: 2, classification: "DATA_UNRELIABLE" }),
    event({ id: 3, classification: "UNUSUAL_SHORELINE_POSITION" }),
  ]);
  assert.deepEqual(classificationCounts(points), {
    DATA_UNRELIABLE: 2,
    UNUSUAL_SHORELINE_POSITION: 1,
  });
});

test("the anomaly chart breaks lines at 45 s like the demo chart", () => {
  const points = toAnomalyPoints([
    event({ id: 1, detected_at: "2026-11-19T14:30:00+00:00", observed_position_px: 800 }),
    event({ id: 2, detected_at: "2026-11-19T14:30:30+00:00", observed_position_px: 801 }),
    event({ id: 3, detected_at: "2026-11-19T14:31:30+00:00", observed_position_px: 802 }),
  ]);
  const rows = buildAnomalyChartData(points, 45_000);
  assert.equal(rows.length, 4);
  assert.equal(rows[2].observed, null);
  assert.equal(rows[3].observed, 802);
});

test("the detector state strip groups consecutive states", () => {
  const points = toAnomalyPoints([
    event({ id: 1, state: "MONITORING" }),
    event({ id: 2, state: "MONITORING", detected_at: "2026-11-19T14:30:10+00:00" }),
    event({ id: 3, state: "CANDIDATE", detected_at: "2026-11-19T14:30:20+00:00" }),
  ]);
  assert.deepEqual(anomalyStateSegments(points), [
    { state: "MONITORING", count: 2 },
    { state: "CANDIDATE", count: 1 },
  ]);
  assert.equal(anomalyStateColor("EVENT"), "#dc2626");
});

test("effective gate is used only when the stored event exposes it", () => {
  const without = toAnomalyPoints([event({ id: 1 })]);
  assert.equal(without[0].effectiveGate, null);
  const withSigma = toAnomalyPoints([
    event({
      id: 2,
      contributing_factors: { sigma_k_px: 20, drawdown_floor_px: 100 },
    }),
  ]);
  assert.equal(withSigma[0].effectiveGate, 100);
  const withGate = toAnomalyPoints([
    event({ id: 3, contributing_factors: { effective_gate_px: 175 } }),
  ]);
  assert.equal(withGate[0].effectiveGate, 175);
});
