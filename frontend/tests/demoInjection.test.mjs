/**
 * DEMO MODE frontend guard tests.
 *
 * The banner/watermark must be unconditional, a recording switch must clear
 * every simulation view, and a limited/not-replayable recording must disable
 * starting a simulation with a visible reason.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { transpileModule } from "typescript";

const sourcePath = fileURLToPath(
  new URL("../features/demo/demoSeries.ts", import.meta.url),
);
const compiled = transpileModule(readFileSync(sourcePath, "utf8"), {
  compilerOptions: { module: 1, target: 7 },
}).outputText;
const module = { exports: {} };
new Function("exports", "module", compiled)(module.exports, module);
const {
  DEMO_BANNER,
  DEMO_WATERMARK,
  chartPoints,
  describeAmplitude,
  stateSegments,
  expectedText,
  statusLabel,
} = module.exports;

const read = (relative) =>
  readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");

test("simulation banner and watermark text are exact", () => {
  assert.equal(DEMO_BANNER, "SIMULATION - not real data");
  assert.equal(DEMO_WATERMARK, "SIMULATION - not real data");
});

test("chart points break the line at a gap larger than 45 s", () => {
  const base = "2026-11-28T02:00:00+00:00";
  const samples = [
    { time: base, real: 1, injected: 2, baseline: 1, z: 0, ck: 0, gate: 100, state: "MONITORING", classification: "X" },
    { time: "2026-11-28T02:00:40+00:00", real: 2, injected: 3, baseline: 1, z: 0, ck: 0, gate: 100, state: "MONITORING", classification: "X" },
    { time: "2026-11-28T02:01:40+00:00", real: 3, injected: 4, baseline: 1, z: 0, ck: 0, gate: 100, state: "MONITORING", classification: "X" },
  ];
  const points = chartPoints(samples);
  assert.equal(points.length, 4);
  assert.equal(points[2].real, null);
  assert.equal(points[3].real, 3);
});

test("amplitude is shown in px and percent of the selected frame height", () => {
  assert.equal(describeAmplitude(41.666666, 720), "300 px = 41.7% of 720 px");
  assert.equal(describeAmplitude(10, 1080), "108 px = 10.0% of 1080 px");
});

test("state strip groups consecutive states and labels are plain", () => {
  const samples = [
    { state: "MONITORING" },
    { state: "MONITORING" },
    { state: "CANDIDATE" },
  ];
  assert.deepEqual(stateSegments(samples), [
    { state: "MONITORING", count: 2 },
    { state: "CANDIDATE", count: 1 },
  ]);
  assert.equal(statusLabel("not_replayable"), "Not replayable");
  assert.equal(statusLabel("limited"), "Limited data");
});

test("preset expected line always shows rate and n", () => {
  assert.match(
    expectedText({ available: true, detection_rate: 1, n_in_bounds: 30, source: "saved offline" }),
    /100% detected, n=30 starts/,
  );
  assert.match(
    expectedText({ available: false, note: "still computing" }),
    /still computing/,
  );
});

test("page and sidebar are gated by the public flag", () => {
  const page = read("../app/demo/injection/page.tsx");
  assert.match(page, /NEXT_PUBLIC_DEMO_INJECTION_ENABLED/);
  assert.match(page, /notFound\(\)/);
  const sidebar = read("../components/Sidebar.tsx");
  assert.match(sidebar, /NEXT_PUBLIC_DEMO_INJECTION_ENABLED/);
  assert.match(sidebar, /\/demo\/injection/);
});

test("banner and chart watermark are unconditional, and switching resets", () => {
  const view = read("../features/demo/DemoInjectionView.tsx");
  assert.match(view, /DEMO_BANNER/);
  assert.match(view, /setSamples\(\[\]\)/);
  assert.match(view, /setResult\(null\)/);
  assert.match(view, /Simulations are disabled for this recording/);
  const charts = read("../features/demo/DemoCharts.tsx");
  assert.match(charts, /DEMO_WATERMARK/);
  const compare = read("../features/demo/ComparePanel.tsx");
  assert.match(compare, /DEMO_WATERMARK/);
  const summary = read("../features/demo/SummaryPanel.tsx");
  assert.match(summary, /DEMO_WATERMARK/);
});

test("no real-data export path is offered by the demo page", () => {
  const view = read("../features/demo/DemoInjectionView.tsx");
  assert.doesNotMatch(view, /export (CSV|JSON|PDF)/i);
  assert.doesNotMatch(view, /download/i);
});
