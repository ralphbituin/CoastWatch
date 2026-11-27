/**
 * Rendering-pipeline test for shoreline segments.
 *
 * There is no browser test runner in this project. This test transpiles the
 * pure geometry module with the TypeScript compiler that is already a dev
 * dependency, then checks the exact polylines the component emits: one SVG
 * <polyline> per segment and nothing for an empty observation.
 *
 * Run: npm run test:shoreline
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { transpileModule } from "typescript";

const sourcePath = fileURLToPath(
  new URL("../features/shoreline/waterlineGeometry.ts", import.meta.url),
);
const source = readFileSync(sourcePath, "utf8");
const compiled = transpileModule(source, {
  compilerOptions: { module: 1, target: 7 }, // CommonJS, ES2020
}).outputText;

const module = { exports: {} };
new Function("exports", "module", compiled)(module.exports, module);
const { observationSegments, decimateSegment, polylinePoints, waterlineBounds } = module.exports;

function renderPolylines(detail) {
  const segments = observationSegments(detail);
  return segments.map(
    (segment) =>
      `<polyline points="${polylinePoints(decimateSegment(segment))}" fill="none" stroke="#22d3ee" />`,
  );
}

test("historical two-segment rows render only the longest single line", () => {
  const detail = {
    shoreline_segments: [
      [
        [10, 40],
        [20, 41],
        [30, 42],
      ],
      [
        [100, 60],
        [110, 61],
      ],
    ],
    shoreline_geometry: [
      [10, 40],
      [20, 41],
      [30, 42],
      [100, 60],
      [110, 61],
    ],
  };
  const polylines = renderPolylines(detail);
  assert.equal(polylines.length, 1);
  assert.match(polylines[0], /^<polyline points="10\.0,40\.0 20\.0,41\.0 30\.0,42\.0"/);
  // The second historical branch is discarded, never drawn as a separate path.
  assert.doesNotMatch(polylines[0], /100\.0,60\.0/);
});

test("empty result renders nothing", () => {
  assert.deepEqual(renderPolylines({ shoreline_segments: [], shoreline_geometry: null }), []);
  assert.deepEqual(renderPolylines(null), []);
  assert.deepEqual(renderPolylines({ shoreline_segments: [[[1, 2]]] }), []);
});

test("legacy flat geometry still renders as one fallback path", () => {
  const polylines = renderPolylines({
    shoreline_geometry: [
      [5, 10],
      [6, 11],
    ],
  });
  assert.equal(polylines.length, 1);
  assert.match(polylines[0], /5\.0,10\.0 6\.0,11\.0/);
});

test("viewBox follows image coordinates and decimation keeps the last point", () => {
  const segment = Array.from({ length: 1000 }, (_, index) => [index, 500 - index]);
  const reduced = decimateSegment(segment, 240);
  assert.ok(reduced.length <= 241);
  assert.deepEqual(reduced[reduced.length - 1], [999, -499]);
  const bounds = waterlineBounds([
    [
      [10, 40],
      [1900, 1000],
    ],
  ]);
  assert.deepEqual(bounds, { width: 1901, height: 1001 });
});
