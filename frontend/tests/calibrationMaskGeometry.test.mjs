/**
 * Pure geometry tests for the calibration monitoring-outline plotter.
 *
 * Run: npm run test:calibration-mask
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { transpileModule } from "typescript";

const sourcePath = fileURLToPath(
  new URL("../features/cameras/calibrationMaskGeometry.ts", import.meta.url),
);
const source = readFileSync(sourcePath, "utf8");
const { outputText } = transpileModule(source, {
  compilerOptions: { module: "esnext", target: "es2020" },
});
const moduleUrl = `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`;
const geometry = await import(moduleUrl);

const {
  clamp01,
  screenToNormalized,
  polygonArea,
  polygonCoveragePercent,
  isSimplePolygon,
  hasMinimumVertices,
  rectangleToPolygon,
  insertVertexOnNearestEdge,
  distanceToSegment,
  validatePolygon,
  polygonWarning,
} = geometry;

const p = (x, y) => ({ x, y });

test("clamp01 clamps to the normalized frame", () => {
  assert.equal(clamp01(-0.2), 0);
  assert.equal(clamp01(0.4), 0.4);
  assert.equal(clamp01(2), 1);
});

test("screenToNormalized maps and clamps client coordinates", () => {
  const rect = { left: 100, top: 50, width: 200, height: 100 };
  assert.deepEqual(screenToNormalized(150, 75, rect), { x: 0.25, y: 0.25 });
  assert.deepEqual(screenToNormalized(50, 500, rect), { x: 0, y: 1 });
  assert.deepEqual(screenToNormalized(150, 75, { ...rect, width: 0 }), { x: 0, y: 0 });
});

test("polygonArea uses the shoelace formula", () => {
  const square = [p(0, 0), p(0.5, 0), p(0.5, 0.5), p(0, 0.5)];
  assert.equal(polygonArea(square), 0.25);
  assert.equal(polygonCoveragePercent(square), 25);
  assert.equal(polygonArea([p(0, 0), p(1, 0)]), 0);
});

test("rectangleToPolygon returns clockwise corners", () => {
  assert.deepEqual(rectangleToPolygon(p(0.8, 0.9), p(0.2, 0.1)), [
    p(0.2, 0.1),
    p(0.8, 0.1),
    p(0.8, 0.9),
    p(0.2, 0.9),
  ]);
});

test("concave polygons are simple and valid", () => {
  const concave = [p(0.1, 0.1), p(0.8, 0.1), p(0.8, 0.8), p(0.5, 0.4), p(0.2, 0.8)];
  assert.equal(isSimplePolygon(concave), true);
  assert.deepEqual(validatePolygon(concave), { ok: true });
});

test("bow-tie self-intersection is rejected", () => {
  const bowTie = [p(0.1, 0.1), p(0.9, 0.9), p(0.9, 0.1), p(0.1, 0.9)];
  assert.equal(isSimplePolygon(bowTie), false);
  const result = validatePolygon(bowTie);
  assert.equal(result.ok, false);
  assert.match(result.reason, /crosses itself/);
});

test("adjacent shared endpoints are not intersections", () => {
  const square = [p(0.1, 0.1), p(0.9, 0.1), p(0.9, 0.9), p(0.1, 0.9)];
  assert.equal(isSimplePolygon(square), true);
});

test("duplicate and zero-area outlines are rejected", () => {
  assert.equal(hasMinimumVertices([p(0, 0), p(1, 0)]), false);
  assert.equal(
    validatePolygon([p(0.2, 0.2), p(0.2, 0.2), p(0.8, 0.2), p(0.8, 0.8)]).ok,
    false,
  );
  assert.match(
    validatePolygon([p(0.2, 0.2), p(0.2, 0.2), p(0.2, 0.2)]).reason,
    /no area|at least|crosses itself/,
  );
});

test("out-of-range points are rejected", () => {
  const result = validatePolygon([p(-0.01, 0.1), p(0.9, 0.1), p(0.9, 0.9)]);
  assert.equal(result.ok, false);
  assert.match(result.reason, /outside/);
});

test("distanceToSegment and edge insertion", () => {
  const square = [p(0, 0), p(1, 0), p(1, 1), p(0, 1)];
  assert.equal(distanceToSegment(p(0.5, 0.1), square[0], square[1]).distance, 0.1);
  const inserted = insertVertexOnNearestEdge(square, p(0.5, 0.02), 0.05);
  assert.ok(inserted);
  assert.equal(inserted.length, 5);
  assert.deepEqual(inserted[1], p(0.5, 0.02));
  assert.equal(insertVertexOnNearestEdge(square, p(0.5, 0.4), 0.05), null);
});

test("tiny and nearly-full outlines warn without blocking", () => {
  const tiny = [p(0.4, 0.4), p(0.403, 0.4), p(0.403, 0.403)];
  assert.match(polygonWarning(tiny), /less than 0.5%/);
  const full = [p(0, 0), p(1, 0), p(1, 1), p(0, 1)];
  assert.match(polygonWarning(full), /almost the whole frame/);
  assert.equal(polygonWarning([p(0.1, 0.1), p(0.4, 0.1), p(0.4, 0.4), p(0.1, 0.4)]), null);
});
