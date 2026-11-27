/** Pure geometry for the calibration monitoring-outline plotter.
 *
 * Points are normalized to the source frame: both x and y are in [0, 1].
 * Nothing here depends on screen size, zoom or device pixel ratio.
 */

export type NormPoint = { x: number; y: number };

export type PolygonValidation = { ok: true } | { ok: false; reason: string };

export const MIN_VERTICES = 3;
export const MAX_VERTICES = 200;
export const MIN_COVERAGE = 0.0001;
/** Coverage below this fraction is usually a mis-click. */
export const WARN_COVERAGE = 0.005;
/** Coverage above this fraction usually means "ignore everything". */
export const WARN_COVERAGE_HIGH = 0.98;

const EPS = 1e-9;

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function screenToNormalized(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
): NormPoint {
  if (rect.width <= 0 || rect.height <= 0) return { x: 0, y: 0 };
  return {
    x: clamp01((clientX - rect.left) / rect.width),
    y: clamp01((clientY - rect.top) / rect.height),
  };
}

export function polygonArea(points: NormPoint[]): number {
  if (points.length < 3) return 0;
  let total = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    total += current.x * next.y - next.x * current.y;
  }
  return Math.abs(total) / 2;
}

export function polygonCoveragePercent(points: NormPoint[]): number {
  return polygonArea(points) * 100;
}

function orientation(a: NormPoint, b: NormPoint, c: NormPoint): number {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

function onSegment(a: NormPoint, b: NormPoint, p: NormPoint): boolean {
  return (
    Math.min(a.x, b.x) - EPS <= p.x &&
    p.x <= Math.max(a.x, b.x) + EPS &&
    Math.min(a.y, b.y) - EPS <= p.y &&
    p.y <= Math.max(a.y, b.y) + EPS
  );
}

export function segmentsIntersect(
  a: NormPoint,
  b: NormPoint,
  c: NormPoint,
  d: NormPoint,
): boolean {
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);

  if (
    ((o1 > EPS && o2 < -EPS) || (o1 < -EPS && o2 > EPS)) &&
    ((o3 > EPS && o4 < -EPS) || (o3 < -EPS && o4 > EPS))
  ) {
    return true;
  }

  if (Math.abs(o1) <= EPS && onSegment(a, b, c)) return true;
  if (Math.abs(o2) <= EPS && onSegment(a, b, d)) return true;
  if (Math.abs(o3) <= EPS && onSegment(c, d, a)) return true;
  if (Math.abs(o4) <= EPS && onSegment(c, d, b)) return true;
  return false;
}

export function isSimplePolygon(points: NormPoint[]): boolean {
  const count = points.length;
  if (count < MIN_VERTICES) return false;
  for (let index = 0; index < count; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % count];
    if (current.x === next.x && current.y === next.y) return false;
  }
  for (let i = 0; i < count; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % count];
    for (let j = i + 1; j < count; j += 1) {
      if (j === i || (j + 1) % count === i || (i + 1) % count === j) continue;
      const c = points[j];
      const d = points[(j + 1) % count];
      if (segmentsIntersect(a, b, c, d)) return false;
    }
  }
  return true;
}

export function hasMinimumVertices(points: NormPoint[], min = MIN_VERTICES): boolean {
  return points.length >= min;
}

export function rectangleToPolygon(anchor: NormPoint, current: NormPoint): NormPoint[] {
  const left = Math.min(anchor.x, current.x);
  const right = Math.max(anchor.x, current.x);
  const top = Math.min(anchor.y, current.y);
  const bottom = Math.max(anchor.y, current.y);
  return [
    { x: left, y: top },
    { x: right, y: top },
    { x: right, y: bottom },
    { x: left, y: bottom },
  ];
}

export function distanceToSegment(
  point: NormPoint,
  a: NormPoint,
  b: NormPoint,
): { distance: number; t: number } {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= EPS) {
    return { distance: Math.hypot(point.x - a.x, point.y - a.y), t: 0 };
  }
  const t = Math.min(
    1,
    Math.max(0, ((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSquared),
  );
  const x = a.x + t * dx;
  const y = a.y + t * dy;
  return { distance: Math.hypot(point.x - x, point.y - y), t };
}

/**
 * Insert a vertex on the nearest edge when the point is within ``maxDistance``.
 * Returns null when no edge is close enough. Adjacent edges are not merged.
 */
export function insertVertexOnNearestEdge(
  points: NormPoint[],
  point: NormPoint,
  maxDistance: number,
): NormPoint[] | null {
  if (points.length < 2) return null;
  let bestIndex = -1;
  let bestDistance = Infinity;
  for (let index = 0; index < points.length; index += 1) {
    const next = (index + 1) % points.length;
    const { distance } = distanceToSegment(point, points[index], points[next]);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }
  if (bestIndex < 0 || bestDistance > maxDistance) return null;
  const copy = points.slice();
  copy.splice(bestIndex + 1, 0, point);
  return copy;
}

export function validatePolygon(points: NormPoint[]): PolygonValidation {
  if (points.length < MIN_VERTICES) {
    return { ok: false, reason: "Add at least 3 points. They connect in order." };
  }
  if (points.length > MAX_VERTICES) {
    return { ok: false, reason: `Too many points (maximum ${MAX_VERTICES}).` };
  }
  for (const point of points) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      return { ok: false, reason: "A point has an invalid coordinate." };
    }
    if (point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) {
      return { ok: false, reason: "A point lies outside the frame." };
    }
  }
  if (!isSimplePolygon(points)) {
    return {
      ok: false,
      reason: "Outline crosses itself or has duplicate points — move a point to fix it.",
    };
  }
  const area = polygonArea(points);
  if (area < MIN_COVERAGE) {
    return { ok: false, reason: "This outline has no area — move the points apart." };
  }
  return { ok: true };
}

export function polygonWarning(points: NormPoint[]): string | null {
  if (points.length < MIN_VERTICES) return null;
  const area = polygonArea(points);
  if (area > 0 && area < WARN_COVERAGE) {
    return "This outline covers less than 0.5% of the frame. Check that it is intentional.";
  }
  if (area > WARN_COVERAGE_HIGH) {
    return "This outline covers almost the whole frame, so the mask will barely restrict detection.";
  }
  return null;
}
