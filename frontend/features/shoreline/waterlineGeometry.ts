/** Pure geometry helpers for rendering ordered shoreline segments. */

export type ShorelinePoint = [number, number];
export type ShorelineSegment = ShorelinePoint[];

export type ShorelineGeometrySource = {
  shoreline_segments?: ShorelineSegment[] | null;
  shoreline_geometry?: ShorelinePoint[] | null;
};

function validSegment(segment: ShorelineSegment | null | undefined): segment is ShorelineSegment {
  return Array.isArray(segment) && segment.length >= 2;
}

/**
 * Return exactly one ordered line. New observations are single-line by
 * contract; for historical multi-segment rows we keep the longest segment and
 * discard the rest so the renderer never draws disconnected polylines.
 */
export function observationSegments(source: ShorelineGeometrySource | null | undefined): ShorelineSegment[] {
  if (!source) return [];
  const segments = (source.shoreline_segments ?? []).filter(validSegment);
  if (segments.length > 0) {
    return [segments.reduce((best, current) => (current.length > best.length ? current : best))];
  }
  return validSegment(source.shoreline_geometry) ? [source.shoreline_geometry] : [];
}

/** Decimate a single segment for SVG; segment breaks are never removed. */
export function decimateSegment(segment: ShorelineSegment, maxPoints = 240): ShorelineSegment {
  if (segment.length <= maxPoints) return segment;
  const stride = Math.ceil(segment.length / maxPoints);
  const reduced: ShorelineSegment = [];
  for (let index = 0; index < segment.length; index += stride) {
    reduced.push(segment[index]);
  }
  const last = segment[segment.length - 1];
  if (reduced[reduced.length - 1] !== last) reduced.push(last);
  return reduced;
}

export function polylinePoints(segment: ShorelineSegment): string {
  return segment.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
}

/** Image-coordinate viewBox covering every segment (y increases downward). */
export function waterlineBounds(segments: ShorelineSegment[]): { width: number; height: number } {
  let maxX = 0;
  let maxY = 0;
  for (const segment of segments) {
    for (const [x, y] of segment) {
      if (Number.isFinite(x) && x > maxX) maxX = x;
      if (Number.isFinite(y) && y > maxY) maxY = y;
    }
  }
  return { width: Math.max(1, Math.ceil(maxX) + 1), height: Math.max(1, Math.ceil(maxY) + 1) };
}
