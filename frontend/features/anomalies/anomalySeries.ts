/** Pure helpers for the tide/position anomaly chart (no React, easy to test). */

import type { AnomalyEvent } from "@/lib/types";

export const ANOMALY_COLORS: Record<string, string> = {
  NORMAL: "#2ecc71",
  UNUSUAL_SHORELINE_POSITION: "#e67e22",
  RAPID_SHORELINE_RETREAT: "#c0392b",
  PERSISTENT_DRAWDOWN: "#8e44ad",
  POSSIBLE_DRAWDOWN: "#d97706",
  DATA_UNRELIABLE: "#7f8c8d",
  PENDING_PERSISTENCE: "#f1c40f",
  EXPLORATORY_MOVEMENT_ONLY: "#3498db",
  EXPLORATORY_ROLLING_BASELINE: "#64748b",
};

export interface AnomalyPoint {
  id: number;
  timestamp: string;
  timeMs: number;
  observed: number | null;
  expected: number | null;
  rollingBaseline: number | null;
  residual: number | null;
  robustZ: number | null;
  cumulativeDrawdown: number | null;
  rollingScale: number | null;
  scaleFloored: boolean;
  tide: number | null;
  movement: number | null;
  classification: string;
  tideAvailable: boolean;
  baselineStatus: string;
  detectorMode: string | null;
  qualityStatus: string | null;
  cameraMotion: string | null;
  reason: string | null;
  state: string;
  effectiveGate: number | null;
}

export function toAnomalyPoints(events: AnomalyEvent[]): AnomalyPoint[] {
  return [...events]
    .sort((left, right) => new Date(left.detected_at).getTime() - new Date(right.detected_at).getTime())
    .map((event) => {
      const rollingBaseline = finiteOrNull(event.rolling_baseline_px);
      return {
        id: event.id,
        timestamp: event.detected_at,
        timeMs: new Date(event.detected_at).getTime(),
        observed: finiteOrNull(event.observed_position_px),
        expected: finiteOrNull(event.expected_position_px),
        rollingBaseline,
        residual: finiteOrNull(event.residual_px),
        robustZ: finiteOrNull(event.robust_z),
        cumulativeDrawdown: finiteOrNull(event.cumulative_drawdown_px),
        rollingScale: finiteOrNull(event.rolling_scale_px),
        scaleFloored: Boolean(event.scale_floored),
        tide: event.tide_data_available ? finiteOrNull(event.tide_height_m) : null,
        movement: finiteOrNull(event.movement_rate_px_s),
        classification: event.classification ?? "DATA_UNRELIABLE",
        tideAvailable: Boolean(event.tide_data_available),
        baselineStatus: event.baseline_status ?? "unavailable",
        detectorMode: event.detector_mode ?? event.resolved_detector_mode ?? null,
        qualityStatus: event.quality_status ?? null,
        cameraMotion:
          typeof event.contributing_factors?.camera_motion === "string"
            ? (event.contributing_factors.camera_motion as string)
            : null,
        reason: event.reason,
        state: event.state,
        effectiveGate: effectiveGate(event.contributing_factors),
      };
    });
}

function factorNumber(
  factors: Record<string, unknown> | null | undefined,
  keys: string[],
): number | null {
  if (!factors) return null;
  for (const key of keys) {
    const value = factors[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

/**
 * The effective movement gate when the stored event exposes it. The current
 * AnomalyEvent API does not persist sigma_k, so this is usually null. The chart
 * only draws a gate when the server actually supplies one.
 */
function effectiveGate(
  factors: Record<string, unknown> | null | undefined,
): number | null {
  const direct = factorNumber(factors, [
    "effective_gate_px",
    "movement_gate_px",
    "gate_px",
  ]);
  if (direct !== null) return direct;
  const sigma = factorNumber(factors, ["sigma_k_px", "sigma_k"]);
  if (sigma === null) return null;
  const floor = factorNumber(factors, ["drawdown_floor_px"]);
  return Math.max(5 * sigma, floor ?? 0);
}

function finiteOrNull(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Split a series into contiguous drawable runs. A run breaks on a missing value
 * and on a time gap larger than ``maxGapMs``, so a missing observation never
 * becomes a straight line between its neighbours.
 */
export function splitRuns(
  points: AnomalyPoint[],
  selector: (point: AnomalyPoint) => number | null,
  maxGapMs = 90_000,
): AnomalyPoint[][] {
  const runs: AnomalyPoint[][] = [];
  let current: AnomalyPoint[] = [];
  let previousTime: number | null = null;
  for (const point of points) {
    const value = selector(point);
    const gap = previousTime === null ? 0 : point.timeMs - previousTime;
    if (value === null || !Number.isFinite(point.timeMs)) {
      if (current.length > 0) runs.push(current);
      current = [];
      previousTime = null;
      continue;
    }
    if (current.length > 0 && gap > maxGapMs) {
      runs.push(current);
      current = [];
    }
    current.push(point);
    previousTime = point.timeMs;
  }
  if (current.length > 0) runs.push(current);
  return runs;
}

export const ANOMALY_GAP_MS = 45_000;

export interface AnomalyChartPoint {
  t: number | null;
  id: number | null;
  observed: number | null;
  baseline: number | null;
  residual: number | null;
  tide: number | null;
  robustZ: number | null;
  cumulativeDrawdown: number | null;
  gate: number | null;
  gateUp: number | null;
  gateDown: number | null;
  classification: string | null;
  state: string | null;
  reason: string | null;
}

/**
 * Build Recharts rows sorted by time, inserting a null row when the next sample
 * is more than 45 s away. The demo chart uses the same gap rule.
 */
export function buildAnomalyChartData(
  points: AnomalyPoint[],
  gapMs = ANOMALY_GAP_MS,
): AnomalyChartPoint[] {
  const rows: AnomalyChartPoint[] = [];
  let previous: number | null = null;
  for (const point of points) {
    if (!Number.isFinite(point.timeMs)) continue;
    if (previous !== null && point.timeMs - previous > gapMs) {
      rows.push({
        t: previous + 1,
        id: null,
        observed: null,
        baseline: null,
        residual: null,
        tide: null,
        robustZ: null,
        cumulativeDrawdown: null,
        gate: null,
        gateUp: null,
        gateDown: null,
        classification: null,
        state: null,
        reason: null,
      });
    }
    const baseline = point.rollingBaseline ?? point.expected;
    const gate = point.effectiveGate;
    rows.push({
      t: point.timeMs,
      id: point.id,
      observed: point.observed,
      baseline,
      residual: point.residual,
      tide: point.tide,
      robustZ: point.robustZ,
      cumulativeDrawdown: point.cumulativeDrawdown,
      gate,
      gateUp: baseline !== null && gate !== null ? baseline + gate : null,
      gateDown: baseline !== null && gate !== null ? baseline - gate : null,
      classification: point.classification,
      state: point.state,
      reason: point.reason,
    });
    previous = point.timeMs;
  }
  return rows;
}

export function anomalyStateSegments(
  points: AnomalyPoint[],
): { state: string; count: number }[] {
  const segments: { state: string; count: number }[] = [];
  for (const point of points) {
    const last = segments[segments.length - 1];
    if (last && last.state === point.state) last.count += 1;
    else segments.push({ state: point.state, count: 1 });
  }
  return segments;
}

export function anomalyStateColor(state: string): string {
  switch (state.toUpperCase()) {
    case "MONITORING":
    case "OBSERVED":
      return "#64748b";
    case "CANDIDATE":
      return "#f59e0b";
    case "PENDING_PERSISTENCE":
      return "#8b5cf6";
    case "EVENT":
    case "ANOMALY_DETECTED":
      return "#dc2626";
    case "COOLDOWN":
      return "#0ea5e9";
    case "DATA_UNRELIABLE":
      return "#94a3b8";
    default:
      return "#cbd5e1";
  }
}

export function finiteBounds(values: Array<number | null>): { min: number; max: number } | null {
  const finite = values.filter((value): value is number => value !== null && Number.isFinite(value));
  if (finite.length === 0) return null;
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  return { min, max };
}

export function scaleToHeight(
  value: number,
  bounds: { min: number; max: number },
  height: number,
  padding = 8,
): number {
  const usable = Math.max(1, height - padding * 2);
  const fraction = (value - bounds.min) / (bounds.max - bounds.min);
  return padding + (1 - fraction) * usable;
}

export function scaleTime(timeMs: number, start: number, end: number, width: number): number {
  if (end <= start) return 0;
  return ((timeMs - start) / (end - start)) * width;
}

export function classificationCounts(points: AnomalyPoint[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const point of points) counts[point.classification] = (counts[point.classification] ?? 0) + 1;
  return counts;
}
