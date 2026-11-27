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
      };
    });
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
