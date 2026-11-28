/** Pure helpers for the DEMO MODE charts and labels. */

export const DEMO_BANNER = "SIMULATION - not real data";
export const DEMO_WATERMARK = "SIMULATION - not real data";
export const GAP_MS = 45_000;
export const Z_THRESHOLD = 3.5;

export type ChartSample = {
  t: number | null;
  real: number | null;
  injected: number | null;
  baseline: number | null;
  gateUp: number | null;
  gateDown: number | null;
  z: number | null;
  ck: number | null;
  gate: number | null;
  state: string | null;
  classification: string | null;
};

export type RawSample = {
  time: string;
  real: number | null;
  injected: number | null;
  baseline: number | null;
  z: number | null;
  ck: number | null;
  gate: number | null;
  state: string;
  classification: string;
};

export function chartPoints(samples: RawSample[], gapMs = GAP_MS): ChartSample[] {
  const ordered = [...samples].sort(
    (left, right) => Date.parse(left.time) - Date.parse(right.time),
  );
  const points: ChartSample[] = [];
  let previous: number | null = null;
  for (const sample of ordered) {
    const t = Date.parse(sample.time);
    if (!Number.isFinite(t)) continue;
    if (previous !== null && t - previous > gapMs) {
      points.push({
        t: previous + 1,
        real: null,
        injected: null,
        baseline: null,
        gateUp: null,
        gateDown: null,
        z: null,
        ck: null,
        gate: null,
        state: null,
        classification: null,
      });
    }
    const gate = sample.gate;
    points.push({
      t,
      real: sample.real,
      injected: sample.injected,
      baseline: sample.baseline,
      gateUp:
        sample.baseline !== null && gate !== null ? sample.baseline + gate : null,
      gateDown:
        sample.baseline !== null && gate !== null ? sample.baseline - gate : null,
      z: sample.z,
      ck: sample.ck,
      gate,
      state: sample.state,
      classification: sample.classification,
    });
    previous = t;
  }
  return points;
}

export function formatManila(
  value: number | null,
  options: { withDate?: boolean } = {},
): string {
  if (value === null || !Number.isFinite(value)) return "";
  return new Date(value).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    ...(options.withDate ? { day: "2-digit", month: "short" } : {}),
    timeZone: "Asia/Manila",
  });
}

export function formatDateManila(value: string | number | null): string {
  if (value === null) return "-";
  const time = typeof value === "number" ? value : Date.parse(value);
  if (!Number.isFinite(time)) return "-";
  return new Date(time).toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Asia/Manila",
  });
}

export function amplitudePx(percent: number, frameHeight: number | null): number {
  return frameHeight ? (percent / 100) * frameHeight : 0;
}

export function amplitudePercent(px: number, frameHeight: number | null): number {
  return frameHeight ? (px / frameHeight) * 100 : 0;
}

export function describeAmplitude(
  percent: number,
  frameHeight: number | null,
): string {
  const px = amplitudePx(percent, frameHeight);
  if (!frameHeight) return `${percent.toFixed(1)}% of frame height`;
  return `${px.toFixed(0)} px = ${percent.toFixed(1)}% of ${frameHeight.toFixed(0)} px`;
}

export type StateSegment = { state: string; count: number };

export function stateSegments(samples: RawSample[]): StateSegment[] {
  const segments: StateSegment[] = [];
  for (const sample of samples) {
    const last = segments[segments.length - 1];
    if (last && last.state === sample.state) last.count += 1;
    else segments.push({ state: sample.state, count: 1 });
  }
  return segments;
}

export function stateColor(state: string): string {
  switch (state) {
    case "MONITORING":
      return "#64748b";
    case "CANDIDATE":
      return "#f59e0b";
    case "PENDING_PERSISTENCE":
      return "#8b5cf6";
    case "EVENT":
      return "#dc2626";
    case "COOLDOWN":
      return "#0ea5e9";
    default:
      return "#94a3b8";
  }
}

export function outcomeText(outcome: string, explanation: string): string {
  if (outcome === "caught") return `Caught - ${explanation}`;
  if (outcome === "missed") return `Missed - ${explanation}`;
  return explanation;
}

export function expectedText(expected: {
  available: boolean;
  detection_rate?: number;
  n_in_bounds?: number;
  source?: string;
  note?: string;
}): string {
  if (!expected.available) return expected.note ?? "Quick estimate not available yet.";
  const rate = Math.round((expected.detection_rate ?? 0) * 100);
  return `${rate}% detected, n=${expected.n_in_bounds ?? 0} starts (${expected.source ?? "quick estimate"})`;
}

export function statusLabel(status: string): string {
  if (status === "replayable") return "Replayable";
  if (status === "limited") return "Limited data";
  return "Not replayable";
}
