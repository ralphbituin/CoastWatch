/** API client for DEMO MODE. Every response is simulation-only. */

import { API_V1 } from "@/lib/config";

export type RecordingStatus = "replayable" | "limited" | "not_replayable";

export type Parity = {
  pass: boolean;
  tolerance: number;
  checked_samples_after_full_history: number;
  max_abs: Record<string, number>;
  classification_mismatches: number;
  state_mismatches: number;
};

export type RecordingProfile = {
  id: string;
  label: string;
  camera: string | null;
  source_type: string | null;
  frame_height_px: number | null;
  frame_width_px: number | null;
  decoded_resolutions: string[];
  nominal_interval_s: number | null;
  duration_s: number | null;
  row_counts: Record<string, number>;
  sessions: string[];
  segment_count: number;
  scored_coverage: number | null;
  longest_valid_window_s: number | null;
  step_std_px: number | null;
  move90_median_px: number | null;
  move90_p95_px: number | null;
  move90_max_px: number | null;
  gate_median_px: number | null;
  gate_p95_px: number | null;
  drawdown_floor_px: number | null;
  scale_floor_px: number | null;
  polygon: boolean;
  notes: string;
  config_hash: string | null;
  config_source: string | null;
  status: RecordingStatus;
  reason: string | null;
  parity: Parity | null;
  valid_start_count: number;
  valid_start_samples: string[];
  reference_min_remaining_s: number;
};

export type Expected = {
  available: boolean;
  detection_rate?: number;
  n_in_bounds?: number;
  n_detected?: number;
  wilson_lo?: number;
  wilson_hi?: number;
  matched_amplitude_px?: number;
  ramp_s?: number;
  source?: string;
  note?: string;
  exact_amplitude?: boolean;
};

export type Preset = {
  id: string;
  label: string;
  description: string;
  params: SimulationParams;
  expected: Expected;
  banner: string;
};

export type SimulationParams = {
  amplitude_pct: number;
  amplitude_px?: number;
  ramp_s: number;
  hold_s: number;
  shape: "retreat_return" | "persistent";
  sign: number;
  speed: number;
  start: string;
  seed?: number;
};

export type SimulationBody = {
  recording_id: string;
  amplitude_pct: number;
  ramp_s: number;
  hold_s: number;
  shape: string;
  sign: number;
  speed: number;
  start: string;
  seed?: number;
};

export type SampleMessage = {
  type: "sample";
  time: string;
  real: number | null;
  injected: number | null;
  baseline: number | null;
  z: number | null;
  ck: number | null;
  gate: number | null;
  state: string;
  classification: string;
  persistence: number | null;
  event: SimEvent | null;
};

export type SimEvent = {
  id: string;
  classification: string;
  detected_at: string;
  latency_s: number | null;
  gate_px: number | null;
  z: number | null;
  ck: number | null;
};

export type SimulationResult = {
  sim_id: string;
  outcome: "caught" | "missed" | string;
  detected: boolean;
  classification: string;
  state_reached: string;
  latency_from_t0_s: number | null;
  latency_from_gate_s: number | null;
  failure_mode: string;
  explanation: string;
  peak_z: number | null;
  peak_ck_px: number | null;
  false_events: number;
  in_bounds: boolean;
  params: SimulationParams;
  label: string;
};

export type SummaryCell = {
  amplitude_px: number;
  amplitude_pct_frame: number;
  ramp_s: number;
  hold_s: number;
  shape: string;
  sign: number;
  n_starts: number;
  n_in_bounds: number;
  n_out_of_bounds: number;
  n_detected: number;
  detection_rate: number;
  wilson_lo: number;
  wilson_hi: number;
  median_latency_s: number | null;
  gate_median_px: number | null;
  failure_counts: Record<string, number>;
  false_events: number;
};

export type SummaryResult = {
  recording_id: string;
  grid: {
    amplitudes_px: number[];
    ramps_s: number[];
    hold_s: number;
    shape: string;
    sign: number;
  };
  cells: SummaryCell[];
  minimum_90: {
    ramp_s: number;
    min_amplitude_px: number | null;
    min_amplitude_pct_frame: number | null;
    detection_rate: number | null;
  }[];
  failure_modes: Record<string, number>;
  false_events_outside_injection: number;
  analysis_starts: { source: string; count: number; starts_utc: string[] };
  offline_comparison: {
    available: boolean;
    checked_cells: number;
    mismatches: {
      amplitude_px: number;
      ramp_s: number;
      differences: Record<string, [number, number]>;
    }[];
    matches?: boolean;
  };
  noise: {
    frame_height_px: number | null;
    step_std_px: number | null;
    move90_median_px: number | null;
    move90_p95_px: number | null;
    move90_max_px: number | null;
    gate_median_px: number | null;
    gate_p95_px: number | null;
    drawdown_floor_px: number | null;
  };
  caveat: string;
};

export type SummaryResponse = {
  recording_id: string;
  status: string;
  reason?: string | null;
  completed_cells?: number;
  total_cells?: number;
  completed_replays?: number;
  total_replays?: number;
  fraction?: number;
  error?: string | null;
  result: SummaryResult | null;
};

export type CompareRow = {
  id: string;
  label: string;
  status: RecordingStatus;
  reason: string | null;
  valid_start_count: number;
  frame_height_px: number | null;
  step_std_px: number | null;
  move90_median_px: number | null;
  move90_p95_px: number | null;
  gate_median_px: number | null;
  gate_p95_px: number | null;
  available: boolean;
  amplitude_px?: number;
  detection_rate?: number;
  n_in_bounds?: number;
  n_starts?: number;
  n_detected?: number;
  wilson_lo?: number;
  wilson_hi?: number;
  median_latency_s?: number | null;
  false_events?: number;
  limited_data?: boolean;
};

export type CompareResult = {
  amplitude_pct: number;
  ramp_s: number;
  shape: string;
  hold_s: number;
  recordings: CompareRow[];
  note: string;
  banner: string;
};

export class DemoApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "DemoApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_V1}${path}`, {
    ...init,
    headers: { Accept: "application/json", "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body?.detail ?? body?.error?.message ?? message;
    } catch {
      // keep default
    }
    throw new DemoApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export const demoApi = {
  recordings: () =>
    request<{ recordings: RecordingProfile[]; banner: string }>("/demo/injection/recordings"),
  presets: (id: string) =>
    request<{ presets: Preset[]; status: RecordingStatus; reason: string | null }>(
      `/demo/injection/recordings/${encodeURIComponent(id)}/presets`,
    ),
  summary: (id: string) =>
    request<SummaryResponse>(`/demo/injection/recordings/${encodeURIComponent(id)}/summary`),
  createSimulation: (body: SimulationBody) =>
    request<{ id: string; stream: string; warnings: string[] }>("/demo/injection/simulations", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  stopSimulation: (id: string) =>
    request<{ id: string; status: string }>(
      `/demo/injection/simulations/${encodeURIComponent(id)}/stop`,
      { method: "POST" },
    ),
  deleteSimulation: (id: string) =>
    request<{ deleted: boolean }>(`/demo/injection/simulations/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  compare: (amplitudePct: number, rampS: number) =>
    request<CompareResult>(
      `/demo/injection/compare?amplitude_pct=${encodeURIComponent(
        amplitudePct,
      )}&ramp_s=${encodeURIComponent(rampS)}`,
    ),
};

export function simulationStreamUrl(id: string): string {
  return `${API_V1}/demo/injection/simulations/${encodeURIComponent(id)}/stream`;
}
