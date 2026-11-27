/** Typed client for the Predictor + Anomaly Test Lab (Phase 7). */

import { API_V1 } from "@/lib/config";
import type { AnomalyEvent, LabRow, LabRowsResponse, LabRun } from "@/lib/types";

async function labRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_V1}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!response.ok) {
    let code = "http_error";
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      code = body?.error?.code ?? code;
      message = body?.error?.message ?? message;
    } catch {
      // Keep the HTTP message when the error envelope is absent.
    }
    throw new Error(`${code}: ${message}`);
  }
  return (await response.json()) as T;
}

export type RollingOptions = {
  window_seconds: number;
  k_seconds: number;
  z_threshold: number;
  persistence: number;
  max_gap_seconds: number;
};

export const DEFAULT_ROLLING: RollingOptions = {
  window_seconds: 1200,
  k_seconds: 90,
  z_threshold: 3.5,
  persistence: 3,
  max_gap_seconds: 45,
};

export function rollingJson(options: RollingOptions): string {
  return JSON.stringify(options);
}

export type ImageRunInput = {
  files: File[];
  manifestText?: string;
  startAt?: string;
  intervalSeconds?: number;
  timezone: string;
  rolling: RollingOptions;
  persist: boolean;
  cameraId?: number | null;
};

export type VideoRunInput = {
  recordingId?: string;
  file?: File | null;
  videoPath?: string;
  sampleIntervalSeconds: number;
  startElapsed: number;
  endElapsed?: number | null;
  maxSamples: number;
  startAt?: string;
  clockAnchors?: string;
  timezone: string;
  rolling: RollingOptions;
  persist: boolean;
  cameraId?: number | null;
};

export type StreamRunInput = {
  cameraId?: number | null;
  youtubeUrl?: string;
  sampleIntervalSeconds: number;
  maxSamples: number;
  maxDurationSeconds: number;
  timezone: string;
  rolling: RollingOptions;
  persist: boolean;
};

export const labApi = {
  status: () =>
    labRequest<Record<string, unknown>>("/lab/status"),

  createImageRun: (input: ImageRunInput) => {
    const form = new FormData();
    for (const file of input.files) form.append("images", file);
    if (input.manifestText?.trim()) form.append("manifest_text", input.manifestText);
    if (input.startAt?.trim()) form.append("start_at", input.startAt);
    if (input.intervalSeconds !== undefined) {
      form.append("interval_seconds", String(input.intervalSeconds));
    }
    form.append("timezone", input.timezone);
    form.append("rolling_json", rollingJson(input.rolling));
    form.append("persist", String(input.persist));
    if (input.cameraId) form.append("camera_id", String(input.cameraId));
    return labRequest<LabRun>("/lab/runs/images", { method: "POST", body: form });
  },

  createVideoRun: (input: VideoRunInput) => {
    const form = new FormData();
    if (input.recordingId) form.append("recording_id", input.recordingId);
    if (input.videoPath) form.append("video_path", input.videoPath);
    if (input.file) form.append("upload", input.file);
    form.append("sample_interval_seconds", String(input.sampleIntervalSeconds));
    form.append("start_elapsed", String(input.startElapsed));
    if (input.endElapsed !== undefined && input.endElapsed !== null) {
      form.append("end_elapsed", String(input.endElapsed));
    }
    form.append("max_samples", String(input.maxSamples));
    if (input.startAt?.trim()) form.append("start_at", input.startAt);
    if (input.clockAnchors?.trim()) form.append("clock_anchors", input.clockAnchors);
    form.append("timezone", input.timezone);
    form.append("rolling_json", rollingJson(input.rolling));
    form.append("persist", String(input.persist));
    if (input.cameraId) form.append("camera_id", String(input.cameraId));
    return labRequest<LabRun>("/lab/runs/video", { method: "POST", body: form });
  },

  createStreamRun: (input: StreamRunInput) =>
    labRequest<LabRun>("/lab/runs/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_id: input.cameraId || null,
        youtube_url: input.youtubeUrl || null,
        sample_interval_seconds: input.sampleIntervalSeconds,
        max_samples: input.maxSamples,
        max_duration_seconds: input.maxDurationSeconds,
        timezone: input.timezone,
        rolling: input.rolling,
        persist: input.persist,
      }),
    }),

  getRun: (runId: string) => labRequest<LabRun>(`/lab/runs/${runId}`),

  getRows: (runId: string, limit = 200, offset = 0) =>
    labRequest<LabRowsResponse>(
      `/lab/runs/${runId}/rows?limit=${limit}&offset=${offset}`,
    ),

  getEvents: (runId: string) =>
    labRequest<{ run_id: string; events: Array<Record<string, unknown>> }>(
      `/lab/runs/${runId}/events`,
    ),

  deleteRun: (runId: string) =>
    labRequest<{ run_id: string; deleted: boolean }>(`/lab/runs/${runId}`, {
      method: "DELETE",
    }),

  artifactUrl: (runId: string, name: string) =>
    `${API_V1}/lab/runs/${runId}/artifacts/${encodeURIComponent(name)}`,
};

export function labRowToAnomalyEvent(row: LabRow, index: number): AnomalyEvent {
  const observedAt = row.observed_at ?? new Date().toISOString();
  return {
    id: index + 1,
    detected_at: observedAt,
    monitoring_station_id: 0,
    camera_id: null,
    recording_id: null,
    frame_index: row.frame_index ?? null,
    state: row.state ?? "DATA_UNRELIABLE",
    classification: (row.classification ?? "DATA_UNRELIABLE") as AnomalyEvent["classification"],
    anomaly_score: null,
    baseline_status: "rolling_unverified",
    tide_adjusted: false,
    tide_data_available: false,
    tide_source: "unavailable",
    tide_height_m: null,
    observed_position_px: row.signed_position_px ?? null,
    expected_position_px: row.rolling_baseline_px ?? null,
    residual_px: row.residual_px ?? null,
    robust_z: row.robust_z ?? null,
    movement_rate_px_s: row.movement_rate_px_s ?? null,
    persistence_count: row.persistence_count ?? null,
    quality_status: row.quality_status ?? null,
    reason: row.reason ?? row.extraction_reason ?? null,
    explanation: null,
    contributing_factors: {
      camera_motion: row.camera_motion_gate ?? null,
      rolling_baseline_px: row.rolling_baseline_px ?? null,
      rolling_scale_px: row.rolling_scale_px ?? null,
      cumulative_drawdown_px: row.cumulative_drawdown_px ?? null,
      baseline_frozen: false,
    },
    acknowledged: false,
    created_at: observedAt,
    detector_mode: "rolling_baseline",
    resolved_detector_mode: "rolling_baseline",
    mode_reason: "test lab rolling-baseline run",
    onset_frame_index: null,
    rolling_baseline_px: row.rolling_baseline_px ?? null,
    rolling_scale_px: row.rolling_scale_px ?? null,
    scale_floored: false,
    cumulative_drawdown_px: row.cumulative_drawdown_px ?? null,
    baseline_frozen: false,
    event_max_classification: row.event_max_classification ?? null,
    onset_time: row.onset_time ?? null,
    detection_time: row.detection_time ?? null,
    detection_latency_s: null,
    event_window_seconds: null,
    cooldown_state: null,
    config_hash: null,
  };
}
