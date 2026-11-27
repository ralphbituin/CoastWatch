/** Typed client for the Phase 6 anomaly-event API. */

import { API_V1 } from "@/lib/config";
import type {
  AnomalyEvent,
  AnomalyEventList,
  AnomalySummary,
  ShorelineObservationDetail,
} from "@/lib/types";

async function anomalyRequest<T>(path: string, init?: RequestInit): Promise<T> {
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

export const anomaliesApi = {
  list: (
    params: {
      camera_id?: number;
      station_id?: number;
      recording_id?: string;
      state?: string;
      classification?: string;
      baseline_status?: string;
      limit?: number;
      offset?: number;
    } = {},
  ) => {
    const q = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") q.set(key, String(value));
    }
    const qs = q.toString();
    return anomalyRequest<AnomalyEventList>(`/anomalies${qs ? `?${qs}` : ""}`);
  },

  summary: (params: { recording_id?: string; camera_id?: number; station_id?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.recording_id) q.set("recording_id", params.recording_id);
    if (params.camera_id) q.set("camera_id", String(params.camera_id));
    if (params.station_id) q.set("station_id", String(params.station_id));
    const qs = q.toString();
    return anomalyRequest<AnomalySummary>(`/anomalies/summary${qs ? `?${qs}` : ""}`);
  },

  current: (params: { camera_id?: number; recording_id?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.camera_id) q.set("camera_id", String(params.camera_id));
    if (params.recording_id) q.set("recording_id", params.recording_id);
    const qs = q.toString();
    return anomalyRequest<AnomalyEvent>(`/anomalies/current${qs ? `?${qs}` : ""}`);
  },

  get: (eventId: number) => anomalyRequest<AnomalyEvent>(`/anomalies/${eventId}`),

  observation: (eventId: number) =>
    anomalyRequest<ShorelineObservationDetail>(`/anomalies/${eventId}/observation`),

  acknowledge: (eventId: number) =>
    anomalyRequest<{ id: number; acknowledged: boolean }>(
      `/anomalies/${eventId}/acknowledge`,
      { method: "POST" },
    ),
};
