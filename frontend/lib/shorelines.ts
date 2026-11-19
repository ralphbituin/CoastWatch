/** Typed client for the CoastWatch shorelines API. */

import { API_V1 } from "@/lib/config";
import type {
  ShorelineCalibration,
  ShorelineModuleStatus,
  ShorelineObservationDetail,
  ShorelineObservationSummary,
  ShorelineObserveResult,
} from "@/lib/types";

async function shorelineRequest<T>(path: string, init?: RequestInit): Promise<T> {
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
      // Non-JSON error body.
    }
    throw new Error(`${code}: ${message}`);
  }

  return (await response.json()) as T;
}

export const shorelinesApi = {
  status: () => shorelineRequest<ShorelineModuleStatus>("/shorelines"),

  listObservations: (params: { camera_id?: number; station_id?: number; limit?: number; offset?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.camera_id) q.set("camera_id", String(params.camera_id));
    if (params.station_id) q.set("station_id", String(params.station_id));
    if (params.limit) q.set("limit", String(params.limit));
    if (params.offset) q.set("offset", String(params.offset));
    const qs = q.toString();
    return shorelineRequest<ShorelineObservationSummary[]>(
      `/shorelines/observations${qs ? `?${qs}` : ""}`,
    );
  },

  latestObservation: (params: { camera_id?: number; station_id?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.camera_id) q.set("camera_id", String(params.camera_id));
    if (params.station_id) q.set("station_id", String(params.station_id));
    const qs = q.toString();
    return shorelineRequest<ShorelineObservationDetail>(
      `/shorelines/observations/latest${qs ? `?${qs}` : ""}`,
    );
  },

  uploadMask: (payload: {
    camera_id: number;
    file: File;
    seaward_axis?: string;
    observed_at?: string;
  }) => {
    const form = new FormData();
    form.append("camera_id", String(payload.camera_id));
    if (payload.seaward_axis) form.append("seaward_axis", payload.seaward_axis);
    if (payload.observed_at) form.append("observed_at", payload.observed_at);
    form.append("file", payload.file);
    return shorelineRequest<ShorelineObservationDetail>("/shorelines/observations", {
      method: "POST",
      body: form,
    });
  },

  observe: (payload: { station_id: number; camera_id: number }) =>
    shorelineRequest<ShorelineObserveResult>("/shorelines/observe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  listCalibrations: () =>
    shorelineRequest<ShorelineCalibration[]>("/shorelines/calibrations"),
};
