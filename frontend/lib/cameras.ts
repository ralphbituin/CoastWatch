/** Typed client for the CoastWatch cameras API. */

import { API_V1 } from "@/lib/config";
import type {
  Camera,
  CameraCalibration,
  CameraStatus,
  CameraTestConnectionResult,
  CameraType,
} from "@/lib/types";

/** Base URL for the JPEG frame endpoint; append a cache-busting query when refreshing. */
export const cameraFrameUrl = (cameraId: number) => `${API_V1}/cameras/${cameraId}/frame`;

async function cameraRequest<T>(path: string, init?: RequestInit): Promise<T> {
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

export const camerasApi = {
  list: () => cameraRequest<Camera[]>("/cameras"),

  get: (cameraId: number) => cameraRequest<Camera>(`/cameras/${cameraId}`),

  create: (payload: {
    code: string;
    name: string;
    monitoring_station_id: number;
    camera_type: CameraType;
    rtsp_url?: string | null;
    device_index?: number | null;
    resolution_width?: number | null;
    resolution_height?: number | null;
    fps?: number | null;
    latitude?: number | null;
    longitude?: number | null;
    orientation_deg?: number | null;
    monitoring_region?: Record<string, unknown> | null;
    enabled?: boolean;
  }) =>
    cameraRequest<Camera>("/cameras", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  status: (cameraId: number) => cameraRequest<CameraStatus>(`/cameras/${cameraId}/status`),

  testConnection: (cameraId: number) =>
    cameraRequest<CameraTestConnectionResult>(`/cameras/${cameraId}/test-connection`, {
      method: "POST",
    }),

  uploadFrame: (cameraId: number, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return cameraRequest<{ message: string }>(`/cameras/${cameraId}/upload-frame`, {
      method: "POST",
      body: form,
    });
  },

  listCalibrations: (cameraId: number) =>
    cameraRequest<CameraCalibration[]>(`/cameras/${cameraId}/calibrations`),

  createCalibration: (
    cameraId: number,
    payload: {
      reference_points?: Record<string, unknown> | null;
      homography_matrix?: unknown[][] | null;
      meters_per_pixel?: number | null;
      notes?: string | null;
      set_active?: boolean;
    },
  ) =>
    cameraRequest<CameraCalibration>(`/cameras/${cameraId}/calibrations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};
