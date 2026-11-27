/** Typed client for archive recordings and live stream sessions (Phase 7). */

import { API_V1 } from "@/lib/config";
import type {
  ShorelineObservationSummary,
  StreamSession,
  VideoRecording,
} from "@/lib/types";

async function recordingRequest<T>(path: string, init?: RequestInit): Promise<T> {
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

export const recordingsApi = {
  listVideos: (limit = 100) =>
    recordingRequest<VideoRecording[]>(`/videos?limit=${limit}`),

  listStreams: (limit = 100) =>
    recordingRequest<StreamSession[]>(`/streams?limit=${limit}`),

  videoObservations: (recordingId: string, limit = 200) =>
    recordingRequest<ShorelineObservationSummary[]>(
      `/videos/${recordingId}/observations?limit=${limit}`,
    ),

  streamObservations: (sessionId: string, limit = 200) =>
    recordingRequest<ShorelineObservationSummary[]>(
      `/streams/${sessionId}/observations?limit=${limit}`,
    ),

  deleteStream: (sessionId: string) =>
    recordingRequest<{ message: string }>(`/streams/${sessionId}`, {
      method: "DELETE",
    }),
};
