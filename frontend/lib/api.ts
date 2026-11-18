/** Minimal typed API client for the CoastWatch backend. */

import { API_V1 } from "@/lib/config";
import type { DatabaseHealthResponse, HealthResponse } from "@/lib/types";

export class ApiRequestError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code?: string,
  ) {
    super(message);
    this.name = "ApiRequestError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_V1}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });

  if (!response.ok) {
    let code: string | undefined;
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      code = body?.error?.code;
      message = body?.error?.message ?? message;
    } catch {
      // Non-JSON error body; keep the default message.
    }
    throw new ApiRequestError(message, response.status, code);
  }

  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  healthDb: () => request<DatabaseHealthResponse>("/health/db"),
};
