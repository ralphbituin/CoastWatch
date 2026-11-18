/**
 * Runtime configuration for the CoastWatch frontend.
 * Values come from public (NEXT_PUBLIC_*) environment variables so the same
 * image can target dev, staging or production backends.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export const API_V1 = `${API_BASE_URL}/api/v1`;

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";

export const WS_URL = `${WS_BASE}/api/v1/ws`;
