/** Shared API types mirroring the backend Pydantic schemas. */

export type HealthStatus = "ok" | "degraded" | "error";

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  timestamp: string;
}

export interface DatabaseHealthResponse {
  status: string;
  database: string;
  reachable: boolean;
  latency_ms: number | null;
  server_version: string | null;
  timescaledb_version: string | null;
  postgis_version: string | null;
  error: string | null;
}

export interface ModuleStatusPayload {
  module: string;
  phase: number;
  implemented: boolean;
  message: string;
  planned_endpoints: string[];
}

export interface ApiError {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}

/** System states as defined by the anomaly engine contract. */
export type SystemState =
  | "NORMAL"
  | "ANOMALY_DETECTED"
  | "DATA_UNRELIABLE"
  | "OFFICIAL_TSUNAMI_ADVISORY";
