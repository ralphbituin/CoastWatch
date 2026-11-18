/** Typed client for the CoastWatch tide API. */

import { API_V1 } from "@/lib/config";
import type {
  ImportCommitResponse,
  ImportPreviewResponse,
  TideChartResponse,
  TidePhaseResponse,
  TideStation,
} from "@/lib/types";

async function tideRequest<T>(path: string, init?: RequestInit): Promise<T> {
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

export const tidesApi = {
  listStations: () => tideRequest<TideStation[]>("/tides/stations"),

  current: (stationCode: string) =>
    tideRequest<TidePhaseResponse>(`/tides/current?station_code=${encodeURIComponent(stationCode)}`),

  at: (stationCode: string, timestampLocal: string, method: "pchip" | "linear" = "pchip") => {
    const params = new URLSearchParams({
      station_code: stationCode,
      timestamp: timestampLocal,
      method,
    });
    return tideRequest<TidePhaseResponse>(`/tides/at?${params}`);
  },

  chart: (
    stationCode: string,
    start: Date,
    end: Date,
    method: "pchip" | "linear" = "pchip",
  ) => {
    const params = new URLSearchParams({
      station_code: stationCode,
      start: start.toISOString(),
      end: end.toISOString(),
      method,
    });
    return tideRequest<TideChartResponse>(`/tides/chart?${params}`);
  },

  listImports: (stationCode?: string, limit = 10) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (stationCode) params.set("station_code", stationCode);
    return tideRequest<{ imports: import("@/lib/types").DatasetImportRead[]; total: number }>(
      `/tides/imports?${params}`,
    );
  },

  previewImport: (file: File, timezone: string, columnMapping?: Record<string, string>) => {
    const form = new FormData();
    form.append("file", file);
    form.append("timezone", timezone);
    if (columnMapping) form.append("column_mapping", JSON.stringify(columnMapping));
    return tideRequest<ImportPreviewResponse>("/tides/imports/preview", {
      method: "POST",
      body: form,
    });
  },

  commitImport: (
    file: File,
    fields: {
      station_code: string;
      name: string;
      timezone?: string;
      unit?: string;
      vertical_datum?: string;
      description?: string;
      source?: string;
      columnMapping?: Record<string, string>;
    },
  ) => {
    const form = new FormData();
    form.append("file", file);
    form.append("station_code", fields.station_code);
    form.append("name", fields.name);
    form.append("timezone", fields.timezone ?? "Asia/Manila");
    form.append("unit", fields.unit ?? "meters");
    if (fields.vertical_datum) form.append("vertical_datum", fields.vertical_datum);
    if (fields.description) form.append("description", fields.description);
    form.append("source", fields.source ?? "NAMRIA_CSV");
    if (fields.columnMapping) form.append("column_mapping", JSON.stringify(fields.columnMapping));
    return tideRequest<ImportCommitResponse>("/tides/imports", { method: "POST", body: form });
  },
};
