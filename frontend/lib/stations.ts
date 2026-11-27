/** Typed client for the CoastWatch stations API. */

import { API_V1 } from "@/lib/config";
import type {
  Country,
  MonitoringStation,
  TideStation,
} from "@/lib/types";

async function stationRequest<T>(path: string, init?: RequestInit): Promise<T> {
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

export const stationsApi = {
  listCountries: () => stationRequest<Country[]>("/countries"),

  listMonitoringStations: () => stationRequest<MonitoringStation[]>("/stations"),

  updateMonitoringStation: (
    stationId: number,
    payload: {
      code?: string;
      name?: string;
      country_iso?: string;
      region?: string | null;
      latitude?: number;
      longitude?: number;
      timezone?: string;
      vertical_datum?: string | null;
      description?: string | null;
    },
  ) =>
    stationRequest<MonitoringStation>(`/stations/${stationId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  deleteMonitoringStation: (stationId: number) =>
    stationRequest<{ message: string }>(`/stations/${stationId}`, {
      method: "DELETE",
    }),

  createMonitoringStation: (payload: {
    code: string;
    name: string;
    country_iso: string;
    region?: string | null;
    latitude: number;
    longitude: number;
    timezone?: string;
    vertical_datum?: string | null;
    description?: string | null;
  }) =>
    stationRequest<MonitoringStation>("/stations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  listTideStations: () => stationRequest<TideStation[]>("/tides/stations"),

  createTideStation: (payload: {
    code: string;
    name: string;
    country_iso: string;
    region?: string | null;
    latitude: number;
    longitude: number;
    timezone?: string;
    data_source?: string;
    vertical_datum?: string | null;
    measurement_unit?: string;
    monitoring_station_id?: number | null;
  }) =>
    stationRequest<TideStation>("/tides/stations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  linkTideStation: (stationId: number, tideStationId: number) =>
    stationRequest<TideStation>(
      `/stations/${stationId}/tide-stations/${tideStationId}`,
      { method: "POST" },
    ),

  unlinkTideStation: (stationId: number, tideStationId: number) =>
    stationRequest<{ message: string }>(
      `/stations/${stationId}/tide-stations/${tideStationId}`,
      { method: "DELETE" },
    ),

  listLinked: (stationId: number) =>
    stationRequest<TideStation[]>(`/stations/${stationId}/tide-stations`),
};
