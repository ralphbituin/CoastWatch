"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Loader2, RefreshCw, Waves } from "lucide-react";
import clsx from "clsx";

import { Card } from "@/components/Card";
import { PageHeader } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { TideAnomalyChart } from "@/features/anomalies/TideAnomalyChart";
import { WaterlineVisualizer } from "@/features/shoreline/WaterlineVisualizer";
import { anomaliesApi } from "@/lib/anomalies";
import { shorelinesApi } from "@/lib/shorelines";
import { stationsApi } from "@/lib/stations";
import { useDisplayTimezone } from "@/lib/settings";
import type {
  AnomalyEvent,
  AnomalySummary,
  MonitoringStation,
  ShorelineBaselineStatus,
  ShorelineObservationDetail,
  ShorelineObservationSummary,
} from "@/lib/types";

function tone(classification: string | null): "ok" | "warn" | "error" | "neutral" {
  if (classification === "NORMAL") return "ok";
  if (classification === "DATA_UNRELIABLE") return "warn";
  if (classification && classification.includes("RETREAT")) return "error";
  if (
    classification?.includes("UNUSUAL") ||
    classification?.includes("DRAWDOWN")
  ) {
    return "warn";
  }
  return "neutral";
}

const OFFICIAL_WARNING_AUTHORITY =
  (process.env.NEXT_PUBLIC_OFFICIAL_WARNING_AUTHORITY ?? "").trim() ||
  "the official national tsunami warning authority";

export function AnomaliesView() {
  const { label: timezoneLabel, formatDateTime } = useDisplayTimezone();
  const [stations, setStations] = useState<MonitoringStation[]>([]);
  const [stationFilter, setStationFilter] = useState<number | "">("");
  const [events, setEvents] = useState<AnomalyEvent[]>([]);
  const [summary, setSummary] = useState<AnomalySummary | null>(null);
  const [observations, setObservations] = useState<ShorelineObservationSummary[]>([]);
  const [baseline, setBaseline] = useState<ShorelineBaselineStatus | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<AnomalyEvent | null>(null);
  const [selectedObservation, setSelectedObservation] =
    useState<ShorelineObservationDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingObservation, setLoadingObservation] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stationsError, setStationsError] = useState<string | null>(null);

  // The station list is independent of the anomaly/observation queries so a
  // failure in one of those never leaves the selector empty.
  const loadStations = useCallback(async () => {
    try {
      const stationList = await stationsApi.listMonitoringStations();
      setStations(stationList);
      setStationsError(null);
    } catch (err) {
      setStationsError(
        err instanceof Error ? err.message : "Failed to load monitoring stations",
      );
    }
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const stationId = stationFilter === "" ? undefined : stationFilter;
      const [eventList, eventSummary, baselineStatus, observationList] =
        await Promise.all([
          anomaliesApi.list({ station_id: stationId, limit: 200 }),
          anomaliesApi.summary({ station_id: stationId }),
          shorelinesApi.baselines({ station_id: stationId }),
          shorelinesApi.listObservations({ station_id: stationId, limit: 200 }),
        ]);
      setEvents(eventList.items);
      setSummary(eventSummary);
      setBaseline(baselineStatus);
      setObservations(observationList);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load anomaly events");
    } finally {
      setLoading(false);
    }
  }, [stationFilter]);

  useEffect(() => {
    void loadStations();
  }, [loadStations]);

  useEffect(() => {
    void load();
  }, [load]);

  const stationById = useMemo(
    () => new Map(stations.map((station) => [station.id, station])),
    [stations],
  );

  const stationRows = useMemo(() => {
    const byStation = new Map<
      number,
      { total: number; classifications: Record<string, number>; latest: string | null }
    >();
    for (const event of events) {
      const row = byStation.get(event.monitoring_station_id) ?? {
        total: 0,
        classifications: {},
        latest: null,
      };
      row.total += 1;
      const key = event.classification ?? event.state;
      row.classifications[key] = (row.classifications[key] ?? 0) + 1;
      if (!row.latest || event.detected_at > row.latest) row.latest = event.detected_at;
      byStation.set(event.monitoring_station_id, row);
    }
    return stations.map((station) => {
      const row = byStation.get(station.id);
      return {
        station,
        total: row?.total ?? 0,
        classifications: row?.classifications ?? {},
        latest: row?.latest ?? null,
      };
    });
  }, [events, stations]);

  const selectEvent = async (event: AnomalyEvent) => {
    setSelectedEvent(event);
    setLoadingObservation(true);
    try {
      const detail = await anomaliesApi.observation(event.id);
      setSelectedObservation(detail);
      setError(null);
    } catch (err) {
      setSelectedObservation(null);
      setError(
        err instanceof Error
          ? `Waterline observation unavailable: ${err.message}`
          : "Waterline observation unavailable",
      );
    } finally {
      setLoadingObservation(false);
    }
  };

  const selectObservation = async (observation: ShorelineObservationSummary) => {
    setSelectedEvent(null);
    setLoadingObservation(true);
    try {
      const detail = await shorelinesApi.observationDetail({
        camera_id: observation.camera_id,
        observed_at: observation.observed_at,
      });
      setSelectedObservation(detail);
      setError(null);
    } catch (err) {
      setSelectedObservation(null);
      setError(
        err instanceof Error
          ? `Waterline observation unavailable: ${err.message}`
          : "Waterline observation unavailable",
      );
    } finally {
      setLoadingObservation(false);
    }
  };

  const showStationColumn = stationFilter === "";

  return (
    <div className="space-y-6">
      <PageHeader
        title="Anomalies"
        description="Per-station anomaly data and stored waterline observations. Experimental rolling-baseline drawdown signal (tide-independent) or tide-conditioned residual when a verified baseline exists."
      />

      <Card className="border-amber-200 bg-amber-50 text-sm text-amber-800">
        <div className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <p>
            Experimental rolling-baseline drawdown detector — not validated for tsunami detection.
            Follow {OFFICIAL_WARNING_AUTHORITY}. A POSSIBLE_DRAWDOWN is an unverified monitoring
            signal, not a confirmed tsunami. Missing data is always reported as DATA_UNRELIABLE,
            never NORMAL, and NORMAL is only possible with a valid tide-conditioned baseline.
          </p>
        </div>
      </Card>

      {error && (
        <Card className="border-red-200 bg-red-50 text-sm text-red-700">{error}</Card>
      )}

      <Card className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <label className="text-xs font-medium text-slate-500" htmlFor="anomaly-station-filter">
              Station
            </label>
            <select
              id="anomaly-station-filter"
              className="min-w-56 rounded-md border border-slate-200 px-3 py-2 text-sm"
              value={stationFilter}
              onChange={(event) =>
                setStationFilter(event.target.value ? Number(event.target.value) : "")
              }
            >
              <option value="">All stations</option>
              {stations.map((station) => (
                <option key={station.id} value={station.id}>
                  {station.code} · {station.name}
                </option>
              ))}
            </select>
            {summary && (
              <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
                {summary.total} event(s)
              </span>
            )}
            <span className="text-xs text-slate-400">times: {timezoneLabel}</span>
            {stationsError && (
              <span role="status" className="text-xs text-red-600">
                Stations unavailable: {stationsError}
              </span>
            )}
          </div>
          <button
            onClick={() => {
              void load();
              void loadStations();
            }}
            disabled={loading}
            className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-40"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            Refresh
          </button>
        </div>

        {stationFilter === "" && stationRows.length > 0 && (
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {stationRows.map(({ station, total, classifications, latest }) => (
              <button
                key={station.id}
                onClick={() => setStationFilter(station.id)}
                className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-left transition-colors hover:border-reef hover:bg-white"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium text-slate-800">
                    {station.code} · {station.name}
                  </span>
                  <span className="rounded bg-white px-2 py-0.5 text-xs text-slate-600">
                    {total} event(s)
                  </span>
                </div>
                <div className="mt-1 flex flex-wrap gap-1 text-[11px] text-slate-500">
                  {Object.entries(classifications).length === 0 ? (
                    <span>no events in the current window</span>
                  ) : (
                    Object.entries(classifications).map(([key, value]) => (
                      <span key={key} className="rounded bg-slate-200/60 px-1.5 py-0.5">
                        {key}: {value}
                      </span>
                    ))
                  )}
                </div>
                <p className="mt-1 text-[11px] text-slate-400">
                  latest {latest ? formatDateTime(latest) : "—"} · {station.timezone}
                </p>
              </button>
            ))}
          </div>
        )}
      </Card>

      <Card
        title={
          baseline?.detector_mode === "rolling_baseline"
            ? "Rolling baseline, shoreline position and drawdown"
            : "Tide, shoreline position and residual"
        }
        subtitle="Sorted by time. Gaps larger than 45 s are not joined, and a detector-state strip is included. The rolling median is self-referential and unverified; click a marker to inspect that event."
        className="space-y-2"
      >
        <div className="mb-3 flex flex-wrap gap-2 text-xs text-slate-500">
          {summary &&
            Object.entries(summary.by_classification).map(([key, value]) => (
              <span key={key} className="rounded bg-slate-100 px-2 py-0.5">
                {key}: {value}
              </span>
            ))}
        </div>
        <TideAnomalyChart
          events={events}
          baselineStatus={baseline?.status}
          baselineReason={baseline?.reason}
          timeZone={timezoneLabel}
          selectedEventId={selectedEvent?.id ?? null}
          onEventClick={(eventId) => {
            const event = events.find((item) => item.id === eventId);
            if (event) void selectEvent(event);
          }}
        />
      </Card>

      <Card
        title="Waterline observation"
        subtitle="Click an anomaly event or a stored observation to render its extracted Ocean/Beach line"
      >
        {loadingObservation ? (
          <p className="flex items-center justify-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-10 text-sm text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            Loading waterline observation…
          </p>
        ) : selectedObservation ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
              <StatusBadge tone="ok">{selectedObservation.camera_code ?? "camera"}</StatusBadge>
              {selectedObservation.station_code ? (
                <span className="rounded bg-slate-100 px-2 py-0.5">
                  station {selectedObservation.station_code}
                </span>
              ) : null}
              <span>{formatDateTime(selectedObservation.observed_at)}</span>
              <span className="text-slate-400">({timezoneLabel})</span>
              {selectedObservation.recording_id ? (
                <span className="rounded bg-slate-100 px-2 py-0.5">
                  {selectedObservation.recording_id}
                  {selectedObservation.frame_index !== null &&
                  selectedObservation.frame_index !== undefined
                    ? ` · frame ${selectedObservation.frame_index}`
                    : ""}
                </span>
              ) : null}
              <span className="rounded bg-slate-100 px-2 py-0.5">
                {selectedObservation.extraction_status ?? "observation"}
                {selectedObservation.segment_count !== null &&
                selectedObservation.segment_count !== undefined
                  ? ` · ${selectedObservation.segment_count} segment(s)`
                  : ""}
              </span>
              {selectedObservation.shoreline_position_px !== null &&
              selectedObservation.shoreline_position_px !== undefined ? (
                <span className="rounded bg-slate-100 px-2 py-0.5">
                  position {selectedObservation.shoreline_position_px.toFixed(1)} px
                </span>
              ) : null}
            </div>
            {selectedEvent && (
              <p className="text-xs text-slate-500">
                Event #{selectedEvent.id} · {selectedEvent.classification ?? selectedEvent.state}
                {selectedEvent.reason ? ` · ${selectedEvent.reason}` : ""}
              </p>
            )}
            <WaterlineVisualizer detail={selectedObservation} heightClass="h-72" />
          </div>
        ) : (
          <div className="space-y-2">
            <WaterlineVisualizer
              detail={null}
              emptyMessage="Select an event in the chart or table, or a stored observation below."
              heightClass="h-56"
            />
            <p className="flex items-center gap-2 text-xs text-slate-500">
              <Waves className="h-4 w-4 text-reef" aria-hidden />
              Each event links back to the exact camera/recording/frame observation when it exists.
            </p>
          </div>
        )}
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card
          title="Recent events"
          subtitle="Scrolls inside the card; click a row to show its waterline observation"
        >
          {events.length === 0 ? (
            <p className="text-sm text-slate-400">
              No anomaly events yet. Run the prototype against a registered recording.
            </p>
          ) : (
            <div className="max-h-[28rem] overflow-auto">
              <table className="min-w-full text-left text-xs">
                <thead className="sticky top-0 z-10 bg-slate-50 text-slate-500">
                  <tr>
                    <th className="py-2 pr-3">Detected</th>
                    {showStationColumn && <th className="py-2 pr-3">Station</th>}
                    <th className="py-2 pr-3">Classification</th>
                    <th className="py-2 pr-3">Observed px</th>
                    <th className="py-2 pr-3">Baseline px</th>
                    <th className="py-2 pr-3">Residual px</th>
                    <th className="py-2 pr-3">Robust z</th>
                    <th className="py-2 pr-3">Drawdown px</th>
                    <th className="py-2 pr-3">Rate px/s</th>
                    <th className="py-2 pr-3">Tide</th>
                    <th className="py-2 pr-3">Persistence</th>
                    <th className="py-2 pr-3">Reason</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {events.map((event) => (
                    <tr
                      key={event.id}
                      onClick={() => void selectEvent(event)}
                      className={clsx(
                        "cursor-pointer transition-colors hover:bg-slate-50",
                        selectedEvent?.id === event.id && "bg-reef/5",
                      )}
                    >
                      <td className="whitespace-nowrap py-2 pr-3 text-slate-600">
                        {formatDateTime(event.detected_at)}
                      </td>
                      {showStationColumn && (
                        <td className="whitespace-nowrap py-2 pr-3 text-slate-600">
                          {stationById.get(event.monitoring_station_id)?.code ??
                            `station #${event.monitoring_station_id}`}
                        </td>
                      )}
                      <td className="py-2 pr-3">
                        <StatusBadge tone={tone(event.classification)}>
                          {event.classification ?? event.state}
                        </StatusBadge>
                      </td>
                      <td className="py-2 pr-3">{event.observed_position_px?.toFixed(1) ?? "—"}</td>
                      <td className="py-2 pr-3">
                        {(event.rolling_baseline_px ?? event.expected_position_px)?.toFixed(1) ?? "—"}
                      </td>
                      <td className="py-2 pr-3">{event.residual_px?.toFixed(1) ?? "—"}</td>
                      <td className="py-2 pr-3">{event.robust_z?.toFixed(2) ?? "—"}</td>
                      <td className="py-2 pr-3">
                        {event.cumulative_drawdown_px?.toFixed(1) ?? "—"}
                      </td>
                      <td className="py-2 pr-3">
                        {event.movement_rate_px_s?.toFixed(3) ?? "—"}
                        {event.movement_rate_px_s !== null &&
                          event.movement_rate_px_s !== undefined && (
                            <span className="ml-1 text-slate-400">
                              {event.movement_rate_px_s > 0 ? "seaward" : "landward"}
                            </span>
                          )}
                      </td>
                      <td className="py-2 pr-3">
                        {event.tide_data_available
                          ? `${event.tide_height_m?.toFixed(3)} m`
                          : "unavailable"}
                      </td>
                      <td className="py-2 pr-3">{event.persistence_count ?? 0}</td>
                      <td className="max-w-[24rem] py-2 pr-3 text-slate-500">
                        {event.reason ?? event.explanation}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card
          title="Station observations"
          subtitle="Stored waterline observations for this station selection; click one to render it"
        >
          {observations.length === 0 ? (
            <p className="text-sm text-slate-400">
              No shoreline observations stored for this station selection.
            </p>
          ) : (
            <div className="max-h-[28rem] overflow-auto">
              <table className="min-w-full text-left text-xs">
                <thead className="sticky top-0 z-10 bg-slate-50 text-slate-500">
                  <tr>
                    <th className="py-2 pr-3">Observed</th>
                    {showStationColumn && <th className="py-2 pr-3">Station</th>}
                    <th className="py-2 pr-3">Camera</th>
                    <th className="py-2 pr-3">Status · segments</th>
                    <th className="py-2 pr-3">Position px</th>
                    <th className="py-2 pr-3">Recording · frame</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {observations.map((observation, index) => {
                    const selected =
                      selectedObservation?.camera_id === observation.camera_id &&
                      new Date(selectedObservation.observed_at).getTime() ===
                        new Date(observation.observed_at).getTime();
                    return (
                      <tr
                        key={`${observation.camera_id}-${observation.observed_at}-${index}`}
                        onClick={() => void selectObservation(observation)}
                        className={clsx(
                          "cursor-pointer transition-colors hover:bg-slate-50",
                          selected && "bg-reef/5",
                        )}
                      >
                        <td className="whitespace-nowrap py-2 pr-3 text-slate-600">
                          {formatDateTime(observation.observed_at)}
                        </td>
                        {showStationColumn && (
                          <td className="whitespace-nowrap py-2 pr-3 text-slate-600">
                            {observation.station_code ??
                              stationById.get(observation.monitoring_station_id)?.code ??
                              `station #${observation.monitoring_station_id}`}
                          </td>
                        )}
                        <td className="py-2 pr-3 text-slate-600">
                          {observation.camera_code ?? `camera #${observation.camera_id}`}
                        </td>
                        <td className="py-2 pr-3">
                          {observation.extraction_status ?? "—"} ·{" "}
                          {observation.segment_count ?? 0} seg
                        </td>
                        <td className="py-2 pr-3">
                          {observation.shoreline_position_px !== null &&
                          observation.shoreline_position_px !== undefined
                            ? observation.shoreline_position_px.toFixed(1)
                            : "—"}
                        </td>
                        <td className="py-2 pr-3 text-slate-500">
                          {observation.recording_id ?? "—"}
                          {observation.frame_index !== null &&
                          observation.frame_index !== undefined
                            ? ` · ${observation.frame_index}`
                            : ""}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
