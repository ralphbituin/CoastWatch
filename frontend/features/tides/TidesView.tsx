"use client";

import { useCallback, useEffect, useState } from "react";
import { CalendarClock, RefreshCw } from "lucide-react";

import { Card } from "@/components/Card";
import { PageHeader } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { ImportPanel } from "@/features/tides/ImportPanel";
import { TideChart } from "@/features/tides/TideChart";
import { TidalSnapshot } from "@/features/tides/TidalSnapshot";
import { tidesApi } from "@/lib/tides";
import type {
  DatasetImportRead,
  TideChartResponse,
  TidePhaseResponse,
  TideStation,
} from "@/lib/types";

function formatForInput(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function TidesView() {
  const [stations, setStations] = useState<TideStation[]>([]);
  const [stationsLoading, setStationsLoading] = useState(true);
  const [selectedCode, setSelectedCode] = useState<string>("");

  const [current, setCurrent] = useState<TidePhaseResponse | null>(null);
  const [currentError, setCurrentError] = useState<string | null>(null);
  const [currentLoading, setCurrentLoading] = useState(false);

  const [selectedAt, setSelectedAt] = useState<TidePhaseResponse | null>(null);
  const [atError, setAtError] = useState<string | null>(null);
  const [atLoading, setAtLoading] = useState(false);
  const [datetimeLocal, setDatetimeLocal] = useState(() => formatForInput(new Date()));

  const [chart, setChart] = useState<TideChartResponse | null>(null);
  const [chartLoading, setChartLoading] = useState(true);
  const [chartError, setChartError] = useState<string | null>(null);
  const [rangeHours, setRangeHours] = useState(48);
  const [method, setMethod] = useState<"pchip" | "linear">("pchip");

  const [imports, setImports] = useState<DatasetImportRead[]>([]);
  const [pageError, setPageError] = useState<string | null>(null);

  const reloadImports = useCallback(async (code: string) => {
    try {
      const { imports: rows } = await tidesApi.listImports(code || undefined, 5);
      setImports(rows);
    } catch {
      // Non-critical: the import list is informational.
      setImports([]);
    }
  }, []);

  const loadStations = useCallback(async () => {
    setStationsLoading(true);
    try {
      const stationList = await tidesApi.listStations();
      setStations(stationList);
      setPageError(null);
      if (stationList.length > 0 && !stationList.some((s) => s.code === selectedCode)) {
        setSelectedCode(stationList[0].code);
      }
    } catch (err) {
      setPageError(err instanceof Error ? err.message : "Failed to load tide stations");
    } finally {
      setStationsLoading(false);
    }
  }, [selectedCode]);

  const refreshCurrent = useCallback(async (code: string) => {
    setCurrentLoading(true);
    try {
      setCurrent(await tidesApi.current(code));
      setCurrentError(null);
    } catch (err) {
      setCurrent(null);
      setCurrentError(err instanceof Error ? err.message : "Failed to load tide prediction");
    } finally {
      setCurrentLoading(false);
    }
  }, []);

  const refreshChart = useCallback(
    async (code: string, hours: number, m: "pchip" | "linear") => {
      setChartLoading(true);
      try {
        const half = (hours / 2) * 3600 * 1000;
        const now = Date.now();
        setChart(await tidesApi.chart(code, new Date(now - half), new Date(now + half), m));
        setChartError(null);
      } catch (err) {
        setChart(null);
        setChartError(err instanceof Error ? err.message : "Failed to load tide chart");
      } finally {
        setChartLoading(false);
      }
    },
    [],
  );

  const predictAt = useCallback(async () => {
    if (!selectedCode || !datetimeLocal) return;
    setAtLoading(true);
    setAtError(null);
    try {
      // Naive local input is interpreted in the station's time zone by the API.
      setSelectedAt(await tidesApi.at(selectedCode, datetimeLocal, method));
    } catch (err) {
      setSelectedAt(null);
      setAtError(err instanceof Error ? err.message : "Failed to load prediction");
    } finally {
      setAtLoading(false);
    }
  }, [selectedCode, datetimeLocal, method]);

  useEffect(() => {
    void loadStations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selectedCode) return;
    void refreshCurrent(selectedCode);
    void refreshChart(selectedCode, rangeHours, method);
    void reloadImports(selectedCode);
    const timer = setInterval(() => void refreshCurrent(selectedCode), 60_000);
    return () => clearInterval(timer);
  }, [selectedCode, rangeHours, method, refreshCurrent, refreshChart, reloadImports]);

  const selectedStation = stations.find((s) => s.code === selectedCode);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Tide predictions"
        description="Tide stations, CSV import, interpolated predictions and high/low water events."
      />

      {pageError && (
        <Card className="border-red-200 bg-red-50 text-sm text-red-700">
          {pageError} — is the backend running on the configured URL?
        </Card>
      )}

      <Card title="Tide station">
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex-1">
            <span className="mb-1 block text-xs font-medium text-slate-500">Station</span>
            <select
              className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
              value={selectedCode}
              onChange={(e) => setSelectedCode(e.target.value)}
              disabled={stationsLoading}
            >
              {stations.map((s) => (
                <option key={s.id} value={s.code}>
                  {s.code} · {s.name}
                </option>
              ))}
            </select>
          </label>
          {selectedStation && (
            <div className="flex flex-wrap gap-2 text-xs text-slate-500">
              <span className="rounded bg-slate-100 px-2 py-1">datum: {selectedStation.vertical_datum ?? "n/a"}</span>
              <span className="rounded bg-slate-100 px-2 py-1">unit: {selectedStation.measurement_unit}</span>
              <span className="rounded bg-slate-100 px-2 py-1">source: {selectedStation.data_source}</span>
              <span className="rounded bg-slate-100 px-2 py-1">tz: {selectedStation.timezone}</span>
              <span className="rounded bg-slate-100 px-2 py-1">{selectedStation.dataset_count} dataset(s)</span>
            </div>
          )}
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <TidalSnapshot title="CURRENT TIME" data={current} loading={currentLoading} error={currentError} />

        <Card title="SELECTED TIME" subtitle="Pick a date & time — predictions use the station's local time zone">
          <div className="mb-3 flex flex-wrap items-end gap-2">
            <label className="flex-1">
              <span className="mb-1 block text-xs font-medium text-slate-500">Date / time (local)</span>
              <input
                type="datetime-local"
                className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                value={datetimeLocal}
                onChange={(e) => setDatetimeLocal(e.target.value)}
              />
            </label>
            <button
              onClick={() => void predictAt()}
              disabled={atLoading || !datetimeLocal}
              className="inline-flex items-center gap-2 rounded-md bg-reef px-4 py-2 text-sm font-medium text-white hover:bg-lagoon disabled:opacity-40"
            >
              <CalendarClock className="h-4 w-4" aria-hidden />
              Predict
            </button>
            <button
              onClick={() => {
                setDatetimeLocal(formatForInput(new Date()));
                void predictAt();
              }}
              className="rounded-md border border-slate-200 p-2 text-slate-500 hover:bg-slate-50"
              aria-label="Jump to now and predict"
            >
              <RefreshCw className="h-4 w-4" aria-hidden />
            </button>
          </div>
          <TidalSnapshot
            title=""
            data={selectedAt}
            loading={atLoading}
            error={atError}
          />
        </Card>
      </div>

      <TideChart
        data={chart}
        loading={chartLoading}
        error={chartError}
        rangeHours={rangeHours}
        onRangeChange={setRangeHours}
        method={method}
        onMethodChange={setMethod}
        onRefresh={() => selectedCode && void refreshChart(selectedCode, rangeHours, method)}
      />

      <ImportPanel
        stations={stations}
        selectedStationCode={selectedCode}
        onSelectStation={setSelectedCode}
        onImported={() => {
          void loadStations();
          if (selectedCode) {
            void refreshCurrent(selectedCode);
            void refreshChart(selectedCode, rangeHours, method);
            void reloadImports(selectedCode);
          }
        }}
      />

      <Card title="Recent imports">
        {imports.length === 0 ? (
          <p className="text-sm text-slate-400">No imports recorded for this station yet.</p>
        ) : (
          <ul className="divide-y divide-slate-100 text-sm">
            {imports.map((row) => (
              <li key={row.id} className="flex flex-wrap items-center justify-between gap-2 py-2">
                <span className="text-slate-700">
                  <span className="mr-2 text-xs text-slate-400">#{row.id}</span>
                  {row.source_filename ?? "unnamed file"}
                </span>
                <span className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                  <span>{row.imported_records}/{row.total_records} imported</span>
                  <StatusBadge
                    tone={
                      row.status === "committed"
                        ? "ok"
                        : row.status === "duplicate"
                          ? "neutral"
                          : "warn"
                    }
                  >
                    {row.status}
                  </StatusBadge>
                  <span>{new Date(row.created_at).toLocaleString("en-PH")}</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
