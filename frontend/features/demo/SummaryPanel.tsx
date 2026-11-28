"use client";

import { useEffect, useState } from "react";

import { demoApi, type SummaryResponse } from "@/features/demo/api";
import { DEMO_WATERMARK, formatDateManila } from "@/features/demo/demoSeries";

const FAILURE_LABELS: Record<string, string> = {
  z_not_reached: "z not reached",
  movement_gate_not_reached: "movement gate not reached",
  z_and_gate_not_simultaneous: "z and gate not simultaneous",
  persistence_not_reached: "persistence not reached",
  no_event_in_window: "no event in window",
};

export function SummaryPanel({ recordingId }: { recordingId: string }) {
  const [state, setState] = useState<SummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function poll() {
      try {
        const next = await demoApi.summary(recordingId);
        if (cancelled) return;
        setState(next);
        if (next.status === "running") {
          timer = setTimeout(poll, 2000);
        }
      } catch (exc) {
        if (!cancelled) {
          setError(exc instanceof Error ? exc.message : "Summary failed");
        }
      }
    }

    setState(null);
    setError(null);
    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [recordingId]);

  const result = state?.result ?? null;
  const fraction = Math.round((state?.fraction ?? 0) * 100);

  return (
    <section className="relative overflow-hidden rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <span className="-rotate-12 text-xl font-bold uppercase tracking-widest text-red-500/10">
          {DEMO_WATERMARK}
        </span>
      </div>
      <header className="mb-3">
        <h2 className="text-sm font-semibold text-slate-900">
          What would a drawdown show on this recording?
        </h2>
        <p className="text-xs text-slate-500">
          Quick estimate: amplitude as percent of frame height, shared start times, independent
          replays. It is cached and runs in the background.
        </p>
      </header>
      {error && <p className="text-xs text-red-700">{error}</p>}
      {state?.status === "running" && (
        <div className="mb-3">
          <div className="h-2 w-full rounded bg-slate-100">
            <div className="h-2 rounded bg-sky-600" style={{ width: `${fraction}%` }} />
          </div>
          <p className="mt-1 text-xs text-slate-500">
            {state.completed_cells ?? 0} / {state.total_cells ?? "?"} cells,{" "}
            {state.completed_replays ?? 0} replays ({fraction}%).
          </p>
        </div>
      )}
      {state?.status === "error" && (
        <p className="text-xs text-red-700">Summary failed: {state.error}</p>
      )}
      {state?.status === "not_replayable" && (
        <p className="text-xs text-amber-700">Not computed: {state.reason}</p>
      )}
      {result && (
        <div className="space-y-3">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  <th className="py-1 pr-3">Ramp</th>
                  <th className="py-1 pr-3">Smallest amplitude with &gt;=90%</th>
                  <th className="py-1 pr-3">Detection rate</th>
                </tr>
              </thead>
              <tbody>
                {result.minimum_90.map((row) => (
                  <tr key={row.ramp_s} className="border-t border-slate-100">
                    <td className="py-1 pr-3">{row.ramp_s.toFixed(0)} s</td>
                    <td className="py-1 pr-3">
                      {row.min_amplitude_px === null
                        ? "none in tested range"
                        : `${row.min_amplitude_px.toFixed(0)} px (${row.min_amplitude_pct_frame?.toFixed(1)}%)`}
                    </td>
                    <td className="py-1 pr-3">
                      {row.detection_rate === null ? "-" : `${(row.detection_rate * 100).toFixed(0)}%`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="grid gap-2 text-xs text-slate-600 sm:grid-cols-2">
            <div className="rounded border border-slate-200 p-2">
              <p className="font-semibold text-slate-900">Recording noise and gate</p>
              <p>sample-to-sample std: {result.noise.step_std_px?.toFixed(1) ?? "-"} px</p>
              <p>median |90 s move|: {result.noise.move90_median_px?.toFixed(1) ?? "-"} px</p>
              <p>p95 |90 s move|: {result.noise.move90_p95_px?.toFixed(1) ?? "-"} px</p>
              <p>effective gate median: {result.noise.gate_median_px?.toFixed(0) ?? "-"} px</p>
              <p>effective gate p95: {result.noise.gate_p95_px?.toFixed(0) ?? "-"} px</p>
            </div>
            <div className="rounded border border-slate-200 p-2">
              <p className="font-semibold text-slate-900">Failure modes and controls</p>
              {Object.entries(result.failure_modes).map(([key, value]) => (
                <p key={key}>
                  {FAILURE_LABELS[key] ?? key}: {value}
                </p>
              ))}
              <p>false events outside injection: {result.false_events_outside_injection}</p>
              <p>
                starts: {result.analysis_starts.count} ({result.analysis_starts.source})
              </p>
            </div>
          </div>
          <div
            className={`rounded border p-2 text-xs ${
              result.offline_comparison.available && result.offline_comparison.mismatches.length === 0
                ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                : "border-amber-200 bg-amber-50 text-amber-800"
            }`}
          >
            {result.offline_comparison.available ? (
              <>
                <p className="font-semibold">
                  Saved offline comparison: {result.offline_comparison.checked_cells} cells checked,{" "}
                  {result.offline_comparison.mismatches.length} mismatch(es).
                </p>
                {result.offline_comparison.mismatches.slice(0, 3).map((item) => (
                  <p key={`${item.amplitude_px}-${item.ramp_s}`}>
                    A {item.amplitude_px} px / R {item.ramp_s} s:{" "}
                    {JSON.stringify(item.differences)}
                  </p>
                ))}
              </>
            ) : (
              <p>No saved offline summary is available for this recording.</p>
            )}
          </div>
          <p className="text-[11px] text-slate-400">{result.caveat}</p>
          <p className="text-[11px] text-slate-400">
            SIMULATION - not real data. First timestamp{" "}
            {result.analysis_starts.starts_utc.length
              ? formatDateManila(result.analysis_starts.starts_utc[0])
              : "-"}{" "}
            Asia/Manila.
          </p>
        </div>
      )}
    </section>
  );
}
