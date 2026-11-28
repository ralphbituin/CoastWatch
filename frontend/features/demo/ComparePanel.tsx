"use client";

import { useState } from "react";

import { demoApi, type CompareResult } from "@/features/demo/api";
import { DEMO_WATERMARK } from "@/features/demo/demoSeries";

export function ComparePanel({
  defaultAmplitude,
  defaultRamp,
}: {
  defaultAmplitude: number;
  defaultRamp: number;
}) {
  const [amplitude, setAmplitude] = useState(defaultAmplitude);
  const [ramp, setRamp] = useState(defaultRamp);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      setResult(await demoApi.compare(amplitude, ramp));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Comparison failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="relative overflow-hidden rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <span className="-rotate-12 text-xl font-bold uppercase tracking-widest text-red-500/10">
          {DEMO_WATERMARK}
        </span>
      </div>
      <header className="mb-3">
        <h2 className="text-sm font-semibold text-slate-900">Compare recordings</h2>
        <p className="text-xs text-slate-500">
          Same amplitude percent and ramp on each recording&apos;s own scene and noise. Recordings
          are not combined.
        </p>
      </header>
      <div className="mb-3 flex flex-wrap items-end gap-3 text-xs">
        <label className="flex flex-col gap-1">
          <span className="font-medium text-slate-600">Amplitude (% frame height)</span>
          <input
            type="number"
            min={5}
            max={60}
            step={0.1}
            value={amplitude}
            onChange={(event) => setAmplitude(Number(event.target.value))}
            className="w-28 rounded border border-slate-300 px-2 py-1"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-medium text-slate-600">Ramp (s)</span>
          <input
            type="number"
            min={1}
            step={1}
            value={ramp}
            onChange={(event) => setRamp(Number(event.target.value))}
            className="w-28 rounded border border-slate-300 px-2 py-1"
          />
        </label>
        <button
          type="button"
          onClick={run}
          disabled={loading}
          className="rounded bg-slate-900 px-3 py-1.5 font-semibold text-white disabled:opacity-50"
        >
          {loading ? "Replaying..." : "Compare"}
        </button>
      </div>
      {error && <p className="mb-2 text-xs text-red-700">{error}</p>}
      {result && (
        <div className="space-y-3">
          <p className="text-xs text-slate-500">
            Amplitude {result.amplitude_pct}% of frame height, ramp {result.ramp_s} s, hold{" "}
            {result.hold_s} s. {result.note}
          </p>
          {result.recordings.map((row) => {
            const rate = row.available ? (row.detection_rate ?? 0) * 100 : 0;
            const low = row.available ? (row.wilson_lo ?? 0) * 100 : 0;
            const high = row.available ? (row.wilson_hi ?? 0) * 100 : 0;
            return (
              <div key={row.id} className="rounded-lg border border-slate-200 p-3">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <p className="text-sm font-semibold text-slate-900">{row.label}</p>
                  <span className="text-xs text-slate-500">
                    {row.available
                      ? `${rate.toFixed(0)}% (95% CI ${low.toFixed(0)}-${high.toFixed(0)}%), n=${row.n_in_bounds ?? 0} / ${row.n_starts ?? 0} starts`
                      : row.reason ?? "not replayable"}
                  </span>
                </div>
                {row.available ? (
                  <>
                    <div className="mt-2 h-3 w-full rounded bg-slate-100">
                      <div
                        className="h-3 rounded bg-sky-600"
                        style={{ width: `${Math.max(0, Math.min(100, rate))}%` }}
                      />
                    </div>
                    <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
                      <span>amplitude {row.amplitude_px?.toFixed(0)} px</span>
                      <span>noise std {row.step_std_px?.toFixed(1) ?? "-"} px</span>
                      <span>median |90 s| {row.move90_median_px?.toFixed(1) ?? "-"} px</span>
                      <span>gate median {row.gate_median_px?.toFixed(0) ?? "-"} px</span>
                      <span>gate p95 {row.gate_p95_px?.toFixed(0) ?? "-"} px</span>
                      <span>false events {row.false_events ?? 0}</span>
                    </div>
                    {row.limited_data && (
                      <p className="mt-1 text-[11px] font-medium text-amber-700">
                        Limited data: only {row.valid_start_count} valid starts. Treat this bar as
                        illustrative, not a rate.
                      </p>
                    )}
                  </>
                ) : (
                  <p className="mt-1 text-xs text-amber-700">
                    No simulation on this recording: {row.reason}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
      {!result && (
        <p className="text-xs text-slate-500">
          Press Compare. Each bar is a simulation on that recording alone; it is not a
          real-world sensitivity or false-alarm claim.
        </p>
      )}
      <p className="mt-3 text-[11px] text-slate-400">
        Time zone Asia/Manila. SIMULATION - not real data.
      </p>
    </section>
  );
}
