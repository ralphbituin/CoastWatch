"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ComparePanel } from "@/features/demo/ComparePanel";
import { DemoCharts } from "@/features/demo/DemoCharts";
import { SummaryPanel } from "@/features/demo/SummaryPanel";
import {
  demoApi,
  simulationStreamUrl,
  type Preset,
  type RecordingProfile,
  type SampleMessage,
  type SimEvent,
  type SimulationResult,
} from "@/features/demo/api";
import {
  DEMO_BANNER,
  DEMO_WATERMARK,
  describeAmplitude,
  expectedText,
  formatDateManila,
  statusLabel,
} from "@/features/demo/demoSeries";

type Params = {
  amplitude_pct: number;
  ramp_s: number;
  hold_s: number;
  shape: "retreat_return" | "persistent";
  sign: number;
  speed: number;
  start: string;
  seed: number;
};

function defaultParams(profile: RecordingProfile): Params {
  const gate = profile.gate_median_px ?? profile.drawdown_floor_px ?? 100;
  const percent = profile.frame_height_px
    ? Math.min(60, Math.max(5, Math.round((1000 * gate) / profile.frame_height_px) / 10))
    : 20;
  return {
    amplitude_pct: percent,
    ramp_s: Math.max(60, profile.nominal_interval_s ?? 10),
    hold_s: 300,
    shape: "retreat_return",
    sign: 1,
    speed: 30,
    start: "random",
    seed: 20261128,
  };
}

function StatusBadge({ status }: { status: string }) {
  const style =
    status === "replayable"
      ? "bg-emerald-100 text-emerald-800"
      : status === "limited"
        ? "bg-amber-100 text-amber-800"
        : "bg-slate-200 text-slate-700";
  return (
    <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${style}`}>
      {statusLabel(status)}
    </span>
  );
}

function ProfileCard({ profile }: { profile: RecordingProfile }) {
  const rows: [string, string][] = [
    ["Camera / source", `${profile.camera ?? "-"} / ${profile.source_type ?? "-"}`],
    ["Frame", profile.frame_height_px ? `${profile.frame_height_px.toFixed(0)} px high` : "unknown"],
    ["Sampling", profile.nominal_interval_s ? `${profile.nominal_interval_s.toFixed(0)} s` : "unknown"],
    [
      "Duration",
      profile.duration_s ? `${(profile.duration_s / 3600).toFixed(2)} h` : "-",
    ],
    [
      "Rows",
      `${profile.row_counts.observations ?? 0} observations, ${profile.row_counts.anomalies ?? 0} detector rows`,
    ],
    ["Segments / sessions", `${profile.segment_count} / ${profile.sessions.length}`],
    [
      "Scored coverage after warm-up",
      profile.scored_coverage === null ? "-" : `${(profile.scored_coverage * 100).toFixed(1)}%`,
    ],
    [
      "Longest valid window",
      profile.longest_valid_window_s
        ? `${(profile.longest_valid_window_s / 60).toFixed(0)} min`
        : "-",
    ],
    ["Noise std", profile.step_std_px === null ? "-" : `${profile.step_std_px.toFixed(1)} px`],
    [
      "Median / p95 |90 s move|",
      `${profile.move90_median_px?.toFixed(1) ?? "-"} / ${profile.move90_p95_px?.toFixed(1) ?? "-"} px`,
    ],
    [
      "Effective gate median / p95",
      `${profile.gate_median_px?.toFixed(0) ?? "-"} / ${profile.gate_p95_px?.toFixed(0) ?? "-"} px`,
    ],
    ["Polygon", profile.polygon ? "yes" : "no"],
    ["Valid starts", `${profile.valid_start_count}`],
    ["Parity", profile.parity ? (profile.parity.pass ? "pass" : "fail") : "-"],
  ];
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold text-slate-900">Recording profile</h2>
        <StatusBadge status={profile.status} />
      </div>
      <dl className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-2 border-b border-slate-100 py-0.5">
            <dt className="text-slate-500">{label}</dt>
            <dd className="text-right font-medium text-slate-800">{value}</dd>
          </div>
        ))}
      </dl>
      {profile.reason && (
        <p className="mt-2 rounded bg-amber-50 p-2 text-xs text-amber-800">{profile.reason}</p>
      )}
      {profile.notes && <p className="mt-2 text-xs text-slate-500">{profile.notes}</p>}
      {profile.parity && (
        <p className="mt-2 text-[11px] text-slate-500">
          Parity checked {profile.parity.checked_samples_after_full_history} samples after full
          history; classification mismatches {profile.parity.classification_mismatches}, state
          mismatches {profile.parity.state_mismatches}.
        </p>
      )}
    </section>
  );
}

type SavedDemo = {
  recording_id: string;
  label: string;
  presets: { id: string; label: string; expected: string }[];
  minimum_90: { ramp_s: number; text: string }[];
  note: string;
};

function SavedDemoPanel({ data, onClose }: { data: SavedDemo; onClose: () => void }) {
  return (
    <section className="rounded-xl border-2 border-red-300 bg-red-50 p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-red-900">Replay saved demo</h2>
          <p className="text-xs text-red-800">
            Pre-computed results only. No detector replay or live computation is running.
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded bg-red-900 px-3 py-1.5 text-xs font-semibold text-white"
        >
          Close saved demo
        </button>
      </div>
      <p className="mt-3 text-sm font-semibold text-slate-900">{data.label}</p>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <div className="rounded border border-red-200 bg-white p-3">
          <p className="text-xs font-semibold text-slate-900">Presets</p>
          {data.presets.map((preset) => (
            <p key={preset.id} className="mt-1 text-xs text-slate-700">
              <span className="font-medium">{preset.label}:</span> {preset.expected}
            </p>
          ))}
        </div>
        <div className="rounded border border-red-200 bg-white p-3">
          <p className="text-xs font-semibold text-slate-900">Quick estimate</p>
          {data.minimum_90.map((row) => (
            <p key={row.ramp_s} className="mt-1 text-xs text-slate-700">
              <span className="font-medium">Ramp {row.ramp_s.toFixed(0)} s:</span> {row.text}
            </p>
          ))}
        </div>
      </div>
      <p className="mt-3 text-xs text-red-900">{data.note}</p>
      <p className="mt-1 text-xs font-semibold text-red-900">{DEMO_BANNER}</p>
    </section>
  );
}

export function DemoInjectionView() {
  const [recordings, setRecordings] = useState<RecordingProfile[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [presets, setPresets] = useState<Preset[]>([]);
  const [params, setParams] = useState<Params | null>(null);
  const [samples, setSamples] = useState<SampleMessage[]>([]);
  const [events, setEvents] = useState<SimEvent[]>([]);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [simId, setSimId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loadingRecordings, setLoadingRecordings] = useState(true);
  const [savedDemo, setSavedDemo] = useState<SavedDemo | null>(null);
  const [savedError, setSavedError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);
  const bufferRef = useRef<SampleMessage[]>([]);
  const flushRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const profile = useMemo(
    () => recordings.find((recording) => recording.id === selectedId) ?? null,
    [recordings, selectedId],
  );
  const canSimulate = Boolean(profile && profile.status !== "not_replayable");

  const stopFlush = useCallback(() => {
    if (flushRef.current) {
      clearInterval(flushRef.current);
      flushRef.current = null;
    }
  }, []);

  const closeStream = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
    stopFlush();
  }, [stopFlush]);

  useEffect(() => {
    let cancelled = false;
    demoApi
      .recordings()
      .then((payload) => {
        if (cancelled) return;
        setRecordings(payload.recordings);
        const first =
          payload.recordings.find((recording) => recording.status === "replayable") ??
          payload.recordings.find((recording) => recording.status === "limited") ??
          payload.recordings[0];
        if (first) setSelectedId(first.id);
      })
      .catch((exc) => setError(exc instanceof Error ? exc.message : "Could not load recordings"))
      .finally(() => setLoadingRecordings(false));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    closeStream();
    setSamples([]);
    setEvents([]);
    setResult(null);
    setWarnings([]);
    setError(null);
    setPresets([]);
    setSimId(null);
    setRunning(false);
    bufferRef.current = [];
    if (!profile) {
      setParams(null);
      return;
    }
    setParams(defaultParams(profile));
    demoApi
      .presets(profile.id)
      .then((payload) => setPresets(payload.presets))
      .catch((exc) => setError(exc instanceof Error ? exc.message : "Could not load presets"));
  }, [profile?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(
    () => () => {
      closeStream();
    },
    [closeStream],
  );

  function selectPreset(preset: Preset) {
    if (!profile) return;
    setParams({
      amplitude_pct: preset.params.amplitude_pct,
      ramp_s: preset.params.ramp_s,
      hold_s: preset.params.hold_s,
      shape: preset.params.shape,
      sign: preset.params.sign,
      speed: preset.params.speed,
      start: "random",
      seed: 20261128,
    });
  }

  async function start() {
    if (!profile || !params) return;
    if (!canSimulate) return;
    closeStream();
    setSamples([]);
    setEvents([]);
    setResult(null);
    setWarnings([]);
    setError(null);
    bufferRef.current = [];
    try {
      const created = await demoApi.createSimulation({
        recording_id: profile.id,
        amplitude_pct: params.amplitude_pct,
        ramp_s: params.ramp_s,
        hold_s: params.hold_s,
        shape: params.shape,
        sign: params.sign,
        speed: params.speed,
        start: params.start,
        seed: params.seed,
      });
      setSimId(created.id);
      setWarnings(created.warnings);
      setRunning(true);
      const source = new EventSource(simulationStreamUrl(created.id));
      sourceRef.current = source;
      flushRef.current = setInterval(() => {
        if (bufferRef.current.length) {
          const batch = bufferRef.current;
          bufferRef.current = [];
          setSamples((previous) => [...previous, ...batch]);
        }
      }, 180);
      source.onmessage = (messageEvent) => {
        const message = JSON.parse(messageEvent.data) as
          | SampleMessage
          | { type: "event"; event: SimEvent }
          | { type: "done"; result: SimulationResult }
          | { type: "stopped" };
        if (message.type === "sample") {
          bufferRef.current.push(message);
        } else if (message.type === "event") {
          setEvents((previous) => [...previous, message.event]);
        } else if (message.type === "done") {
          if (bufferRef.current.length) {
            const batch = bufferRef.current;
            bufferRef.current = [];
            setSamples((previous) => [...previous, ...batch]);
          }
          setResult(message.result);
          setRunning(false);
          closeStream();
        } else if (message.type === "stopped") {
          setRunning(false);
          closeStream();
        }
      };
      source.onerror = () => {
        setRunning(false);
        closeStream();
      };
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Simulation could not start");
      setRunning(false);
    }
  }

  async function pause() {
    closeStream();
    setRunning(false);
    if (simId) {
      try {
        await demoApi.stopSimulation(simId);
      } catch {
        // Best effort; the simulation is in memory only.
      }
    }
  }

  async function reset() {
    closeStream();
    setSamples([]);
    setEvents([]);
    setResult(null);
    setWarnings([]);
    setError(null);
    setRunning(false);
    bufferRef.current = [];
    if (simId) {
      try {
        await demoApi.deleteSimulation(simId);
      } catch {
        // Best effort.
      }
    }
    setSimId(null);
  }

  async function loadSavedDemo() {
    setSavedError(null);
    try {
      const response = await fetch("/demo/injection-saved.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`Saved demo not found (${response.status})`);
      setSavedDemo((await response.json()) as SavedDemo);
      closeStream();
      setRunning(false);
      setResult(null);
      setSamples([]);
    } catch (exc) {
      setSavedError(exc instanceof Error ? exc.message : "Saved demo could not be loaded");
    }
  }

  if (loadingRecordings) {
    return <p className="text-sm text-slate-500">Loading recorded runs...</p>;
  }

  return (
    <div className="space-y-5">
      <div className="sticky top-0 z-20 -mx-2 rounded-lg border-2 border-red-500 bg-red-600 px-4 py-3 text-white shadow-lg">
        <p className="text-sm font-bold uppercase tracking-wide">{DEMO_BANNER}</p>
        <p className="text-xs text-red-100">
          Offline position-level injection on recorded noise. Not a real event, not a tsunami
          validation, and not a real-world sensitivity or false-alarm claim.
        </p>
      </div>

      <header>
        <h1 className="text-2xl font-semibold text-slate-900">Drawdown injection demo</h1>
        <p className="mt-1 text-sm text-slate-500">
          Replay any eligible recorded run through the production rolling-baseline detector with a
          synthetic drawdown. Simulation only.
        </p>
      </header>

      <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex min-w-[260px] flex-col gap-1 text-xs">
            <span className="font-medium text-slate-600">Recording</span>
            <select
              value={selectedId}
              onChange={(event) => setSelectedId(event.target.value)}
              className="rounded border border-slate-300 px-2 py-1.5 text-sm"
            >
              {recordings.map((recording) => (
                <option key={recording.id} value={recording.id}>
                  {recording.label} - {statusLabel(recording.status)}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={loadSavedDemo}
            className="rounded border border-red-300 bg-red-50 px-3 py-1.5 text-xs font-semibold text-red-800"
          >
            Replay saved demo
          </button>
        </div>
        {savedError && <p className="mt-2 text-xs text-red-700">{savedError}</p>}
      </section>

      {savedDemo && <SavedDemoPanel data={savedDemo} onClose={() => setSavedDemo(null)} />}

      {profile && <ProfileCard profile={profile} />}

      {profile && !canSimulate && (
        <section className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
          Simulations are disabled for this recording. {profile.reason}
        </section>
      )}

      {profile && canSimulate && params && (
        <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="mb-2 text-sm font-semibold text-slate-900">Presets</h2>
          <div className="flex flex-wrap gap-2">
            {presets.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => selectPreset(preset)}
                className="rounded border border-slate-300 bg-slate-50 px-3 py-1.5 text-left text-xs hover:bg-slate-100"
              >
                <span className="block font-semibold text-slate-900">{preset.label}</span>
                <span className="block text-slate-500">{preset.description}</span>
                <span className="block text-[11px] text-sky-700">
                  expected: {expectedText(preset.expected)}
                </span>
              </button>
            ))}
          </div>

          <div className="mt-4 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
            <label className="flex flex-col gap-1">
              <span className="font-medium text-slate-600">Amplitude (% of frame height)</span>
              <input
                type="number"
                min={0}
                max={60}
                step={0.1}
                value={params.amplitude_pct}
                onChange={(event) =>
                  setParams({ ...params, amplitude_pct: Number(event.target.value) })
                }
                className="rounded border border-slate-300 px-2 py-1"
              />
              <span className="text-[11px] text-slate-500">
                {describeAmplitude(params.amplitude_pct, profile.frame_height_px)}
              </span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="font-medium text-slate-600">Ramp (s)</span>
              <input
                type="number"
                min={1}
                step={1}
                value={params.ramp_s}
                onChange={(event) => setParams({ ...params, ramp_s: Number(event.target.value) })}
                className="rounded border border-slate-300 px-2 py-1"
              />
              <span className="text-[11px] text-slate-500">
                Hold {params.hold_s} s. Ramp under 3 samples warns.
              </span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="font-medium text-slate-600">Shape</span>
              <select
                value={params.shape}
                onChange={(event) =>
                  setParams({ ...params, shape: event.target.value as Params["shape"] })
                }
                className="rounded border border-slate-300 px-2 py-1"
              >
                <option value="retreat_return">retreat and return</option>
                <option value="persistent">persistent</option>
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="font-medium text-slate-600">Direction</span>
              <select
                value={params.sign}
                onChange={(event) => setParams({ ...params, sign: Number(event.target.value) })}
                className="rounded border border-slate-300 px-2 py-1"
              >
                <option value={1}>seaward (+)</option>
                <option value={-1}>landward (-)</option>
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="font-medium text-slate-600">Start</span>
              <select
                value={params.start}
                onChange={(event) => setParams({ ...params, start: event.target.value })}
                className="rounded border border-slate-300 px-2 py-1"
              >
                <option value="random">Random (seed {params.seed})</option>
                {profile.valid_start_samples.map((value) => (
                  <option key={value} value={value}>
                    {formatDateManila(value)} Asia/Manila
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="font-medium text-slate-600">Speed</span>
              <select
                value={params.speed}
                onChange={(event) => setParams({ ...params, speed: Number(event.target.value) })}
                className="rounded border border-slate-300 px-2 py-1"
              >
                {[1, 10, 30, 60].map((speed) => (
                  <option key={speed} value={speed}>
                    {speed}x (original timestamps)
                  </option>
                ))}
              </select>
            </label>
            <div className="flex items-end gap-2">
              <button
                type="button"
                onClick={start}
                disabled={running || profile.status === "not_replayable"}
                className="rounded bg-slate-900 px-4 py-2 font-semibold text-white disabled:opacity-40"
              >
                Start
              </button>
              <button
                type="button"
                onClick={pause}
                disabled={!running}
                className="rounded border border-slate-300 px-4 py-2 font-semibold text-slate-700 disabled:opacity-40"
              >
                Pause
              </button>
              <button
                type="button"
                onClick={reset}
                className="rounded border border-slate-300 px-4 py-2 font-semibold text-slate-700"
              >
                Reset
              </button>
            </div>
          </div>

          {profile.status === "limited" && (
            <p className="mt-2 rounded bg-amber-50 p-2 text-xs text-amber-800">
              Limited data: {profile.valid_start_count} valid starts. This is a single illustrative
              run, not a rate.
            </p>
          )}
          {warnings.map((warning) => (
            <p key={warning} className="mt-2 rounded bg-amber-50 p-2 text-xs text-amber-800">
              {warning}
            </p>
          ))}
          {error && <p className="mt-2 rounded bg-red-50 p-2 text-xs text-red-700">{error}</p>}
        </section>
      )}

      {profile && (
        <>
          <DemoCharts
            samples={samples}
            event={events[events.length - 1] ?? null}
            result={result}
            frameHeight={profile.frame_height_px}
          />
          <SummaryPanel recordingId={profile.id} />
        </>
      )}

      <ComparePanel
        key={`${selectedId}-${params?.amplitude_pct}-${params?.ramp_s}`}
        defaultAmplitude={params?.amplitude_pct ?? 20}
        defaultRamp={params?.ramp_s ?? 60}
      />

      <section className="relative overflow-hidden rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="-rotate-12 text-lg font-bold uppercase tracking-widest text-red-500/10">
            {DEMO_WATERMARK}
          </span>
        </div>
        <h2 className="text-sm font-semibold text-slate-900">What this shows / what it does not</h2>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-slate-600">
          <li>It shows position-level injection on recorded noise, replayed by the real detector.</li>
          <li>
            It does not test segmentation behaviour during a real retreat (wet sand, scene change).
          </li>
          <li>Each result applies to that recording&apos;s scene and conditions only.</li>
          <li>Recordings are never combined into a single rate.</li>
          <li>It is not a tsunami validation and makes no real-world sensitivity claim.</li>
        </ul>
        <p className="mt-3 text-xs font-semibold text-red-700">{DEMO_BANNER}</p>
      </section>
    </div>
  );
}
