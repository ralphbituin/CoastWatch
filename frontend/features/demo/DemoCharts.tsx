"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  DEMO_WATERMARK,
  Z_THRESHOLD,
  chartPoints,
  formatManila,
  stateColor,
  stateSegments,
  type RawSample,
} from "@/features/demo/demoSeries";
import type { SimEvent, SimulationResult } from "@/features/demo/api";

function Watermark() {
  return (
    <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
      <span className="-rotate-12 select-none text-2xl font-bold uppercase tracking-widest text-red-500/20">
        {DEMO_WATERMARK}
      </span>
    </div>
  );
}

function panelDomain(points: { t: number | null }[]): [number, number] {
  const values = points
    .map((point) => point.t)
    .filter((value): value is number => value !== null);
  if (!values.length) return [0, 1];
  return [Math.min(...values), Math.max(...values)];
}

function ChartFrame({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="relative overflow-hidden rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <header className="mb-2">
        <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
        {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
      </header>
      <Watermark />
      {children}
    </section>
  );
}

export function DemoCharts({
  samples,
  event,
  result,
  frameHeight,
}: {
  samples: RawSample[];
  event: SimEvent | null;
  result: SimulationResult | null;
  frameHeight: number | null;
}) {
  const points = useMemo(() => chartPoints(samples), [samples]);
  const domain = useMemo(() => panelDomain(points), [points]);
  const segments = useMemo(() => stateSegments(samples), [samples]);

  if (!samples.length) {
    return (
      <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-sm text-slate-500">
        Start a simulation to see the detector replay. Every chart is a simulation.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <ChartFrame
        title="Position (px, positive = seaward)"
        subtitle={`Real recorded position, injected position, detector baseline and the effective gate band${frameHeight ? ` (frame height ${frameHeight.toFixed(0)} px)` : ""}.`}
      >
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={points} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="t"
              type="number"
              domain={domain}
              tickFormatter={(value: number) => formatManila(value)}
              minTickGap={28}
              fontSize={11}
            />
            <YAxis fontSize={11} width={50} />
            <Tooltip
              labelFormatter={(value) => formatManila(Number(value), { withDate: true })}
              formatter={(value: number | string, name: string) => [value, name]}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line dataKey="gateUp" name="gate band" stroke="#f59e0b" dot={false} strokeDasharray="4 4" strokeWidth={1} />
            <Line dataKey="gateDown" name="gate band" stroke="#f59e0b" dot={false} strokeDasharray="4 4" strokeWidth={1} legendType="none" />
            <Line dataKey="baseline" name="baseline" stroke="#0f766e" dot={false} strokeWidth={2} />
            <Line dataKey="real" name="real position" stroke="#334155" dot={false} strokeWidth={1} />
            <Line dataKey="injected" name="injected (SIM)" stroke="#2563eb" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </ChartFrame>

      <ChartFrame
        title="Robust z"
        subtitle={`Production detector robust z; the 3.5 threshold is the line the detector uses.`}
      >
        <ResponsiveContainer width="100%" height={190}>
          <LineChart data={points} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="t"
              type="number"
              domain={domain}
              tickFormatter={(value: number) => formatManila(value)}
              minTickGap={28}
              fontSize={11}
            />
            <YAxis fontSize={11} width={50} />
            <Tooltip
              labelFormatter={(value) => formatManila(Number(value), { withDate: true })}
              formatter={(value: number | string, name: string) => [value, name]}
            />
            <ReferenceLine y={Z_THRESHOLD} stroke="#dc2626" strokeDasharray="5 4" label={{ value: "3.5", fontSize: 10, fill: "#dc2626" }} />
            <Line dataKey="z" name="robust z" stroke="#7c3aed" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </ChartFrame>

      <ChartFrame
        title="Cumulative drawdown vs effective movement gate"
        subtitle="This is the chart that explains most misses: the detector needs C_k at or above the recording's gate."
      >
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={points} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="t"
              type="number"
              domain={domain}
              tickFormatter={(value: number) => formatManila(value)}
              minTickGap={28}
              fontSize={11}
            />
            <YAxis fontSize={11} width={50} />
            <Tooltip
              labelFormatter={(value) => formatManila(Number(value), { withDate: true })}
              formatter={(value: number | string, name: string) => [value, name]}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line dataKey="ck" name="cumulative drawdown" stroke="#0ea5e9" dot={false} strokeWidth={2} />
            <Line dataKey="gate" name="effective gate" stroke="#f97316" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </ChartFrame>

      <section className="relative overflow-hidden rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <header className="mb-2 flex items-baseline justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Detector state strip</h3>
            <p className="text-xs text-slate-500">One strip segment per replayed sample.</p>
          </div>
          {event && (
            <span className="rounded-full bg-red-600 px-3 py-1 text-xs font-semibold text-white">
              SIM event {event.id}
            </span>
          )}
        </header>
        <Watermark />
        <div className="flex h-7 w-full overflow-hidden rounded border border-slate-200">
          {segments.map((segment, index) => (
            <div
              key={`${segment.state}-${index}`}
              title={`${segment.state} x${segment.count}`}
              style={{
                background: stateColor(segment.state),
                flexGrow: segment.count,
              }}
            />
          ))}
        </div>
        <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-600">
          {["MONITORING", "CANDIDATE", "PENDING_PERSISTENCE", "EVENT", "COOLDOWN"].map((state) => (
            <span key={state} className="inline-flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: stateColor(state) }} />
              {state}
            </span>
          ))}
        </div>
      </section>

      {result && (
        <section
          className={`rounded-xl border p-4 shadow-sm ${
            result.outcome === "caught"
              ? "border-red-300 bg-red-50"
              : "border-amber-300 bg-amber-50"
          }`}
        >
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">
            SIMULATION - not real data
          </p>
          <p className="mt-1 text-lg font-semibold text-slate-900">
            {result.outcome === "caught" ? "Caught" : "Missed"}
          </p>
          <p className="mt-1 text-sm text-slate-700">{result.explanation}</p>
          <dl className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-600 sm:grid-cols-4">
            <div>
              <dt className="font-semibold">Classification</dt>
              <dd>{result.classification}</dd>
            </div>
            <div>
              <dt className="font-semibold">Latency</dt>
              <dd>{result.latency_from_t0_s === null ? "-" : `${result.latency_from_t0_s.toFixed(0)} s`}</dd>
            </div>
            <div>
              <dt className="font-semibold">Peak z</dt>
              <dd>{result.peak_z === null ? "-" : result.peak_z.toFixed(2)}</dd>
            </div>
            <div>
              <dt className="font-semibold">False events</dt>
              <dd>{result.false_events}</dd>
            </div>
          </dl>
        </section>
      )}
    </div>
  );
}
