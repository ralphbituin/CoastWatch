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
  ANOMALY_COLORS,
  ANOMALY_GAP_MS,
  anomalyStateColor,
  anomalyStateSegments,
  buildAnomalyChartData,
  classificationCounts,
  finiteBounds,
  toAnomalyPoints,
  type AnomalyChartPoint,
  type AnomalyPoint,
} from "@/features/anomalies/anomalySeries";
import type { AnomalyEvent } from "@/lib/types";

type Props = {
  events: AnomalyEvent[];
  baselineStatus?: string | null;
  baselineReason?: string | null;
  timeZone?: string;
  selectedEventId?: number | null;
  onEventClick?: (eventId: number) => void;
};

type PlotRow = AnomalyChartPoint & { eventY: number | null };

const OFFICIAL_WARNING_AUTHORITY =
  (process.env.NEXT_PUBLIC_OFFICIAL_WARNING_AUTHORITY ?? "").trim() ||
  "the official national tsunami warning authority";

const Z_THRESHOLD = 3.5;
const EVENT_CLASSIFICATIONS = [
  "UNUSUAL_SHORELINE_POSITION",
  "RAPID_SHORELINE_RETREAT",
  "PERSISTENT_DRAWDOWN",
  "POSSIBLE_DRAWDOWN",
];

function formatAxisTime(timeMs: number, timeZone: string): string {
  return new Date(timeMs).toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone,
  });
}

function tooltipTime(timeMs: number, timeZone: string): string {
  return new Date(timeMs).toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone,
  });
}

function valueText(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "—";
}

function AnomalyTooltip({
  active,
  payload,
  timeZone,
}: {
  active?: boolean;
  payload?: Array<{ payload: PlotRow }>;
  timeZone: string;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  if (row.t === null) return null;
  return (
    <div className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg">
      <p className="font-semibold text-slate-800">{tooltipTime(row.t, timeZone)}</p>
      <p className="text-slate-500">
        {row.classification ?? "unknown"} · {row.state ?? "unknown state"}
      </p>
      <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 text-slate-600">
        <span>observed</span>
        <span className="text-right">{valueText(row.observed)} px</span>
        <span>baseline</span>
        <span className="text-right">{valueText(row.baseline)} px</span>
        <span>residual</span>
        <span className="text-right">{valueText(row.residual)} px</span>
        <span>robust z</span>
        <span className="text-right">{valueText(row.robustZ)}</span>
        <span>cumulative Ck</span>
        <span className="text-right">{valueText(row.cumulativeDrawdown)} px</span>
        <span>effective gate</span>
        <span className="text-right">{valueText(row.gate)} px</span>
        <span>tide</span>
        <span className="text-right">{valueText(row.tide)} m</span>
      </div>
      {row.reason && <p className="mt-1 max-w-72 text-[11px] text-slate-500">{row.reason}</p>}
    </div>
  );
}

function ChartFrame({
  title,
  unit,
  children,
}: {
  title: string;
  unit: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-slate-200 bg-white">
      <div className="flex items-baseline justify-between border-b border-slate-100 px-3 py-2">
        <p className="text-xs font-semibold text-slate-700">{title}</p>
        <p className="text-[10px] text-slate-400">{unit}</p>
      </div>
      <div className="px-1 py-1">{children}</div>
    </div>
  );
}

function EventDot({
  cx,
  cy,
  payload,
  selectedEventId,
  onEventClick,
}: {
  cx?: number;
  cy?: number;
  payload?: PlotRow;
  selectedEventId?: number | null;
  onEventClick?: (eventId: number) => void;
}) {
  if (
    cx === undefined ||
    cy === undefined ||
    !payload ||
    !payload.classification ||
    !EVENT_CLASSIFICATIONS.includes(payload.classification) ||
    payload.id === null
  ) {
    return <g />;
  }
  const selected = selectedEventId === payload.id;
  const color = ANOMALY_COLORS[payload.classification] ?? "#c0392b";
  return (
    <g
      role={onEventClick ? "button" : undefined}
      tabIndex={onEventClick ? 0 : undefined}
      aria-label={`${payload.classification} event ${payload.id}`}
      className={onEventClick ? "cursor-pointer" : undefined}
      onClick={() => onEventClick?.(payload.id as number)}
      onKeyDown={(event) => {
        if (onEventClick && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          onEventClick(payload.id as number);
        }
      }}
    >
      <circle
        cx={cx}
        cy={cy}
        r={selected ? 7 : 5}
        fill={color}
        stroke={selected ? "#facc15" : "#111827"}
        strokeWidth={selected ? 2.5 : 0.8}
      >
        <title>{`${payload.classification}: ${payload.reason ?? ""}`}</title>
      </circle>
    </g>
  );
}

function StateStrip({ points }: { points: AnomalyPoint[] }) {
  const segments = useMemo(() => anomalyStateSegments(points), [points]);
  const states = useMemo(
    () => Array.from(new Set(points.map((point) => point.state))),
    [points],
  );
  return (
    <div className="rounded-md border border-slate-200 bg-white px-3 py-2">
      <div className="mb-1 flex items-baseline justify-between">
        <p className="text-xs font-semibold text-slate-700">Detector state strip</p>
        <p className="text-[10px] text-slate-400">one segment per stored detector row</p>
      </div>
      <div className="flex h-6 w-full overflow-hidden rounded border border-slate-200">
        {segments.map((segment, index) => (
          <div
            key={`${segment.state}-${index}`}
            title={`${segment.state} x${segment.count}`}
            style={{ background: anomalyStateColor(segment.state), flexGrow: segment.count }}
          />
        ))}
      </div>
      <div className="mt-1 flex flex-wrap gap-3 text-[10px] text-slate-500">
        {states.map((state) => (
          <span key={state} className="inline-flex items-center gap-1">
            <span
              className="h-2 w-2 rounded-full"
              style={{ background: anomalyStateColor(state) }}
            />
            {state}
          </span>
        ))}
      </div>
    </div>
  );
}

export function TideAnomalyChart({
  events,
  baselineStatus,
  baselineReason,
  timeZone = "GMT",
  selectedEventId = null,
  onEventClick,
}: Props) {
  const points = useMemo(() => toAnomalyPoints(events), [events]);
  const counts = useMemo(() => classificationCounts(points), [points]);
  const rolling = useMemo(
    () =>
      baselineStatus === "rolling_unverified" ||
      points.some((point) => point.detectorMode === "rolling_baseline"),
    [baselineStatus, points],
  );
  const chartData = useMemo<PlotRow[]>(() => {
    return buildAnomalyChartData(points, ANOMALY_GAP_MS).map((row) => ({
      ...row,
      eventY:
        row.classification && EVENT_CLASSIFICATIONS.includes(row.classification)
          ? (row.observed ?? row.baseline ?? row.residual)
          : null,
    }));
  }, [points]);

  if (points.length === 0) {
    return (
      <p
        role="status"
        className="rounded-md border border-slate-200 bg-slate-50 px-3 py-8 text-center text-sm text-slate-500"
      >
        No anomaly/quality observations recorded for this selection.
      </p>
    );
  }

  const times = chartData
    .map((row) => row.t)
    .filter((value): value is number => value !== null);
  const xStart = Math.min(...times);
  const xEnd = Math.max(...times) === xStart ? xStart + 60_000 : Math.max(...times);
  const hasGate = chartData.some((row) => row.gate !== null);

  const positionValues = chartData.flatMap((row) => [
    row.observed,
    row.baseline,
    row.gateUp,
    row.gateDown,
  ]);
  const positionBounds = finiteBounds(positionValues) ?? { min: 0, max: 1 };
  const tidePoints = chartData.filter((row) => row.tide !== null);
  const tideBounds = finiteBounds(tidePoints.map((row) => row.tide));
  const residualPoints = chartData.filter((row) => row.residual !== null);
  const residualBounds = finiteBounds(residualPoints.map((row) => row.residual)) ?? {
    min: -1,
    max: 1,
  };
  const zPoints = chartData.filter((row) => row.robustZ !== null);
  const zBounds = finiteBounds([...zPoints.map((row) => row.robustZ), 0, Z_THRESHOLD]) ?? {
    min: -1,
    max: 4,
  };
  const cumulativePoints = chartData.filter((row) => row.cumulativeDrawdown !== null);
  const cumulativeBounds =
    finiteBounds([
      ...cumulativePoints.map((row) => row.cumulativeDrawdown),
      ...chartData.map((row) => row.gate),
      0,
    ]) ?? { min: 0, max: 1 };

  const axis = (
    <XAxis
      dataKey="t"
      type="number"
      domain={[xStart, xEnd]}
      tickFormatter={(value: number) => formatAxisTime(value, timeZone)}
      minTickGap={36}
      fontSize={10}
      tick={{ fill: "#64748b" }}
    />
  );
  const grid = <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />;
  const tooltip = (
    <Tooltip content={<AnomalyTooltip timeZone={timeZone} />} />
  );

  return (
    <div className="space-y-2">
      <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
        {rolling
          ? `Experimental rolling-baseline drawdown detector — not validated for tsunami detection. Follow ${OFFICIAL_WARNING_AUTHORITY}.`
          : "Experimental baseline — insufficient recording history for validated anomaly detection."}
        {baselineStatus === "unavailable" && (
          <span className="ml-1">
            Baseline unavailable{baselineReason ? `: ${baselineReason}` : "."}
          </span>
        )}
        {rolling && (
          <span className="ml-1">
            The rolling median is self-referential and unverified; it is not a tide model.
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-4 rounded bg-slate-700" /> observed position
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-4 rounded bg-teal-600" />
          {rolling ? "Rolling baseline (unverified; no tide data)" : "expected (provisional)"}
        </span>
        {tideBounds && (
          <span className="inline-flex items-center gap-1">
            <span className="h-2 w-4 rounded bg-blue-600" /> tide
          </span>
        )}
        {rolling && (
          <span className="inline-flex items-center gap-1">
            <span className="h-2 w-4 rounded bg-orange-500" /> cumulative drawdown
          </span>
        )}
        {hasGate && (
          <span className="inline-flex items-center gap-1">
            <span className="h-2 w-4 rounded bg-amber-500" /> effective gate
          </span>
        )}
        {!hasGate && rolling && (
          <span className="text-slate-400">effective gate not exposed by the stored event API</span>
        )}
        {Object.entries(counts)
          .filter(([classification]) => !(rolling && classification === "NORMAL"))
          .map(([classification, count]) => (
            <span key={classification} className="rounded bg-slate-100 px-2 py-0.5">
              {classification}: {count}
            </span>
          ))}
        <span className="text-slate-400">
          times: {timeZone} · gaps &gt; {ANOMALY_GAP_MS / 1000} s are not joined
        </span>
      </div>

      <ChartFrame title="Shoreline position" unit="px · + = seaward">
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
            {grid}
            {axis}
            <YAxis domain={[positionBounds.min, positionBounds.max]} fontSize={10} width={52} />
            {tooltip}
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {hasGate && (
              <Line
                dataKey="gateUp"
                name="effective gate band"
                stroke="#f59e0b"
                strokeDasharray="4 4"
                strokeWidth={1}
                dot={false}
                activeDot={false}
              />
            )}
            {hasGate && (
              <Line
                dataKey="gateDown"
                name="gate band"
                stroke="#f59e0b"
                strokeDasharray="4 4"
                strokeWidth={1}
                dot={false}
                activeDot={false}
                legendType="none"
              />
            )}
            <Line
              dataKey="baseline"
              name={rolling ? "rolling baseline" : "expected"}
              stroke="#0f766e"
              strokeDasharray="5 3"
              strokeWidth={1.6}
              dot={false}
              activeDot={false}
            />
            <Line
              dataKey="observed"
              name="observed position"
              stroke="#334155"
              strokeWidth={1.8}
              dot={false}
              activeDot={{ r: 3 }}
            />
            <Line
              dataKey="eventY"
              name="event marker"
              stroke="transparent"
              dot={(props) => (
                <EventDot
                  {...props}
                  selectedEventId={selectedEventId}
                  onEventClick={onEventClick}
                />
              )}
              activeDot={false}
              legendType="none"
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartFrame>

      {tideBounds && (
        <ChartFrame title="Tide height" unit="m">
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={chartData} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
              {grid}
              {axis}
              <YAxis domain={[tideBounds.min, tideBounds.max]} fontSize={10} width={52} />
              {tooltip}
              <Line
                dataKey="tide"
                name="tide"
                stroke="#2563eb"
                strokeWidth={1.8}
                dot={false}
                activeDot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartFrame>
      )}

      <ChartFrame
        title={rolling ? "Residual from rolling baseline" : "Residual from expected"}
        unit="px · + = seaward"
      >
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={chartData} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
            {grid}
            {axis}
            <YAxis domain={[residualBounds.min, residualBounds.max]} fontSize={10} width={52} />
            {tooltip}
            <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
            <Line
              dataKey="residual"
              name="residual"
              stroke="#c0392b"
              strokeWidth={1.8}
              dot={false}
              activeDot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
        {residualPoints.length === 0 && (
          <p className="px-2 pb-2 text-xs text-slate-500">
            {rolling
              ? "Residual not recorded for this selection."
              : "Residual unavailable — no verified tide-conditioned baseline."}
          </p>
        )}
      </ChartFrame>

      <ChartFrame title="Robust z-score" unit="σ">
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={chartData} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
            {grid}
            {axis}
            <YAxis domain={[zBounds.min, zBounds.max]} fontSize={10} width={52} />
            {tooltip}
            <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
            <ReferenceLine
              y={Z_THRESHOLD}
              stroke="#dc2626"
              strokeDasharray="5 4"
              label={{ value: "3.5", fontSize: 10, fill: "#dc2626" }}
            />
            <Line
              dataKey="robustZ"
              name="robust z"
              stroke="#2563eb"
              strokeWidth={1.8}
              dot={false}
              activeDot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
        {zPoints.length === 0 && (
          <p className="px-2 pb-2 text-xs text-slate-500">
            Robust z unavailable for this selection.
          </p>
        )}
      </ChartFrame>

      <ChartFrame title="Cumulative drawdown Ck" unit="px · + = seaward">
        <ResponsiveContainer width="100%" height={170}>
          <LineChart data={chartData} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
            {grid}
            {axis}
            <YAxis
              domain={[cumulativeBounds.min, cumulativeBounds.max]}
              fontSize={10}
              width={52}
            />
            {tooltip}
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
            {hasGate && (
              <Line
                dataKey="gate"
                name="effective gate"
                stroke="#f97316"
                strokeWidth={1.8}
                dot={false}
                activeDot={false}
              />
            )}
            <Line
              dataKey="cumulativeDrawdown"
              name="cumulative drawdown"
              stroke="#0ea5e9"
              strokeWidth={1.8}
              dot={false}
              activeDot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
        {cumulativePoints.length === 0 && (
          <p className="px-2 pb-2 text-xs text-slate-500">
            Cumulative drawdown is recorded per valid observation in the rolling output.
          </p>
        )}
      </ChartFrame>

      <StateStrip points={points} />

      {!tideBounds && (
        <p className="text-xs text-slate-500">
          Tide height unavailable. No verified, geographically appropriate Thai tide data is
          associated with this footage, so no tide graph is shown.
        </p>
      )}
    </div>
  );
}
