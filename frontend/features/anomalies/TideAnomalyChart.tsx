"use client";

import { useMemo } from "react";

import {
  ANOMALY_COLORS,
  classificationCounts,
  finiteBounds,
  scaleTime,
  scaleToHeight,
  splitRuns,
  toAnomalyPoints,
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

type Marker = {
  id: number;
  x: number;
  y: number;
  color: string;
  classification: string;
  reason: string | null;
};

type PanelProps = {
  title: string;
  unit: string;
  points: AnomalyPoint[];
  selector: (point: AnomalyPoint) => number | null;
  secondarySelector?: (point: AnomalyPoint) => number | null;
  secondaryColor?: string;
  secondaryDash?: string;
  bounds: { min: number; max: number };
  height: number;
  color: string;
  dash?: string;
  zeroLine?: boolean;
  markers?: Marker[];
  selectedEventId?: number | null;
  onEventClick?: (eventId: number) => void;
  unreliable?: AnomalyPoint[];
  showTicks?: boolean;
  xStart: number;
  xEnd: number;
  ticks: number[];
  timeZone: string;
  emptyText?: string;
};

const OFFICIAL_WARNING_AUTHORITY =
  (process.env.NEXT_PUBLIC_OFFICIAL_WARNING_AUTHORITY ?? "").trim() ||
  "the official national tsunami warning authority";

const WIDTH = 960;
const PLOT_TOP = 20;
const POSITION_HEIGHT = 210;
const TIDE_HEIGHT = 100;
const RESIDUAL_HEIGHT = 100;
const Z_HEIGHT = 90;
const CUMULATIVE_HEIGHT = 100;
const PADDING = { left: 58, right: 62, top: 18, bottom: 26 };
const GAP_MS = 90_000;
const EVENT_CLASSIFICATIONS = [
  "UNUSUAL_SHORELINE_POSITION",
  "RAPID_SHORELINE_RETREAT",
  "PERSISTENT_DRAWDOWN",
  "POSSIBLE_DRAWDOWN",
];

function formatTime(timeMs: number, timeZone: string): string {
  return new Date(timeMs).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone,
  });
}

function linePath(
  points: AnomalyPoint[],
  selector: (point: AnomalyPoint) => number | null,
  bounds: { min: number; max: number },
  height: number,
  xStart: number,
  xEnd: number,
  offsetY = 0,
): string {
  return splitRuns(points, selector, GAP_MS)
    .flatMap((run) => {
      const coordinates = run.map((point) => {
        const value = selector(point) as number;
        const x = PADDING.left + scaleTime(point.timeMs, xStart, xEnd, WIDTH - PADDING.left - PADDING.right);
        const y = offsetY + scaleToHeight(value, bounds, height, PADDING.top);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      return coordinates.length > 1 ? [`M${coordinates.join(" L")}`] : [];
    })
    .join(" ");
}

function markers(
  points: AnomalyPoint[],
  bounds: { min: number; max: number },
  height: number,
  xStart: number,
  xEnd: number,
  offsetY = 0,
): Marker[] {
  return points.flatMap((point) => {
    if (!EVENT_CLASSIFICATIONS.includes(point.classification)) return [];
    const value = point.observed ?? point.rollingBaseline ?? point.expected;
    if (value === null) return [];
    const x = PADDING.left + scaleTime(point.timeMs, xStart, xEnd, WIDTH - PADDING.left - PADDING.right);
    const y = offsetY + scaleToHeight(value, bounds, height, PADDING.top);
    return [
      {
        id: point.id,
        x,
        y,
        color: ANOMALY_COLORS[point.classification] ?? "#c0392b",
        classification: point.classification,
        reason: point.reason,
      },
    ];
  });
}

function ChartPanel({
  title,
  unit,
  points,
  selector,
  secondarySelector,
  secondaryColor,
  secondaryDash,
  bounds,
  height,
  color,
  dash,
  zeroLine = false,
  markers: panelMarkers = [],
  selectedEventId = null,
  onEventClick,
  unreliable = [],
  showTicks = false,
  xStart,
  xEnd,
  ticks,
  timeZone,
  emptyText,
}: PanelProps) {
  const path = linePath(points, selector, bounds, height, xStart, xEnd, PLOT_TOP);
  const secondaryPath = secondarySelector
    ? linePath(points, secondarySelector, bounds, height, xStart, xEnd, PLOT_TOP)
    : "";
  const hasData = path.length > 0 || secondaryPath.length > 0;
  const bottom = PLOT_TOP + height;
  const svgHeight = bottom + (showTicks ? 30 : 8);
  const zeroY = zeroLine
    ? PLOT_TOP + scaleToHeight(0, bounds, height, PADDING.top)
    : null;

  return (
    <div className="rounded-md border border-slate-200 bg-white">
      <svg
        viewBox={`0 0 ${WIDTH} ${svgHeight}`}
        className="w-full"
        role="img"
        aria-label={title}
      >
        <text x={PADDING.left} y={13} className="fill-slate-600" fontSize={11} fontWeight={600}>
          {title}
        </text>
        <text x={WIDTH - PADDING.right} y={13} textAnchor="end" className="fill-slate-400" fontSize={10}>
          {unit}
        </text>
        <text x={PADDING.left - 6} y={PLOT_TOP + 4} textAnchor="end" className="fill-slate-400" fontSize={9}>
          {bounds.max.toFixed(2)}
        </text>
        <text x={PADDING.left - 6} y={bottom - 2} textAnchor="end" className="fill-slate-400" fontSize={9}>
          {bounds.min.toFixed(2)}
        </text>
        {zeroY !== null && (
          <line
            x1={PADDING.left}
            x2={WIDTH - PADDING.right}
            y1={zeroY}
            y2={zeroY}
            stroke="#94a3b8"
            strokeDasharray="3 3"
          />
        )}
        {path && (
          <path
            d={path}
            fill="none"
            stroke={color}
            strokeWidth={1.8}
            strokeDasharray={dash}
          />
        )}
        {secondaryPath && secondaryColor && (
          <path
            d={secondaryPath}
            fill="none"
            stroke={secondaryColor}
            strokeWidth={1.5}
            strokeDasharray={secondaryDash}
          />
        )}
        {!hasData && emptyText && (
          <text x={WIDTH / 2} y={PLOT_TOP + height / 2} textAnchor="middle" className="fill-slate-400" fontSize={11}>
            {emptyText}
          </text>
        )}
        {panelMarkers.map((marker) => (
          <g
            key={marker.id}
            role={onEventClick ? "button" : undefined}
            tabIndex={onEventClick ? 0 : undefined}
            aria-label={`${marker.classification} event ${marker.id}`}
            className={onEventClick ? "cursor-pointer" : undefined}
            onClick={() => onEventClick?.(marker.id)}
            onKeyDown={(event) => {
              if (onEventClick && (event.key === "Enter" || event.key === " ")) {
                event.preventDefault();
                onEventClick(marker.id);
              }
            }}
          >
            <circle
              cx={marker.x}
              cy={marker.y}
              r={selectedEventId === marker.id ? 7 : 5}
              fill={marker.color}
              stroke={selectedEventId === marker.id ? "#facc15" : "#111827"}
              strokeWidth={selectedEventId === marker.id ? 2.5 : 0.8}
            >
              <title>{`${marker.classification}: ${marker.reason ?? ""}`}</title>
            </circle>
          </g>
        ))}
        {unreliable.map((point) => {
          const x = PADDING.left + scaleTime(point.timeMs, xStart, xEnd, WIDTH - PADDING.left - PADDING.right);
          return (
            <g key={`unreliable-${point.id}`}>
              <line x1={x} x2={x} y1={bottom - 6} y2={bottom} stroke="#94a3b8" strokeWidth={1} />
              <title>{`DATA_UNRELIABLE: ${point.reason ?? "no shoreline/tide evidence"}`}</title>
            </g>
          );
        })}
        {showTicks &&
          ticks.map((tick, index) => (
            <text
              key={tick}
              x={PADDING.left + scaleTime(tick, xStart, xEnd, WIDTH - PADDING.left - PADDING.right)}
              y={svgHeight - 8}
              textAnchor={index === 0 ? "start" : index === ticks.length - 1 ? "end" : "middle"}
              className="fill-slate-500"
              fontSize={10}
            >
              {formatTime(tick, timeZone)}
            </text>
          ))}
      </svg>
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
    () => baselineStatus === "rolling_unverified" || points.some((point) => point.detectorMode === "rolling_baseline"),
    [baselineStatus, points],
  );
  const positionLines = useMemo(
    () => points.filter((point) => point.observed !== null || point.rollingBaseline !== null || point.expected !== null),
    [points],
  );

  if (points.length === 0) {
    return (
      <p role="status" className="rounded-md border border-slate-200 bg-slate-50 px-3 py-8 text-center text-sm text-slate-500">
        No anomaly/quality observations recorded for this selection.
      </p>
    );
  }

  const xStart = points[0].timeMs;
  const xEnd = points[points.length - 1].timeMs === xStart ? xStart + 60_000 : points[points.length - 1].timeMs;
  const baselineSelector = (point: AnomalyPoint) => point.rollingBaseline ?? point.expected;
  const positionBounds =
    finiteBounds(
      positionLines.flatMap((point) => [point.observed, point.rollingBaseline, point.expected]),
    ) ?? { min: 0, max: 1 };
  const tidePoints = points.filter((point) => point.tide !== null);
  const tideBounds = finiteBounds(tidePoints.map((point) => point.tide));
  const residualPoints = points.filter((point) => point.residual !== null);
  const residualBounds = finiteBounds(residualPoints.map((point) => point.residual)) ?? { min: -1, max: 1 };
  const zPoints = points.filter((point) => point.robustZ !== null);
  const zBounds = finiteBounds(zPoints.map((point) => point.robustZ)) ?? { min: -1, max: 1 };
  const cumulativePoints = points.filter((point) => point.cumulativeDrawdown !== null);
  const cumulativeValues = cumulativePoints.map((point) => point.cumulativeDrawdown as number);
  const cumulativeBounds = finiteBounds([...cumulativeValues, 0]) ?? { min: 0, max: 1 };
  const positionMarkers = markers(points, positionBounds, POSITION_HEIGHT, xStart, xEnd, PLOT_TOP);
  const unreliablePoints = points.filter((point) => point.classification === "DATA_UNRELIABLE");
  const ticks = [xStart, xStart + (xEnd - xStart) / 2, xEnd];

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
        <span className="inline-flex items-center gap-1"><span className="h-2 w-4 rounded bg-slate-700" /> observed position</span>
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-4 rounded bg-teal-600" />
          {rolling ? "Rolling baseline (unverified; no tide data)" : "expected (provisional)"}
        </span>
        {tideBounds && (
          <span className="inline-flex items-center gap-1"><span className="h-2 w-4 rounded bg-blue-600" /> tide</span>
        )}
        {rolling && (
          <span className="inline-flex items-center gap-1"><span className="h-2 w-4 rounded bg-orange-500" /> cumulative drawdown</span>
        )}
        {Object.entries(counts)
          .filter(([classification]) => !(rolling && classification === "NORMAL"))
          .map(([classification, count]) => (
            <span key={classification} className="rounded bg-slate-100 px-2 py-0.5">
              {classification}: {count}
            </span>
          ))}
        <span className="text-slate-400">times: {timeZone}</span>
      </div>

      <ChartPanel
        title="Shoreline position"
        unit="px · + = seaward"
        points={positionLines}
        selector={(point) => point.observed}
        secondarySelector={baselineSelector}
        secondaryColor="#0f766e"
        secondaryDash="5 3"
        bounds={positionBounds}
        height={POSITION_HEIGHT}
        color="#334155"
        markers={positionMarkers}
        selectedEventId={selectedEventId}
        onEventClick={onEventClick}
        unreliable={unreliablePoints}
        xStart={xStart}
        xEnd={xEnd}
        ticks={ticks}
        timeZone={timeZone}
      />
      {tideBounds && (
        <ChartPanel
          title="Tide height"
          unit="m"
          points={tidePoints}
          selector={(point) => point.tide}
          bounds={tideBounds}
          height={TIDE_HEIGHT}
          color="#2563eb"
          xStart={xStart}
          xEnd={xEnd}
          ticks={ticks}
          timeZone={timeZone}
        />
      )}
      <ChartPanel
        title={rolling ? "Residual from rolling baseline" : "Residual from expected"}
        unit="px · + = seaward"
        points={residualPoints}
        selector={(point) => point.residual}
        bounds={residualBounds}
        height={RESIDUAL_HEIGHT}
        color="#c0392b"
        zeroLine
        xStart={xStart}
        xEnd={xEnd}
        ticks={ticks}
        timeZone={timeZone}
        emptyText={
          rolling
            ? "Residual not recorded for this selection."
            : "Residual unavailable — no verified tide-conditioned baseline."
        }
      />
      <ChartPanel
        title="Robust z-score"
        unit="σ"
        points={zPoints}
        selector={(point) => point.robustZ}
        bounds={zBounds}
        height={Z_HEIGHT}
        color="#2563eb"
        zeroLine
        xStart={xStart}
        xEnd={xEnd}
        ticks={ticks}
        timeZone={timeZone}
        emptyText="Robust z unavailable for this selection."
      />
      <ChartPanel
        title="Cumulative drawdown Ck"
        unit="px · + = seaward"
        points={cumulativePoints}
        selector={(point) => point.cumulativeDrawdown}
        bounds={cumulativeBounds}
        height={CUMULATIVE_HEIGHT}
        color="#f97316"
        zeroLine
        showTicks
        xStart={xStart}
        xEnd={xEnd}
        ticks={ticks}
        timeZone={timeZone}
        emptyText="Cumulative drawdown is recorded per valid observation in the rolling output."
      />

      {!tideBounds && (
        <p className="text-xs text-slate-500">
          Tide height unavailable. No verified, geographically appropriate Thai tide data is associated
          with this footage, so no tide graph is shown.
        </p>
      )}
    </div>
  );
}
