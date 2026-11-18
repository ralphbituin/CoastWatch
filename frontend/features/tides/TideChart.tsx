"use client";

import { RefreshCw } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import clsx from "clsx";

import { Card } from "@/components/Card";
import type { TideChartResponse } from "@/lib/types";

const RANGE_OPTIONS = [
  { label: "24 h", hours: 24 },
  { label: "48 h", hours: 48 },
  { label: "7 days", hours: 168 },
];

function formatTick(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleString("en-PH", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function TideChart({
  data,
  loading,
  error,
  rangeHours,
  onRangeChange,
  method,
  onMethodChange,
  onRefresh,
}: {
  data: TideChartResponse | null;
  loading: boolean;
  error: string | null;
  rangeHours: number;
  onRangeChange: (hours: number) => void;
  method: "pchip" | "linear";
  onMethodChange: (method: "pchip" | "linear") => void;
  onRefresh: () => void;
}) {
  const nowIso = new Date().toISOString();

  return (
    <Card title="Tide chart" subtitle="Predicted heights with high/low water events">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5">
          {RANGE_OPTIONS.map(({ label, hours }) => (
            <button
              key={hours}
              onClick={() => onRangeChange(hours)}
              className={clsx(
                "rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
                rangeHours === hours
                  ? "border-reef bg-reef text-white"
                  : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50",
              )}
            >
              {label}
            </button>
          ))}
          {data && !data.sufficient_for_7d && rangeHours === 168 && (
            <span className="self-center text-xs text-amber-600">
              (insufficient 7-day coverage — showing available data)
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {(["pchip", "linear"] as const).map((m) => (
            <button
              key={m}
              onClick={() => onMethodChange(m)}
              className={clsx(
                "rounded-md border px-3 py-1.5 text-xs font-medium capitalize transition-colors",
                method === m
                  ? "border-deep bg-deep text-white"
                  : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50",
              )}
            >
              {m}
            </button>
          ))}
          <button
            onClick={onRefresh}
            className="rounded-md border border-slate-200 p-1.5 text-slate-500 hover:bg-slate-50"
            aria-label="Refresh chart"
          >
            <RefreshCw className="h-4 w-4" aria-hidden />
          </button>
        </div>
      </div>

      {error ? (
        <p className="text-sm text-red-600">{error}</p>
      ) : loading && !data ? (
        <p className="text-sm text-slate-400">Loading chart…</p>
      ) : data && data.points.length > 0 ? (
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.points} margin={{ top: 12, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatTick}
                tick={{ fontSize: 10 }}
                interval="preserveStartEnd"
                minTickGap={48}
              />
              <YAxis
                width={48}
                tick={{ fontSize: 11 }}
                unit=" m"
                domain={[
                  (dataMin: number) => Math.floor((dataMin - 0.2) * 10) / 10,
                  (dataMax: number) => Math.ceil((dataMax + 0.2) * 10) / 10,
                ]}
              />
              <Tooltip
                labelFormatter={(label: string) =>
                  new Date(label).toLocaleString("en-PH", {
                    dateStyle: "medium",
                    timeStyle: "short",
                  })
                }
                formatter={(value: number) => [`${value.toFixed(2)} m`, "height"]}
              />
              <ReferenceLine x={nowIso} stroke="#f97316" strokeDasharray="4 4" label={{ value: "now", fontSize: 10, fill: "#f97316", position: "insideTopLeft" }} />
              <Line
                type="monotone"
                dataKey="height"
                stroke="#0891b2"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              {data.events.map((event) => (
                <ReferenceDot
                  key={`${event.type}-${event.timestamp}`}
                  x={event.timestamp}
                  y={event.height}
                  r={4}
                  fill={event.type === "high" ? "#10b981" : "#ef4444"}
                  stroke="#ffffff"
                  strokeWidth={1.5}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className="text-sm text-slate-400">
          No prediction data in this range. Import a NAMRIA-style CSV first.
        </p>
      )}

      {data && (
        <div className="mt-2 flex flex-wrap gap-4 text-xs text-slate-500">
          <span>coverage: {data.coverage_start ? new Date(data.coverage_start).toLocaleDateString() : "—"} → {data.coverage_end ? new Date(data.coverage_end).toLocaleDateString() : "—"}</span>
          <span>{data.count} predictions</span>
          <span>datum: {data.points[0]?.datum ?? "n/a"}</span>
        </div>
      )}
    </Card>
  );
}
