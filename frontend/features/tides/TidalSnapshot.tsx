"use client";

import {
  ArrowDown,
  ArrowUp,
  ArrowRight,
  Anchor,
  AlertCircle,
} from "lucide-react";

import { Card } from "@/components/Card";
import { StatusBadge } from "@/components/StatusBadge";
import type { TidePhaseResponse } from "@/lib/types";

function directionMeta(direction: string): { icon: typeof ArrowUp; label: string; tone: "ok" | "warn" | "neutral" } {
  switch (direction) {
    case "rising":
      return { icon: ArrowUp, label: "Rising tide", tone: "ok" };
    case "falling":
      return { icon: ArrowDown, label: "Falling tide", tone: "warn" };
    case "slack_high":
      return { icon: ArrowRight, label: "Slack (high water)", tone: "neutral" };
    case "slack_low":
      return { icon: ArrowRight, label: "Slack (low water)", tone: "neutral" };
    default:
      return { icon: ArrowRight, label: "Unknown / slack", tone: "neutral" };
  }
}

const QUALITY_TONE: Record<string, "ok" | "warn" | "error" | "neutral"> = {
  ok: "ok",
  interpolated: "warn",
  estimated: "warn",
  unknown: "neutral",
  unavailable: "error",
};

export function TidalSnapshot({
  title,
  data,
  loading,
  error,
}: {
  title?: string;
  data: TidePhaseResponse | null;
  loading: boolean;
  error: string | null;
}) {
  return (
    <Card title={title || undefined}>
      {error ? (
        <div className="flex items-start gap-2 text-sm text-red-600">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <p>{error}</p>
        </div>
      ) : loading && !data ? (
        <p className="text-sm text-slate-400">Loading prediction…</p>
      ) : data ? (
        <div className="space-y-4">
          <div className="flex items-end gap-4">
            <div>
              <p className="text-5xl font-semibold tracking-tight text-slate-900">
                {data.predicted_height_m.toFixed(2)}
                <span className="ml-1 text-xl font-normal text-slate-400">{data.unit}</span>
              </p>
              <p className="mt-1 text-xs text-slate-500">
                {new Date(data.local_timestamp).toLocaleString("en-PH", {
                  year: "numeric",
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                  timeZoneName: "short",
                })}{" "}
                · method {data.interpolation_method}
              </p>
            </div>
            {(() => {
              const { icon: Icon, label, tone } = directionMeta(data.direction);
              return (
                <StatusBadge tone={tone}>
                  <Icon className="mr-1 h-3.5 w-3.5" aria-hidden />
                  {label}
                </StatusBadge>
              );
            })()}
          </div>

          <div className="flex flex-wrap gap-2 text-xs">
            <StatusBadge tone={QUALITY_TONE[data.quality] ?? "neutral"}>
              quality: {data.quality}
            </StatusBadge>
            <StatusBadge tone="neutral">
              <Anchor className="mr-1 h-3.5 w-3.5" aria-hidden />
              {data.vertical_datum ?? "datum n/a"}
            </StatusBadge>
            <StatusBadge tone="neutral">{data.source ?? "source n/a"}</StatusBadge>
          </div>
          <p className="text-xs text-slate-400">{data.quality_note}</p>

          <div className="grid grid-cols-2 gap-3 border-t border-slate-100 pt-3 text-sm">
            <div>
              <p className="text-xs font-medium uppercase text-slate-400">Previous event</p>
              {data.previous_event ? (
                <p className="mt-1 text-slate-800">
                  {data.previous_event.type === "high" ? "High" : "Low"} ·{" "}
                  {new Date(data.previous_event.timestamp).toLocaleString("en-PH", {
                    dateStyle: "short",
                    timeStyle: "short",
                  })}{" "}
                  · {data.previous_event.height.toFixed(2)} m
                </p>
              ) : (
                <p className="mt-1 text-slate-400">none in coverage</p>
              )}
            </div>
            <div>
              <p className="text-xs font-medium uppercase text-slate-400">Next event</p>
              {data.next_event ? (
                <p className="mt-1 text-slate-800">
                  {data.next_event.type === "high" ? "High" : "Low"} ·{" "}
                  {new Date(data.next_event.timestamp).toLocaleString("en-PH", {
                    dateStyle: "short",
                    timeStyle: "short",
                  })}{" "}
                  · {data.next_event.height.toFixed(2)} m
                </p>
              ) : (
                <p className="mt-1 text-slate-400">none in coverage</p>
              )}
            </div>
          </div>
        </div>
      ) : (
        <p className="text-sm text-slate-400">No prediction yet.</p>
      )}
    </Card>
  );
}
