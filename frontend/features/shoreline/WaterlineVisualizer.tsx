"use client";

import { useMemo } from "react";

import {
  decimateSegment,
  observationSegments,
  polylinePoints,
  waterlineBounds,
} from "@/features/shoreline/waterlineGeometry";
import type { ShorelineObservationDetail } from "@/lib/types";

type Props = {
  detail: ShorelineObservationDetail | null;
  emptyMessage?: string;
  heightClass?: string;
};

/**
 * Renders the ordered Ocean/Beach waterline segments for one stored
 * observation. Coordinates are image pixels (y increases downward), so the SVG
 * keeps the native aspect ratio and never mirrors the shoreline.
 */
export function WaterlineVisualizer({
  detail,
  emptyMessage = "No waterline observation is selected.",
  heightClass = "h-64",
}: Props) {
  const segments = useMemo(
    () => (detail ? observationSegments(detail).map((segment) => decimateSegment(segment)) : []),
    [detail],
  );
  const bounds = useMemo(() => waterlineBounds(segments), [segments]);
  const pointCount = useMemo(
    () => segments.reduce((total, segment) => total + segment.length, 0),
    [segments],
  );

  if (!detail) {
    return (
      <p
        role="status"
        className={`rounded-md border border-dashed border-slate-200 bg-slate-50 px-3 py-8 text-center text-sm text-slate-500 ${heightClass}`}
      >
        {emptyMessage}
      </p>
    );
  }

  if (segments.length === 0) {
    return (
      <p
        role="status"
        className={`rounded-md border border-amber-200 bg-amber-50 px-3 py-8 text-center text-sm text-amber-800 ${heightClass}`}
      >
        This observation has no extracted shoreline geometry
        {detail.extraction_reason ? ` (${detail.extraction_reason})` : ""}. No line is fabricated.
      </p>
    );
  }

  const gridStep = Math.max(40, Math.round(Math.max(bounds.width, bounds.height) / 8));

  return (
    <div className="space-y-2">
      <svg
        viewBox={`0 0 ${bounds.width} ${bounds.height}`}
        preserveAspectRatio="xMidYMid meet"
        className={`w-full rounded-md border border-slate-200 bg-slate-950 ${heightClass}`}
        role="img"
        aria-label={`Extracted waterline: ${segments.length} segment(s), ${pointCount} points`}
      >
        <defs>
          <pattern
            id="waterline-grid"
            width={gridStep}
            height={gridStep}
            patternUnits="userSpaceOnUse"
          >
            <path
              d={`M ${gridStep} 0 L 0 0 0 ${gridStep}`}
              fill="none"
              stroke="#1e293b"
              strokeWidth={1}
            />
          </pattern>
        </defs>
        <rect width={bounds.width} height={bounds.height} fill="url(#waterline-grid)" />
        {segments.map((segment, index) => {
          const first = segment[0];
          const last = segment[segment.length - 1];
          return (
            <g key={index}>
              <polyline
                points={polylinePoints(segment)}
                fill="none"
                stroke="#22d3ee"
                strokeWidth={2.5}
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
              <circle cx={first[0]} cy={first[1]} r={3} fill="#34d399" stroke="#064e3b" strokeWidth={1} />
              <circle cx={last[0]} cy={last[1]} r={3} fill="#f87171" stroke="#7f1d1d" strokeWidth={1} />
            </g>
          );
        })}
      </svg>
      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-4 bg-cyan-400" /> waterline
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-emerald-400" /> start
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-red-400" /> end
        </span>
        <span>
          {segments.length} segment(s) · {pointCount} rendered point(s)
        </span>
      </div>
    </div>
  );
}
