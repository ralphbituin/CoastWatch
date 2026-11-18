"use client";

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card } from "@/components/Card";

/**
 * Demonstration tide chart. Phase 1 renders a clearly-labelled *synthetic*
 * sinusoid to validate the charting stack; real predicted tide data from the
 * Phase 2 engine replaces this.
 */
export function TideChartDemo() {
  const [hours] = useState(48);

  const data = useMemo(() => {
    const start = new Date();
    start.setMinutes(0, 0, 0);
    return Array.from({ length: hours }, (_, i) => {
      const t = new Date(start.getTime() + i * 3600_000);
      const height = 0.9 * Math.sin((i / 12.4) * Math.PI) + 0.1; // synthetic
      return { time: `${t.getHours().toString().padStart(2, "0")}:00`, height: Number(height.toFixed(2)) };
    });
  }, [hours]);

  return (
    <Card
      title="Tide chart (synthetic placeholder)"
      subtitle="Not real tide data — replaced by the Phase 2 prediction engine"
    >
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: -16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="time" tick={{ fontSize: 11 }} interval={5} />
            <YAxis tick={{ fontSize: 11 }} unit=" m" />
            <Tooltip />
            <ReferenceLine y={0} stroke="#94a3b8" />
            <Line type="monotone" dataKey="height" stroke="#0891b2" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
