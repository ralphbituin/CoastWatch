"use client";

import { PageHeader } from "@/components/PageHeader";
import { PhasePlaceholder } from "@/components/PhasePlaceholder";
import { TideChartDemo } from "@/features/tides/TideChart";

export function TidesView() {
  return (
    <div>
      <PageHeader
        title="Tide predictions"
        description="Tide stations, CSV import, interpolation and phase classification."
      />
      <div className="space-y-6">
        <PhasePlaceholder
          module="Tides"
          phase={2}
          description="Import validated NAMRIA-style CSV datasets, interpolate predicted tide heights, and classify high/low and rising/falling conditions for any supported timestamp."
          planned={[
            "CSV import with configurable column mapping and preview",
            "Linear and PCHIP interpolation (no silent extrapolation)",
            "High/low water detection and tide-phase classification",
            "24h / 48h / 7-day charts with current-time marker",
            "Import history and validation reporting",
          ]}
        />
        <TideChartDemo />
      </div>
    </div>
  );
}
