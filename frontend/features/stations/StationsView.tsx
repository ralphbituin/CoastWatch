import { PageHeader } from "@/components/PageHeader";
import { PhasePlaceholder } from "@/components/PhasePlaceholder";

export function StationsView() {
  return (
    <div>
      <PageHeader
        title="Monitoring stations"
        description="Coastal sites, tide-station associations and countries."
      />
      <PhasePlaceholder
        module="Stations"
        phase={2}
        description="Create and manage monitoring locations, associate them with tide stations, and keep per-station time zones, datums and units separate."
        planned={[
          "Create / edit monitoring stations with coordinates and timezone",
          "Link stations to tide stations (never blend unrelated datums)",
          "Country reference data (PH today, SEA later)",
          "Per-station data-quality status",
        ]}
      />
    </div>
  );
}
