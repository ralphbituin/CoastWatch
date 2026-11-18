"use client";

import { RefreshCw } from "lucide-react";

import { Card } from "@/components/Card";
import { PageHeader } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { DataFlow } from "@/features/dashboard/DataFlow";
import { useHealth } from "@/hooks/useHealth";
import { useRealtimeStatus } from "@/hooks/useWebSocket";
import { API_BASE_URL } from "@/lib/config";

const ROADMAP = [
  { phase: 1, module: "Foundation", status: "complete" },
  { phase: 2, module: "Tide prediction engine", status: "planned" },
  { phase: 3, module: "Camera integration", status: "planned" },
  { phase: 4, module: "Shoreline computer vision", status: "planned" },
  { phase: 5, module: "Tide-aware baseline", status: "planned" },
  { phase: 6, module: "Anomaly detection", status: "planned" },
  { phase: 7, module: "Monitoring dashboard", status: "planned" },
  { phase: 8, module: "Testing & validation", status: "planned" },
  { phase: 9, module: "Edge deployment", status: "planned" },
];

export function DashboardOverview() {
  const { service, db, loading, error, refresh } = useHealth();
  const realtime = useRealtimeStatus();

  const serviceTone = error ? "error" : service ? "ok" : "neutral";
  const dbTone = db?.reachable ? "ok" : db ? "error" : "neutral";
  const realtimeTone =
    realtime.status === "open" ? "ok" : realtime.status === "connecting" ? "warn" : "error";

  return (
    <div>
      <PageHeader
        title="Monitoring dashboard"
        description="Phase 1 foundation — environment, database and realtime connectivity."
      />

      <div className="mb-6 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card title="Backend service">
          <div className="flex items-center justify-between">
            <StatusBadge tone={serviceTone}>
              {error ? "unreachable" : service?.status ?? (loading ? "checking…" : "unknown")}
            </StatusBadge>
            <button
              onClick={refresh}
              className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-slate-800"
              aria-label="Refresh health"
            >
              <RefreshCw className="h-3.5 w-3.5" aria-hidden /> refresh
            </button>
          </div>
          <p className="mt-3 text-xs text-slate-500">
            {service ? `v${service.version} · ${service.environment}` : API_BASE_URL}
          </p>
        </Card>

        <Card title="Database">
          <StatusBadge tone={dbTone}>
            {db ? (db.reachable ? "connected" : "unreachable") : "unknown"}
          </StatusBadge>
          <dl className="mt-3 space-y-1 text-xs text-slate-500">
            <div className="flex justify-between">
              <dt>PostgreSQL</dt>
              <dd>{db?.server_version ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt>TimescaleDB</dt>
              <dd>{db?.timescaledb_version ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt>PostGIS</dt>
              <dd>{db?.postgis_version ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt>Latency</dt>
              <dd>{db?.latency_ms != null ? `${db.latency_ms} ms` : "—"}</dd>
            </div>
          </dl>
        </Card>

        <Card title="Realtime channel">
          <StatusBadge tone={realtimeTone}>{realtime.status}</StatusBadge>
          <p className="mt-3 text-xs text-slate-500">
            Push channel for observations, tide updates and anomaly events
            (used from Phase 7).
          </p>
        </Card>

        <Card title="System state">
          <StatusBadge tone="neutral">NORMAL</StatusBadge>
          <p className="mt-3 text-xs text-slate-500">
            Anomaly states are produced from Phase 6. An anomaly never confirms a
            tsunami — official advisories remain authoritative.
          </p>
        </Card>
      </div>

      <Card title="Data flow" subtitle="camera → waterline → baseline → anomaly → dashboard" className="mb-6">
        <DataFlow />
      </Card>

      <Card title="Development roadmap">
        <ul className="divide-y divide-slate-100">
          {ROADMAP.map((item) => (
            <li key={item.phase} className="flex items-center justify-between py-2 text-sm">
              <span className="text-slate-700">
                <span className="mr-2 inline-block w-16 text-xs font-medium text-slate-400">
                  Phase {item.phase}
                </span>
                {item.module}
              </span>
              <StatusBadge tone={item.status === "complete" ? "ok" : "neutral"}>
                {item.status}
              </StatusBadge>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
