"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Clock, RotateCcw, XCircle } from "lucide-react";

import { Card } from "@/components/Card";
import { PageHeader } from "@/components/PageHeader";
import {
  DEFAULT_TIMEZONE,
  isValidTimeZone,
  normalizeTimeZone,
  timezoneLabel,
  useDisplayTimezone,
} from "@/lib/settings";

const COMMON_TIMEZONES = [
  "GMT",
  "UTC",
  "Asia/Manila",
  "Asia/Bangkok",
  "Asia/Tokyo",
  "Australia/Sydney",
  "Europe/London",
  "America/New_York",
  "America/Los_Angeles",
];

export function SettingsView() {
  const { timezone, label, setTimezone, formatDateTime } = useDisplayTimezone();
  const [draft, setDraft] = useState(timezone);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    setDraft(timezone);
  }, [timezone]);

  const report = (message: string) => {
    setNotice(message);
    setTimeout(() => setNotice(null), 3500);
  };

  const save = () => {
    const candidate = draft.trim() || DEFAULT_TIMEZONE;
    if (!isValidTimeZone(candidate)) {
      setError(
        `"${candidate}" is not a valid IANA timezone. Use GMT/UTC or a value such as Asia/Bangkok.`,
      );
      return;
    }
    setError(null);
    setTimezone(candidate);
    report(`Display timezone set to ${timezoneLabel(normalizeTimeZone(candidate))}.`);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        description="Display preferences for live stream sessions and anomaly event timestamps."
      />

      {error && (
        <Card className="border-red-200 bg-red-50 text-sm text-red-700">
          <div className="flex items-start gap-2">
            <XCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            {error}
          </div>
        </Card>
      )}
      {notice && (
        <Card className="border-emerald-200 bg-emerald-50 text-sm text-emerald-700">
          <div className="flex items-start gap-2">
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            {notice}
          </div>
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-2">
        <Card
          title="Live stream session timezone"
          subtitle="Stored timestamps are always UTC; this only changes how they are displayed"
        >
          <div className="space-y-3">
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">
                Timezone (IANA name or GMT/UTC)
              </span>
              <input
                className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="GMT"
                list="timezone-options"
              />
              <datalist id="timezone-options">
                {COMMON_TIMEZONES.map((zone) => (
                  <option key={zone} value={zone} />
                ))}
              </datalist>
            </label>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={save}
                className="inline-flex items-center gap-2 rounded-md bg-deep px-4 py-2 text-sm font-medium text-white hover:bg-lagoon"
              >
                <Clock className="h-4 w-4" aria-hidden />
                Save timezone
              </button>
              <button
                onClick={() => {
                  setDraft(DEFAULT_TIMEZONE);
                  setTimezone(DEFAULT_TIMEZONE);
                  setError(null);
                  report("Display timezone reset to GMT.");
                }}
                className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50"
              >
                <RotateCcw className="h-4 w-4" aria-hidden />
                Use GMT
              </button>
            </div>
            <p className="text-xs text-slate-500">
              Live-capture rows use wall-clock receipt time in UTC. Archive rows keep their own
              derived clock provenance. Changing this setting never rewrites stored timestamps.
            </p>
          </div>
        </Card>

        <Card title="Preview" subtitle="The current preference applied to the same instant">
          <dl className="space-y-2 text-sm">
            <div className="flex items-center justify-between gap-3">
              <dt className="text-slate-500">Active timezone</dt>
              <dd className="font-medium text-slate-800">{label}</dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="text-slate-500">UTC instant</dt>
              <dd className="font-mono text-xs text-slate-700">2026-11-26T00:00:00Z</dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="text-slate-500">Rendered</dt>
              <dd className="font-mono text-xs text-slate-700">
                {formatDateTime("2026-11-26T00:00:00Z")}
              </dd>
            </div>
          </dl>
          <p className="mt-3 text-xs text-slate-500">
            Default is <span className="font-medium">GMT</span>, matching the live-capture
            provenance. The setting is stored in this browser.
          </p>
        </Card>
      </div>
    </div>
  );
}
