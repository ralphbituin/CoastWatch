"use client";

import { useState } from "react";
import { CheckCircle2, FileUp, Loader2, XCircle } from "lucide-react";
import clsx from "clsx";

import { Card } from "@/components/Card";
import { StatusBadge } from "@/components/StatusBadge";
import { tidesApi } from "@/lib/tides";
import type { ImportCommitResponse, ImportPreviewResponse, TideStation } from "@/lib/types";

const UNITS = ["meters", "feet", "centimeters"];

export function ImportPanel({
  stations,
  selectedStationCode,
  onSelectStation,
  onImported,
}: {
  stations: TideStation[];
  selectedStationCode: string;
  onSelectStation: (code: string) => void;
  onImported: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [timezone, setTimezone] = useState("Asia/Manila");
  const [unit, setUnit] = useState("meters");
  const [datum, setDatum] = useState("");
  const [mappingHeight, setMappingHeight] = useState("tide_height");
  const [mappingTimestamp, setMappingTimestamp] = useState("timestamp");
  const [mappingEvent, setMappingEvent] = useState("event_type");

  const [preview, setPreview] = useState<ImportPreviewResponse | null>(null);
  const [commit, setCommit] = useState<ImportCommitResponse | null>(null);
  const [busy, setBusy] = useState<"preview" | "commit" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mapping = {
    timestamp: mappingTimestamp,
    height: mappingHeight,
    event_type: mappingEvent || undefined,
  } as Record<string, string>;

  const runPreview = async () => {
    if (!file) return;
    setBusy("preview");
    setError(null);
    setCommit(null);
    try {
      const result = await tidesApi.previewImport(file, timezone, mapping);
      setPreview(result);
    } catch (err) {
      setPreview(null);
      setError(err instanceof Error ? err.message : "Preview failed");
    } finally {
      setBusy(null);
    }
  };

  const runCommit = async () => {
    if (!file || !selectedStationCode || !name.trim()) return;
    setBusy("commit");
    setError(null);
    try {
      const result = await tidesApi.commitImport(file, {
        station_code: selectedStationCode,
        name: name.trim(),
        timezone,
        unit,
        vertical_datum: datum.trim() || undefined,
        source: "NAMRIA_CSV",
        columnMapping: mapping,
      });
      setCommit(result);
      setPreview(null);
      setFile(null);
      setName("");
      onImported();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Import failed");
    } finally {
      setBusy(null);
    }
  };

  return (
    <Card
      title="Import NAMRIA tide CSV"
      subtitle="Values are stored with station, dataset, unit and datum — datums are never blended"
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-3 text-sm">
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500">Tide station</span>
            <select
              className="w-full rounded-md border border-slate-200 px-3 py-2"
              value={selectedStationCode}
              onChange={(e) => onSelectStation(e.target.value)}
            >
              <option value="">— select station —</option>
              {stations.map((s) => (
                <option key={s.id} value={s.code}>
                  {s.code} · {s.name}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500">Dataset name</span>
            <input
              className="w-full rounded-md border border-slate-200 px-3 py-2"
              placeholder="e.g. NAMRIA Manila Tide Table 2026"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">Time zone (IANA)</span>
              <input
                className="w-full rounded-md border border-slate-200 px-3 py-2"
                value={timezone}
                onChange={(e) => setTimezone(e.target.value)}
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">Unit</span>
              <select
                className="w-full rounded-md border border-slate-200 px-3 py-2"
                value={unit}
                onChange={(e) => setUnit(e.target.value)}
              >
                {UNITS.map((u) => (
                  <option key={u} value={u}>
                    {u}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500">Vertical datum (kept per station)</span>
            <input
              className="w-full rounded-md border border-slate-200 px-3 py-2"
              placeholder="e.g. MLWS, MLLW, LAT"
              value={datum}
              onChange={(e) => setDatum(e.target.value)}
            />
          </label>

          <div className="grid grid-cols-3 gap-2">
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">Timestamp col.</span>
              <input
                className="w-full rounded-md border border-slate-200 px-2 py-2 text-xs"
                value={mappingTimestamp}
                onChange={(e) => setMappingTimestamp(e.target.value)}
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">Height col.</span>
              <input
                className="w-full rounded-md border border-slate-200 px-2 py-2 text-xs"
                value={mappingHeight}
                onChange={(e) => setMappingHeight(e.target.value)}
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">Event col. (opt.)</span>
              <input
                className="w-full rounded-md border border-slate-200 px-2 py-2 text-xs"
                value={mappingEvent}
                onChange={(e) => setMappingEvent(e.target.value)}
              />
            </label>
          </div>

          <label className="flex cursor-pointer items-center justify-center gap-2 rounded-md border border-dashed border-slate-300 px-4 py-6 text-slate-500 hover:border-reef hover:text-reef">
            <FileUp className="h-5 w-5" aria-hidden />
            <span className="text-sm">{file ? file.name : "Choose NAMRIA-style CSV…"}</span>
            <input
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>

          <div className="flex gap-2">
            <button
              onClick={runPreview}
              disabled={!file || busy !== null}
              className="inline-flex items-center gap-2 rounded-md border border-reef px-4 py-2 text-sm font-medium text-reef hover:bg-reef hover:text-white disabled:opacity-40"
            >
              {busy === "preview" ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Preview
            </button>
            <button
              onClick={runCommit}
              disabled={!file || !selectedStationCode || !name.trim() || busy !== null}
              className="inline-flex items-center gap-2 rounded-md bg-deep px-4 py-2 text-sm font-medium text-white hover:bg-lagoon disabled:opacity-40"
            >
              {busy === "commit" ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Commit import
            </button>
          </div>

          {error && (
            <p className="flex items-start gap-2 text-sm text-red-600">
              <XCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              {error}
            </p>
          )}
          {commit && (
            <p className="flex items-start gap-2 text-sm text-emerald-700">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              Import #{commit.import_id}: status <strong>{commit.status}</strong> —{" "}
              {commit.imported_records} inserted, {commit.rejected_records} rejected.
            </p>
          )}
        </div>

        <div className="space-y-3">
          {preview ? (
            <>
              <div className="flex flex-wrap gap-2">
                <StatusBadge tone={preview.valid_records > 0 ? "ok" : "error"}>
                  {preview.valid_records} valid
                </StatusBadge>
                <StatusBadge tone={preview.rejected_records > 0 ? "warn" : "neutral"}>
                  {preview.rejected_records} rejected
                </StatusBadge>
                <StatusBadge tone="neutral">{preview.total_records} total</StatusBadge>
              </div>
              <p className="text-xs text-slate-500">{preview.message}</p>
              <p className="text-xs text-slate-500">
                detected columns: {preview.columns_detected.join(", ")}
              </p>
              <p className="text-xs text-slate-500">
                coverage:{" "}
                {preview.coverage_start ? new Date(preview.coverage_start).toLocaleString() : "—"}{" "}
                → {preview.coverage_end ? new Date(preview.coverage_end).toLocaleString() : "—"}
              </p>
              {Object.keys((preview.rejection_summary ?? {}) as Record<string, unknown>).length >
                0 && (
                <div className="rounded-md bg-amber-50 p-3 text-xs text-amber-800">
                  <p className="font-semibold">Rejected records by reason</p>
                  <ul className="mt-1 list-inside list-disc">
                    {Object.entries(
                      (preview.rejection_summary as { by_reason?: Record<string, number> })
                        .by_reason ?? {},
                    ).map(([reason, count]) => (
                      <li key={reason}>
                        {reason}: {String(count)}
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2">
                    Rejected rows are kept (bounded to {preview.rejection_samples.length} shown raw)
                    — nothing is silently discarded.
                  </p>
                </div>
              )}
            </>
          ) : (
            <div className="rounded-md border border-slate-100 bg-slate-50 p-4 text-sm text-slate-500">
              <p className="font-medium text-slate-700">Preview before commit</p>
              <p className="mt-1">
                Choose a file and press <em>Preview</em>. The importer validates dates, duplicates,
                heights, time zones, datum metadata and ordering; rejected rows are shown with
                reasons before anything is written to the database.
              </p>
              <p className="mt-2 text-xs">
                Timestamps may be naive local time (interpreted in the station timezone) or explicit
                offsets. Heights accept decimal comma or dot.
              </p>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
