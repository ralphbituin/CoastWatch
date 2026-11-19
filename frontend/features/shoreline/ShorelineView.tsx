"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2, Play, RefreshCw, Upload, XCircle } from "lucide-react";

import { Card } from "@/components/Card";
import { PageHeader } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { camerasApi } from "@/lib/cameras";
import { shorelinesApi } from "@/lib/shorelines";
import { stationsApi } from "@/lib/stations";
import type {
  Camera,
  ShorelineModuleStatus,
  ShorelineObservationDetail,
  ShorelineObservationSummary,
  ShorelineObserveResult,
} from "@/lib/types";

function WaterlinePreview({ detail }: { detail: ShorelineObservationDetail }) {
  const geometry = detail.shoreline_geometry;
  const polyline = useMemo(() => {
    if (!geometry || geometry.length < 2) return null;
    // Decimate dense polylines for SVG rendering.
    const stride = Math.ceil(geometry.length / 240);
    return geometry
      .filter((_, i) => i % stride === 0)
      .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
      .join(" ");
  }, [geometry]);
  if (!geometry || geometry.length < 2 || !polyline) return null;
  const maxX = Math.max(...geometry.map((p) => p[0]), 1);
  const maxY = Math.max(...geometry.map((p) => p[1]), 1);
  return (
    <svg
      viewBox={`0 0 ${maxX} ${maxY}`}
      className="mt-2 h-44 w-full rounded-md border border-slate-200 bg-slate-900"
      role="img"
      aria-label="Extracted waterline geometry"
    >
      <polyline
        points={polyline}
        fill="none"
        stroke="#22d3ee"
        strokeWidth={Math.max(2, maxX / 320)}
      />
    </svg>
  );
}

export function ShorelineView() {
  const [status, setStatus] = useState<ShorelineModuleStatus | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<number | "">("");
  const [observations, setObservations] = useState<ShorelineObservationSummary[]>([]);
  const [latest, setLatest] = useState<ShorelineObservationDetail | null>(null);
  const [observeResult, setObserveResult] = useState<ShorelineObserveResult | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [stationId, setStationId] = useState<number | null>(null);

  const [maskForm, setMaskForm] = useState({ seaward_axis: "", observed_at: "" });

  const loadAll = useCallback(async () => {
    try {
      const [moduleStatus, cameraList] = await Promise.all([
        shorelinesApi.status(),
        camerasApi.list(),
      ]);
      setStatus(moduleStatus);
      setCameras(cameraList);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load shorelines module");
    }
  }, []);

  const loadObservations = useCallback(async (cameraId?: number) => {
    try {
      const [list, latestResult] = await Promise.allSettled([
        shorelinesApi.listObservations({ camera_id: cameraId, limit: 20 }),
        shorelinesApi.latestObservation(cameraId ? { camera_id: cameraId } : {}),
      ]);
      setObservations(list.status === "fulfilled" ? list.value : []);
      setLatest(latestResult.status === "fulfilled" ? latestResult.value : null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load observations");
    }
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  useEffect(() => {
    const cameraId = selectedCameraId === "" ? undefined : selectedCameraId;
    void loadObservations(cameraId);
    const cam = cameras.find((c) => c.id === cameraId);
    setStationId(cam?.monitoring_station_id ?? null);
  }, [selectedCameraId, cameras, loadObservations]);

  const report = (message: string) => {
    setNotice(message);
    setTimeout(() => setNotice(null), 4000);
  };

  const runObserve = async () => {
    if (stationId === null || selectedCameraId === "") return;
    setBusy(true);
    setError(null);
    try {
      const result = await shorelinesApi.observe({ station_id: stationId, camera_id: selectedCameraId });
      setObserveResult(result);
      await loadObservations(selectedCameraId);
      if (result.state === "DATA_UNRELIABLE") {
        setError(`Observation unavailable: ${(result.details as { message?: string }).message ?? result.details["reason"]}`);
      } else {
        report("Observation persisted.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Observation cycle failed");
    } finally {
      setBusy(false);
    }
  };

  const uploadMask = async (file: File) => {
    if (selectedCameraId === "") return;
    setBusy(true);
    setError(null);
    try {
      await shorelinesApi.uploadMask({
        camera_id: selectedCameraId,
        file,
        seaward_axis: maskForm.seaward_axis.trim() || undefined,
        observed_at: maskForm.observed_at.trim() || undefined,
      });
      setMaskForm({ seaward_axis: "", observed_at: "" });
      report("Mask uploaded and observation recorded.");
      await loadObservations(selectedCameraId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Mask upload failed");
    } finally {
      setBusy(false);
    }
  };

  const stateTone = (state?: string) =>
    state === "OBSERVED" ? "ok" : state === "DATA_UNRELIABLE" ? "warn" : "neutral";

  return (
    <div className="space-y-6">
      <PageHeader
        title="Shoreline"
        description="Waterline extraction from operator-labelled masks or (once trained) the segmentation model."
      />

      {status && (
        <Card className={status.segmentation_model_available ? "border-emerald-200" : "border-amber-200 bg-amber-50"}>
          <div className="flex items-start gap-2 text-sm">
            {status.segmentation_model_available ? (
              <CheckCircle2 className="mt-0.5 h-4 w-4 text-emerald-600" aria-hidden />
            ) : (
              <AlertTriangle className="mt-0.5 h-4 w-4 text-amber-600" aria-hidden />
            )}
            <span className={status.segmentation_model_available ? "text-emerald-800" : "text-amber-800"}>
              {status.segmentation_model_available
                ? "Segmentation model artifact detected."
                : "No segmentation model is provisioned yet — waterlines require operator-labelled masks or a trained model (see docs/MODEL_DOCUMENTATION.md)."}
            </span>
          </div>
        </Card>
      )}

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
        <Card title="Observations" subtitle="Most recent calibrated positions first">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <select
              className="rounded-md border border-slate-200 px-3 py-2 text-sm"
              value={selectedCameraId}
              onChange={(e) => setSelectedCameraId(e.target.value ? Number(e.target.value) : "")}
            >
              <option value="">All cameras</option>
              {cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.code} · {c.name}
                </option>
              ))}
            </select>
            <button
              onClick={() => void loadObservations(selectedCameraId === "" ? undefined : selectedCameraId)}
              className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50"
            >
              <RefreshCw className="h-4 w-4" aria-hidden />
              Refresh
            </button>
          </div>

          {observations.length === 0 ? (
            <p className="text-sm text-slate-400">
              No observations yet — upload an operator-labelled mask or run an observation cycle.
            </p>
          ) : (
            <ul className="divide-y divide-slate-100 text-sm">
              {observations.map((obs) => (
                <li key={`${obs.camera_id}-${obs.observed_at}`} className="flex flex-wrap items-center justify-between gap-2 py-2">
                  <span className="text-slate-700">
                    <span className="mr-2 text-xs text-slate-400">{obs.camera_code ?? `camera #${obs.camera_id}`}</span>
                    {new Date(obs.observed_at).toLocaleString()}
                  </span>
                  <span className="flex flex-wrap gap-2 text-xs text-slate-500">
                    <span className="rounded bg-slate-100 px-2 py-0.5">
                      {obs.calibrated_position_m !== null ? `${obs.calibrated_position_m.toFixed(3)} m` : "uncalibrated"}
                    </span>
                    <span className="rounded bg-slate-100 px-2 py-0.5">
                      confidence {obs.segmentation_confidence ?? "n/a"}
                    </span>
                    {obs.frame_quality_flags && obs.frame_quality_flags.length > 0 && (
                      <StatusBadge tone="warn">{obs.frame_quality_flags.join(", ")}</StatusBadge>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Latest waterline" subtitle="Extracted polyline from the most recent observation">
          {latest ? (
            <div>
              <div className="flex flex-wrap items-center gap-3 text-sm">
                <StatusBadge tone="ok">{latest.camera_code}</StatusBadge>
                <span className="text-slate-500">
                  {new Date(latest.observed_at).toLocaleString()}
                  {latest.calibrated_position_m !== null
                    ? ` · ${latest.calibrated_position_m.toFixed(3)} m seaward (calibrated v${latest.calibration_id})`
                    : " · uncalibrated"}
                </span>
              </div>
              <WaterlinePreview detail={latest} />
            </div>
          ) : (
            <p className="text-sm text-slate-400">No observations recorded yet.</p>
          )}

          <div className="mt-4 border-t border-slate-100 pt-4">
            <h3 className="text-xs font-semibold text-slate-500">Run observation cycle</h3>
            <p className="mt-1 text-xs text-slate-500">
              Captures a frame, attempts segmentation, and persists the result. Without a model the
              result is honestly reported as DATA_UNRELIABLE.
            </p>
            <button
              onClick={() => void runObserve()}
              disabled={busy || selectedCameraId === ""}
              className="mt-2 inline-flex items-center gap-2 rounded-md border border-reef px-3 py-2 text-sm font-medium text-reef hover:bg-reef hover:text-white disabled:opacity-40"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" aria-hidden />}
              Observe now
            </button>
            {observeResult && (
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-600">
                <StatusBadge tone={stateTone(observeResult.state)}>{observeResult.state}</StatusBadge>
                <span className="text-slate-500">
                  {(observeResult.details as { message?: string }).message ??
                    String(observeResult.details["reason"] ?? "")}
                </span>
              </div>
            )}
          </div>

          <div className="mt-4 border-t border-slate-100 pt-4">
            <h3 className="text-xs font-semibold text-slate-500">Operator-labelled water mask</h3>
            <p className="mt-1 text-xs text-slate-500">
              PNG/JPEG where water = white. Extracted immediately; camera must be selected above.
            </p>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-slate-500">Seaward axis (dx,dy, optional)</span>
                <input
                  className="w-full rounded-md border border-slate-200 px-3 py-2 text-xs"
                  placeholder="0.94,0.34"
                  value={maskForm.seaward_axis}
                  onChange={(e) => setMaskForm({ ...maskForm, seaward_axis: e.target.value })}
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-slate-500">Observed at (ISO, optional)</span>
                <input
                  className="w-full rounded-md border border-slate-200 px-3 py-2 text-xs"
                  placeholder="2026-11-19T06:30:00+00:00"
                  value={maskForm.observed_at}
                  onChange={(e) => setMaskForm({ ...maskForm, observed_at: e.target.value })}
                />
              </label>
            </div>
            <label className="mt-2 inline-flex cursor-pointer items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-xs text-slate-600 hover:bg-slate-50">
              <Upload className="h-4 w-4" aria-hidden />
              Upload mask (PNG/JPEG)
              <input
                type="file"
                accept=".png,.jpg,.jpeg"
                className="hidden"
                disabled={selectedCameraId === ""}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void uploadMask(file);
                  e.target.value = "";
                }}
              />
            </label>
          </div>
        </Card>
      </div>

      <Card title="Notes & limits" className="text-xs text-slate-500">
        <ul className="list-inside list-disc space-y-1">
          <li>No segmentation model ships with the repo. Masks come from operators today; a U-Net/DeepLabV3+ trained on coastal footage can be dropped in later without API changes.</li>
          <li>Contour extraction uses per-column water extents with sea-side detection (the half of the frame carrying more water is assumed to be sea).</li>
          <li>Calibrated positions require an active calibration profile with a homography and a seaward axis; otherwise positions stay uncalibrated (never defaulted to 0).</li>
          <li>Baselines (expected shoreline given the tide) land in Phase 5; anomaly scoring in Phase 6.</li>
        </ul>
      </Card>
    </div>
  );
}
