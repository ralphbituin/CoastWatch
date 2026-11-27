"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2, Play, RefreshCw, Upload, XCircle } from "lucide-react";

import { Card } from "@/components/Card";
import { PageHeader } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { LiveShorelineVisualizer } from "@/features/shoreline/LiveShorelineVisualizer";
import {
  decimateSegment,
  observationSegments,
  polylinePoints,
  waterlineBounds,
} from "@/features/shoreline/waterlineGeometry";
import { camerasApi } from "@/lib/cameras";
import { shorelinesApi } from "@/lib/shorelines";
import { stationsApi } from "@/lib/stations";
import type {
  Camera,
  ShorelineBaselineStatus,
  ShorelineModuleStatus,
  ShorelineObservationDetail,
  ShorelineObservationSummary,
  ShorelineObserveResult,
} from "@/lib/types";

function WaterlinePreview({ detail }: { detail: ShorelineObservationDetail }) {
  const segments = useMemo(
    () => observationSegments(detail).map((segment) => decimateSegment(segment)),
    [detail],
  );
  const bounds = useMemo(() => waterlineBounds(segments), [segments]);

  if (segments.length === 0) {
    return (
      <p role="status" className="mt-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-6 text-center text-sm text-slate-500">
        No shoreline detected for this observation.
      </p>
    );
  }

  return (
    <>
      <svg
        viewBox={`0 0 ${bounds.width} ${bounds.height}`}
        preserveAspectRatio="xMidYMid meet"
        className="mt-2 h-44 w-full rounded-md border border-slate-200 bg-slate-900"
        role="img"
        aria-label={`Extracted waterline: one continuous ${detail.shoreline_axis ?? "image"} line`}
      >
        {segments.map((segment, index) => (
          <polyline
            key={index}
            points={polylinePoints(segment)}
            fill="none"
            stroke="#22d3ee"
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-500">
        <span>Axis: {detail.shoreline_axis ?? "unknown"}</span>
        <span>
          Measured coverage:{" "}
          {detail.measured_coverage == null
            ? "unknown"
            : `${Math.round(detail.measured_coverage * 100)}%`}
        </span>
        <span>Bridged spans: {detail.interpolated_ranges?.length ?? 0}</span>
      </div>
    </>
  );
}

export function ShorelineView() {
  const [status, setStatus] = useState<ShorelineModuleStatus | null>(null);
  const [baselineStatus, setBaselineStatus] = useState<ShorelineBaselineStatus | null>(null);
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
      const [moduleStatus, cameraList, baselines] = await Promise.all([
        shorelinesApi.status(),
        camerasApi.list(),
        shorelinesApi.baselines().catch(() => null),
      ]);
      setStatus(moduleStatus);
      setCameras(cameraList);
      setBaselineStatus(baselines);
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
            <div className={status.segmentation_model_available ? "text-emerald-800" : "text-amber-800"}>
              <p>
                {status.segmentation_model_available
                  ? `${status.architecture ?? "Segmentation"} ${status.encoder ?? ""} artifact detected.`.trim()
                  : "No segmentation model is provisioned yet — waterlines require operator-labelled masks or a trained model (see docs/MODEL_DOCUMENTATION.md)."}
              </p>
              {status.segmentation_model_available && (
                <p className="mt-1 text-xs">
                  Classes: {(status.class_names ?? []).map((name, index) => `${index}=${name}`).join(" · ")}
                  {status.water_classes ? ` · binary water = Ocean only (${status.water_classes.join(", ")})` : ""}
                  {status.checkpoint_sha256
                    ? ` · checkpoint ${status.checkpoint_sha256.slice(0, 12)}…`
                    : ""}
                </p>
              )}
            </div>
          </div>
        </Card>
      )}

      <Card className="border-amber-200 bg-amber-50 text-sm text-amber-800">
        <div className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <div>
            <p>Experimental baseline — insufficient recording history for validated anomaly detection.</p>
            <p className="mt-1 text-xs">
              {baselineStatus?.available
                ? "A provisional tide-conditioned baseline exists for this camera/station."
                : `Tide-conditioned baseline unavailable${baselineStatus?.reason ? `: ${baselineStatus.reason}` : "."}`}
            </p>
          </div>
        </div>
      </Card>

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

      <LiveShorelineVisualizer
        cameras={cameras}
        selectedCameraId={selectedCameraId}
        onCameraChange={setSelectedCameraId}
        modelAvailable={Boolean(status?.segmentation_model_available)}
      />

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
              {latest.recording_id && (
                <p className="mt-1 text-xs text-slate-500">
                  Recording {latest.recording_id}
                  {latest.frame_index !== null && latest.frame_index !== undefined
                    ? ` · frame ${latest.frame_index}`
                    : ""}
                  {latest.signed_position_px !== null && latest.signed_position_px !== undefined
                    ? ` · signed position ${latest.signed_position_px.toFixed(1)} px (${latest.position_axis ?? "y"}, + = seaward)`
                    : ""}
                  {latest.extraction_status ? ` · ${latest.extraction_status}` : ""}
                  {latest.extraction_reason ? ` (${latest.extraction_reason})` : ""}
                </p>
              )}
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
          <li>
            Model weights are never committed. The active artifact is SegFormer MiT-B2 under
            <code> MODEL_PATH/shoreline-segmentation/</code>; observation cycles and the live
            preview use it. Without it, masks come from operators.
          </li>
          <li>
            The model predicts Beach 0 / Ocean 1 / Sky 2. The binary water mask is Ocean only; the
            shoreline is the shared Ocean/Beach boundary, never the Sky/Ocean horizon.
          </li>
          <li>
            If a frame has no Ocean/Beach contact (for example an unlabelled swash zone), the result
            is reported as no shoreline rather than a fabricated line.
          </li>
          <li>Calibrated positions require an active calibration profile with a homography and a seaward axis; otherwise positions stay uncalibrated (never defaulted to 0).</li>
          <li>Phase 5/6 prototype: observed positions are stored in pixels with a seaward-positive sign. A tide-conditioned baseline is fitted only when verified, geographically appropriate tide data supports it; otherwise analysis is explicitly movement-only and never reports NORMAL.</li>
          {status?.limitations?.length ? (
            <li>
              <span className="font-medium">Known limitations:</span> {status.limitations.join(" ")}
            </li>
          ) : null}
        </ul>
      </Card>
    </div>
  );
}
