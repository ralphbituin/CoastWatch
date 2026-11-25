"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { FlaskConical, Loader2, Play, Trash2, XCircle } from "lucide-react";
import clsx from "clsx";

import { Card } from "@/components/Card";
import { PageHeader } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { TideAnomalyChart } from "@/features/anomalies/TideAnomalyChart";
import { camerasApi } from "@/lib/cameras";
import {
  DEFAULT_ROLLING,
  labApi,
  labRowToAnomalyEvent,
  type RollingOptions,
} from "@/lib/lab";
import { recordingsApi } from "@/lib/recordings";
import type {
  AnomalyEvent,
  Camera,
  LabRow,
  LabRun,
  VideoRecording,
} from "@/lib/types";

type Tab = "images" | "video" | "stream";

const DEPLOYMENT_OFFSET = "+07:00";

const OFFICIAL_WARNING_AUTHORITY =
  (process.env.NEXT_PUBLIC_OFFICIAL_WARNING_AUTHORITY ?? "").trim() ||
  "the official national tsunami warning authority";

function localInputToIso(value: string): string {
  if (!value) return "";
  return /[zZ]|[+-]\d{2}:\d{2}$/.test(value) ? value : `${value}:00${DEPLOYMENT_OFFSET}`;
}

function fmt(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

export function LabView() {
  const [tab, setTab] = useState<Tab>("images");
  const [run, setRun] = useState<LabRun | null>(null);
  const [rows, setRows] = useState<LabRow[]>([]);
  const [labEvents, setLabEvents] = useState<Array<Record<string, unknown>>>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [recordings, setRecordings] = useState<VideoRecording[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [timezone, setTimezone] = useState("Asia/Bangkok");
  const [rolling, setRolling] = useState<RollingOptions>(DEFAULT_ROLLING);

  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [manifestText, setManifestText] = useState("");
  const [imageStart, setImageStart] = useState("");
  const [imageInterval, setImageInterval] = useState("60");
  const [persist, setPersist] = useState(false);
  const [cameraId, setCameraId] = useState("");

  const [videoRecordingId, setVideoRecordingId] = useState("");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoStart, setVideoStart] = useState("");
  const [videoInterval, setVideoInterval] = useState("30");
  const [videoStartElapsed, setVideoStartElapsed] = useState("0");
  const [videoEndElapsed, setVideoEndElapsed] = useState("");
  const [videoMaxSamples, setVideoMaxSamples] = useState("12");
  const [clockAnchors, setClockAnchors] = useState("");

  const [streamCameraId, setStreamCameraId] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [streamInterval, setStreamInterval] = useState("30");
  const [streamMaxSamples, setStreamMaxSamples] = useState("12");
  const [streamDuration, setStreamDuration] = useState("300");

  const loadReferences = useCallback(async () => {
    const [cameraResult, recordingResult] = await Promise.allSettled([
      camerasApi.list(),
      recordingsApi.listVideos(),
    ]);
    if (cameraResult.status === "fulfilled") setCameras(cameraResult.value);
    if (recordingResult.status === "fulfilled") setRecordings(recordingResult.value);
  }, []);

  useEffect(() => {
    void loadReferences();
  }, [loadReferences]);

  const liveCameras = useMemo(
    () => cameras.filter((camera) => ["rtsp", "usb", "video", "youtube_live"].includes(camera.camera_type)),
    [cameras],
  );

  const poll = useCallback(async (runId: string) => {
    for (let attempt = 0; attempt < 400; attempt += 1) {
      const current = await labApi.getRun(runId);
      setRun(current);
      if (["completed", "failed", "cancelled"].includes(current.status)) {
        const [rowResult, eventResult] = await Promise.all([
          labApi.getRows(runId, 200),
          labApi.getEvents(runId),
        ]);
        setRows(rowResult.items);
        setLabEvents(eventResult.events);
        return current;
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    return null;
  }, []);

  const launch = async (factory: () => Promise<LabRun>) => {
    setBusy(true);
    setError(null);
    setRows([]);
    setLabEvents([]);
    try {
      const created = await factory();
      setRun(created);
      await poll(created.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Lab run failed");
    } finally {
      setBusy(false);
    }
  };

  const startImageRun = () =>
    launch(() =>
      labApi.createImageRun({
        files: imageFiles,
        manifestText: manifestText || undefined,
        startAt: manifestText ? undefined : localInputToIso(imageStart),
        intervalSeconds: manifestText ? undefined : Number(imageInterval),
        timezone,
        rolling,
        persist,
        cameraId: cameraId ? Number(cameraId) : null,
      }),
    );

  const startVideoRun = () =>
    launch(() =>
      labApi.createVideoRun({
        recordingId: videoRecordingId || undefined,
        file: videoFile,
        startAt: localInputToIso(videoStart) || undefined,
        clockAnchors: clockAnchors || undefined,
        sampleIntervalSeconds: Number(videoInterval),
        startElapsed: Number(videoStartElapsed || 0),
        endElapsed: videoEndElapsed ? Number(videoEndElapsed) : null,
        maxSamples: Number(videoMaxSamples),
        timezone,
        rolling,
        persist,
        cameraId: cameraId ? Number(cameraId) : null,
      }),
    );

  const startStreamRun = () =>
    launch(() =>
      labApi.createStreamRun({
        cameraId: streamCameraId ? Number(streamCameraId) : null,
        youtubeUrl: streamCameraId ? undefined : youtubeUrl,
        sampleIntervalSeconds: Number(streamInterval),
        maxSamples: Number(streamMaxSamples),
        maxDurationSeconds: Number(streamDuration),
        timezone,
        rolling,
        persist: persist && Boolean(streamCameraId),
      }),
    );

  const chartEvents = useMemo<AnomalyEvent[]>(
    () => rows.map(labRowToAnomalyEvent),
    [rows],
  );

  const rollingInput = (key: keyof RollingOptions, label: string) => (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-500">{label}</span>
      <input
        type="number"
        step="any"
        className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
        value={rolling[key]}
        onChange={(event) =>
          setRolling({ ...rolling, [key]: Number(event.target.value) })
        }
      />
    </label>
  );

  const input = (
    label: string,
    value: string,
    onChange: (value: string) => void,
    type = "text",
    placeholder?: string,
  ) => (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-500">{label}</span>
      <input
        className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
        type={type}
        step={type === "number" ? "any" : undefined}
        placeholder={placeholder}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Predictor & Anomaly Test Lab"
        description="Run the production SegFormer shoreline predictor and the rolling-baseline detector on still images, a video file, or a bounded live stream. Persistence is off by default."
      />

      <Card className="border-amber-200 bg-amber-50 text-xs text-amber-800">
        Experimental rolling-baseline drawdown detector — not validated for tsunami detection.
        Follow {OFFICIAL_WARNING_AUTHORITY}. No tide data is fabricated. Live YouTube rows use
        wall-clock receipt time (confidence ≤ 0.6), not a per-frame capture clock.
      </Card>

      {error && (
        <Card className="border-red-200 bg-red-50 text-sm text-red-700">
          <div className="flex items-start gap-2">
            <XCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            {error}
          </div>
        </Card>
      )}

      <div className="flex flex-wrap gap-2">
        {(["images", "video", "stream"] as Tab[]).map((item) => (
          <button
            key={item}
            onClick={() => setTab(item)}
            className={clsx(
              "rounded-md px-3 py-2 text-sm font-medium",
              tab === item ? "bg-deep text-white" : "border border-slate-200 text-slate-600",
            )}
          >
            {item === "images" ? "Batch images" : item === "video" ? "Video" : "Live stream / camera"}
          </button>
        ))}
      </div>

      <Card title="Detector configuration" subtitle="Time-based rolling median/MAD; no tide baseline">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {rollingInput("window_seconds", "Window (s)")}
          {rollingInput("k_seconds", "k (s)")}
          {rollingInput("z_threshold", "Robust z threshold")}
          {rollingInput("persistence", "Persistence samples")}
          {rollingInput("max_gap_seconds", "Max gap (s)")}
          {input("Deployment timezone", timezone, setTimezone)}
        </div>
        <label className="mt-3 flex items-center gap-2 text-sm text-slate-600">
          <input
            type="checkbox"
            checked={persist}
            onChange={(event) => setPersist(event.target.checked)}
          />
          Persist to the production tables (explicit opt-in; needs a registered camera)
        </label>
        <label className="mt-2 block">
          <span className="mb-1 block text-xs font-medium text-slate-500">
            Persistence camera (optional)
          </span>
          <select
            className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
            value={cameraId}
            onChange={(event) => setCameraId(event.target.value)}
          >
            <option value="">— none (lab only) —</option>
            {cameras.map((camera) => (
              <option key={camera.id} value={camera.id}>
                {camera.code} · {camera.name}
              </option>
            ))}
          </select>
        </label>
      </Card>

      {tab === "images" && (
        <Card title="Batch images" subtitle="Manifest or operator start time + interval is mandatory">
          <div
            className="rounded-md border-2 border-dashed border-slate-200 px-4 py-6 text-center text-sm text-slate-500"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              setImageFiles(Array.from(event.dataTransfer.files));
            }}
          >
            <input
              type="file"
              multiple
              accept=".jpg,.jpeg,.png"
              onChange={(event) => setImageFiles(Array.from(event.target.files ?? []))}
            />
            <p className="mt-2 text-xs text-slate-400">
              Drag/drop or choose images (max 50). Uploaded order is the timestamp order.
            </p>
            {imageFiles.length > 0 && (
              <ol className="mt-2 list-inside list-decimal text-left text-xs text-slate-600">
                {imageFiles.map((file) => (
                  <li key={`${file.name}-${file.size}`}>{file.name}</li>
                ))}
              </ol>
            )}
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <label className="block md:col-span-2">
              <span className="mb-1 block text-xs font-medium text-slate-500">
                Per-file timestamp manifest (JSON/CSV; optional if start+interval is used)
              </span>
              <textarea
                className="h-24 w-full rounded-md border border-slate-200 px-3 py-2 font-mono text-xs"
                placeholder='{"a.jpg": "2026-11-25T06:00:00+07:00"} or a.jpg,2026-11-25T06:00:00+07:00'
                value={manifestText}
                onChange={(event) => setManifestText(event.target.value)}
              />
            </label>
            {input("Operator start time (local)", imageStart, setImageStart, "datetime-local")}
            {input("Interval (s)", imageInterval, setImageInterval, "number")}
          </div>
          <button
            onClick={() => void startImageRun()}
            disabled={busy || imageFiles.length === 0}
            className="mt-3 inline-flex items-center gap-2 rounded-md bg-deep px-4 py-2 text-sm font-medium text-white hover:bg-lagoon disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run images
          </button>
        </Card>
      )}

      {tab === "video" && (
        <Card title="Video" subtitle="Registered recording or upload; timestamps require anchors or a start time">
          <div className="grid gap-3 md:grid-cols-2">
            <label className="block md:col-span-2">
              <span className="mb-1 block text-xs font-medium text-slate-500">
                Registered recording
              </span>
              <select
                className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                value={videoRecordingId}
                onChange={(event) => setVideoRecordingId(event.target.value)}
              >
                <option value="">— choose a recording —</option>
                {recordings.map((recording) => (
                  <option key={recording.recording_id} value={recording.recording_id}>
                    {recording.recording_id} · {recording.file_path.split(/[\\/]/).pop()}
                  </option>
                ))}
              </select>
            </label>
            <label className="block md:col-span-2">
              <span className="mb-1 block text-xs font-medium text-slate-500">
                …or upload a video file
              </span>
              <input
                type="file"
                accept=".mp4,.mov,.avi,.mkv,.webm"
                onChange={(event) => setVideoFile(event.target.files?.[0] ?? null)}
              />
            </label>
            {input("Operator start time (local)", videoStart, setVideoStart, "datetime-local")}
            {input("Sample interval (s)", videoInterval, setVideoInterval, "number")}
            {input("Start elapsed (s)", videoStartElapsed, setVideoStartElapsed, "number")}
            {input("End elapsed (s, optional)", videoEndElapsed, setVideoEndElapsed, "number")}
            {input("Max samples", videoMaxSamples, setVideoMaxSamples, "number")}
            <label className="block md:col-span-2">
              <span className="mb-1 block text-xs font-medium text-slate-500">
                Verified visible-clock anchors (JSON list, optional)
              </span>
              <textarea
                className="h-20 w-full rounded-md border border-slate-200 px-3 py-2 font-mono text-xs"
                placeholder='[{"elapsed_seconds": 0, "timestamp": "2026-11-25T06:00:00+07:00", "confidence": 0.9}]'
                value={clockAnchors}
                onChange={(event) => setClockAnchors(event.target.value)}
              />
            </label>
          </div>
          <button
            onClick={() => void startVideoRun()}
            disabled={busy || (!videoRecordingId && !videoFile)}
            className="mt-3 inline-flex items-center gap-2 rounded-md bg-deep px-4 py-2 text-sm font-medium text-white hover:bg-lagoon disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run video
          </button>
        </Card>
      )}

      {tab === "stream" && (
        <Card title="Live stream / camera" subtitle="Bounded run; no unbounded streaming or frame storage">
          <div className="grid gap-3 md:grid-cols-2">
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">
                Registered live camera
              </span>
              <select
                className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                value={streamCameraId}
                onChange={(event) => setStreamCameraId(event.target.value)}
              >
                <option value="">— none —</option>
                {liveCameras.map((camera) => (
                  <option key={camera.id} value={camera.id}>
                    {camera.code} · {camera.camera_type}
                  </option>
                ))}
              </select>
            </label>
            {input(
              "…or public YouTube Live URL",
              youtubeUrl,
              setYoutubeUrl,
              "url",
              "https://youtu.be/…",
            )}
            {input("Sample interval (s)", streamInterval, setStreamInterval, "number")}
            {input("Max samples", streamMaxSamples, setStreamMaxSamples, "number")}
            {input("Max duration (s)", streamDuration, setStreamDuration, "number")}
          </div>
          <p className="mt-2 text-xs text-slate-500">
            A lost stream is reported as DATA_UNRELIABLE; no row is fabricated. Persistence requires
            a registered camera and the explicit opt-in above.
          </p>
          <button
            onClick={() => void startStreamRun()}
            disabled={busy || (!streamCameraId && !youtubeUrl)}
            className="mt-3 inline-flex items-center gap-2 rounded-md bg-deep px-4 py-2 text-sm font-medium text-white hover:bg-lagoon disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run bounded stream
          </button>
        </Card>
      )}

      {run && (
        <Card title={`Run ${run.run_id}`} subtitle={`${run.kind} · ${run.status}`}>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <StatusBadge tone={run.status === "completed" ? "ok" : run.status === "failed" ? "warn" : "neutral"}>
              {run.status}
            </StatusBadge>
            <span className="text-slate-500">
              {run.row_count} row(s) · {run.event_count} event(s) · persistence {run.persist ? "on" : "off"}
            </span>
            {Object.entries(run.progress).slice(0, 6).map(([key, value]) => (
              <span key={key} className="rounded bg-slate-100 px-2 py-0.5 text-xs">
                {key}: {String(value)}
              </span>
            ))}
            <button
              onClick={() => {
                void labApi.deleteRun(run.run_id).then(() => {
                  setRun(null);
                  setRows([]);
                  setLabEvents([]);
                });
              }}
              className="ml-auto inline-flex items-center gap-1 text-xs text-slate-500 hover:text-red-600"
            >
              <Trash2 className="h-3.5 w-3.5" aria-hidden /> delete run/artifacts
            </button>
          </div>
          {run.errors.length > 0 && (
            <p className="mt-2 text-xs text-red-700">{run.errors.join("; ")}</p>
          )}
          {run.warnings.length > 0 && (
            <p className="mt-2 text-xs text-amber-700">{run.warnings.join("; ")}</p>
          )}
          {run.artifacts.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {run.artifacts.map((name) =>
                name.startsWith("overlay_") ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    key={name}
                    src={labApi.artifactUrl(run.run_id, name)}
                    alt={name}
                    className="h-32 rounded border border-slate-200 object-cover"
                  />
                ) : (
                  <a
                    key={name}
                    className="rounded border border-slate-200 px-2 py-1 text-xs text-reef hover:bg-slate-50"
                    href={labApi.artifactUrl(run.run_id, name)}
                  >
                    {name}
                  </a>
                ),
              )}
            </div>
          )}
        </Card>
      )}

      {rows.length > 0 && (
        <Card title="Per-timestamp rows" subtitle="UTC and deployment timezone; provenance on every row">
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead className="bg-slate-50 text-left text-slate-500">
                <tr>
                  <th className="px-2 py-1">#</th>
                  <th className="px-2 py-1">observed_at (UTC / local)</th>
                  <th className="px-2 py-1">source · conf</th>
                  <th className="px-2 py-1">shoreline px</th>
                  <th className="px-2 py-1">signed px</th>
                  <th className="px-2 py-1">baseline</th>
                  <th className="px-2 py-1">scale</th>
                  <th className="px-2 py-1">residual</th>
                  <th className="px-2 py-1">z</th>
                  <th className="px-2 py-1">Ck</th>
                  <th className="px-2 py-1">classification</th>
                  <th className="px-2 py-1">status / segments</th>
                  <th className="px-2 py-1">reason</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={`${row.frame_index ?? index}-${index}`} className="border-t border-slate-100">
                    <td className="px-2 py-1 text-slate-400">{row.frame_index ?? index}</td>
                    <td className="px-2 py-1">
                      <span className="block font-mono text-[11px]">{row.observed_at ?? "—"}</span>
                      <span className="block text-slate-400">{row.observed_at_local ?? "—"}</span>
                    </td>
                    <td className="px-2 py-1">
                      {row.timestamp_source ?? "—"} · {fmt(row.timestamp_confidence, 2)}
                    </td>
                    <td className="px-2 py-1">{fmt(row.shoreline_position_px)}</td>
                    <td className="px-2 py-1">{fmt(row.signed_position_px)}</td>
                    <td className="px-2 py-1">{fmt(row.rolling_baseline_px)}</td>
                    <td className="px-2 py-1">{fmt(row.rolling_scale_px)}</td>
                    <td className="px-2 py-1">{fmt(row.residual_px)}</td>
                    <td className="px-2 py-1">{fmt(row.robust_z, 2)}</td>
                    <td className="px-2 py-1">{fmt(row.cumulative_drawdown_px)}</td>
                    <td className="px-2 py-1 font-medium">{row.classification ?? "DATA_UNRELIABLE"}</td>
                    <td className="px-2 py-1">
                      {row.extraction_status ?? "—"} · {row.segment_count ?? 0} seg
                      {row.camera_motion_gate && row.camera_motion_gate !== "unavailable"
                        ? ` · motion ${row.camera_motion_gate}`
                        : ""}
                    </td>
                    <td className="px-2 py-1 text-slate-500">
                      {row.reason ?? row.extraction_reason ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {chartEvents.length > 0 && (
        <Card title="Rolling series" subtitle="Observed, rolling baseline, residual, robust z and cumulative drawdown">
          <TideAnomalyChart
            events={chartEvents}
            baselineStatus="rolling_unverified"
            baselineReason="Test-lab rolling baseline; no tide data is used."
          />
        </Card>
      )}

      {labEvents.length > 0 && (
        <Card title="Rolling events" subtitle="Drawdown candidates from this run (never labelled tsunami)">
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead className="bg-slate-50 text-left text-slate-500">
                <tr>
                  <th className="px-2 py-1">onset</th>
                  <th className="px-2 py-1">detection</th>
                  <th className="px-2 py-1">latency (s)</th>
                  <th className="px-2 py-1">persistence</th>
                  <th className="px-2 py-1">max classification</th>
                  <th className="px-2 py-1">peak drawdown (px)</th>
                  <th className="px-2 py-1">end reason</th>
                </tr>
              </thead>
              <tbody>
                {labEvents.map((event, index) => (
                  <tr key={index} className="border-t border-slate-100">
                    <td className="px-2 py-1">{String(event.onset_time ?? "—")}</td>
                    <td className="px-2 py-1">{String(event.detection_time ?? "—")}</td>
                    <td className="px-2 py-1">{fmt(Number(event.detection_latency_s), 1)}</td>
                    <td className="px-2 py-1">{String(event.persistence ?? "—")}</td>
                    <td className="px-2 py-1">{String(event.max_classification ?? "—")}</td>
                    <td className="px-2 py-1">{fmt(Number(event.peak_cumulative_drawdown_px))}</td>
                    <td className="px-2 py-1">{String(event.end_reason ?? "—")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <p className="flex items-center gap-2 text-xs text-slate-400">
        <FlaskConical className="h-3.5 w-3.5" aria-hidden />
        Lab runs are bounded and temporary. Delete a run to remove its artifacts immediately; finished
        runs also expire automatically.
      </p>
    </div>
  );
}
