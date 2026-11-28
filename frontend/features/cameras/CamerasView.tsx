"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Camera,
  CheckCircle2,
  Loader2,
  Pencil,
  PlugZap,
  RefreshCw,
  Trash2,
  Upload,
  XCircle,
} from "lucide-react";
import clsx from "clsx";

import { Card } from "@/components/Card";
import { PageHeader } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { camerasApi, cameraFrameUrl } from "@/lib/cameras";
import { stationsApi } from "@/lib/stations";
import { CalibrationMaskPlotter } from "./CalibrationMaskPlotter";
import type {
  Camera as CameraRow,
  CameraCalibration,
  CameraDevice,
  CameraStatus,
  CameraTestConnectionResult,
  CameraType,
  MonitoringStation,
} from "@/lib/types";

const CAMERA_TYPES: CameraType[] = ["rtsp", "usb", "upload", "video", "youtube_live"];

const BLANK_CAMERA_FORM = {
  code: "",
  name: "",
  monitoring_station_id: "",
  camera_type: "rtsp" as CameraType,
  rtsp_url: "",
  device_index: "",
  video_path: "",
  youtube_url: "",
  stream_quality: "",
  stream_format: "",
  resolution_width: "",
  resolution_height: "",
  fps: "",
  latitude: "",
  longitude: "",
  orientation_deg: "",
  enabled: true,
};


export function CamerasView() {
  const [stations, setStations] = useState<MonitoringStation[]>([]);
  const [cameras, setCameras] = useState<CameraRow[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [status, setStatus] = useState<CameraStatus | null>(null);
  const [testResult, setTestResult] = useState<CameraTestConnectionResult | null>(null);
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [deviceWarnings, setDeviceWarnings] = useState<string[]>([]);
  const [detecting, setDetecting] = useState(false);
  const [calibrations, setCalibrations] = useState<CameraCalibration[]>([]);
  const [frameStamp, setFrameStamp] = useState(0);
  const [frameError, setFrameError] = useState<string | null>(null);
  const [readyCameraId, setReadyCameraId] = useState<number | null>(null);
  const [plotterDirty, setPlotterDirty] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [editingCameraId, setEditingCameraId] = useState<number | null>(null);

  const [form, setForm] = useState({
    code: "",
    name: "",
    monitoring_station_id: "",
    camera_type: "rtsp" as CameraType,
    rtsp_url: "",
    device_index: "",
    video_path: "",
    youtube_url: "",
    stream_quality: "",
    stream_format: "",
    resolution_width: "",
    resolution_height: "",
    fps: "",
    latitude: "",
    longitude: "",
    orientation_deg: "",
    enabled: true,
  });

  const [calibrationForm, setCalibrationForm] = useState({
    meters_per_pixel: "",
    reference_points: "",
    homography_matrix: "",
    notes: "",
  });

  const selected = useMemo(
    () => cameras.find((c) => c.id === selectedId) ?? null,
    [cameras, selectedId],
  );

  const loadAll = useCallback(async () => {
    try {
      const [stationList, cameraList] = await Promise.all([
        stationsApi.listMonitoringStations(),
        camerasApi.list(),
      ]);
      setStations(stationList);
      setCameras(cameraList);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load cameras");
    }
  }, []);

  const loadSelected = useCallback(async (cameraId: number) => {
    const [statusResult, calibrationResult] = await Promise.allSettled([
      camerasApi.status(cameraId),
      camerasApi.listCalibrations(cameraId),
    ]);
    setStatus(statusResult.status === "fulfilled" ? statusResult.value : null);
    setCalibrations(calibrationResult.status === "fulfilled" ? calibrationResult.value : []);
    setTestResult(null);
    setFrameError(null);
    setFrameStamp(Date.now());
    setReadyCameraId(cameraId);
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  useEffect(() => {
    if (!selectedId) {
      setReadyCameraId(null);
      return;
    }
    setReadyCameraId(null);
    void loadSelected(selectedId);
  }, [selectedId, loadSelected]);

  const report = (message: string) => {
    setNotice(message);
    setTimeout(() => setNotice(null), 4000);
  };

  const resetForm = () => {
    setForm({ ...BLANK_CAMERA_FORM });
    setEditingCameraId(null);
  };

  const detectDevices = async () => {
    setDetecting(true);
    setError(null);
    setDeviceWarnings([]);
    try {
      const result = await camerasApi.detectDevices(true);
      setDevices(result.devices);
      setDeviceWarnings(result.warnings);
      if (result.devices.length === 0 && result.warnings.length === 0) {
        report("No cameras detected on the backend host.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Camera detection failed");
    } finally {
      setDetecting(false);
    }
  };

  const applyDetectedDevice = (device: CameraDevice) => {
    if (!device.available) {
      setError(`${device.name} is present but could not be opened. Close other apps using it and retry.`);
      return;
    }
    const next = { ...form };
    if (device.camera_type === "usb" && device.device_index !== null) {
      next.camera_type = "usb";
      next.device_index = String(device.device_index);
      next.rtsp_url = "";
    } else if (device.stream_url) {
      // Host-agent cameras are MJPEG streams read through the network provider.
      next.camera_type = "rtsp";
      next.rtsp_url = device.stream_url;
      next.device_index = "";
    } else {
      setError(`No usable source was reported for ${device.name}.`);
      return;
    }
    if (!next.name.trim()) next.name = device.name;
    if (!next.code.trim()) {
      const slug = device.name
        .toUpperCase()
        .replace(/[^A-Z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 56);
      next.code = `CAM-${slug || "USB"}`;
    }
    setForm(next);
    report(`Using ${device.name} (${device.source === "host_agent" ? "host agent" : "local"}).`);
  };

  const beginEdit = (camera: CameraRow) => {
    setForm({
      code: camera.code,
      name: camera.name,
      monitoring_station_id: String(camera.monitoring_station_id),
      camera_type: camera.camera_type,
      rtsp_url: "",
      device_index: camera.device_index === null ? "" : String(camera.device_index),
      video_path: camera.video_path ?? "",
      youtube_url: "",
      stream_quality: camera.stream_quality ?? "",
      stream_format: camera.stream_format ?? "",
      resolution_width: camera.resolution_width === null ? "" : String(camera.resolution_width),
      resolution_height: camera.resolution_height === null ? "" : String(camera.resolution_height),
      fps: camera.fps === null ? "" : String(camera.fps),
      latitude: camera.latitude === null ? "" : String(camera.latitude),
      longitude: camera.longitude === null ? "" : String(camera.longitude),
      orientation_deg: camera.orientation_deg === null ? "" : String(camera.orientation_deg),
      enabled: camera.enabled,
    });
    setEditingCameraId(camera.id);
    setError(null);
    if (typeof window !== "undefined") window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const removeCamera = async (camera: CameraRow) => {
    if (
      !window.confirm(
        `Remove camera ${camera.code}? This deletes its calibration profiles, stored shoreline observations and linked live sessions. Anomaly events are kept with the camera link cleared.`,
      )
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await camerasApi.remove(camera.id);
      if (selectedId === camera.id) setSelectedId(null);
      if (editingCameraId === camera.id) resetForm();
      report(`Camera ${camera.code} removed.`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to remove camera");
    } finally {
      setBusy(false);
    }
  };

  const saveCamera = async () => {
    setBusy(true);
    setError(null);
    try {
      const stationId = Number(form.monitoring_station_id);
      const num = (v: string) => (v.trim() === "" ? null : Number(v));
      if (!form.code.trim() || !form.name.trim() || Number.isNaN(stationId) || stationId <= 0) {
        throw new Error("Code, name and a monitoring station are required.");
      }
      const deviceIndex = num(form.device_index);
      if (form.camera_type === "rtsp" && !editingCameraId && !form.rtsp_url.trim()) {
        throw new Error("A stream URL is required for network cameras.");
      }
      if (form.camera_type === "usb" && (deviceIndex === null || Number.isNaN(deviceIndex))) {
        throw new Error("A device index is required for USB cameras.");
      }
      if (form.camera_type === "video" && !form.video_path.trim()) {
        throw new Error("A video file path is required for video cameras (e.g. clip.mp4).");
      }
      if (form.camera_type === "youtube_live" && !editingCameraId && !form.youtube_url.trim()) {
        throw new Error("A public https:// YouTube Live URL is required.");
      }

      const payload = {
        code: form.code.trim(),
        name: form.name.trim(),
        monitoring_station_id: stationId,
        camera_type: form.camera_type,
        rtsp_url: null as string | null,
        device_index: form.camera_type === "usb" ? deviceIndex : null,
        video_path: form.camera_type === "video" ? form.video_path.trim() : null,
        youtube_url: null as string | null,
        stream_quality:
          form.camera_type === "youtube_live" && form.stream_quality.trim()
            ? form.stream_quality.trim()
            : null,
        stream_format:
          form.camera_type === "youtube_live" && form.stream_format.trim()
            ? form.stream_format.trim()
            : null,
        resolution_width: num(form.resolution_width),
        resolution_height: num(form.resolution_height),
        fps: num(form.fps),
        latitude: num(form.latitude),
        longitude: num(form.longitude),
        orientation_deg: num(form.orientation_deg),
        enabled: form.enabled,
      };
      // Secrets are never returned by the API. Omit blank inputs so the stored
      // value survives an edit of unrelated fields.
      if (form.camera_type === "rtsp" && form.rtsp_url.trim()) {
        payload.rtsp_url = form.rtsp_url.trim();
      }
      if (form.camera_type === "youtube_live" && form.youtube_url.trim()) {
        payload.youtube_url = form.youtube_url.trim();
      }

      if (editingCameraId) {
        const updatePayload: Parameters<typeof camerasApi.update>[1] = { ...payload };
        // Omitted secrets must survive an edit; sending null would clear them.
        if (!form.rtsp_url.trim()) delete updatePayload.rtsp_url;
        if (!form.youtube_url.trim()) delete updatePayload.youtube_url;
        await camerasApi.update(editingCameraId, updatePayload);
        report(`Camera ${form.code.trim()} updated.`);
      } else {
        await camerasApi.create(payload);
        report("Camera registered.");
      }
      resetForm();
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save camera");
    } finally {
      setBusy(false);
    }
  };

  const runTestConnection = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      setTestResult(await camerasApi.testConnection(selected.id));
      await loadSelected(selected.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Test connection failed");
    } finally {
      setBusy(false);
    }
  };

  const refreshFrame = () => {
    setFrameError(null);
    setFrameStamp(Date.now());
  };

  const startStream = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      await camerasApi.startStream(selected.id);
      report("Live capture started.");
      await loadSelected(selected.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start live capture");
    } finally {
      setBusy(false);
    }
  };

  const stopStream = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      await camerasApi.stopStream(selected.id);
      report("Live capture stopped.");
      await loadSelected(selected.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not stop live capture");
    } finally {
      setBusy(false);
    }
  };

  const uploadFrame = async (file: File) => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      await camerasApi.uploadFrame(selected.id, file);
      report("Frame uploaded.");
      refreshFrame();
      await loadSelected(selected.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Frame upload failed");
    } finally {
      setBusy(false);
    }
  };

  const uploadVideo = async (file: File) => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      await camerasApi.uploadVideo(selected.id, file);
      report("Video uploaded and linked to the camera.");
      setFrameStamp(Date.now());
      await loadSelected(selected.id);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Video upload failed");
    } finally {
      setBusy(false);
    }
  };

  const createCalibration = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      let referencePoints: Record<string, unknown> | null = null;
      if (calibrationForm.reference_points.trim()) {
        referencePoints = JSON.parse(calibrationForm.reference_points) as Record<string, unknown>;
      }
      let homography: unknown[][] | null = null;
      if (calibrationForm.homography_matrix.trim()) {
        homography = JSON.parse(calibrationForm.homography_matrix) as unknown[][];
      }
      const metersPerPixel =
        calibrationForm.meters_per_pixel.trim() === ""
          ? null
          : Number(calibrationForm.meters_per_pixel);
      if (metersPerPixel !== null && Number.isNaN(metersPerPixel)) {
        throw new Error("meters_per_pixel must be a number");
      }
      await camerasApi.createCalibration(selected.id, {
        reference_points: referencePoints,
        homography_matrix: homography,
        meters_per_pixel: metersPerPixel,
        notes: calibrationForm.notes.trim() || null,
      });
      setCalibrationForm({ meters_per_pixel: "", reference_points: "", homography_matrix: "", notes: "" });
      report("Calibration profile stored.");
      await loadSelected(selected.id);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Calibration failed");
    } finally {
      setBusy(false);
    }
  };

  const input = (
    label: string,
    value: string,
    onChange: (v: string) => void,
    opts: { placeholder?: string; type?: string } = {},
  ) => (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-500">{label}</span>
      <input
        className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
        placeholder={opts.placeholder}
        type={opts.type ?? "text"}
        step={opts.type === "number" ? "any" : undefined}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );

  const stationOptions = stations.map((s) => (
    <option key={s.id} value={s.id}>
      {s.code} · {s.name}
    </option>
  ));

  const typeBadge = (t: CameraType) =>
    t === "rtsp"
      ? "RTSP / MJPEG"
      : t === "usb"
        ? "USB"
        : t === "video"
          ? "Video"
          : t === "youtube_live"
            ? "YouTube Live"
            : "Upload";

  return (
    <div className="space-y-6">
      <PageHeader
        title="Cameras"
        description="RTSP and USB camera acquisition, upload stills, health monitoring and calibration profiles."
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
        {/* Camera list */}
        <Card title="Cameras" subtitle="Click a camera to inspect health, frame and calibration">
          {cameras.length === 0 ? (
            <p className="text-sm text-slate-400">No cameras yet — register one on the right.</p>
          ) : (
            <ul className="divide-y divide-slate-100">
              {cameras.map((camera) => (
                <li key={camera.id}>
                  <button
                    onClick={() => {
                      if (
                        selectedId !== camera.id &&
                        plotterDirty &&
                        !window.confirm(
                          "Discard the unsaved monitoring outline changes for this camera?",
                        )
                      ) {
                        return;
                      }
                      setSelectedId(camera.id);
                    }}
                    className={clsx(
                      "flex w-full items-center justify-between gap-3 rounded-md px-2 py-3 text-left transition-colors hover:bg-slate-50",
                      selectedId === camera.id && "bg-reef/5 ring-1 ring-reef",
                    )}
                  >
                    <span>
                      <span className="block text-sm font-medium text-slate-800">
                        <Camera className="mr-1 inline h-4 w-4 text-reef" aria-hidden />
                        {camera.code} · {camera.name}
                      </span>
                      <span className="mt-0.5 block pl-5 text-xs text-slate-500">
                        {camera.station_code ?? `station #${camera.monitoring_station_id}`} ·{" "}
                        {typeBadge(camera.camera_type)}
                        {camera.camera_type === "rtsp" && camera.rtsp_url_redacted
                          ? ` · ${camera.rtsp_url_redacted}`
                          : ""}
                        {camera.camera_type === "usb" ? ` · device ${camera.device_index}` : ""}
                        {camera.camera_type === "video" ? ` · ${camera.video_path ?? "no video linked"}` : ""}
                        {camera.camera_type === "youtube_live"
                          ? ` · ${camera.youtube_url_redacted ?? "no YouTube URL"}`
                          : ""}
                      </span>
                    </span>
                    <StatusBadge tone={camera.enabled ? "ok" : "neutral"}>
                      {camera.enabled ? "enabled" : "disabled"}
                    </StatusBadge>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>

        {/* Create or edit camera */}
        <Card
          title={editingCameraId ? `Edit camera #${editingCameraId}` : "Register camera"}
          subtitle={
            editingCameraId
              ? "Leave RTSP/YouTube fields blank to keep the stored secret"
              : "Each camera keeps its own transport, resolution and calibration"
          }
        >
          <div className="grid grid-cols-2 gap-3">
            {input("Code", form.code, (v) => setForm({ ...form, code: v }), { placeholder: "e.g. CAM-MNL-001" })}
            {input("Name", form.name, (v) => setForm({ ...form, name: v }))}
            <label className="block col-span-2">
              <span className="mb-1 block text-xs font-medium text-slate-500">Monitoring station</span>
              <select
                className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                value={form.monitoring_station_id}
                onChange={(e) => setForm({ ...form, monitoring_station_id: e.target.value })}
              >
                <option value="">— select a station —</option>
                {stationOptions}
              </select>
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">Transport</span>
              <select
                className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                value={form.camera_type}
                onChange={(e) => setForm({ ...form, camera_type: e.target.value as CameraType })}
              >
                {CAMERA_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {typeBadge(t)}
                  </option>
                ))}
              </select>
            </label>
            <div className="col-span-2 rounded-md border border-slate-200 bg-slate-50 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="text-xs font-medium text-slate-600">Detected cameras</p>
                  <p className="text-xs text-slate-400">
                    Local USB devices and, when configured, webcams exposed by the Windows host agent.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void detectDevices()}
                  disabled={detecting}
                  className="inline-flex items-center gap-2 rounded-md border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-40"
                >
                  {detecting ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5" aria-hidden />
                  )}
                  {detecting ? "Detecting…" : "Detect cameras"}
                </button>
              </div>
              {devices.length > 0 ? (
                <ul className="mt-2 space-y-1">
                  {devices.map((device) => (
                    <li key={`${device.source}-${device.backend}-${device.index ?? device.name}`}>
                      <button
                        type="button"
                        onClick={() => applyDetectedDevice(device)}
                        disabled={!device.available}
                        className="flex w-full items-center justify-between gap-3 rounded border border-slate-200 bg-white px-3 py-2 text-left text-xs hover:border-reef hover:bg-reef/5 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        <span>
                          <span className="block font-medium text-slate-800">
                            {device.name}
                            {device.is_virtual ? (
                              <span className="ml-2 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-normal text-slate-500">
                                virtual
                              </span>
                            ) : null}
                          </span>
                          <span className="block text-slate-400">
                            {device.source === "host_agent" ? "host agent" : "local"}
                            {device.index !== null ? ` · index ${device.index}` : ""}
                            {device.width && device.height ? ` · ${device.width}×${device.height}` : ""}
                            {device.fps ? ` · ${device.fps.toFixed(0)} fps` : ""}
                            {device.available ? "" : " · unavailable"}
                          </span>
                        </span>
                        <span className="shrink-0 text-reef">
                          {device.camera_type === "usb" ? "Use USB" : "Use stream"}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-xs text-slate-400">
                  Click Detect cameras. In Docker Desktop on Windows, start the host agent first so
                  webcams are visible inside the container.
                </p>
              )}
              {deviceWarnings.map((warning) => (
                <p key={warning} className="mt-2 text-xs text-amber-700">
                  {warning}
                </p>
              ))}
            </div>
            {form.camera_type === "rtsp" &&
              input("Stream URL (RTSP or HTTP MJPEG)", form.rtsp_url, (v) => setForm({ ...form, rtsp_url: v }), {
                placeholder: editingCameraId
                  ? "leave blank to keep the stored stream URL"
                  : "rtsp://user:pass@host/stream or http://host:8765/stream/0.mjpg",
              })}
            {form.camera_type === "usb" &&
              input("Device index", form.device_index, (v) => setForm({ ...form, device_index: v }), {
                type: "number",
                placeholder: "0",
              })}
            {form.camera_type === "video" && (
              <div className="col-span-2">
                {input(
                  "Video file (relative to DATA_DIR/videos/)",
                  form.video_path,
                  (v) => setForm({ ...form, video_path: v }),
                  { placeholder: "coastline_day1.mp4" },
                )}
                <p className="mt-1 text-xs text-slate-400">
                  Or create the camera, then upload the file below — the upload replaces this path.
                </p>
              </div>
            )}
            {form.camera_type === "youtube_live" && (
              <div className="col-span-2 space-y-2">
                {input(
                  "YouTube Live URL (public/authorized https only)",
                  form.youtube_url,
                  (v) => setForm({ ...form, youtube_url: v }),
                  { placeholder: editingCameraId
                    ? "leave blank to keep the stored YouTube URL"
                    : "https://www.youtube.com/watch?v=… or https://youtu.be/…" },
                )}
                <div className="grid grid-cols-2 gap-3">
                  {input(
                    "Preferred quality",
                    form.stream_quality,
                    (v) => setForm({ ...form, stream_quality: v }),
                    { placeholder: "best or 720p" },
                  )}
                  {input(
                    "Preferred format (optional)",
                    form.stream_format,
                    (v) => setForm({ ...form, stream_format: v }),
                    { placeholder: "e.g. 96 or mp4" },
                  )}
                </div>
                <p className="text-xs text-slate-400">
                  Resolution uses yt-dlp. Only public/authorized streams; comply with YouTube&apos;s
                  terms and your deployment&apos;s legal review. The API returns a redacted URL.
                </p>
              </div>
            )}
            {form.camera_type === "upload" && (
              <p className="col-span-1 self-end text-xs text-slate-500">
                Stills are uploaded from this page once the camera exists.
              </p>
            )}
            {input("Resolution width", form.resolution_width, (v) => setForm({ ...form, resolution_width: v }), { type: "number", placeholder: "1920" })}
            {input("Resolution height", form.resolution_height, (v) => setForm({ ...form, resolution_height: v }), { type: "number", placeholder: "1080" })}
            {input("FPS", form.fps, (v) => setForm({ ...form, fps: v }), { type: "number", placeholder: "15" })}
            {input("Orientation (deg)", form.orientation_deg, (v) => setForm({ ...form, orientation_deg: v }), { type: "number", placeholder: "90" })}
            {input("Latitude", form.latitude, (v) => setForm({ ...form, latitude: v }), { type: "number", placeholder: "14.5995" })}
            {input("Longitude", form.longitude, (v) => setForm({ ...form, longitude: v }), { type: "number", placeholder: "120.9842" })}
            <label className="col-span-2 flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-slate-300"
                checked={form.enabled}
                onChange={(event) => setForm({ ...form, enabled: event.target.checked })}
              />
              Camera enabled (disabled cameras reject capture and preview)
            </label>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={() => void saveCamera()}
              disabled={busy}
              className="inline-flex items-center gap-2 rounded-md bg-deep px-4 py-2 text-sm font-medium text-white hover:bg-lagoon disabled:opacity-40"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              {editingCameraId ? "Save camera changes" : "Register camera"}
            </button>
            {editingCameraId && (
              <button
                onClick={resetForm}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-40"
              >
                Cancel edit
              </button>
            )}
          </div>
        </Card>
      </div>

      {/* Selected camera detail */}
      {selected && (
        <div className="grid gap-6 xl:grid-cols-2">
          <Card
            title={`${selected.code} — health`}
            subtitle="Live status reflects only this browser's backend process"
          >
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <button
                onClick={() => beginEdit(selected)}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-md border border-reef px-3 py-2 text-sm font-medium text-reef hover:bg-reef hover:text-white disabled:opacity-40"
              >
                <Pencil className="h-4 w-4" aria-hidden />
                Edit camera
              </button>
              <button
                onClick={() => void removeCamera(selected)}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-md border border-red-200 px-3 py-2 text-sm text-red-600 hover:bg-red-50 disabled:opacity-40"
              >
                <Trash2 className="h-4 w-4" aria-hidden />
                Remove camera
              </button>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <StatusBadge tone={status?.healthy ? "ok" : status?.connected ? "warn" : "neutral"}>
                {status?.healthy ? "healthy" : status?.connected ? "connected" : "not connected"}
              </StatusBadge>
              {status && (
                <span className="text-slate-500">
                  {status.message} · {status.frames_captured} frame(s) captured
                  {status.last_frame_at ? ` · last ${new Date(status.last_frame_at).toLocaleString()}` : ""}
                </span>
              )}
            </div>
            {status && (
              <div className="mt-2 space-y-1 text-xs text-slate-500">
                {selected.camera_type === "youtube_live" && (
                  <p>
                    Source: {selected.youtube_url_redacted ?? "—"} · live:{" "}
                    {status.is_live === null || status.is_live === undefined
                      ? "unknown"
                      : status.is_live
                        ? "yes"
                        : "no"}{" "}
                    · format: {status.resolved_format ?? "—"}
                  </p>
                )}
                <p>
                  Live capture: {status.stream_state ?? "stopped"}
                  {status.stream_session_id ? ` · session ${status.stream_session_id}` : ""}
                  {typeof status.frames_persisted === "number"
                    ? ` · ${status.frames_persisted} persisted`
                    : ""}
                  {typeof status.reconnect_count === "number" && status.reconnect_count > 0
                    ? ` · ${status.reconnect_count} reconnect(s)`
                    : ""}
                </p>
                {status.last_rolling_classification && (
                  <p>Rolling verdict: {status.last_rolling_classification}</p>
                )}
                {status.last_error && <p className="text-amber-700">Last error: {status.last_error}</p>}
              </div>
            )}
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                onClick={() => void runTestConnection()}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-md border border-reef px-3 py-2 text-sm font-medium text-reef hover:bg-reef hover:text-white disabled:opacity-40"
              >
                <PlugZap className="h-4 w-4" aria-hidden />
                Test connection
              </button>
              {["rtsp", "usb", "video", "youtube_live"].includes(selected.camera_type) && (
                <>
                  <button
                    onClick={() => void startStream()}
                    disabled={busy}
                    className="inline-flex items-center gap-2 rounded-md border border-emerald-600 px-3 py-2 text-sm font-medium text-emerald-700 hover:bg-emerald-600 hover:text-white disabled:opacity-40"
                  >
                    Start live capture
                  </button>
                  <button
                    onClick={() => void stopStream()}
                    disabled={busy}
                    className="inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-600 hover:bg-slate-100 disabled:opacity-40"
                  >
                    Stop
                  </button>
                </>
              )}
              <button
                onClick={() => void loadSelected(selected.id)}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-40"
              >
                <RefreshCw className="h-4 w-4" aria-hidden />
                Refresh status
              </button>
            </div>
            {testResult && (
              <div className="mt-3 rounded-md border border-slate-100 bg-slate-50 p-3 text-xs text-slate-600">
                <p className="font-medium text-slate-800">
                  {testResult.ok ? "✓ Connection ok" : "✗ Connection failed"} — {testResult.message}
                </p>
                {testResult.latency_ms !== null && <p>latency: {testResult.latency_ms} ms</p>}
                {testResult.frame_width && (
                  <p>
                    frame: {testResult.frame_width}×{testResult.frame_height}
                    {testResult.captured_at ? ` · ${new Date(testResult.captured_at).toLocaleString()}` : ""}
                  </p>
                )}
                {testResult.quality && (
                  <p>
                    quality: {testResult.quality.score}
                    {testResult.quality.flags.length > 0 ? ` · flags: ${testResult.quality.flags.join(", ")}` : " · no flags"}
                  </p>
                )}
                {testResult.is_live !== null && testResult.is_live !== undefined && (
                  <p>
                    live: {testResult.is_live ? "yes" : "no"}
                    {testResult.resolved_format ? ` · format ${testResult.resolved_format}` : ""}
                    {testResult.resolved_quality ? ` · ${testResult.resolved_quality}` : ""}
                  </p>
                )}
                {testResult.source_url_redacted && <p>source: {testResult.source_url_redacted}</p>}
              </div>
            )}

            {/* Frame preview */}
            <div className="mt-4">
              <div className="mb-1 flex items-center justify-between">
                <h3 className="text-xs font-semibold text-slate-500">Latest frame</h3>
                <button
                  onClick={refreshFrame}
                  className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700"
                >
                  <RefreshCw className="h-3.5 w-3.5" aria-hidden /> refresh
                </button>
              </div>
              {frameError ? (
                <p className="text-xs text-amber-700">{frameError}</p>
              ) : (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  key={frameStamp}
                  src={`${cameraFrameUrl(selected.id)}?t=${frameStamp}`}
                  alt={`Latest frame from ${selected.code}`}
                  className="max-h-64 w-full rounded-md border border-slate-200 object-contain bg-slate-900"
                  onError={() => setFrameError("No frame available yet (upload one, or check the connection).")}
                />
              )}
              {selected.camera_type === "upload" && (
                <label className="mt-2 inline-flex cursor-pointer items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-xs text-slate-600 hover:bg-slate-50">
                  <Upload className="h-4 w-4" aria-hidden />
                  Upload still (JPEG/PNG)
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) void uploadFrame(file);
                      e.target.value = "";
                    }}
                  />
                </label>
              )}
              {selected.camera_type === "video" && (
                <label className="mt-2 inline-flex cursor-pointer items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-xs text-slate-600 hover:bg-slate-50">
                  <Upload className="h-4 w-4" aria-hidden />
                  Upload video (MP4/MOV/AVI/MKV/WebM)
                  <input
                    type="file"
                    accept=".mp4,.mov,.avi,.mkv,.webm"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) void uploadVideo(file);
                      e.target.value = "";
                    }}
                  />
                </label>
              )}
            </div>
          </Card>

          <Card
            title="Calibration profiles"
            subtitle="Per-camera, versioned — never shared between cameras"
          >
            {calibrations.length === 0 ? (
              <p className="text-sm text-slate-400">No calibration profiles yet.</p>
            ) : (
              <ul className="mb-3 divide-y divide-slate-100 text-sm">
                {calibrations.map((cal) => (
                  <li key={cal.id} className="flex items-center justify-between gap-3 py-2">
                    <span className="text-slate-700">
                      v{cal.version}
                      {cal.meters_per_pixel ? ` · ${cal.meters_per_pixel} m/px` : ""}
                      {cal.roi_polygon
                        ? ` · outline ${cal.roi_polygon.points.length} pts`
                        : " · no outline"}
                      {cal.notes ? (
                        <span className="block text-xs text-slate-400">{cal.notes}</span>
                      ) : null}
                    </span>
                    {cal.is_active && <StatusBadge tone="ok">active</StatusBadge>}
                  </li>
                ))}
              </ul>
            )}
            <div className="grid gap-2">
              {input(
                "Meters per pixel",
                calibrationForm.meters_per_pixel,
                (v) => setCalibrationForm({ ...calibrationForm, meters_per_pixel: v }),
                { type: "number", placeholder: "0.05" },
              )}
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-slate-500">
                  Reference points (JSON, optional)
                </span>
                <textarea
                  className="h-16 w-full rounded-md border border-slate-200 px-3 py-2 font-mono text-xs"
                  placeholder='{"points":[{"pixel":[120,80],"world_m":[0.0,0.0]}]}'
                  value={calibrationForm.reference_points}
                  onChange={(e) => setCalibrationForm({ ...calibrationForm, reference_points: e.target.value })}
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-slate-500">
                  Homography 3×3 (JSON, optional)
                </span>
                <textarea
                  className="h-16 w-full rounded-md border border-slate-200 px-3 py-2 font-mono text-xs"
                  placeholder="[[1,0,0],[0,1,0],[0,0,1]]"
                  value={calibrationForm.homography_matrix}
                  onChange={(e) => setCalibrationForm({ ...calibrationForm, homography_matrix: e.target.value })}
                />
              </label>
              {input("Notes", calibrationForm.notes, (v) => setCalibrationForm({ ...calibrationForm, notes: v }))}
              <button
                onClick={() => void createCalibration()}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-md bg-deep px-4 py-2 text-sm font-medium text-white hover:bg-lagoon disabled:opacity-40"
              >
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                Save calibration profile
              </button>
            </div>
          </Card>
        </div>
      )}

      {selected && readyCameraId === selected.id && (
        <CalibrationMaskPlotter
          key={selected.id}
          camera={selected}
          calibration={calibrations.find((cal) => cal.is_active) ?? null}
          onDirtyChange={setPlotterDirty}
          onSaved={async () => {
            await loadSelected(selected.id);
            await loadAll();
          }}
        />
      )}
    </div>
  );
}
