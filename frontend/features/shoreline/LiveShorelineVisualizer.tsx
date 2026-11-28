"use client";

import { useEffect, useState } from "react";
import { Loader2, Pause, Play, Radio } from "lucide-react";

import { Card } from "@/components/Card";
import { API_V1 } from "@/lib/config";
import type { Camera } from "@/lib/types";

type Props = {
  cameras: Camera[];
  selectedCameraId: number | "";
  onCameraChange: (cameraId: number | "") => void;
  modelAvailable: boolean;
};

/** Missing/blank headers are unknown, never a fabricated zero. */
function numericHeader(headers: Headers, name: string): number | null {
  const raw = headers.get(name);
  if (raw === null || raw.trim() === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

export function LiveShorelineVisualizer({
  cameras,
  selectedCameraId,
  onCameraChange,
  modelAvailable,
}: Props) {
  const [running, setRunning] = useState(true);
  const [showMask, setShowMask] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [polyline, setPolyline] = useState<string | null>(null);
  const [frameWidth, setFrameWidth] = useState<number | null>(null);
  const [frameHeight, setFrameHeight] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [capturedAt, setCapturedAt] = useState<string | null>(null);
  const [architecture, setArchitecture] = useState<string | null>(null);
  const [shorelineAvailable, setShorelineAvailable] = useState<boolean | null>(null);
  const [segments, setSegments] = useState<number | null>(null);
  const [shorelineAxis, setShorelineAxis] = useState<string | null>(null);
  const [interpolatedRanges, setInterpolatedRanges] = useState<number | null>(null);
  const [measuredCoverage, setMeasuredCoverage] = useState<number | null>(null);
  const [shorelineReason, setShorelineReason] = useState<string | null>(null);
  const [waitingForSample, setWaitingForSample] = useState(false);
  const [stale, setStale] = useState(false);
  const [sampleAgeMs, setSampleAgeMs] = useState<number | null>(null);
  const camera = cameras.find((item) => item.id === selectedCameraId);
  const polylineSegments = polyline
    ? polyline
        .split(";")
        .map((segment) => segment.trim())
        .filter(Boolean)
    : [];

  useEffect(() => {
    setImageUrl(null);
    setPolyline(null);
    setFrameWidth(null);
    setFrameHeight(null);
    setError(null);
    setConfidence(null);
    setInferenceMs(null);
    setCapturedAt(null);
    setArchitecture(null);
    setShorelineAvailable(null);
    setSegments(null);
    setShorelineAxis(null);
    setInterpolatedRanges(null);
    setMeasuredCoverage(null);
    setShorelineReason(null);
    setWaitingForSample(false);
    setStale(false);
    setSampleAgeMs(null);
  }, [selectedCameraId]);

  useEffect(() => {
    if (!running || !modelAvailable || selectedCameraId === "") return;

    let active = true;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let currentUrl: string | null = null;
    const controller = new AbortController();
    // The worker samples once per interval; never poll faster than it produces.
    const FALLBACK_INTERVAL_MS = 10_000;

    const refresh = async () => {
      const startedAt = Date.now();
      let nextDelayMs = FALLBACK_INTERVAL_MS;
      try {
        const response = await fetch(
          `${API_V1}/cameras/${selectedCameraId}/shoreline-preview?mask=${showMask ? "1" : "0"}&ts=${Date.now()}`,
          { cache: "no-store", signal: controller.signal },
        );
        const intervalHeader = numericHeader(response.headers, "X-Preview-Sample-Interval-Ms");
        if (intervalHeader !== null && intervalHeader > 0) {
          nextDelayMs = intervalHeader;
        }

        if (response.status === 503) {
          // The worker has not produced a frame yet: keep the last image and
          // wait for the next sample instead of showing an error.
          let retryAfterMs = Math.min(nextDelayMs, 5000);
          const retryAfter = Number(response.headers.get("Retry-After"));
          if (Number.isFinite(retryAfter) && retryAfter > 0) {
            retryAfterMs = retryAfter * 1000;
          }
          if (!active) return;
          setWaitingForSample(true);
          setError(null);
          nextDelayMs = retryAfterMs;
          return;
        }
        if (!response.ok) {
          let message = `Preview failed (${response.status})`;
          try {
            const body = await response.json();
            message = body?.error?.message ?? message;
          } catch {
            // Keep the HTTP status when the response isn't the API error envelope.
          }
          throw new Error(message);
        }

        const blob = await response.blob();
        if (!active) return;
        const nextUrl = URL.createObjectURL(blob);
        if (currentUrl) URL.revokeObjectURL(currentUrl);
        currentUrl = nextUrl;
        setImageUrl(nextUrl);
        setError(null);
        setWaitingForSample(false);
        setStale(response.headers.get("X-Preview-Stale") === "1");
        setSampleAgeMs(numericHeader(response.headers, "X-Preview-Age-Ms"));
        setConfidence(numericHeader(response.headers, "X-Segmentation-Confidence"));
        setInferenceMs(numericHeader(response.headers, "X-Inference-Ms"));
        setCapturedAt(response.headers.get("X-Frame-Captured-At"));
        const available = response.headers.get("X-Shoreline-Available");
        setShorelineAvailable(available === null ? null : available === "1");
        setSegments(numericHeader(response.headers, "X-Shoreline-Segments"));
        setShorelineAxis(response.headers.get("X-Shoreline-Axis") || null);
        setInterpolatedRanges(
          numericHeader(response.headers, "X-Shoreline-Interpolated-Ranges"),
        );
        setMeasuredCoverage(
          numericHeader(response.headers, "X-Shoreline-Measured-Coverage"),
        );
        setShorelineReason(response.headers.get("X-Shoreline-Reason") || null);
        setPolyline(response.headers.get("X-Shoreline-Polyline") || null);
        setFrameWidth(numericHeader(response.headers, "X-Frame-Width"));
        setFrameHeight(numericHeader(response.headers, "X-Frame-Height"));
        setArchitecture(
          [response.headers.get("X-Model-Architecture"), response.headers.get("X-Model-Encoder")]
            .filter(Boolean)
            .join(" ") || null,
        );
      } catch (err) {
        if (active && !controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Could not load shoreline preview");
        }
      } finally {
        if (active) {
          // Align the next request to the producer's cadence, not to latency.
          const elapsed = Date.now() - startedAt;
          const delay = Math.max(1000, nextDelayMs - elapsed);
          timeout = setTimeout(() => void refresh(), delay);
        }
      }
    };

    void refresh();
    return () => {
      active = false;
      controller.abort();
      if (timeout) clearTimeout(timeout);
      if (currentUrl) URL.revokeObjectURL(currentUrl);
    };
  }, [modelAvailable, running, selectedCameraId, showMask]);

  return (
    <Card
      title="Live shoreline visualizer"
      subtitle="The model prediction is drawn directly over each camera frame."
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <label className="sr-only" htmlFor="shoreline-preview-camera">
          Camera
        </label>
        <select
          id="shoreline-preview-camera"
          className="min-w-56 rounded-md border border-slate-200 px-3 py-2 text-sm"
          value={selectedCameraId}
          onChange={(event) =>
            onCameraChange(event.target.value ? Number(event.target.value) : "")
          }
        >
          <option value="">Select a camera</option>
          {cameras.map((item) => (
            <option key={item.id} value={item.id}>
              {item.code} · {item.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => setRunning((value) => !value)}
          disabled={!modelAvailable || selectedCameraId === ""}
          className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-700 hover:bg-slate-50 disabled:opacity-40"
        >
          {running ? (
            <Pause className="h-4 w-4" aria-hidden />
          ) : (
            <Play className="h-4 w-4" aria-hidden />
          )}
          {running ? "Pause" : "Resume"}
        </button>
        <label className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-700">
          <input
            type="checkbox"
            className="h-4 w-4 accent-reef"
            checked={showMask}
            onChange={(event) => setShowMask(event.target.checked)}
          />
          Mask
        </label>
        {running && modelAvailable && selectedCameraId !== "" && (
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-700">
            <Radio className="h-3.5 w-3.5 animate-pulse" aria-hidden />
            Live
          </span>
        )}
        {imageUrl && stale && (
          <span className="inline-flex items-center gap-1.5 rounded bg-amber-50 px-2 py-1 text-xs font-medium text-amber-700">
            Stale sample
          </span>
        )}
      </div>

      <div className="relative flex min-h-56 items-center justify-center overflow-hidden rounded-lg bg-slate-950">
        {imageUrl ? (
          <div className="relative w-full">
            {/* Blob URLs change per frame; a plain img avoids Next image optimization for a live feed. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl}
              alt={`Live ${camera?.name ?? "camera"} frame with predicted shoreline overlay`}
              className="block max-h-[70vh] w-full object-contain"
            />
            {!showMask && polylineSegments.length > 0 && frameWidth && frameHeight ? (
              <svg
                className="pointer-events-none absolute inset-0 h-full w-full"
                viewBox={`0 0 ${frameWidth} ${frameHeight}`}
                preserveAspectRatio="xMidYMid meet"
                aria-hidden
              >
                {polylineSegments.map((segment, index) => (
                  <g key={index}>
                    <polyline
                      points={segment}
                      fill="none"
                      stroke="#000000"
                      strokeWidth={6}
                      vectorEffect="non-scaling-stroke"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <polyline
                      points={segment}
                      fill="none"
                      stroke="#00f5ff"
                      strokeWidth={3}
                      vectorEffect="non-scaling-stroke"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </g>
                ))}
              </svg>
            ) : null}
          </div>
        ) : (
          <div className="px-6 py-12 text-center text-sm text-slate-400">
            {selectedCameraId === "" ? (
              "Select a camera to start the shoreline preview."
            ) : !modelAvailable ? (
              "A usable segmentation model is required to preview shoreline predictions."
            ) : running ? (
              <span className="inline-flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                {waitingForSample
                  ? "Waiting for the live worker's next sample…"
                  : "Waiting for the first prediction…"}
              </span>
            ) : (
              "Preview paused. Resume to continue."
            )}
          </div>
        )}
        {imageUrl && running && (
          <span className="absolute left-3 top-3 rounded bg-black/70 px-2 py-1 text-xs font-medium text-white">
            ● SEGFORMER 3-CLASS · SHORELINE
          </span>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
        {showMask && (
          <>
            <span className="inline-flex items-center gap-1.5 text-slate-600">
              <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: "#d6b589" }} />
              Beach
            </span>
            <span className="inline-flex items-center gap-1.5 text-slate-600">
              <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: "#2070be" }} />
              Ocean
            </span>
            <span className="inline-flex items-center gap-1.5 text-slate-600">
              <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: "#87ceeb" }} />
              Sky
            </span>
          </>
        )}
        <span className="inline-flex items-center gap-1.5 text-slate-600">
          <span className="h-0.5 w-4" style={{ backgroundColor: "#00f5ff" }} />
          Ocean/Beach shoreline
        </span>
      </div>

      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs">
        {error ? (
          <span role="status" className="text-amber-700">
            Preview unavailable: {error}
          </span>
        ) : imageUrl ? (
          <span className="text-slate-500">
            {architecture ? `${architecture} · ` : ""}
            {confidence !== null
              ? `Model confidence ${(confidence * 100).toFixed(1)}%`
              : "Prediction received"}
            {inferenceMs !== null ? ` · inference ${inferenceMs.toFixed(1)} ms` : ""}
            {segments !== null ? ` · ${segments} shoreline line` : ""}
            {shorelineAxis ? ` (${shorelineAxis})` : ""}
            {measuredCoverage !== null ? ` · measured ${(measuredCoverage * 100).toFixed(0)}%` : ""}
            {interpolatedRanges !== null && interpolatedRanges > 0 ? ` · ${interpolatedRanges} bridged span(s)` : ""}
            {capturedAt ? ` · frame ${new Date(capturedAt).toLocaleTimeString()}` : ""}
            {sampleAgeMs !== null ? ` · sample age ${(sampleAgeMs / 1000).toFixed(1)} s` : ""}
          </span>
        ) : (
          <span className="text-slate-400">
            {showMask
              ? "Colours are the model's three classes; the yellow line is the extracted Ocean/Beach boundary."
              : "The yellow line is the extracted Ocean/Beach boundary; the class mask is hidden."}
          </span>
        )}
      </div>

      {imageUrl && shorelineAvailable === false && (
        <p role="status" className="mt-2 text-xs text-amber-700">
          No Ocean/Beach contact in this frame
          {shorelineReason ? ` (${shorelineReason})` : ""}. No shoreline is fabricated; the class
          overlay is still shown.
        </p>
      )}
    </Card>
  );
}
