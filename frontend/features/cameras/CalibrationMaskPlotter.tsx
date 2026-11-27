"use client";

import {
  AlertTriangle,
  CheckCircle2,
  Grid3x3,
  Loader2,
  MousePointer2,
  Pause,
  Play,
  Redo2,
  RefreshCw,
  Save,
  Square,
  Trash2,
  Undo2,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import clsx from "clsx";

import { Card } from "@/components/Card";
import { camerasApi, cameraFrameUrl } from "@/lib/cameras";
import type { Camera, CameraCalibration, CameraRoiPolygon } from "@/lib/types";
import {
  MIN_VERTICES,
  insertVertexOnNearestEdge,
  polygonArea,
  polygonCoveragePercent,
  polygonWarning,
  rectangleToPolygon,
  screenToNormalized,
  validatePolygon,
  type NormPoint,
} from "./calibrationMaskGeometry";

type Tool = "point" | "box";

type Draft = {
  points: NormPoint[];
  closed: boolean;
};

type FrameInfo = {
  url: string | null;
  width: number;
  height: number;
  capturedAt: string | null;
  status: "loading" | "live" | "frozen" | "stale";
  error: string | null;
};

type Props = {
  camera: Camera;
  calibration: CameraCalibration | null;
  onSaved: (saved: CameraCalibration) => void | Promise<void>;
  onDirtyChange?: (dirty: boolean) => void;
};

const HISTORY_LIMIT = 50;
const GRID_DIVISIONS = 10;

const EMPTY_DRAFT: Draft = { points: [], closed: false };

function fromCalibration(calibration: CameraCalibration | null): Draft {
  const points = calibration?.roi_polygon?.points;
  if (!points || points.length === 0) return EMPTY_DRAFT;
  return {
    points: points.map(([x, y]) => ({ x, y })),
    closed: true,
  };
}

function sameDraft(a: Draft, b: Draft): boolean {
  if (a.closed !== b.closed || a.points.length !== b.points.length) return false;
  return a.points.every(
    (point, index) =>
      point.x === b.points[index].x && point.y === b.points[index].y,
  );
}

function formatCapturedAt(value: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleTimeString();
}

export function CalibrationMaskPlotter({
  camera,
  calibration,
  onSaved,
  onDirtyChange,
}: Props) {
  const [draft, setDraft] = useState<Draft>(() => fromCalibration(calibration));
  const [tool, setTool] = useState<Tool>("point");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [gridOn, setGridOn] = useState(true);
  const [frozen, setFrozen] = useState(false);
  const [pageVisible, setPageVisible] = useState(true);
  const [refreshToken, setRefreshToken] = useState(0);
  const [frame, setFrame] = useState<FrameInfo>({
    url: null,
    width: 0,
    height: 0,
    capturedAt: null,
    status: "loading",
    error: null,
  });
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const [boxPreview, setBoxPreview] = useState<{ a: NormPoint; b: NormPoint } | null>(
    null,
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);

  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const blobUrlRef = useRef<string | null>(null);
  const draftRef = useRef(draft);
  draftRef.current = draft;
  const pastRef = useRef<Draft[]>([]);
  const futureRef = useRef<Draft[]>([]);
  const dragRef = useRef<{
    index: number;
    start: Draft;
    moved: boolean;
  } | null>(null);
  const boxRef = useRef<{ anchor: NormPoint; start: Draft } | null>(null);
  const loadedCalibrationRef = useRef<number | null>(calibration?.id ?? null);

  const checked = validatePolygon(draft.points);
  const areaPercent = polygonCoveragePercent(draft.points);
  const warning = polygonWarning(draft.points);

  // Load the active profile once per profile change. A saved outline becomes
  // the new baseline; an in-progress draft is not silently overwritten.
  useEffect(() => {
    const nextId = calibration?.id ?? null;
    if (loadedCalibrationRef.current === nextId) return;
    loadedCalibrationRef.current = nextId;
    setDraft(fromCalibration(calibration));
    pastRef.current = [];
    futureRef.current = [];
    setSelectedIndex(null);
    setDirty(false);
    setError(null);
    setNotice(null);
  }, [calibration]);

  useEffect(() => {
    onDirtyChange?.(dirty);
  }, [dirty, onDirtyChange]);

  useEffect(() => {
    return () => onDirtyChange?.(false);
  }, [onDirtyChange]);

  useEffect(() => {
    if (!dirty) return;
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [dirty]);

  useEffect(() => {
    const update = () => setPageVisible(!document.hidden);
    update();
    document.addEventListener("visibilitychange", update);
    return () => document.removeEventListener("visibilitychange", update);
  }, []);

  // Poll the existing JPEG frame endpoint. Requests never overlap; the last
  // good frame is retained if the camera fails.
  useEffect(() => {
    if (frozen || !pageVisible) return;
    let active = true;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();

    const refresh = async () => {
      try {
        const response = await fetch(
          `${cameraFrameUrl(camera.id)}?t=${Date.now()}`,
          { cache: "no-store", signal: controller.signal },
        );
        if (!response.ok) {
          let message = `Frame unavailable (${response.status})`;
          try {
            const body = await response.json();
            message = body?.error?.message ?? message;
          } catch {
            // Keep the HTTP status when the body is not the API error envelope.
          }
          throw new Error(message);
        }
        const blob = await response.blob();
        if (!active) return;
        const url = URL.createObjectURL(blob);
        if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = url;
        setFrame((previous) => ({
          ...previous,
          url,
          capturedAt: response.headers.get("X-Frame-Captured-At") ?? previous.capturedAt,
          status: "live",
          error: null,
        }));
      } catch (err) {
        if (active && !controller.signal.aborted) {
          const message =
            err instanceof Error ? err.message : "Could not load the camera frame";
          setFrame((previous) => ({
            ...previous,
            status: previous.url ? "stale" : "loading",
            error: message,
          }));
        }
      } finally {
        if (active) timeout = setTimeout(() => void refresh(), 1200);
      }
    };

    void refresh();
    return () => {
      active = false;
      controller.abort();
      if (timeout) clearTimeout(timeout);
    };
  }, [camera.id, frozen, pageVisible, refreshToken]);

  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    };
  }, []);

  // Track the rendered content box so pointer coordinates and drawing use CSS
  // pixels while saved points stay normalized.
  useEffect(() => {
    const element = wrapperRef.current;
    if (!element) return;
    const measure = () => {
      const rect = element.getBoundingClientRect();
      setCanvasSize({ width: rect.width, height: rect.height });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    window.addEventListener("resize", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);

  const contextFor = (event: React.PointerEvent<HTMLCanvasElement> | React.MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { rect, point: screenToNormalized(event.clientX, event.clientY, rect) };
  };

  const nearestVertex = (event: React.PointerEvent<HTMLCanvasElement>): number | null => {
    const { rect, point } = contextFor(event);
    const radius = event.pointerType === "touch" ? 24 : 12;
    let best: number | null = null;
    let bestDistance = Infinity;
    draftRef.current.points.forEach((vertex, index) => {
      const distance = Math.hypot(
        (vertex.x - point.x) * rect.width,
        (vertex.y - point.y) * rect.height,
      );
      if (distance <= radius && distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    });
    return best;
  };

  const commit = useCallback((updater: (previous: Draft) => Draft) => {
    setDraft((previous) => {
      const next = updater(previous);
      if (sameDraft(previous, next)) return previous;
      pastRef.current = [...pastRef.current.slice(-(HISTORY_LIMIT - 1)), previous];
      futureRef.current = [];
      return next;
    });
    setDirty(true);
    setNotice(null);
  }, []);

  const undo = () => {
    const previous = pastRef.current.pop();
    if (!previous) return;
    futureRef.current.push(draftRef.current);
    setDraft(previous);
    setSelectedIndex(null);
    setDirty(true);
    setError(null);
    setNotice(null);
  };

  const redo = () => {
    const next = futureRef.current.pop();
    if (!next) return;
    pastRef.current.push(draftRef.current);
    setDraft(next);
    setSelectedIndex(null);
    setDirty(true);
    setError(null);
    setNotice(null);
  };

  const closeOutline = () => {
    const result = validatePolygon(draftRef.current.points);
    if (!result.ok) {
      setError(result.reason);
      return;
    }
    commit((previous) => ({ ...previous, closed: true }));
    setError(null);
  };

  const deleteSelected = () => {
    if (selectedIndex === null) return;
    const points = draftRef.current.points.filter((_, index) => index !== selectedIndex);
    commit(() => ({
      points,
      closed: points.length >= MIN_VERTICES ? draftRef.current.closed : false,
    }));
    setSelectedIndex(null);
  };

  const clearOutline = () => {
    if (!window.confirm("Clear this outline? The saved calibration is not removed until you save.")) {
      return;
    }
    commit(() => EMPTY_DRAFT);
    setSelectedIndex(null);
    setError(null);
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (saving) return;
    event.preventDefault();
    wrapperRef.current?.focus();
    const { point } = contextFor(event);
    const current = draftRef.current;

    if (tool === "box") {
      if (current.points.length > 0) {
        const replace = window.confirm(
          "Replace the current outline with this rectangle?",
        );
        if (!replace) return;
      }
      boxRef.current = { anchor: point, start: current };
      setBoxPreview({ a: point, b: point });
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // Some synthetic/embedded pointer events have no active pointer.
      }
      return;
    }

    const index = nearestVertex(event);
    if (index !== null) {
      setSelectedIndex(index);
      dragRef.current = { index, start: current, moved: false };
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // Some synthetic/embedded pointer events have no active pointer.
      }
      return;
    }

    if (current.closed) {
      setNotice("Outline is closed. Press Edit points to add vertices, or drag an existing point.");
      return;
    }
    const nextPoints = [...current.points, point];
    commit(() => ({ points: nextPoints, closed: false }));
    setSelectedIndex(nextPoints.length - 1);
    setError(null);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const { point } = contextFor(event);
    if (dragRef.current) {
      const { index } = dragRef.current;
      dragRef.current.moved = true;
      setDraft((previous) => ({
        ...previous,
        points: previous.points.map((vertex, vertexIndex) =>
          vertexIndex === index ? point : vertex,
        ),
      }));
      setDirty(true);
      return;
    }
    if (boxRef.current) {
      setBoxPreview({ a: boxRef.current.anchor, b: point });
    }
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragRef.current) {
      const drag = dragRef.current;
      dragRef.current = null;
      if (drag.moved) {
        pastRef.current = [
          ...pastRef.current.slice(-(HISTORY_LIMIT - 1)),
          drag.start,
        ];
        futureRef.current = [];
        setError(null);
      } else if (
        drag.index === 0 &&
        !draftRef.current.closed &&
        draftRef.current.points.length >= MIN_VERTICES
      ) {
        closeOutline();
      }
      try {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
      } catch {
        // Pointer capture is best-effort.
      }
      return;
    }

    if (boxRef.current) {
      const anchor = boxRef.current.anchor;
      const { point } = contextFor(event);
      boxRef.current = null;
      setBoxPreview(null);
      try {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
      } catch {
        // Pointer capture is best-effort.
      }
      const rectangle = rectangleToPolygon(anchor, point);
      if (polygonArea(rectangle) < 0.00005) {
        setError("Drag a larger rectangle to create an outline.");
        return;
      }
      commit(() => ({ points: rectangle, closed: true }));
      setTool("point");
      setSelectedIndex(null);
      setError(null);
    }
  };

  const handleDoubleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (saving || tool === "box") return;
    const { rect, point } = contextFor(event);
    const tolerance = 14 / Math.max(rect.width, 1);
    const inserted = insertVertexOnNearestEdge(
      draftRef.current.points,
      point,
      tolerance,
    );
    if (!inserted) return;
    commit(() => ({ points: inserted, closed: draftRef.current.closed }));
    setError(null);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const meta = event.ctrlKey || event.metaKey;
    if (meta && event.key.toLowerCase() === "z") {
      event.preventDefault();
      if (event.shiftKey) redo();
      else undo();
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      if (selectedIndex !== null) {
        event.preventDefault();
        deleteSelected();
      }
      return;
    }
    if (event.key === "Escape") {
      setSelectedIndex(null);
      setBoxPreview(null);
      boxRef.current = null;
      setError(null);
      return;
    }
    if (event.key === "Enter" && !draftRef.current.closed) {
      event.preventDefault();
      closeOutline();
    }
  };

  // Draw the outline and the 50%-transparent grid in CSS pixels.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvasSize.width <= 0 || canvasSize.height <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvasSize.width;
    const height = canvasSize.height;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);

    if (gridOn) {
      context.save();
      context.lineWidth = 1;
      for (let index = 1; index < GRID_DIVISIONS; index += 1) {
        const x = Math.round((width * index) / GRID_DIVISIONS) + 0.5;
        const y = Math.round((height * index) / GRID_DIVISIONS) + 0.5;
        context.strokeStyle = "rgba(0, 0, 0, 0.35)";
        context.beginPath();
        context.moveTo(x + 1, 0);
        context.lineTo(x + 1, height);
        context.moveTo(0, y + 1);
        context.lineTo(width, y + 1);
        context.stroke();
        context.globalAlpha = 0.5;
        context.strokeStyle = "rgba(255, 255, 255, 0.85)";
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
        context.globalAlpha = 1;
      }
      context.restore();
    }

    const points = draft.points.map((point) => ({
      x: point.x * width,
      y: point.y * height,
    }));

    if (points.length >= 2) {
      context.beginPath();
      context.moveTo(points[0].x, points[0].y);
      for (let index = 1; index < points.length; index += 1) {
        context.lineTo(points[index].x, points[index].y);
      }
      if (draft.closed && points.length >= 3) {
        context.closePath();
        context.fillStyle = "rgba(20, 184, 166, 0.16)";
        context.fill();
      }
      context.lineJoin = "round";
      context.lineCap = "round";
      context.strokeStyle = "rgba(255, 255, 255, 0.85)";
      context.lineWidth = 4;
      context.stroke();
      context.strokeStyle = "#0d9488";
      context.lineWidth = 2;
      context.stroke();
      if (!draft.closed && points.length >= 3) {
        context.setLineDash([6, 5]);
        context.strokeStyle = "rgba(13, 148, 136, 0.75)";
        context.lineWidth = 1.5;
        context.beginPath();
        context.moveTo(points[points.length - 1].x, points[points.length - 1].y);
        context.lineTo(points[0].x, points[0].y);
        context.stroke();
        context.setLineDash([]);
      }
    } else if (points.length === 1) {
      // Keep a visible dot until there are enough vertices for a line.
      context.fillStyle = "#0d9488";
      context.beginPath();
      context.arc(points[0].x, points[0].y, 3, 0, Math.PI * 2);
      context.fill();
    }

    const handleSize = 9;
    points.forEach((point, index) => {
      const selected = index === selectedIndex;
      const first = index === 0;
      const size = first && !draft.closed ? handleSize + 3 : handleSize;
      context.fillStyle = selected ? "#0d9488" : "rgba(255, 255, 255, 0.95)";
      context.strokeStyle = selected ? "#ffffff" : "#0f766e";
      context.lineWidth = selected ? 2 : 1.5;
      context.beginPath();
      context.rect(
        point.x - size / 2,
        point.y - size / 2,
        size,
        size,
      );
      context.fill();
      context.stroke();
      if (first && !draft.closed && points.length >= MIN_VERTICES) {
        context.strokeStyle = "rgba(255, 255, 255, 0.95)";
        context.lineWidth = 1.5;
        context.beginPath();
        context.arc(point.x, point.y, size + 4, 0, Math.PI * 2);
        context.stroke();
      }
    });

    if (boxPreview) {
      const a = { x: boxPreview.a.x * width, y: boxPreview.a.y * height };
      const b = { x: boxPreview.b.x * width, y: boxPreview.b.y * height };
      context.setLineDash([5, 4]);
      context.strokeStyle = "#0d9488";
      context.lineWidth = 1.5;
      context.fillStyle = "rgba(20, 184, 166, 0.12)";
      context.beginPath();
      context.rect(
        Math.min(a.x, b.x),
        Math.min(a.y, b.y),
        Math.abs(b.x - a.x),
        Math.abs(b.y - a.y),
      );
      context.fill();
      context.stroke();
      context.setLineDash([]);
    }
  }, [boxPreview, canvasSize, draft, gridOn, selectedIndex]);

  const saveOutline = async () => {
    const result = validatePolygon(draft.points);
    if (!result.ok) {
      setError(result.reason);
      return;
    }
    if (!frame.url) {
      setError("Wait for a camera frame before saving the outline.");
      return;
    }
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const polygon: CameraRoiPolygon = {
        coordinate_space: "normalized",
        points: draft.points.map((point) => [point.x, point.y]),
        frame_width: frame.width || null,
        frame_height: frame.height || null,
      };
      const saved = calibration
        ? await camerasApi.saveRoiPolygon(camera.id, calibration.id, {
            roi_polygon: polygon,
          })
        : await camerasApi.createCalibration(camera.id, {
            roi_polygon: polygon,
            set_active: true,
            notes: "Monitoring outline created in the camera calibration editor.",
          });
      setDirty(false);
      setNotice(
        `Saved as calibration v${saved.version}. Only this outline is used for shoreline detection.`,
      );
      loadedCalibrationRef.current = saved.id;
      await onSaved(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save the outline");
    } finally {
      setSaving(false);
    }
  };

  const removeOutline = async () => {
    if (!calibration || !calibration.roi_polygon) return;
    if (
      !window.confirm(
        "Remove the monitoring outline? This re-enables full-frame detection for this camera and is saved as a new calibration version.",
      )
    ) {
      return;
    }
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const saved = await camerasApi.saveRoiPolygon(camera.id, calibration.id, {
        roi_polygon: null,
      });
      setDraft(EMPTY_DRAFT);
      setSelectedIndex(null);
      pastRef.current = [];
      futureRef.current = [];
      setDirty(false);
      setNotice(
        `Saved as calibration v${saved.version} with no outline. The full frame is used again.`,
      );
      loadedCalibrationRef.current = saved.id;
      await onSaved(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not remove the outline");
    } finally {
      setSaving(false);
    }
  };

  const frameDimensions =
    frame.width > 0 && frame.height > 0
      ? `${frame.width}×${frame.height}`
      : null;
  const savedDimensions =
    calibration?.roi_polygon?.frame_width && calibration.roi_polygon.frame_height
      ? `${calibration.roi_polygon.frame_width}×${calibration.roi_polygon.frame_height}`
      : null;
  const scaledNotice =
    savedDimensions &&
    frameDimensions &&
    savedDimensions !== frameDimensions
      ? `Outline scaled from ${savedDimensions} to ${frameDimensions}.`
      : null;

  const badge = frozen
    ? "FROZEN"
    : frame.status === "stale"
      ? "STALE"
      : camera.camera_type === "upload"
        ? "LATEST FRAME"
        : "LIVE";
  const capturedLabel = formatCapturedAt(frame.capturedAt);
  const hasSavedOutline = Boolean(calibration?.roi_polygon);

  return (
    <Card
      title={`Monitoring area — ${camera.code}`}
      subtitle="Only the shaded outline is used for shoreline detection."
    >
      <p className="mb-3 text-sm text-slate-600">
        Only the shaded area is measured. Everything outside the outline is ignored.
      </p>

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded-md border border-slate-200 p-0.5">
          <button
            type="button"
            onClick={() => setTool("point")}
            aria-pressed={tool === "point"}
            className={clsx(
              "inline-flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs font-medium",
              tool === "point"
                ? "bg-reef text-white"
                : "text-slate-600 hover:bg-slate-50",
            )}
          >
            <MousePointer2 className="h-3.5 w-3.5" aria-hidden />
            Point
          </button>
          <button
            type="button"
            onClick={() => setTool("box")}
            aria-pressed={tool === "box"}
            className={clsx(
              "inline-flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs font-medium",
              tool === "box"
                ? "bg-reef text-white"
                : "text-slate-600 hover:bg-slate-50",
            )}
          >
            <Square className="h-3.5 w-3.5" aria-hidden />
            Box
          </button>
        </div>

        <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-slate-600">
          <input
            type="checkbox"
            className="h-4 w-4 rounded border-slate-300"
            checked={gridOn}
            onChange={(event) => setGridOn(event.target.checked)}
          />
          <Grid3x3 className="h-3.5 w-3.5" aria-hidden />
          Grid (50% opacity)
        </label>

        <button
          type="button"
          onClick={() => setFrozen((value) => !value)}
          className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs text-slate-700 hover:bg-slate-50"
        >
          {frozen ? (
            <Play className="h-3.5 w-3.5" aria-hidden />
          ) : (
            <Pause className="h-3.5 w-3.5" aria-hidden />
          )}
          {frozen ? "Resume live feed" : "Freeze frame"}
        </button>
        <button
          type="button"
          onClick={() => {
            setFrozen(false);
            setRefreshToken((value) => value + 1);
          }}
          className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs text-slate-700 hover:bg-slate-50"
        >
          <RefreshCw className="h-3.5 w-3.5" aria-hidden />
          Refresh feed
        </button>
      </div>

      <div
        ref={wrapperRef}
        className="relative w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-950"
        style={{
          aspectRatio:
            frame.width > 0 && frame.height > 0
              ? `${frame.width} / ${frame.height}`
              : "16 / 9",
        }}
        tabIndex={0}
        role="application"
        aria-label="Monitoring outline plotter. Use the Point tool to add connected points or Box to draw a rectangle."
        onKeyDown={handleKeyDown}
      >
        {frame.url ? (
          // A blob URL changes per frame; a plain img avoids Next optimization.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={frame.url}
            alt={`Camera frame from ${camera.code}`}
            draggable={false}
            className="pointer-events-none absolute inset-0 h-full w-full select-none object-contain"
            onLoad={(event) => {
              const image = event.currentTarget;
              if (
                image.naturalWidth > 0 &&
                (image.naturalWidth !== frame.width ||
                  image.naturalHeight !== frame.height)
              ) {
                setFrame((previous) => ({
                  ...previous,
                  width: image.naturalWidth,
                  height: image.naturalHeight,
                }));
              }
            }}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-sm text-slate-400">
            {frame.error ? (
              <span className="text-amber-300">{frame.error}</span>
            ) : (
              <span className="inline-flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                Waiting for the camera frame…
              </span>
            )}
          </div>
        )}

        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full touch-none"
          style={{ cursor: tool === "box" ? "crosshair" : "crosshair" }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          onDoubleClick={handleDoubleClick}
        />

        <span className="pointer-events-none absolute left-3 top-3 rounded bg-black/70 px-2 py-1 text-[11px] font-semibold text-white">
          {badge}
        </span>
        {capturedLabel && (
          <span className="pointer-events-none absolute bottom-3 right-3 rounded bg-black/60 px-2 py-1 text-[11px] text-white">
            frame {capturedLabel}
          </span>
        )}
        {frame.status === "frozen" && (
          <span className="pointer-events-none absolute right-3 top-3 rounded bg-amber-500/90 px-2 py-1 text-[11px] font-medium text-white">
            Feed paused
          </span>
        )}
      </div>

      {frame.error && frame.url && (
        <p role="status" className="mt-2 text-xs text-amber-700">
          Feed problem: {frame.error}. The last good frame is shown; drawing is still possible.
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-slate-600">
        <span aria-live="polite">
          {draft.points.length} point{draft.points.length === 1 ? "" : "s"} ·{" "}
          {draft.closed ? "closed outline" : "open outline"} · area{" "}
          {areaPercent.toFixed(1)}% of frame
        </span>
        <span className="text-slate-400">
          {calibration
            ? `Editing outline from calibration v${calibration.version}`
            : "No calibration profile yet"}
        </span>
        {scaledNotice && <span className="text-amber-700">{scaledNotice}</span>}
      </div>

      {warning && (
        <p role="status" className="mt-2 flex items-start gap-1.5 text-xs text-amber-700">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          {warning}
        </p>
      )}
      {!checked.ok && draft.points.length > 0 && (
        <p role="status" className="mt-2 flex items-start gap-1.5 text-xs text-red-700">
          <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          {checked.reason}
        </p>
      )}
      {notice && (
        <p role="status" className="mt-2 flex items-start gap-1.5 text-xs text-emerald-700">
          <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          {notice}
        </p>
      )}
      {error && (
        <p role="alert" className="mt-2 flex items-start gap-1.5 text-xs text-red-700">
          <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          {error}
        </p>
      )}

      <div className="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => void saveOutline()}
          disabled={saving || !checked.ok || !frame.url || !dirty}
          className="inline-flex items-center gap-2 rounded-md bg-deep px-4 py-2 text-sm font-medium text-white hover:bg-lagoon disabled:opacity-40"
        >
          {saving ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          ) : (
            <Save className="h-4 w-4" aria-hidden />
          )}
          Save monitoring outline
        </button>
        {!draft.closed && draft.points.length >= MIN_VERTICES && (
          <button
            type="button"
            onClick={closeOutline}
            className="inline-flex items-center gap-2 rounded-md border border-reef px-3 py-2 text-sm font-medium text-reef hover:bg-reef hover:text-white"
          >
            Finish outline
          </button>
        )}
        {draft.closed && (
          <button
            type="button"
            onClick={() => commit((previous) => ({ ...previous, closed: false }))}
            className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50"
          >
            Edit points
          </button>
        )}
        <button
          type="button"
          onClick={undo}
          disabled={pastRef.current.length === 0}
          className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-40"
        >
          <Undo2 className="h-4 w-4" aria-hidden />
          Undo
        </button>
        <button
          type="button"
          onClick={redo}
          disabled={futureRef.current.length === 0}
          className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-40"
        >
          <Redo2 className="h-4 w-4" aria-hidden />
          Redo
        </button>
        <button
          type="button"
          onClick={deleteSelected}
          disabled={selectedIndex === null}
          className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-40"
        >
          <Trash2 className="h-4 w-4" aria-hidden />
          Delete point
        </button>
        <button
          type="button"
          onClick={clearOutline}
          className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50"
        >
          Clear
        </button>
        {hasSavedOutline && (
          <button
            type="button"
            onClick={() => void removeOutline()}
            disabled={saving}
            className="inline-flex items-center gap-2 rounded-md border border-red-200 px-3 py-2 text-sm text-red-600 hover:bg-red-50 disabled:opacity-40"
          >
            Remove outline
          </button>
        )}
      </div>

      {!hasSavedOutline && draft.points.length === 0 && (
        <p className="mt-2 text-xs text-slate-500">
          No monitoring outline — the full frame is currently used.
        </p>
      )}

      <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-3 text-xs leading-relaxed text-slate-600">
        <p className="mb-1 font-semibold text-slate-800">How to use this outline</p>
        <p>
          Draw one outline around the shoreline you want CoastWatch to measure. Click the
          live frame to add points; they connect in order. Drag a point to adjust it, or use{" "}
          <strong>Box</strong> to start from a rectangle. The shaded polygon is the{" "}
          <strong>only area used for shoreline detection</strong> — Ocean and Beach outside
          it are ignored, so distant water, other beaches, and the sky cannot produce a false
          shoreline.
        </p>
        <p className="mt-2">
          Keep the outline tight around the coast this camera is watching: include the sand
          and water where the shoreline can move, but exclude roads, buildings, piers, and
          neighbouring beaches. This is a preview frame, not a recording. Freeze the frame if
          the scene is moving while you draw.
        </p>
        <p className="mt-2">
          If the camera is physically moved, re-draw the outline. If its resolution changes,
          the outline is scaled automatically, but check that it still matches the coast
          before relying on new measurements.
        </p>
      </div>
    </Card>
  );
}
