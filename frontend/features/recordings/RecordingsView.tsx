"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Film, Loader2, Radio, RefreshCw, Trash2, Waves, XCircle } from "lucide-react";
import clsx from "clsx";

import { Card } from "@/components/Card";
import { PageHeader } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { TideAnomalyChart } from "@/features/anomalies/TideAnomalyChart";
import { WaterlineVisualizer } from "@/features/shoreline/WaterlineVisualizer";
import { anomaliesApi } from "@/lib/anomalies";
import { recordingsApi } from "@/lib/recordings";
import { shorelinesApi } from "@/lib/shorelines";
import { useDisplayTimezone } from "@/lib/settings";
import type {
  AnomalyEvent,
  ShorelineObservationDetail,
  ShorelineObservationSummary,
  StreamSession,
  VideoRecording,
} from "@/lib/types";

type Selection = {
  kind: "archive" | "stream";
  id: string;
  recordingId: string;
  label: string;
};

export function RecordingsView() {
  const { label: timezoneLabel, formatDateTime } = useDisplayTimezone();
  const [videos, setVideos] = useState<VideoRecording[]>([]);
  const [streams, setStreams] = useState<StreamSession[]>([]);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [observations, setObservations] = useState<ShorelineObservationSummary[]>([]);
  const [events, setEvents] = useState<AnomalyEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<AnomalyEvent | null>(null);
  const [selectedObservation, setSelectedObservation] =
    useState<ShorelineObservationDetail | null>(null);
  const [loadingObservation, setLoadingObservation] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const loadLists = useCallback(async () => {
    const [videoResult, streamResult] = await Promise.allSettled([
      recordingsApi.listVideos(),
      recordingsApi.listStreams(),
    ]);
    if (videoResult.status === "fulfilled") setVideos(videoResult.value);
    if (streamResult.status === "fulfilled") setStreams(streamResult.value);
    if (videoResult.status === "rejected" && streamResult.status === "rejected") {
      setError("Could not load recordings or stream sessions.");
    }
  }, []);

  useEffect(() => {
    void loadLists();
  }, [loadLists]);

  const loadSelection = useCallback(async (item: Selection) => {
    setBusy(true);
    setError(null);
    setSelection(item);
    setSelectedEvent(null);
    setSelectedObservation(null);
    try {
      const [rows, eventList] = await Promise.all([
        item.kind === "archive"
          ? recordingsApi.videoObservations(item.id)
          : recordingsApi.streamObservations(item.id),
        anomaliesApi.list({ recording_id: item.recordingId, limit: 200 }),
      ]);
      setObservations(rows);
      setEvents(eventList.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load this recording");
      setObservations([]);
      setEvents([]);
    } finally {
      setBusy(false);
    }
  }, []);

  const selectEvent = async (event: AnomalyEvent) => {
    setSelectedEvent(event);
    setLoadingObservation(true);
    try {
      setSelectedObservation(await anomaliesApi.observation(event.id));
      setError(null);
    } catch (err) {
      setSelectedObservation(null);
      setError(
        err instanceof Error
          ? `Waterline observation unavailable: ${err.message}`
          : "Waterline observation unavailable",
      );
    } finally {
      setLoadingObservation(false);
    }
  };

  const selectObservation = async (observation: ShorelineObservationSummary) => {
    setSelectedEvent(null);
    setLoadingObservation(true);
    try {
      setSelectedObservation(
        await shorelinesApi.observationDetail({
          camera_id: observation.camera_id,
          observed_at: observation.observed_at,
        }),
      );
      setError(null);
    } catch (err) {
      setSelectedObservation(null);
      setError(
        err instanceof Error
          ? `Waterline observation unavailable: ${err.message}`
          : "Waterline observation unavailable",
      );
    } finally {
      setLoadingObservation(false);
    }
  };

  const removeStream = async (session: StreamSession) => {
    if (
      !window.confirm(
        `Remove live stream ${session.session_id}? This stops it if running and deletes its observations and anomaly rows for this session.`,
      )
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await recordingsApi.deleteStream(session.session_id);
      if (selection?.kind === "stream" && selection.id === session.session_id) {
        setSelection(null);
        setObservations([]);
        setEvents([]);
        setSelectedEvent(null);
        setSelectedObservation(null);
      }
      await loadLists();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not remove live stream");
    } finally {
      setBusy(false);
    }
  };

  const rolling = useMemo(
    () => events.some((event) => event.detector_mode === "rolling_baseline"),
    [events],
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Recordings & live sessions"
        description="Per-timestamp shoreline observations and the rolling baseline/residual/event marks. Archive rows keep their derived clock provenance; live rows use wall-clock receipt time."
      />

      {error && (
        <Card className="border-red-200 bg-red-50 text-sm text-red-700">
          <div className="flex items-start gap-2">
            <XCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            {error}
          </div>
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Archive recordings" subtitle="video_recordings; stable recording_id">
          {videos.length === 0 ? (
            <p className="text-sm text-slate-400">No archive recordings registered.</p>
          ) : (
            <ul className="divide-y divide-slate-100">
              {videos.map((recording) => {
                const item: Selection = {
                  kind: "archive",
                  id: recording.recording_id,
                  recordingId: recording.recording_id,
                  label: recording.file_path.split(/[\\/]/).pop() ?? recording.recording_id,
                };
                return (
                  <li key={recording.recording_id}>
                    <button
                      onClick={() => void loadSelection(item)}
                      className={clsx(
                        "flex w-full items-center justify-between gap-3 rounded-md px-2 py-3 text-left hover:bg-slate-50",
                        selection?.kind === "archive" &&
                          selection.id === recording.recording_id &&
                          "bg-reef/5 ring-1 ring-reef",
                      )}
                    >
                      <span>
                        <span className="block text-sm font-medium text-slate-800">
                          <Film className="mr-1 inline h-4 w-4 text-reef" aria-hidden />
                          {item.label}
                        </span>
                        <span className="mt-0.5 block pl-5 text-xs text-slate-500">
                          {recording.recording_id} · {recording.timestamp_source ?? "unavailable"}
                          {recording.timestamp_confidence !== null
                            ? ` (${recording.timestamp_confidence})`
                            : ""}
                        </span>
                      </span>
                      <StatusBadge tone="neutral">{recording.processing_status ?? "registered"}</StatusBadge>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </Card>

        <Card title="Live stream sessions" subtitle="One stable session/recording id per capture run">
          {streams.length === 0 ? (
            <p className="text-sm text-slate-400">
              No live sessions yet. Start live capture on a camera, or run a bounded stream in the{" "}
              <Link className="text-reef underline" href="/lab">
                Test Lab
              </Link>
              .
            </p>
          ) : (
            <ul className="divide-y divide-slate-100">
              {streams.map((session) => {
                const item: Selection = {
                  kind: "stream",
                  id: session.session_id,
                  recordingId: session.recording_id,
                  label: session.camera_code ?? `camera ${session.camera_id}`,
                };
                return (
                  <li key={session.session_id} className="flex items-start gap-1">
                    <button
                      onClick={() => void loadSelection(item)}
                      className={clsx(
                        "flex flex-1 items-center justify-between gap-3 rounded-md px-2 py-3 text-left hover:bg-slate-50",
                        selection?.kind === "stream" &&
                          selection.id === session.session_id &&
                          "bg-reef/5 ring-1 ring-reef",
                      )}
                    >
                      <span>
                        <span className="block text-sm font-medium text-slate-800">
                          <Radio className="mr-1 inline h-4 w-4 text-reef" aria-hidden />
                          {item.label}
                        </span>
                        <span className="mt-0.5 block pl-5 text-xs text-slate-500">
                          {session.session_id} · {session.frames_persisted} persisted /{" "}
                          {session.frames_rejected} rejected
                          {session.last_rolling_classification
                            ? ` · ${session.last_rolling_classification}`
                            : ""}
                        </span>
                        <span className="mt-0.5 block pl-5 text-xs text-slate-400">
                          started {formatDateTime(session.started_at)} ({timezoneLabel})
                          {session.ended_at ? ` · ended ${formatDateTime(session.ended_at)}` : ""}
                        </span>
                      </span>
                      <StatusBadge tone={session.status === "running" ? "ok" : "neutral"}>
                        {session.status}
                      </StatusBadge>
                    </button>
                    <button
                      onClick={() => void removeStream(session)}
                      disabled={busy}
                      title={`Remove live stream ${session.session_id}`}
                      aria-label={`Remove live stream ${session.session_id}`}
                      className="mt-2 inline-flex items-center gap-1 rounded-md border border-red-200 px-2 py-1 text-xs text-red-600 hover:bg-red-50 disabled:opacity-40"
                    >
                      <Trash2 className="h-3.5 w-3.5" aria-hidden />
                      Remove
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </Card>
      </div>

      {selection && (
        <>
          <Card
            title={`${selection.label} — observations`}
            subtitle={`recording_id ${selection.recordingId}`}
          >
            <div className="mb-3 flex items-center gap-2">
              <button
                onClick={() => void loadSelection(selection)}
                disabled={busy}
                className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700 disabled:opacity-40"
              >
                {busy ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <RefreshCw className="h-3.5 w-3.5" />
                )}
                refresh
              </button>
              <span className="text-xs text-slate-400">
                {observations.length} observation(s) · {events.length} anomaly/quality row(s)
              </span>
            </div>
            {observations.length === 0 ? (
              <p className="text-sm text-slate-400">No observations recorded for this identity.</p>
            ) : (
              <div className="max-h-96 overflow-auto">
                <table className="min-w-full text-xs">
                  <thead className="sticky top-0 bg-slate-50 text-left text-slate-500">
                    <tr>
                      <th className="px-2 py-1">#</th>
                      <th className="px-2 py-1">observed_at ({timezoneLabel})</th>
                      <th className="px-2 py-1">source · conf</th>
                      <th className="px-2 py-1">shoreline px</th>
                      <th className="px-2 py-1">signed px</th>
                      <th className="px-2 py-1">status · segments</th>
                      <th className="px-2 py-1">reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {observations.map((row, index) => {
                      const selected =
                        selectedObservation?.camera_id === row.camera_id &&
                        new Date(selectedObservation.observed_at).getTime() ===
                          new Date(row.observed_at).getTime();
                      return (
                        <tr
                          key={`${row.observed_at}-${index}`}
                          onClick={() => void selectObservation(row)}
                          className={clsx(
                            "cursor-pointer border-t border-slate-100 hover:bg-slate-50",
                            selected && "bg-reef/5",
                          )}
                        >
                          <td className="px-2 py-1 text-slate-400">{row.frame_index ?? index}</td>
                          <td className="px-2 py-1 font-mono text-[11px]">
                            {formatDateTime(row.observed_at)}
                          </td>
                          <td className="px-2 py-1">
                            {row.capture_timestamp_source ?? "—"} ·{" "}
                            {typeof row.capture_timestamp_confidence === "number"
                              ? row.capture_timestamp_confidence.toFixed(2)
                              : "—"}
                          </td>
                          <td className="px-2 py-1">
                            {row.shoreline_position_px === null || row.shoreline_position_px === undefined
                              ? "—"
                              : row.shoreline_position_px.toFixed(1)}
                          </td>
                          <td className="px-2 py-1">
                            {row.signed_position_px === null || row.signed_position_px === undefined
                              ? "—"
                              : row.signed_position_px.toFixed(1)}
                          </td>
                          <td className="px-2 py-1">
                            {row.extraction_status ?? "—"} · {row.segment_count ?? 0} seg
                          </td>
                          <td className="px-2 py-1 text-slate-500">{row.extraction_reason ?? "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <Card
            title="Rolling baseline & events"
            subtitle={
              rolling
                ? "Rolling-baseline mode; NORMAL is never shown. Click a marker or event to inspect its waterline."
                : "No rolling rows"
            }
          >
            {events.length === 0 ? (
              <p className="text-sm text-slate-400">
                No anomaly/quality rows stored for this identity yet. Missing data is not normality.
              </p>
            ) : (
              <>
                <TideAnomalyChart
                  events={events}
                  baselineStatus={rolling ? "rolling_unverified" : null}
                  baselineReason={rolling ? "Tide-independent rolling median; no tide data is used." : null}
                  timeZone={timezoneLabel}
                  selectedEventId={selectedEvent?.id ?? null}
                  onEventClick={(eventId) => {
                    const event = events.find((item) => item.id === eventId);
                    if (event) void selectEvent(event);
                  }}
                />
                <div className="mt-3 max-h-56 overflow-auto rounded-md border border-slate-200">
                  <table className="min-w-full text-left text-xs">
                    <thead className="sticky top-0 bg-slate-50 text-slate-500">
                      <tr>
                        <th className="px-2 py-1">Detected</th>
                        <th className="px-2 py-1">Classification</th>
                        <th className="px-2 py-1">Observed px</th>
                        <th className="px-2 py-1">Residual px</th>
                        <th className="px-2 py-1">Reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {events.map((event) => (
                        <tr
                          key={event.id}
                          onClick={() => void selectEvent(event)}
                          className={clsx(
                            "cursor-pointer border-t border-slate-100 hover:bg-slate-50",
                            selectedEvent?.id === event.id && "bg-reef/5",
                          )}
                        >
                          <td className="whitespace-nowrap px-2 py-1 font-mono text-[11px]">
                            {formatDateTime(event.detected_at)}
                          </td>
                          <td className="px-2 py-1">
                            <StatusBadge tone={event.classification?.includes("RETREAT") ? "error" : "neutral"}>
                              {event.classification ?? event.state}
                            </StatusBadge>
                          </td>
                          <td className="px-2 py-1">
                            {event.observed_position_px?.toFixed(1) ?? "—"}
                          </td>
                          <td className="px-2 py-1">{event.residual_px?.toFixed(1) ?? "—"}</td>
                          <td className="px-2 py-1 text-slate-500">
                            {event.reason ?? event.explanation ?? "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </Card>

          <Card
            title="Waterline observation"
            subtitle="Click an event or observation row to render its stored Ocean/Beach line"
          >
            {loadingObservation ? (
              <p className="flex items-center justify-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-10 text-sm text-slate-500">
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                Loading waterline observation…
              </p>
            ) : selectedObservation ? (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
                  <StatusBadge tone="ok">{selectedObservation.camera_code ?? "camera"}</StatusBadge>
                  <span>{formatDateTime(selectedObservation.observed_at)}</span>
                  <span className="text-slate-400">({timezoneLabel})</span>
                  {selectedObservation.recording_id ? (
                    <span className="rounded bg-slate-100 px-2 py-0.5">
                      {selectedObservation.recording_id}
                      {selectedObservation.frame_index !== null &&
                      selectedObservation.frame_index !== undefined
                        ? ` · frame ${selectedObservation.frame_index}`
                        : ""}
                    </span>
                  ) : null}
                  <span className="rounded bg-slate-100 px-2 py-0.5">
                    {selectedObservation.extraction_status ?? "observation"} ·{" "}
                    {selectedObservation.segment_count ?? 0} segment(s)
                  </span>
                  {selectedEvent && (
                    <span className="text-slate-500">
                      event #{selectedEvent.id} · {selectedEvent.classification ?? selectedEvent.state}
                    </span>
                  )}
                </div>
                <WaterlineVisualizer detail={selectedObservation} heightClass="h-72" />
              </div>
            ) : (
              <div className="space-y-2">
                <WaterlineVisualizer
                  detail={null}
                  emptyMessage="Click an anomaly event or an observation row to render its waterline."
                  heightClass="h-56"
                />
                <p className="flex items-center gap-2 text-xs text-slate-500">
                  <Waves className="h-4 w-4 text-reef" aria-hidden />
                  The visualizer renders the stored ordered segments for the exact row you select.
                </p>
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
