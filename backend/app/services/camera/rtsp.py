"""RTSP camera provider (Phase 3).

RTSP URL configuration, connection timeout, automatic reconnection and
stream health monitoring. Frames are acquired via OpenCV's FFMPEG-backed
``VideoCapture``.

A background drain thread keeps calling ``grab()`` so the transport socket is
never allowed to back up between the (deliberately slow) sampling reads. The
newest decoded frame is retained and returned by :meth:`read_frame`.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime
from typing import Any

import cv2

from app.core.logging import get_logger
from app.services.camera.base import (
    CameraConnectionError,
    CameraMetadata,
    CameraProvider,
    CameraStatus,
    Frame,
    FrameUnavailableError,
)

logger = get_logger(__name__)

# After this many consecutive read failures the provider drops the connection
# and reconnects on the next read/connect call.
_MAX_CONSECUTIVE_FAILURES = 3

# TCP applies backpressure instead of discarding packets when the reader is
# slower than the publisher. Set before any VideoCapture is constructed.
_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp"


def _force_tcp_transport() -> None:
    """Ensure OpenCV's FFMPEG backend opens RTSP over TCP."""
    existing = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    options = [part for part in existing.split(";") if part]
    if "rtsp_transport" in options:
        index = options.index("rtsp_transport")
        if index + 1 < len(options):
            options[index + 1] = "tcp"
        else:
            options.append("tcp")
    else:
        options.extend(_FFMPEG_CAPTURE_OPTIONS.split(";"))
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ";".join(options)


_force_tcp_transport()


class RTSPCameraProvider(CameraProvider):
    def __init__(
        self,
        url: str,
        *,
        camera_id: int = 0,
        timeout_s: float = 5.0,
        max_reconnect_attempts: int = 3,
        reconnect_delay_s: float = 0.5,
        stall_timeout_s: float = 60.0,
    ) -> None:
        if not url or not url.strip():
            raise ValueError("rtsp url must not be blank")
        self._url = url
        self._camera_id = camera_id
        self._timeout_s = timeout_s
        self._max_reconnect_attempts = max(1, max_reconnect_attempts)
        self._reconnect_delay_s = max(0.0, reconnect_delay_s)
        # Two default 30 s sampling intervals: longer than this without a new
        # frame means the transport is stalled even if the socket still opens.
        self._stall_timeout_s = max(1.0, stall_timeout_s)

        self._cap: Any | None = None
        self._lock = threading.Lock()
        self._grab_lock = threading.Lock()
        self._connect_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._drain_stop = threading.Event()
        self._drain_thread: threading.Thread | None = None

        self._latest_image: Any | None = None
        self._latest_captured_at: datetime | None = None
        self._latest_monotonic: float | None = None
        self._drain_failures = 0

        self._frames_captured = 0
        self._last_frame_at: datetime | None = None
        self._consecutive_failures = 0
        self._sequence = 0
        self._last_error: str | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        self._open_capture()

    def disconnect(self) -> None:
        self._release()

    def _release(self) -> None:
        self._stop_drain()
        with self._grab_lock:
            self._release_capture_locked()

    def _release_capture_locked(self) -> None:
        """Drop the capture; caller holds ``_grab_lock``."""
        with self._lock:
            cap = self._cap
            self._cap = None
            self._latest_image = None
            self._latest_captured_at = None
            self._latest_monotonic = None
        if cap is not None:
            try:
                cap.release()
            except Exception:  # pragma: no cover - backend dependent
                logger.debug("rtsp camera %s: release failed", self._camera_id)

    def _stop_drain(self) -> None:
        thread = self._drain_thread
        self._drain_thread = None
        self._drain_stop.set()
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=max(1.0, self._timeout_s + 1.0))

    def _start_drain(self) -> None:
        self._drain_stop.clear()
        thread = threading.Thread(
            target=self._drain_loop,
            name=f"coastwatch-rtsp-drain-{self._camera_id}",
            daemon=True,
        )
        self._drain_thread = thread
        thread.start()

    def _open_capture(self) -> None:
        self._stop_drain()
        with self._grab_lock:
            self._release_capture_locked()
            _force_tcp_transport()
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            # Best-effort open timeout (supported by the FFMPEG backend on
            # OpenCV >= 4.8). Ignored silently on builds that lack the property.
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self._timeout_s * 1000))
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self._timeout_s * 1000))
            except Exception:  # pragma: no cover - backend dependent
                logger.debug("rtsp timeout properties not supported by this OpenCV build")
            if not cap.isOpened():
                cap.release()
                self._last_error = f"could not open RTSP stream (timeout {self._timeout_s}s)"
                raise CameraConnectionError(self._last_error)
            try:
                # Keep at most one decoded frame queued inside the backend.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # pragma: no cover - backend dependent
                logger.debug("rtsp CAP_PROP_BUFFERSIZE not supported by this build")
            with self._lock:
                self._cap = cap
                self._consecutive_failures = 0
                self._drain_failures = 0
                self._last_error = None
                self._latest_image = None
                self._latest_captured_at = None
                self._latest_monotonic = None
        self._frame_ready.clear()
        self._start_drain()
        logger.info("rtsp camera %s connected (tcp)", self._camera_id)

    def _ensure_connected(self) -> None:
        with self._connect_lock:
            with self._lock:
                cap = self._cap
                connected = cap is not None and cap.isOpened()
            if connected:
                return
            attempts = 0
            while attempts < self._max_reconnect_attempts:
                attempts += 1
                try:
                    self._open_capture()
                    return
                except CameraConnectionError:
                    if attempts < self._max_reconnect_attempts:
                        time.sleep(self._reconnect_delay_s)
            raise CameraConnectionError(
                self._last_error or f"RTSP stream unavailable after {attempts} attempt(s)"
            )

    # ------------------------------------------------------------------
    # Background drain
    # ------------------------------------------------------------------
    def _drain_loop(self) -> None:
        while not self._drain_stop.is_set():
            ok = False
            image = None
            with self._grab_lock:
                if self._drain_stop.is_set():
                    break
                with self._lock:
                    cap = self._cap
                if cap is None or not cap.isOpened():
                    break
                try:
                    ok = bool(cap.grab())
                    if ok:
                        ok, image = cap.retrieve()
                except Exception:  # pragma: no cover - backend dependent
                    ok = False
                    image = None
            if ok and image is not None:
                with self._lock:
                    self._latest_image = image
                    self._latest_captured_at = datetime.now(UTC)
                    self._latest_monotonic = time.monotonic()
                    self._drain_failures = 0
                    self._last_error = None
                self._frame_ready.set()
                self._drain_stop.wait(0.01)
                continue

            with self._lock:
                self._drain_failures += 1
                self._last_error = "frame read failed"
                failures = self._drain_failures
            if failures >= _MAX_CONSECUTIVE_FAILURES:
                # Drop the stream; the next read reconnects from scratch.
                with self._grab_lock:
                    self._release_capture_locked()
                break
            self._drain_stop.wait(0.05)

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------
    def read_frame(self) -> Frame:
        self._ensure_connected()
        deadline = time.monotonic() + max(1.0, self._timeout_s)
        while True:
            with self._lock:
                image = self._latest_image
                captured_at = self._latest_captured_at
                failures = self._drain_failures
            if image is not None and captured_at is not None and failures == 0:
                with self._lock:
                    self._consecutive_failures = 0
                    self._last_error = None
                    self._frames_captured += 1
                    self._sequence += 1
                    self._last_frame_at = captured_at
                    sequence = self._sequence
                return Frame(
                    image=image,
                    captured_at=captured_at,
                    metadata=self.get_metadata(),
                    sequence=sequence,
                )
            if failures > 0:
                self._register_failure("frame read failed")
                raise FrameUnavailableError(
                    f"could not read a frame from the RTSP stream "
                    f"({self._consecutive_failures} consecutive failure(s))"
                )
            if time.monotonic() >= deadline:
                self._register_failure("stream stalled")
                raise FrameUnavailableError(
                    "timed out waiting for the RTSP stream to produce a frame"
                )
            self._frame_ready.wait(0.05)

    def _register_failure(self, message: str) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._last_error = message
            release = self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES
        if release:
            self._release()

    # ------------------------------------------------------------------
    # Status / metadata
    # ------------------------------------------------------------------
    def get_status(self) -> CameraStatus:
        with self._lock:
            connected = self._cap is not None and bool(self._cap.isOpened())
            stalled = bool(
                connected
                and self._latest_monotonic is not None
                and (time.monotonic() - self._latest_monotonic) > self._stall_timeout_s
            )
            healthy = (
                connected
                and self._consecutive_failures == 0
                and self._drain_failures == 0
                and not stalled
            )
            if stalled:
                message = "stream stalled"
            elif connected and self._last_frame_at is not None:
                message = "stream healthy"
            elif connected:
                message = "connected, no frames read yet"
            elif self._last_error:
                message = self._last_error
            else:
                message = "not connected"
            return CameraStatus(
                connected=connected,
                healthy=healthy,
                message=message,
                last_frame_at=self._last_frame_at,
                frames_captured=self._frames_captured,
            )

    def get_metadata(self) -> CameraMetadata:
        width = height = None
        fps = None
        with self._lock:
            cap = self._cap
        if cap is not None and cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = float(raw_fps) if raw_fps and raw_fps > 0 else None
        return CameraMetadata(
            camera_id=self._camera_id,
            camera_type="rtsp",
            width=width,
            height=height,
            fps=fps,
        )
