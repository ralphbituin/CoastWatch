"""Windows host camera agent for CoastWatch.

Why this exists
---------------
Docker Desktop on Windows runs Linux containers and does not expose Windows USB
webcams to them. This small agent runs **natively on Windows**, enumerates the
DirectShow cameras (including "OBS Virtual Camera"), and re-serves them as MJPEG
over HTTP. The Dockerized backend can then register/consume them as ordinary
network cameras; the backend never needs `/dev/video*` passthrough.

Run (from the repository root):

    powershell -ExecutionPolicy Bypass -File scripts/host-camera-agent.ps1

or directly:

    backend\\.venv\\Scripts\\python.exe tools\\host_camera_agent.py --host 0.0.0.0 --port 8765

Then start the stack. The backend discovers the agent at
``HOST_CAMERA_AGENT_URL`` (default ``http://host.docker.internal:8765`` in
docker-compose.yml).

Security
--------
The agent has **no authentication** and serves live camera frames. Bind it to
``127.0.0.1`` only if the backend runs natively on the same machine. For Docker
Desktop, ``0.0.0.0`` is required so the container can reach the host; restrict
port 8765 with Windows Firewall to your local/Docker networks.

Only one client can open a physical camera at a time. If another application
(OBS, Teams, browser) owns the device, the stream returns 409 with a reason.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2

VIRTUAL_HINTS = ("obs virtual", "virtual camera", "loopback", "dummy", "obs-cam")
_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_FPS = 15
_DEFAULT_QUALITY = 80

_locks: dict[int, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(index: int) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(index, threading.Lock())


def _is_virtual(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in VIRTUAL_HINTS)


def _parse_ffmpeg_dshow_devices(output: str) -> list[str]:
    """Support both ffmpeg layouts: ``"Name" (video)`` and section headers."""
    names: list[str] = []
    in_video_section = False
    for line in output.splitlines():
        if "DirectShow video devices" in line:
            in_video_section = True
            continue
        if "DirectShow audio devices" in line:
            in_video_section = False
            continue
        if "Alternative name" in line:
            continue
        typed = re.search(r'"([^"]+)"\s*\((video|audio)\)\s*$', line)
        if typed:
            if typed.group(2) == "video":
                names.append(typed.group(1).strip())
            continue
        if in_video_section:
            match = re.search(r'"([^"]+)"\s*$', line)
            if match:
                names.append(match.group(1).strip())
    return names


def _ffmpeg_dshow_devices() -> list[str]:
    if shutil.which("ffmpeg") is None:
        return []
    try:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return _parse_ffmpeg_dshow_devices(completed.stderr or "")


def _pygrabber_devices() -> list[str]:
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        return [str(name) for name in FilterGraph().get_input_devices()]
    except Exception:
        return []


def enumerate_devices() -> list[str]:
    """DirectShow video device names, in OpenCV index order."""
    if platform.system() != "Windows":
        return []
    names = _pygrabber_devices()
    if names:
        return names
    return _ffmpeg_dshow_devices()


def _probe(index: int) -> dict[str, Any]:
    """Open one device briefly and report whether it is usable."""
    lock = _lock_for(index)
    if not lock.acquire(blocking=False):
        return {"available": True, "detail": "currently streaming to another client"}

    cap = None
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return {
                "available": False,
                "detail": "device could not be opened (in use or unavailable)",
            }
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if raw_fps and raw_fps > 0 else None
        ok, _ = cap.read()
        return {
            "available": bool(ok),
            "width": width,
            "height": height,
            "fps": fps,
            "detail": None if ok else "device opened but returned no frame",
        }
    finally:
        if cap is not None:
            cap.release()
        lock.release()


def device_payload(probe: bool = False) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for index, name in enumerate(enumerate_devices()):
        item: dict[str, Any] = {
            "index": index,
            "name": name,
            "backend": "directshow->mjpeg",
            "available": True,
            "is_virtual": _is_virtual(name),
            "stream_url": f"/stream/{index}.mjpg",
            "snapshot_url": f"/snapshot/{index}.jpg",
        }
        if probe:
            item.update(_probe(index))
        devices.append(item)
    return devices


class Handler(BaseHTTPRequestHandler):
    server_version = "CoastWatchHostCamera/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if self.server.verbose:  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _camera_index(self) -> int | None:
        match = re.match(r"^/(?:stream|snapshot)/(\d+)(?:\.(?:mjpg|jpg))?$", self.path)
        return int(match.group(1)) if match else None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            payload = {
                "ok": True,
                "platform": platform.platform(),
                "devices": len(enumerate_devices()),
            }
            self._json(payload)
            return
        if parsed.path == "/devices":
            probe = parse_qs(parsed.query).get("probe", ["0"])[0].lower() in {"1", "true", "yes"}
            self._json({"devices": device_payload(probe=probe)})
            return

        index = self._camera_index()
        if index is None:
            self._json({"error": "not found"}, status=404)
            return
        names = enumerate_devices()
        if index < 0 or index >= len(names):
            self._json({"error": f"camera index {index} is not present"}, status=404)
            return
        if parsed.path.startswith("/snapshot/"):
            self._snapshot(index)
        else:
            self._stream(index, parse_qs(parsed.query))

    def _open_capture(self, index: int, query: dict[str, list[str]]) -> Any:
        width = int(query.get("width", [_DEFAULT_WIDTH])[0])
        height = int(query.get("height", [_DEFAULT_HEIGHT])[0])
        fps = float(query.get("fps", [_DEFAULT_FPS])[0])
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _snapshot(self, index: int) -> None:
        lock = _lock_for(index)
        if not lock.acquire(blocking=False):
            self._json(
                {"error": f"camera {index} is already streaming to another client"},
                status=409,
            )
            return
        cap = None
        try:
            cap = self._open_capture(index, {})
            if cap is None:
                self._json(
                    {"error": f"camera {index} could not be opened (in use or unavailable)"},
                    status=409,
                )
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                self._json({"error": f"camera {index} returned no frame"}, status=503)
                return
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _DEFAULT_QUALITY])
            if not ok:
                self._json({"error": "JPEG encode failed"}, status=500)
                return
            body = encoded.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        finally:
            if cap is not None:
                cap.release()
            lock.release()

    def _stream(self, index: int, query: dict[str, list[str]]) -> None:
        lock = _lock_for(index)
        if not lock.acquire(blocking=False):
            self._json(
                {"error": f"camera {index} is already streaming to another client"},
                status=409,
            )
            return

        cap = None
        try:
            cap = self._open_capture(index, query)
            if cap is None:
                self._json(
                    {"error": f"camera {index} could not be opened (in use or unavailable)"},
                    status=409,
                )
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                self._json({"error": f"camera {index} returned no frame"}, status=503)
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            quality = int(query.get("quality", [_DEFAULT_QUALITY])[0])
            while True:
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, max(10, min(95, quality))]
                )
                if not ok:
                    break
                data = encoded.tobytes()
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
        finally:
            if cap is not None:
                cap.release()
            lock.release()


class AgentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, verbose: bool = False) -> None:
        super().__init__(address, Handler)
        self.verbose = verbose


def main() -> None:
    parser = argparse.ArgumentParser(description="CoastWatch Windows host camera agent")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default: 8765)")
    parser.add_argument("--verbose", action="store_true", help="log every HTTP request")
    args = parser.parse_args()

    names = enumerate_devices()
    print(f"CoastWatch host camera agent on http://{args.host}:{args.port}")
    print(f"Platform: {platform.platform()}")
    if names:
        print("Detected cameras:")
        for index, name in enumerate(names):
            print(f"  [{index}] {name}")
    else:
        print("No DirectShow camera names detected.")
        print("Install ffmpeg or pygrabber on the host so names can be enumerated.")

    server = AgentServer((args.host, args.port), verbose=args.verbose)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
