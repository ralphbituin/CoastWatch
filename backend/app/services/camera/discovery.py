"""Camera device discovery.

Two sources are supported:

* **local** — devices visible to the backend process itself:
  - Linux: ``/dev/video*`` with names read from
    ``/sys/class/video4linux/videoN/name`` (V4L2).
  - Windows: DirectShow names from ``pygrabber`` when installed, otherwise
    parsed from ``ffmpeg -f dshow -list_devices``.
  - macOS: ``ffmpeg -f avfoundation -list_devices``.

* **host_agent** — a small helper process running on the Docker host (see
  ``tools/host_camera_agent.py``). Docker Desktop on Windows cannot pass a USB
  webcam into a Linux container, so the host agent enumerates DirectShow
  devices and re-serves them as MJPEG. The backend fetches its ``/devices``
  list and registers the stream URL as an ordinary network (RTSP/HTTP) camera.

Discovery never invents a device. A probe that cannot open a node is reported
as ``available=false``; the UI lets the operator decide what to register.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import urlopen

import cv2

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Names that are clearly software/virtual devices. They are still listed (OBS
# Virtual Camera is useful), just marked so the UI can label them.
VIRTUAL_HINTS = ("obs virtual", "virtual camera", "loopback", "dummy", "obs-cam")


@dataclass(slots=True)
class DiscoveredCamera:
    """One camera device visible from the backend or the host agent."""

    index: int | None
    name: str
    source: str  # "local" | "host_agent"
    camera_type: str  # "usb" | "rtsp"
    backend: str
    available: bool = True
    is_virtual: bool = False
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    device_index: int | None = None
    stream_url: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "source": self.source,
            "camera_type": self.camera_type,
            "backend": self.backend,
            "available": self.available,
            "is_virtual": self.is_virtual,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "device_index": self.device_index,
            "stream_url": self.stream_url,
            "detail": self.detail,
        }


@dataclass(slots=True)
class DiscoveryResult:
    devices: list[DiscoveredCamera] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    host_agent_configured: bool = False
    host_agent_available: bool = False


def _is_virtual(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in VIRTUAL_HINTS)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _probe(index: int, backend: int) -> tuple[bool, float | None, int | None, int | None] | None:
    """Try to open one device; return ``(ok, fps, width, height)``.

    ``None`` means the probe itself is unavailable (OpenCV backend missing).
    Opening a camera can block, so callers only run this on explicit request.
    """
    try:
        cap = cv2.VideoCapture(index, backend)
    except Exception:  # pragma: no cover - backend dependent
        return None
    try:
        if not cap.isOpened():
            return False, None, None, None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if raw_fps and raw_fps > 0 else None
        return True, fps, width, height
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Linux / V4L2
# ---------------------------------------------------------------------------


def _linux_local_cameras(probe: bool) -> list[DiscoveredCamera]:
    root = Path("/sys/class/video4linux")
    if not root.is_dir():
        return []

    nodes: list[tuple[int, str, str]] = []
    for entry in root.glob("video*"):
        suffix = entry.name[len("video") :]
        if not suffix.isdigit():
            continue
        index = int(suffix)
        device_path = Path(f"/dev/video{index}")
        if not device_path.exists():
            continue
        try:
            name = (entry / "name").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            name = ""
        nodes.append((index, name or f"Video device {index}", str(device_path)))

    nodes.sort(key=lambda item: item[0])
    cameras: list[DiscoveredCamera] = []
    for index, name, device_path in nodes:
        detail = device_path
        available = True
        fps: float | None = None
        width: int | None = None
        height: int | None = None
        if probe:
            result = _probe(index, cv2.CAP_V4L2)
            if result is not None:
                available, fps, width, height = result
                if not available:
                    detail = f"{device_path} (could not be opened)"
        cameras.append(
            DiscoveredCamera(
                index=index,
                name=name,
                source="local",
                camera_type="usb",
                backend="v4l2",
                available=available,
                is_virtual=_is_virtual(name),
                width=width,
                height=height,
                fps=fps,
                device_index=index,
                detail=detail,
            )
        )
    return cameras


# ---------------------------------------------------------------------------
# Windows / DirectShow
# ---------------------------------------------------------------------------


def _parse_ffmpeg_dshow_devices(output: str) -> list[str]:
    """Extract video device names from ``ffmpeg -f dshow -list_devices`` output.

    ffmpeg versions differ: newer builds append ``(video)``/``(audio)`` to each
    line, older ones rely on "DirectShow video devices" / "… audio devices"
    section headers. Support both.
    """
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
    if not _ffmpeg_available():
        return []
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - host dependent
        logger.debug("ffmpeg dshow enumeration failed: %s", exc)
        return []
    return _parse_ffmpeg_dshow_devices(completed.stderr or "")


def _pygrabber_devices() -> list[str]:
    try:  # Optional Windows-only dependency.
        from pygrabber.dshow_graph import FilterGraph  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        return [str(name) for name in FilterGraph().get_input_devices()]
    except Exception as exc:  # pragma: no cover - COM dependent
        logger.debug("pygrabber enumeration failed: %s", exc)
        return []


def _windows_local_cameras(probe: bool) -> list[DiscoveredCamera]:
    names = _pygrabber_devices()
    backend = "directshow"
    if not names:
        names = _ffmpeg_dshow_devices()
        backend = "directshow (ffmpeg)"

    if not names:
        # No DirectShow-name source is available. Do not invent model names;
        # report nothing rather than guessing at indices.
        return []

    cameras: list[DiscoveredCamera] = []
    for index, name in enumerate(names):
        available = True
        fps: float | None = None
        width: int | None = None
        height: int | None = None
        detail: str | None = None
        if probe:
            result = _probe(index, cv2.CAP_DSHOW)
            if result is not None:
                available, fps, width, height = result
                if not available:
                    detail = "DirectShow device could not be opened (in use or unavailable)"
        cameras.append(
            DiscoveredCamera(
                index=index,
                name=name,
                source="local",
                camera_type="usb",
                backend=backend,
                available=available,
                is_virtual=_is_virtual(name),
                width=width,
                height=height,
                fps=fps,
                device_index=index,
                detail=detail,
            )
        )
    return cameras


# ---------------------------------------------------------------------------
# macOS / AVFoundation
# ---------------------------------------------------------------------------


def _parse_ffmpeg_avfoundation_devices(output: str) -> list[tuple[int, str]]:
    devices: list[tuple[int, str]] = []
    for line in output.splitlines():
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if not match:
            continue
        label = match.group(2).strip()
        if label.lower().startswith("avfoundation"):
            continue
        devices.append((int(match.group(1)), label))
    return devices


def _darwin_local_cameras(probe: bool) -> list[DiscoveredCamera]:
    if not _ffmpeg_available():
        return []
    try:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - host dependent
        logger.debug("ffmpeg avfoundation enumeration failed: %s", exc)
        return []
    cameras: list[DiscoveredCamera] = []
    for index, name in _parse_ffmpeg_avfoundation_devices(completed.stderr or ""):
        available = True
        fps: float | None = None
        width: int | None = None
        height: int | None = None
        if probe:
            result = _probe(index, cv2.CAP_AVFOUNDATION)
            if result is not None:
                available, fps, width, height = result
        cameras.append(
            DiscoveredCamera(
                index=index,
                name=name,
                source="local",
                camera_type="usb",
                backend="avfoundation",
                available=available,
                is_virtual=_is_virtual(name),
                width=width,
                height=height,
                fps=fps,
                device_index=index,
            )
        )
    return cameras


def _local_cameras(probe: bool) -> list[DiscoveredCamera]:
    system = platform.system()
    if system == "Linux":
        cameras = _linux_local_cameras(probe)
    elif system == "Windows":
        cameras = _windows_local_cameras(probe)
    elif system == "Darwin":
        cameras = _darwin_local_cameras(probe)
    else:  # pragma: no cover - unsupported platform
        cameras = []

    # V4L2 often exposes several nodes per physical camera with the same name.
    # Keep the first openable node per name (or the first node when not probing).
    unique: dict[str, DiscoveredCamera] = {}
    for camera in cameras:
        existing = unique.get(camera.name)
        if existing is None:
            unique[camera.name] = camera
            continue
        preferred = camera.available and (
            not existing.available or (camera.index or 0) < (existing.index or 0)
        )
        if preferred:
            unique[camera.name] = camera
    return sorted(unique.values(), key=lambda item: (item.index is None, item.index or 0))


# ---------------------------------------------------------------------------
# Host agent (Windows Docker bridge)
# ---------------------------------------------------------------------------


def _host_agent_cameras(
    base_url: str, timeout: float, probe: bool = False
) -> list[DiscoveredCamera]:
    url = urljoin(base_url.rstrip("/") + "/", "devices")
    if probe:
        url = f"{url}?probe=1"
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - configured URL
        payload = json.loads(response.read().decode("utf-8"))

    raw_devices = payload.get("devices") if isinstance(payload, dict) else payload
    if not isinstance(raw_devices, list):
        raise ValueError("host camera agent returned an unexpected /devices payload")

    cameras: list[DiscoveredCamera] = []
    root = base_url.rstrip("/")
    for item in raw_devices:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        name = str(item.get("name") or f"Camera {index}")
        stream = item.get("stream_url") or item.get("url")
        if isinstance(stream, str) and stream.startswith("/"):
            stream = root + stream
        if not isinstance(stream, str) or not stream:
            continue
        cameras.append(
            DiscoveredCamera(
                index=int(index) if isinstance(index, int) else None,
                name=name,
                source="host_agent",
                camera_type="rtsp",  # the agent serves MJPEG; the network provider reads it
                backend=str(item.get("backend") or "dshow->mjpeg"),
                available=bool(item.get("available", True)),
                is_virtual=_is_virtual(name) or bool(item.get("is_virtual")),
                width=item.get("width") if isinstance(item.get("width"), int) else None,
                height=item.get("height") if isinstance(item.get("height"), int) else None,
                fps=float(item["fps"]) if isinstance(item.get("fps"), int | float) else None,
                device_index=None,  # host indices are not valid inside the container
                stream_url=stream,
                detail=item.get("detail"),
            )
        )
    return cameras


def discover_cameras(*, probe: bool = False, include_host_agent: bool = True) -> DiscoveryResult:
    """Enumerate camera devices without registering anything.

    A failure to reach the configured host agent is reported in ``warnings`` and
    does not hide locally visible devices.
    """
    result = DiscoveryResult()
    try:
        result.devices.extend(_local_cameras(probe))
    except Exception as exc:  # pragma: no cover - defensive, platform dependent
        logger.warning("local camera discovery failed: %s", exc)
        result.warnings.append(f"local discovery failed: {exc}")

    settings = get_settings()
    base_url = (settings.host_camera_agent_url or "").strip()
    result.host_agent_configured = bool(base_url)
    if include_host_agent and base_url:
        try:
            agent_devices = _host_agent_cameras(
                base_url, settings.host_camera_agent_timeout_seconds, probe
            )
            result.devices.extend(agent_devices)
            result.host_agent_available = True
        except Exception as exc:
            logger.info("host camera agent unavailable at %s: %s", base_url, exc)
            result.warnings.append(
                f"host camera agent unavailable at {base_url}: {exc}. "
                "Start tools/host_camera_agent.py on the host to see webcams."
            )

    # When the backend runs natively on the same Windows host as the agent, the
    # same webcam appears in both lists. Prefer the direct local device; the
    # host-agent entry only matters when the backend cannot see it (Docker).
    local_names = {item.name.strip().lower() for item in result.devices if item.source == "local"}
    result.devices = [
        item
        for item in result.devices
        if item.source == "local" or item.name.strip().lower() not in local_names
    ]

    # Stable ordering: local first, then host agent, alphabetical.
    result.devices.sort(
        key=lambda item: (item.source != "local", item.name.lower(), item.index or 0)
    )
    return result
