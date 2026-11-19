"""Common camera provider interface (Phase 3).

Both RTSP and USB providers return standardized frames with timestamps and
camera metadata. Upload cameras read previously captured stills from disk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class CameraError(Exception):
    """Base class for camera acquisition failures."""


class CameraConnectionError(CameraError):
    """The camera could not be opened or reached."""


class FrameUnavailableError(CameraError):
    """A frame was requested but none could be acquired."""


@dataclass(slots=True)
class CameraMetadata:
    camera_id: int
    camera_type: str  # "rtsp" | "usb" | "upload"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Frame:
    """A single acquired frame plus its metadata."""

    image: Any  # numpy.ndarray (H, W, 3) BGR
    captured_at: datetime
    metadata: CameraMetadata
    sequence: int = 0


@dataclass(slots=True)
class CameraStatus:
    connected: bool
    healthy: bool
    message: str
    last_frame_at: datetime | None = None
    frames_captured: int = 0


class CameraProvider(ABC):
    """Abstract camera interface shared by all transports."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read_frame(self) -> Frame:
        """Return the next frame, or raise :class:`FrameUnavailableError`."""
        ...

    @abstractmethod
    def get_status(self) -> CameraStatus: ...

    @abstractmethod
    def get_metadata(self) -> CameraMetadata: ...
