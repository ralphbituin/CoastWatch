"""Streaming frame sampling for large video archives.

Design rules:

* never load the whole video or all frames into memory;
* always decode sequentially with ``cv2.VideoCapture.grab()`` and only
  ``retrieve()`` a target frame;
* compute the target frame index from **video metadata** (``elapsed * fps``),
  never from wall-clock processing time;
* preserve the native frame (no resizing here);
* make resume explicit: the last committed frame index is passed in, the
  capture seeks there once, and decoding then continues sequentially;
* record extraction failures instead of silently skipping them.

The expected number of samples for ``webcam_6h.mp4`` at 30 s is 517
(``0, 30, ..., 15480`` seconds).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class VideoMetadata:
    path: str
    file_size_bytes: int
    duration_seconds: float
    fps: float
    average_fps: float
    frame_count: int
    width: int
    height: int
    video_codec: str | None = None
    audio_codec: str | None = None
    container_tags: dict[str, str] = field(default_factory=dict)
    creation_time: str | None = None
    probe_source: str = "opencv"

    @property
    def duration_text(self) -> str:
        seconds = int(round(self.duration_seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours}h {minutes}m {secs}s"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_size_bytes": self.file_size_bytes,
            "duration_seconds": round(self.duration_seconds, 3),
            "duration_text": self.duration_text,
            "fps": self.fps,
            "average_fps": round(self.average_fps, 6),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "container_tags": self.container_tags,
            "creation_time": self.creation_time,
            "probe_source": self.probe_source,
        }


def _ffprobe(path: Path) -> dict[str, Any] | None:
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        (
            "format=duration,size,format_name:format_tags:"
            "stream=index,codec_name,codec_type,width,height,"
            "r_frame_rate,avg_frame_rate,nb_frames,duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def _fraction(text: str | None) -> float | None:
    if not text or "/" not in text:
        return None
    numerator, denominator = text.split("/", 1)
    try:
        value = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def probe_video(path: str | Path) -> VideoMetadata:
    """Return authoritative container metadata, or raise if unreadable."""
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    probe = _ffprobe(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV cannot open video: {video_path}")
    try:
        cv_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        cv_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cv_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        cv_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    duration = (cv_frames / cv_fps) if cv_fps > 0 and cv_frames > 0 else 0.0
    average_fps = cv_fps
    fps = cv_fps
    width, height = cv_width, cv_height
    video_codec = audio_codec = None
    tags: dict[str, str] = {}
    creation_time = None
    source = "opencv"

    if probe:
        source = "ffprobe+opencv"
        fmt = probe.get("format", {})
        with contextlib.suppress(TypeError, ValueError):
            duration = float(fmt.get("duration") or duration)
        tags = {str(k): str(v) for k, v in (fmt.get("tags") or {}).items()}
        creation_time = tags.get("creation_time") or tags.get("creation_time_utc")
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video" and video_codec is None:
                video_codec = stream.get("codec_name")
                width = int(stream.get("width") or width)
                height = int(stream.get("height") or height)
                fps = _fraction(stream.get("r_frame_rate")) or fps
                average_fps = _fraction(stream.get("avg_frame_rate")) or average_fps
                with contextlib.suppress(TypeError, ValueError):
                    cv_frames = int(stream.get("nb_frames") or cv_frames)
            elif stream.get("codec_type") == "audio" and audio_codec is None:
                audio_codec = stream.get("codec_name")

    if fps <= 0 or cv_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"incomplete video metadata for {video_path}")
    if duration <= 0:
        duration = cv_frames / fps

    return VideoMetadata(
        path=str(video_path),
        file_size_bytes=video_path.stat().st_size,
        duration_seconds=duration,
        fps=fps,
        average_fps=average_fps,
        frame_count=cv_frames,
        width=width,
        height=height,
        video_codec=video_codec,
        audio_codec=audio_codec,
        container_tags=tags,
        creation_time=creation_time,
        probe_source=source,
    )


def stable_recording_id(path: str | Path, metadata: VideoMetadata | None = None) -> str:
    """Filesystem identity hash; stable across renames of the *same* file.

    Uses size + mtime + resolved path. This is not a content hash; a full
    SHA-256 is optional and stored separately when requested.
    """
    video_path = Path(path).resolve()
    stat = video_path.stat()
    identity = f"{video_path}|{stat.st_size}|{stat.st_mtime_ns}"
    return "rec-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def target_samples(
    metadata: VideoMetadata,
    *,
    sample_interval_seconds: float,
    start_elapsed: float = 0.0,
    end_elapsed: float | None = None,
    max_samples: int | None = None,
) -> list[tuple[float, int]]:
    """Return ``(elapsed_seconds, frame_index)`` pairs from video metadata."""
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be > 0")
    duration = metadata.duration_seconds
    start = max(0.0, float(start_elapsed))
    end = min(duration, float(end_elapsed)) if end_elapsed is not None else duration
    if end <= start:
        return []
    elapsed = start
    samples: list[tuple[float, int]] = []
    seen: set[int] = set()
    # Include the start sample even when start is not a multiple of the interval.
    while elapsed <= end + 1e-6:
        frame_index = int(round(elapsed * metadata.fps))
        if 0 <= frame_index < metadata.frame_count and frame_index not in seen:
            seen.add(frame_index)
            samples.append((round(elapsed, 3), frame_index))
            if max_samples is not None and len(samples) >= int(max_samples):
                break
        elapsed += sample_interval_seconds
    return samples


@dataclass(slots=True)
class SampledFrame:
    frame_index: int
    elapsed_seconds: float
    status: str  # "ok" | "grab_failed" | "retrieve_failed"
    frame: np.ndarray | None = None
    error: str | None = None


def iter_sampled_frames(
    path: str | Path,
    metadata: VideoMetadata,
    *,
    sample_interval_seconds: float = 30.0,
    start_elapsed: float = 0.0,
    end_elapsed: float | None = None,
    max_samples: int | None = None,
    resume_from_frame: int | None = None,
    skip_frame_indices: set[int] | None = None,
) -> Iterator[SampledFrame]:
    """Yield sampled frames while decoding the archive sequentially."""
    samples = target_samples(
        metadata,
        sample_interval_seconds=sample_interval_seconds,
        start_elapsed=start_elapsed,
        end_elapsed=end_elapsed,
        max_samples=max_samples,
    )
    skip = set(skip_frame_indices or set())
    targets = [(elapsed, index) for elapsed, index in samples if index not in skip]
    if not targets:
        return

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV cannot open video: {path}")
    target_by_index = {index: elapsed for elapsed, index in targets}
    first_target = targets[0][1]
    if resume_from_frame is not None and resume_from_frame > 0:
        seek_to = max(resume_from_frame, first_target)
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(seek_to))
        current = seek_to
    else:
        current = 0

    pending = sorted(target_by_index)
    position = 0
    try:
        while position < len(pending):
            next_target = pending[position]
            if current > next_target:
                # A seek landed past the target; report it rather than looping.
                yield SampledFrame(
                    frame_index=next_target,
                    elapsed_seconds=target_by_index[next_target],
                    status="grab_failed",
                    error=f"seek overshot target frame {next_target} (at {current})",
                )
                position += 1
                continue
            while current < next_target:
                if not capture.grab():
                    yield SampledFrame(
                        frame_index=next_target,
                        elapsed_seconds=target_by_index[next_target],
                        status="grab_failed",
                        error=f"video ended or grab failed before frame {next_target}",
                    )
                    return
                current += 1
            elapsed = target_by_index[next_target]
            if not capture.grab():
                yield SampledFrame(
                    frame_index=next_target,
                    elapsed_seconds=elapsed,
                    status="grab_failed",
                    error=f"video ended or grab failed at frame {next_target}",
                )
                return
            current += 1
            ok, frame = capture.retrieve()
            if not ok or frame is None or frame.size == 0:
                yield SampledFrame(
                    frame_index=next_target,
                    elapsed_seconds=elapsed,
                    status="retrieve_failed",
                    error=f"frame {next_target} could not be decoded",
                )
            else:
                yield SampledFrame(
                    frame_index=next_target,
                    elapsed_seconds=elapsed,
                    status="ok",
                    frame=frame,
                )
            position += 1
    finally:
        capture.release()


def frame_sha256(frame: np.ndarray) -> str:
    """Hash the decoded pixels (not the re-encoded file bytes)."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


def sample_count_estimate(metadata: VideoMetadata, interval_seconds: float) -> int:
    return int(math.floor(metadata.duration_seconds / interval_seconds)) + 1
