"""Frame preprocessing and quality assessment (Phase 3).

Frame sampling, resizing, noise reduction, contrast enhancement, optional
stabilization and region-of-interest selection — plus frame-quality checks
used to reject unusable captures before they reach the vision stages.

Quality checks are deliberately conservative heuristics on pixel statistics;
they are not a substitute for human review of a camera site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import cv2
import numpy as np

# --- Thresholds (documented heuristics, tune per deployment) ---
_BLACK_MEAN_THRESHOLD = 8.0
# Mean absolute per-pixel difference below this means the scene did not change.
_FROZEN_DIFF_THRESHOLD = 1.0
# Variance of the Laplacian below this means the frame carries no usable detail.
_BLUR_LAPLACIAN_VAR_THRESHOLD = 100.0
# Standard deviation below this (but not black) suggests a covered/obstructed lens.
_OBSTRUCTED_STD_THRESHOLD = 12.0


@dataclass(slots=True)
class PreprocessConfig:
    """Operator-tunable knobs for :func:`preprocess`."""

    # Resize output to exactly (width, height) if both are set.
    width: int | None = None
    height: int | None = None
    # Otherwise downscale so the longest side is at most this many pixels.
    max_dimension: int | None = 1920
    denoise: bool = True
    enhance_contrast: bool = True
    # Optional region of interest as (x, y, w, h) in pixels. Applied before resize.
    roi: tuple[int, int, int, int] | None = None
    # Keep every Nth frame when sampling a stream; 1 keeps all frames.
    sample_every: int = 1


@dataclass(slots=True)
class FrameQuality:
    score: float
    is_black: bool = False
    is_frozen: bool = False
    is_blurry: bool = False
    is_obstructed: bool = False
    is_timestamp_inconsistent: bool = False
    flags: list[str] = field(default_factory=list)


def preprocess(frame: Any, config: PreprocessConfig | None = None) -> Any:
    """Return a cleaned copy of ``frame`` (a BGR ``numpy.ndarray``).

    Steps, in order: ROI crop → resize → noise reduction → contrast
    enhancement (CLAHE on the L channel of LAB).
    """
    if frame is None:
        raise ValueError("frame must not be None")
    cfg = config or PreprocessConfig()
    image = frame

    if cfg.roi is not None:
        x, y, w, h = cfg.roi
        if w <= 0 or h <= 0:
            raise ValueError("roi width/height must be positive")
        frame_h, frame_w = image.shape[:2]
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            raise ValueError(
                f"roi (x={x}, y={y}, w={w}, h={h}) lies outside the {frame_w}x{frame_h} frame"
            )
        image = image[y : y + h, x : x + w]

    if cfg.width is not None and cfg.height is not None:
        image = cv2.resize(image, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
    elif cfg.max_dimension:
        longest = max(image.shape[:2])
        if longest > cfg.max_dimension:
            scale = cfg.max_dimension / longest
            new_size = (
                max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))),
            )
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    if cfg.denoise:
        image = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)

    if cfg.enhance_contrast:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        image = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    return image


def assess_quality(
    frame: Any,
    *,
    previous_frame: Any | None = None,
    captured_at: datetime | None = None,
    previous_captured_at: datetime | None = None,
) -> FrameQuality:
    """Score a frame and flag unusable captures.

    ``previous_frame`` enables the frozen-frame check; ``captured_at`` /
    ``previous_captured_at`` enable the timestamp-consistency check (frames
    must arrive in monotonic, non-decreasing order).
    """
    if frame is None:
        raise ValueError("frame must not be None")

    flags: list[str] = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    mean_intensity = float(gray.mean())
    std_intensity = float(gray.std())
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_black = mean_intensity < _BLACK_MEAN_THRESHOLD
    if is_black:
        flags.append("black_frame")

    is_frozen = False
    if previous_frame is not None:
        prev_gray = (
            cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            if previous_frame.ndim == 3
            else previous_frame
        )
        if prev_gray.shape == gray.shape:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            is_frozen = diff < _FROZEN_DIFF_THRESHOLD
            if is_frozen:
                flags.append("frozen_frame")

    is_blurry = laplacian_var < _BLUR_LAPLACIAN_VAR_THRESHOLD
    if is_blurry:
        flags.append("severe_blur")

    is_obstructed = (not is_black) and std_intensity < _OBSTRUCTED_STD_THRESHOLD
    if is_obstructed:
        flags.append("obstruction")

    is_timestamp_inconsistent = (
        captured_at is not None
        and previous_captured_at is not None
        and captured_at <= previous_captured_at
    )
    if is_timestamp_inconsistent:
        flags.append("timestamp_inconsistent")

    score = 1.0
    if is_black:
        score -= 0.6
    if is_frozen:
        score -= 0.4
    if is_blurry:
        score -= 0.3
    if is_obstructed:
        score -= 0.5
    if is_timestamp_inconsistent:
        score -= 0.2
    score = round(min(1.0, max(0.0, score)), 3)

    return FrameQuality(
        score=score,
        is_black=is_black,
        is_frozen=is_frozen,
        is_blurry=is_blurry,
        is_obstructed=is_obstructed,
        is_timestamp_inconsistent=is_timestamp_inconsistent,
        flags=flags,
    )
