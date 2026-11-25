"""Shoreline extraction from a three-class segmentation mask.

This module is intentionally separate from segmentation and from any physical
tide/bathymetry model. It answers one question: *where is the Ocean/Beach
interface in image coordinates?*

Class ids: 0 = Beach, 1 = Ocean, 2 = Sky, 255 = ignore.

The production implementation is
``backend/app/services/vision/shoreline.py:extract_shoreline_from_classes``.
This reference mirrors its class-aware scanline algorithm so offline evaluation
and visualisation produce the same geometry:

* close thin occluders in the Ocean mask only;
* keep every large Ocean and Beach component, rejecting small artifacts;
* pick one Ocean pixel per scanline that touches Beach on the land side, so
  separate interface branches can never be joined by a median;
* ignore the frame margin and out-of-view seaward edges;
* median-filter along the scanline, split at gaps, drop short segments.

The Sky/Ocean horizon is never used: Sky is not Beach, so no contact exists
there. Coordinates are original-frame pixels. Calibration is a separate hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

BEACH = 0
OCEAN = 1
SKY = 2
IGNORE = 255

ORIENTATIONS = ("land_below", "land_above", "land_left", "land_right")


@dataclass
class ShorelineResult:
    """Image-space shoreline result. All distances are pixels."""

    points: np.ndarray  # concatenation of ordered segments, shape (N, 2) as (x, y)
    per_column: np.ndarray  # alias of points (legacy name)
    boundary_mask: np.ndarray  # uint8 HxW, 1 = scanline shoreline raster
    valid_columns: int
    ocean_pixels: int
    beach_pixels: int
    rejected_ocean_components: int
    rejected_beach_components: int
    segments: list[np.ndarray] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_world(self, homography: np.ndarray) -> np.ndarray:
        """Apply a 3x3 image-to-world homography (calibration hook)."""
        matrix = np.asarray(homography, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("homography must be a 3x3 matrix")
        if self.points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        xy = self.points.astype(np.float64)
        homogeneous = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
        world = homogeneous @ matrix.T
        w = world[:, 2:3]
        if np.any(np.abs(w) < 1e-12):
            raise ValueError("homography maps a shoreline point to infinity")
        return world[:, :2] / w

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid_columns": self.valid_columns,
            "ocean_pixels": self.ocean_pixels,
            "beach_pixels": self.beach_pixels,
            "rejected_ocean_components": self.rejected_ocean_components,
            "rejected_beach_components": self.rejected_beach_components,
            "points": self.points.tolist(),
            "per_column": self.per_column.tolist(),
            "segments": [segment.tolist() for segment in self.segments],
            "metadata": self.metadata,
        }


def _keep_large_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool), 0, 0
    keep = np.zeros(count, dtype=bool)
    for label_id in range(1, count):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= int(min_area):
            keep[label_id] = True
    area = int(stats[keep, cv2.CC_STAT_AREA].sum()) if bool(keep.any()) else 0
    rejected = int(count - 1 - int(keep.sum()))
    if not bool(keep.any()):
        return np.zeros_like(mask, dtype=bool), 0, rejected
    return keep[labels], area, rejected


def _land_fraction(beach: np.ndarray, orientation: str, depth: int) -> np.ndarray:
    height, width = beach.shape
    depth = max(
        1,
        min(int(depth), height if orientation in ("land_below", "land_above") else width),
    )
    fraction = np.zeros((height, width), dtype=np.float32)
    for step in range(1, depth + 1):
        if orientation == "land_below":
            fraction[: height - step] += beach[step:]
        elif orientation == "land_above":
            fraction[step:] += beach[: height - step]
        elif orientation == "land_left":
            fraction[:, step:] += beach[:, : width - step]
        else:
            fraction[:, : width - step] += beach[:, step:]
    return fraction / float(depth)


def _infer_orientation(ocean: np.ndarray, beach: np.ndarray, *, depth: int) -> str:
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = ocean & (cv2.erode(ocean.astype(np.uint8), kernel, iterations=1) == 0)
    if not boundary.any():
        return "land_below"
    counts = {
        orientation: int(
            np.count_nonzero(boundary & (_land_fraction(beach, orientation, depth) >= 0.5))
        )
        for orientation in ORIENTATIONS
    }
    foreground = ("land_below", "land_left", "land_right")
    best_side = max(foreground, key=lambda name: counts[name])
    best_count = max(1, counts[best_side])
    if counts["land_below"] > 0 and counts["land_below"] >= 0.75 * best_count:
        return "land_below"
    return best_side


def _scanline_boundary(
    ocean: np.ndarray,
    land_fraction: np.ndarray,
    *,
    orientation: str,
    margin: int,
) -> np.ndarray:
    height, width = ocean.shape
    margin = max(0, int(margin))
    points: list[tuple[float, float]] = []
    if orientation in ("land_below", "land_above"):
        for x in range(margin, width - margin):
            rows = np.flatnonzero(ocean[:, x])
            if rows.size == 0:
                continue
            if orientation == "land_below":
                if int(rows.max()) >= height - margin:
                    continue
                valid = rows[
                    (rows >= margin)
                    & (rows < height - margin)
                    & (land_fraction[rows, x] >= 0.5)
                ]
                if valid.size:
                    points.append((float(x), float(valid.max())))
            else:
                if int(rows.min()) < margin:
                    continue
                valid = rows[
                    (rows >= margin)
                    & (rows < height - margin)
                    & (land_fraction[rows, x] >= 0.5)
                ]
                if valid.size:
                    points.append((float(x), float(valid.min())))
        return np.asarray(points, dtype=np.float64) if points else np.empty((0, 2))

    for y in range(margin, height - margin):
        cols = np.flatnonzero(ocean[y, :])
        if cols.size == 0:
            continue
        if orientation == "land_left":
            if int(cols.min()) < margin:
                continue
            valid = cols[
                (cols >= margin)
                & (cols < width - margin)
                & (land_fraction[y, cols] >= 0.5)
            ]
            if valid.size:
                points.append((float(valid.min()), float(y)))
        else:
            if int(cols.max()) >= width - margin:
                continue
            valid = cols[
                (cols >= margin)
                & (cols < width - margin)
                & (land_fraction[y, cols] >= 0.5)
            ]
            if valid.size:
                points.append((float(valid.max()), float(y)))
    return np.asarray(points, dtype=np.float64) if points else np.empty((0, 2))


def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values
    window = min(int(window), int(values.size))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    half = window // 2
    return np.asarray(
        [
            float(np.median(values[max(0, i - half) : min(values.size, i + half + 1)]))
            for i in range(values.size)
        ],
        dtype=np.float64,
    )


def _build_segments(
    points: np.ndarray,
    *,
    orientation: str,
    max_gap: float,
    min_segment_px: int,
    smoothing: int,
) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    vertical = orientation in ("land_below", "land_above")
    scan = points[:, 0] if vertical else points[:, 1]
    values = points[:, 1] if vertical else points[:, 0]
    smoothed = values.astype(np.float64, copy=True)
    start = 0
    for index in range(1, len(points) + 1):
        if index == len(points) or scan[index] - scan[index - 1] != 1.0:
            smoothed[start:index] = _median_filter_1d(values[start:index], smoothing)
            start = index
    ordered = points.copy()
    ordered[:, 1 if vertical else 0] = smoothed

    segments: list[np.ndarray] = []
    start = 0
    for index in range(1, len(ordered)):
        if float(np.linalg.norm(ordered[index] - ordered[index - 1])) > max_gap:
            segments.append(ordered[start:index])
            start = index
    segments.append(ordered[start:])
    return [segment for segment in segments if len(segment) >= max(2, int(min_segment_px))]


def extract_shoreline(
    mask: np.ndarray,
    *,
    ocean_id: int = OCEAN,
    beach_id: int = BEACH,
    ignore_index: int = IGNORE,
    min_ocean_component_px: int | None = None,
    min_beach_component_px: int | None = None,
    close_kernel: int | None = None,
    orientation: str = "auto",
    margin: int | None = None,
    max_gap: float | None = None,
    min_segment_px: int | None = None,
    smoothing: int | None = None,
    min_valid_columns: int = 8,
) -> ShorelineResult:
    """Extract ordered Ocean/Beach segments from a class-id mask.

    Thresholds default to frame-relative values matching the backend settings.
    ``smoothing`` is an optional absolute median-kernel width; ``close_kernel``
    is an optional absolute closing kernel. Neither modifies the input mask.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {array.shape}")
    height, width = array.shape
    frame_pixels = height * width

    ocean_raw = array == ocean_id
    beach_raw = array == beach_id
    empty = np.zeros((height, width), dtype=np.uint8)
    if not ocean_raw.any() or not beach_raw.any():
        reason = "no ocean pixels" if not ocean_raw.any() else "no beach pixels"
        return ShorelineResult(
            points=np.empty((0, 2), dtype=np.float64),
            per_column=np.empty((0, 2), dtype=np.float64),
            boundary_mask=empty,
            valid_columns=0,
            ocean_pixels=int(ocean_raw.sum()),
            beach_pixels=int(beach_raw.sum()),
            rejected_ocean_components=0,
            rejected_beach_components=0,
            metadata={"available": False, "reason": reason},
        )

    ocean_min = int(min_ocean_component_px or max(1, round(frame_pixels * 0.005)))
    beach_min = int(min_beach_component_px or max(1, round(frame_pixels * 0.0005)))
    kernel_px = int(close_kernel or max(3, round(min(height, width) * 0.03)))
    if kernel_px % 2 == 0:
        kernel_px += 1
    if kernel_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
        ocean_closed = cv2.morphologyEx(
            ocean_raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    else:
        ocean_closed = ocean_raw

    ocean, ocean_area, rejected_ocean = _keep_large_components(ocean_closed, ocean_min)
    beach, beach_area, rejected_beach = _keep_large_components(beach_raw, beach_min)
    if not ocean.any() or ocean_area < ocean_min or not beach.any() or beach_area < beach_min:
        reason = (
            "dominant ocean component too small"
            if ocean_area < ocean_min
            else "no usable beach"
        )
        return ShorelineResult(
            points=np.empty((0, 2), dtype=np.float64),
            per_column=np.empty((0, 2), dtype=np.float64),
            boundary_mask=empty,
            valid_columns=0,
            ocean_pixels=int(ocean_area),
            beach_pixels=int(beach_area),
            rejected_ocean_components=rejected_ocean,
            rejected_beach_components=rejected_beach,
            metadata={"available": False, "reason": reason},
        )

    depth = max(3, round(min(height, width) * 0.02))
    orientation_value = str(orientation).strip().lower()
    if orientation_value == "auto":
        orientation_value = _infer_orientation(ocean, beach, depth=depth)
    if orientation_value not in ORIENTATIONS:
        raise ValueError(f"orientation must be 'auto' or one of {ORIENTATIONS}")
    scan_length = height if orientation_value in ("land_below", "land_above") else width
    margin_value = max(0, int(margin if margin is not None else round(min(height, width) * 0.005)))
    gap_value = float(max_gap if max_gap is not None else max(8.0, scan_length * 0.03))
    min_segment = int(
        min_segment_px if min_segment_px is not None else max(4, round(scan_length * 0.015))
    )
    smoothing_value = int(
        smoothing
        if smoothing is not None
        else max(3, round(min(height, width) * 0.008))
    )
    if smoothing_value % 2 == 0:
        smoothing_value += 1

    scanline = _scanline_boundary(
        ocean,
        _land_fraction(beach, orientation_value, depth),
        orientation=orientation_value,
        margin=margin_value,
    )
    segments = _build_segments(
        scanline,
        orientation=orientation_value,
        max_gap=gap_value,
        min_segment_px=min_segment,
        smoothing=smoothing_value,
    )
    valid = int(sum(len(segment) for segment in segments))
    if valid < min_valid_columns:
        return ShorelineResult(
            points=np.empty((0, 2), dtype=np.float64),
            per_column=np.empty((0, 2), dtype=np.float64),
            boundary_mask=empty,
            valid_columns=valid,
            ocean_pixels=int(ocean_area),
            beach_pixels=int(beach_area),
            rejected_ocean_components=rejected_ocean,
            rejected_beach_components=rejected_beach,
            metadata={
                "available": False,
                "reason": "too few columns contain an Ocean/Beach interface",
                "orientation": orientation_value,
            },
        )

    points = np.concatenate(segments, axis=0)
    boundary_mask = np.zeros((height, width), dtype=np.uint8)
    for segment in segments:
        for x, y in segment:
            if 0 <= int(y) < height and 0 <= int(x) < width:
                boundary_mask[int(y), int(x)] = 1
    return ShorelineResult(
        points=points,
        per_column=points,
        boundary_mask=boundary_mask,
        valid_columns=valid,
        ocean_pixels=int(ocean_area),
        beach_pixels=int(beach_area),
        rejected_ocean_components=rejected_ocean,
        rejected_beach_components=rejected_beach,
        segments=segments,
        metadata={
            "available": True,
            "image_size": [int(width), int(height)],
            "orientation": orientation_value,
            "close_kernel_px": int(kernel_px),
            "smoothing_px": int(smoothing_value),
            "max_gap_px": float(gap_value),
            "min_segment_px": int(min_segment),
            "margin_px": int(margin_value),
            "segments": len(segments),
        },
    )


def shoreline_columns(
    result: ShorelineResult, *, x_min: int | None = None, x_max: int | None = None
) -> np.ndarray:
    """Return ordered points, optionally restricted to an x range."""
    points = result.points
    if x_min is not None:
        points = points[points[:, 0] >= x_min]
    if x_max is not None:
        points = points[points[:, 0] <= x_max]
    return points
