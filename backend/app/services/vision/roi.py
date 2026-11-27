"""Operator-drawn monitoring area (calibration ROI).

A monitoring area is a single simple polygon stored in normalized frame
coordinates (``x``, ``y`` in ``[0, 1]``). It is an *operator constraint*, not a
model prediction: every pixel outside the polygon is forced to Sky (or zero in a
binary water mask), so Ocean/Beach outside the area can never produce a
shoreline.

The polygon is rasterized on a copy of the inference mask. Stored frames and
observations are never modified.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

COORDINATE_SPACE = "normalized"
MIN_VERTICES = 3
MAX_VERTICES = 200
MIN_COVERAGE = 0.0001
_EPS = 1e-9


class RoiPolygonError(ValueError):
    """Raised when a monitoring polygon cannot be validated or rasterized."""


def _as_point(entry: Any, index: int) -> tuple[float, float]:
    if isinstance(entry, Mapping):
        try:
            x, y = float(entry["x"]), float(entry["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RoiPolygonError(f"point #{index} must contain numeric x and y") from exc
    else:
        try:
            x, y = float(entry[0]), float(entry[1])
        except (IndexError, TypeError, ValueError) as exc:
            raise RoiPolygonError(f"point #{index} must be a pair of numbers") from exc
        if isinstance(entry, str | bytes):
            raise RoiPolygonError(f"point #{index} must be a pair of numbers")
    if not (math.isfinite(x) and math.isfinite(y)):
        raise RoiPolygonError(f"point #{index} contains a non-finite value")
    if not (-_EPS <= x <= 1.0 + _EPS and -_EPS <= y <= 1.0 + _EPS):
        raise RoiPolygonError(
            f"point #{index} ({x}, {y}) lies outside normalized coordinates [0, 1]"
        )
    return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    return (
        min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS
        and min(a[1], b[1]) - _EPS <= p[1] <= max(a[1], b[1]) + _EPS
    )


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """Return True for a proper crossing or a collinear overlap."""
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if ((o1 > _EPS and o2 < -_EPS) or (o1 < -_EPS and o2 > _EPS)) and (
        (o3 > _EPS and o4 < -_EPS) or (o3 < -_EPS and o4 > _EPS)
    ):
        return True

    if abs(o1) <= _EPS and _on_segment(a, b, c):
        return True
    if abs(o2) <= _EPS and _on_segment(a, b, d):
        return True
    if abs(o3) <= _EPS and _on_segment(c, d, a):
        return True
    return bool(abs(o4) <= _EPS and _on_segment(c, d, b))


def polygon_is_simple(points: Sequence[tuple[float, float]]) -> bool:
    """A simple ring has no self-intersections and no duplicate vertices."""
    count = len(points)
    if count < MIN_VERTICES:
        return False
    for index, point in enumerate(points):
        if point == points[(index + 1) % count]:
            return False
    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        for j in range(i + 1, count):
            if j == i or (j + 1) % count == i or (i + 1) % count == j:
                continue
            c, d = points[j], points[(j + 1) % count]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """Absolute shoelace area in normalized frame units (frame area = 1)."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x, y) in enumerate(points):
        nx, ny = points[(index + 1) % len(points)]
        total += x * ny - nx * y
    return abs(total) / 2.0


def normalize_roi_polygon(raw: Any) -> dict[str, Any] | None:
    """Validate and canonicalize a stored/request ROI polygon.

    Returns ``None`` for ``None``. Raises :class:`RoiPolygonError` otherwise.
    """
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, Mapping):
        raise RoiPolygonError("roi_polygon must be an object")

    coordinate_space = raw.get("coordinate_space", COORDINATE_SPACE)
    if coordinate_space != COORDINATE_SPACE:
        raise RoiPolygonError(f"roi_polygon.coordinate_space must be {COORDINATE_SPACE!r}")

    raw_points = raw.get("points")
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, str | bytes):
        raise RoiPolygonError("roi_polygon.points must be a list of [x, y] pairs")
    if len(raw_points) < MIN_VERTICES:
        raise RoiPolygonError(f"roi_polygon needs at least {MIN_VERTICES} points")
    if len(raw_points) > MAX_VERTICES:
        raise RoiPolygonError(f"roi_polygon has too many points (max {MAX_VERTICES})")

    points = [_as_point(entry, index) for index, entry in enumerate(raw_points)]
    area = polygon_area(points)
    if area < MIN_COVERAGE:
        raise RoiPolygonError("roi_polygon covers no meaningful area (draw a larger outline)")
    if not polygon_is_simple(points):
        raise RoiPolygonError("roi_polygon is self-intersecting or has duplicate vertices")

    frame_width = raw.get("frame_width")
    frame_height = raw.get("frame_height")
    for label, value in (("frame_width", frame_width), ("frame_height", frame_height)):
        if value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise RoiPolygonError(f"roi_polygon.{label} must be an integer") from exc
        if number < 1 or number > 16384:
            raise RoiPolygonError(f"roi_polygon.{label} is out of range")
        if label == "frame_width":
            frame_width = number
        else:
            frame_height = number

    return {
        "coordinate_space": COORDINATE_SPACE,
        "points": [[x, y] for x, y in points],
        "frame_width": frame_width if frame_width is not None else None,
        "frame_height": frame_height if frame_height is not None else None,
    }


def _roi_points(roi_polygon: Any) -> list[tuple[float, float]] | None:
    normalized = normalize_roi_polygon(roi_polygon)
    if normalized is None:
        return None
    return [(float(x), float(y)) for x, y in normalized["points"]]


def polygon_mask(points: Sequence[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize normalized points into a ``uint8`` 0/255 mask of ``shape``."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise RoiPolygonError("mask shape must be positive")
    # Include the last row/column when a vertex sits at 1.0.
    xs = [min(width - 1, max(0, int(round(x * (width - 1))))) for x, _ in points]
    ys = [min(height - 1, max(0, int(round(y * (height - 1))))) for _, y in points]
    vertices = np.asarray(list(zip(xs, ys, strict=True)), dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [vertices], 255)  # type: ignore[call-overload]
    return mask


def apply_roi_to_class_mask(
    class_mask: Any,
    roi_polygon: Any,
    *,
    outside_value: int = 2,
) -> np.ndarray:
    """Copy ``class_mask`` and set every pixel outside the ROI to ``outside_value``.

    Sky (2) is the default because it preserves the 3-class contract and is
    never Beach, so an Ocean pixel at the polygon boundary cannot create a false
    Ocean/Beach contact. The input array is never mutated.
    """
    array = np.asarray(class_mask)
    if array.ndim != 2:
        raise RoiPolygonError("class_mask must be a 2-D array")
    if roi_polygon is None:
        return array
    points = _roi_points(roi_polygon)
    if points is None:
        return array
    output = np.array(array, copy=True)
    mask = polygon_mask(points, output.shape) > 0
    output[~mask] = outside_value
    return output


def apply_roi_to_binary_mask(water_mask: Any, roi_polygon: Any) -> np.ndarray:
    """Copy a binary water mask and set every pixel outside the ROI to 0."""
    array = np.asarray(water_mask)
    if array.ndim != 2:
        raise RoiPolygonError("water_mask must be a 2-D array")
    if roi_polygon is None:
        return array
    points = _roi_points(roi_polygon)
    if points is None:
        return array
    output = np.array(array, copy=True)
    mask = polygon_mask(points, output.shape) > 0
    output[~mask] = 0
    return output


def load_active_roi(session: Any, camera: Any) -> dict[str, Any] | None:
    """Load and validate the active calibration's ROI for ``camera``.

    Invalid stored data is logged and treated as no ROI only after failing
    loudly in the log; the caller still runs full-frame rather than crash a
    live worker. New saves are validated at the API boundary.
    """
    calibration_id = getattr(camera, "active_calibration_id", None)
    if not calibration_id:
        return None
    from app.models.camera import CameraCalibration

    row = session.get(CameraCalibration, calibration_id)
    if row is None or row.camera_id != getattr(camera, "id", None):
        return None
    try:
        return normalize_roi_polygon(row.roi_polygon)
    except RoiPolygonError:
        logger.exception(
            "camera %s has an invalid stored monitoring outline (calibration %s); "
            "running with the full frame",
            getattr(camera, "id", None),
            calibration_id,
        )
        return None
