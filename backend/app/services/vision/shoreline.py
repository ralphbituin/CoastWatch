"""Shoreline ("waterline") extraction — Phase 4.

BOUNDARY CONTRACT: this module answers **"where is the waterline *now*"**.
It does not know about tides and does not compare against a baseline. It turns
a three-class mask (active path) or a binary water mask (operator path) into a
shoreline geometry and, when the camera is calibrated, a calibrated position in
metres (positive = seaward).

The active path is :func:`extract_shoreline_from_classes`: it consumes the
3-class SegFormer mask (0 Beach, 1 Ocean, 2 Sky) and extracts the shared
**Ocean/Beach** boundary. The Sky/Ocean horizon is never used as the shoreline,
and a frame with no Beach/Ocean contact returns ``shoreline_available=False``
instead of a fabricated line.

The operator-mask path remains :func:`extract_shoreline`. It consumes a binary
water mask (non-zero = water) and treats it as ground truth
(``segmentation_confidence`` defaults to 1.0). Its algorithm is per-column water
extent → sea-side detection → far water edge → smoothing → image-space polyline.

Both paths keep calibration separate and unchanged: a caller-supplied
:class:`CalibrationProfile` maps image points to world metres. No tide,
bathymetry or metric claim is made here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.core.config import get_settings
from app.services.vision.calibration import CalibrationProfile, apply_homography
from app.services.vision.roi import apply_roi_to_binary_mask, apply_roi_to_class_mask

BEACH = 0
OCEAN = 1
SKY = 2
IGNORE = 255

# Orientation of the Beach/Ocean contact in image coordinates.
#   land_below : Beach is below Ocean (typical beach camera) -> scan columns,
#                take the lowest Ocean pixel with Beach underneath.
#   land_above : Beach is above Ocean -> scan columns, take the highest.
#   land_left  : Beach is left of Ocean -> scan rows, take the leftmost.
#   land_right : Beach is right of Ocean -> scan rows, take the rightmost.
SCAN_ORIENTATIONS = ("land_below", "land_above", "land_left", "land_right")


@dataclass(slots=True)
class ShorelineObservationResult:
    points: list[list[float]]  # image-space [[x, y], ...]
    calibrated_position_m: float | None
    segmentation_confidence: float
    calibration_version: int | None = None
    # Backward-compatible extensions for the class-aware path.
    segments: list[list[list[float]]] = field(default_factory=list)
    shoreline_available: bool = True
    reason: str | None = None
    valid_columns: int = 0  # scanlines with a valid contact (columns or rows)
    ocean_pixels: int | None = None
    beach_pixels: int | None = None
    rejected_ocean_components: int = 0
    rejected_beach_components: int = 0
    # Number of area-qualified Ocean components kept by the coastline
    # selector, and how many were discarded for being landward/enclosed.
    ocean_components_kept: int | None = None
    ocean_components_rejected_geometry: int = 0
    # Secondary shoreline branches dropped because their local seaward water
    # body was thinner than the configured depth floor.
    branches_rejected_depth: int = 0
    # Single-line contract diagnostics.
    shoreline_axis: str | None = None
    interpolated_ranges: list[list[int]] = field(default_factory=list)
    measured_coverage: float | None = None
    # Diagnostics for the active extraction parameters (class-aware path).
    orientation_used: str | None = None
    bridge_kernel_px: int | None = None
    median_kernel_px: int | None = None
    max_gap_px: float | None = None
    margin_px: int | None = None


class ShorelineExtractionError(ValueError):
    """Raised when a mask yields no usable waterline."""


def _as_water_mask(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY) if array.shape[2] == 3 else array[:, :, 0]
    if array.ndim != 2:
        raise ShorelineExtractionError("mask must be a 2-D single-channel image")
    return (array > 127).astype(np.uint8)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values
    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def extract_shoreline(
    mask: Any,
    *,
    calibration: CalibrationProfile | None = None,
    seaward_axis: tuple[float, float] | None = None,
    confidence: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    monitoring_polygon: Any | None = None,
    smoothing_window: int = 5,
    min_water_columns: int = 8,
) -> ShorelineObservationResult:
    """Extract the waterline polyline from a water mask.

    ``calibration`` maps pixel→world metres; when combined with
    ``seaward_axis`` (a unit vector in world coordinates), each waterline
    point is projected onto that axis and the mean projection becomes
    ``calibrated_position_m`` (relative, positive = seaward).

    ``roi`` is a legacy integer crop. ``monitoring_polygon`` is the operator's
    normalized calibration outline; pixels outside it are treated as no water
    before any extraction work.
    """
    water = _as_water_mask(mask)
    water = apply_roi_to_binary_mask(water, monitoring_polygon)
    if roi is not None:
        x, y, w, h = roi
        height, width = water.shape
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ShorelineExtractionError(
                f"roi (x={x}, y={y}, w={w}, h={h}) lies outside the {width}x{height} mask"
            )
        water = water[y : y + h, x : x + w]
        x_offset, y_offset = x, y
    else:
        x_offset = y_offset = 0

    # Speckle filter: remove isolated water blobs smaller than a few pixels.
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    height, width = water.shape

    # Per-column water extent.
    column_top = np.full(width, np.nan, dtype=np.float64)
    column_bot = np.full(width, np.nan, dtype=np.float64)
    for x in range(width):
        ys = np.flatnonzero(water[:, x])
        if ys.size:
            column_top[x] = ys.min()
            column_bot[x] = ys.max()

    valid = ~np.isnan(column_top)
    if int(valid.sum()) < min_water_columns:
        raise ShorelineExtractionError(
            f"mask contains water in only {int(valid.sum())} column(s); "
            f"need at least {min_water_columns}"
        )

    # Sea-side detection: the half carrying more water mass is the sea side;
    # the shoreline is the far water edge on that side.
    top_mass = float(water[: height // 2].sum())
    bottom_mass = float(water[height // 2 :].sum())
    sea_below = bottom_mass >= top_mass
    line = column_top if sea_below else column_bot

    # Interpolate missing columns from neighbours, then smooth.
    xs = np.arange(width, dtype=np.float64)
    line = np.interp(xs, xs[valid], line[valid])
    line = _moving_average(line, smoothing_window)

    points = [[float(x + x_offset), float(y + y_offset)] for x, y in zip(xs, line, strict=True)]

    calibrated_position_m: float | None = None
    if calibration is not None and calibration.homography is not None and seaward_axis is not None:
        axis = np.asarray(seaward_axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ShorelineExtractionError("seaward_axis must be a non-zero vector")
        axis = axis / norm
        world = apply_homography(calibration.homography, [(p[0], p[1]) for p in points])
        calibrated_position_m = float(np.mean(world @ axis))

    segmentation_confidence = 1.0 if confidence is None else float(confidence)
    return ShorelineObservationResult(
        points=points,
        calibrated_position_m=calibrated_position_m,
        segmentation_confidence=segmentation_confidence,
        calibration_version=calibration.version if calibration is not None else None,
        # The binary operator mask yields one ordered polyline; expose it as a
        # single segment so chart consumers do not have to special-case it.
        segments=[points],
        valid_columns=int(valid.sum()),
    )


# ---------------------------------------------------------------------------
# Class-aware Ocean/Beach extraction (SegFormer 3-class mask)
# ---------------------------------------------------------------------------


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return the largest 8-connected component, its area and component count."""
    array = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool), 0, 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest, int(stats[largest, cv2.CC_STAT_AREA]), count - 1


def _keep_large_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int, int]:
    """Keep every 8-connected component at least ``min_area`` pixels.

    Returns ``(kept_mask, kept_area, rejected_component_count)``. Small Beach
    patches enclosed by Ocean (people, foam, glare, reflections) are rejected
    here, before they can reach the scanline.
    """
    array = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool), 0, 0
    keep = np.zeros(count, dtype=bool)
    for label_id in range(1, count):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= int(min_area):
            keep[label_id] = True
    kept_area = int(stats[keep, cv2.CC_STAT_AREA].sum()) if bool(keep.any()) else 0
    rejected = int(count - 1 - int(keep.sum()))
    if not bool(keep.any()):
        return np.zeros_like(mask, dtype=bool), 0, rejected
    return keep[labels], kept_area, rejected


def _select_coastline_components(
    ocean: np.ndarray,
    beach: np.ndarray,
    sky: np.ndarray,
    *,
    orientation: str,
) -> tuple[np.ndarray, int, int]:
    """Keep only Ocean components that belong to the real coastline.

    The area filter deliberately keeps every large Ocean component, because a
    wide occluder can split one real sea band into two honest segments. That
    rule also keeps a large false-Ocean patch (blue deck furniture, a rock
    shadow, a pool, an on-screen sign) on the land side. This selector is a
    second, geometric filter on top of the area filter:

    * the dominant Beach component is the largest kept Beach region;
    * an Ocean component is eligible only when it shares a border with that
      Beach and lies on its seaward side (using the inferred scan orientation);
    * the eligible component with the longest shared Beach border is the anchor
      (the real coastline);
    * other eligible components are kept only when they are open water, i.e.
      they touch Sky or the frame border. A patch fully enclosed by Beach is a
      pool/rock/shadow, not the shoreline.

    Sky adjacency is used *only* to identify open sea for component selection;
    it is never used to draw the line on the Sky/Ocean horizon. When no
    eligible component exists (for example the two-Beach side-view geometry),
    the area-filtered mask is returned unchanged, so single-component frames
    are unaffected.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ocean.astype(np.uint8), connectivity=8
    )
    if count <= 2:
        return ocean, max(0, count - 1), 0
    beach_count, beach_labels, beach_stats, beach_centroids = cv2.connectedComponentsWithStats(
        beach.astype(np.uint8), connectivity=8
    )
    if beach_count <= 1:
        return ocean, count - 1, 0
    dominant_label = 1 + int(np.argmax(beach_stats[1:, cv2.CC_STAT_AREA]))
    dominant_beach = beach_labels == dominant_label
    beach_cx, beach_cy = beach_centroids[dominant_label]

    def is_seaward(cx: float, cy: float) -> bool:
        if orientation == "land_below":
            return cy < beach_cy
        if orientation == "land_above":
            return cy > beach_cy
        if orientation == "land_left":
            return cx > beach_cx
        return cx < beach_cx

    kernel = np.ones((3, 3), dtype=np.uint8)
    candidates: list[dict[str, int]] = []
    for label_id in range(1, count):
        component = labels == label_id
        dilated = cv2.dilate(component.astype(np.uint8), kernel) > 0
        shared = int(np.count_nonzero(dilated & dominant_beach))
        if shared <= 0:
            continue
        cx, cy = centroids[label_id]
        if not is_seaward(float(cx), float(cy)):
            continue
        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        candidates.append(
            {
                "label": label_id,
                "area": int(stats[label_id, cv2.CC_STAT_AREA]),
                "shared": shared,
                "sky_border": int(np.count_nonzero(dilated & sky)),
                "open_border": int(
                    left <= 0
                    or top <= 0
                    or left + width >= ocean.shape[1]
                    or top + height >= ocean.shape[0]
                ),
            }
        )
    if not candidates:
        return ocean, count - 1, 0
    anchor = max(candidates, key=lambda row: (row["shared"], row["area"]))
    keep = {anchor["label"]}
    for row in candidates:
        if row["label"] == anchor["label"]:
            continue
        if row["sky_border"] > 0 or row["open_border"]:
            keep.add(row["label"])
    if len(keep) == count - 1:
        return ocean, count - 1, 0
    selected = np.isin(labels, list(keep))
    return selected, len(keep), count - 1 - len(keep)


_SEAWARD_VECTORS = {
    "land_below": (0.0, -1.0),
    "land_above": (0.0, 1.0),
    "land_left": (1.0, 0.0),
    "land_right": (-1.0, 0.0),
}


def _branch_ocean_depth(
    segment: np.ndarray, ocean: np.ndarray, orientation: str, max_steps: int
) -> float:
    """90th-percentile open-water depth on the seaward side of a branch.

    The depth is measured perpendicular to the local shoreline tangent, not
    along one global axis, so a diagonal/curved coastline is handled locally.
    A thin Ocean strip between the Beach and Sky has a small depth; an open sea
    branch has a large one.

    The normal/step search is vectorised over points and steps. It is the exact
    same integer sampling as the original per-point loop (``round`` to nearest,
    stop at the first non-Ocean sample).
    """
    points = np.asarray(segment, dtype=np.float64)
    if len(points) == 0 or max_steps <= 0:
        return 0.0
    height, width = ocean.shape
    seaward = np.asarray(_SEAWARD_VECTORS[orientation], dtype=np.float64)
    indices = np.arange(len(points))
    previous = points[np.maximum(indices - 1, 0)]
    following = points[np.minimum(indices + 1, len(points) - 1)]
    tangent = following - previous
    norms = np.linalg.norm(tangent, axis=1)
    normals = np.empty_like(points)
    valid = norms >= 1e-9
    if valid.any():
        unit = tangent[valid] / norms[valid, None]
        normals[valid, 0] = -unit[:, 1]
        normals[valid, 1] = unit[:, 0]
    normals[~valid] = seaward
    flip = (normals @ seaward) < 0
    normals[flip] = -normals[flip]

    steps = np.arange(1, int(max_steps) + 1, dtype=np.float64)
    xx = np.rint(points[:, 0, None] + normals[:, 0, None] * steps[None, :]).astype(np.int64)
    yy = np.rint(points[:, 1, None] + normals[:, 1, None] * steps[None, :]).astype(np.int64)
    inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    sampled = np.zeros_like(inside, dtype=bool)
    if inside.any():
        sampled[inside] = ocean[yy[inside], xx[inside]]
    continuous = (inside & sampled).astype(np.uint8)
    runs = np.cumprod(continuous, axis=1).sum(axis=1, dtype=np.float64)
    return float(np.percentile(runs, 90))


def _filter_thin_ocean_branches(
    segments: list[np.ndarray],
    ocean: np.ndarray,
    orientation: str,
    min_depth_ratio: float,
) -> tuple[list[np.ndarray], int, list[float]]:
    """Drop secondary branches whose seaward water body is too thin.

    The anchor branch (the one with the deepest local open water) is always
    kept. This is the branch that carries the real coastline. Other branches
    are kept only when their local seaward depth reaches ``min_depth_ratio`` of
    the short frame side. A pier-split real ocean has two deep halves and both
    survive; a thin landward strip between Beach and Sky does not.
    """
    if not segments:
        return [], 0, []
    short = min(ocean.shape)
    max_steps = max(8, int(round(short * max(0.12, min_depth_ratio * 2))))
    depths = [_branch_ocean_depth(segment, ocean, orientation, max_steps) for segment in segments]
    if len(segments) < 2 or min_depth_ratio <= 0:
        return segments, 0, depths
    anchor_index = max(
        range(len(segments)), key=lambda index: (depths[index], len(segments[index]))
    )
    threshold = float(min_depth_ratio) * short
    kept_indices = [
        index
        for index, _segment in enumerate(segments)
        if index == anchor_index or depths[index] >= threshold
    ]
    return (
        [segments[index] for index in kept_indices],
        len(segments) - len(kept_indices),
        [depths[index] for index in kept_indices],
    )


def _component_solidity(component: np.ndarray) -> float:
    """Area divided by convex-hull area; compact open water is close to 1."""
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    return float(cv2.contourArea(contour)) / hull_area if hull_area > 0 else 0.0


def _bridge_scanline_points(
    points: np.ndarray, orientation: str
) -> tuple[np.ndarray, list[list[int]], int]:
    """Return one point per scanline from first to last measured scanline.

    Missing scanlines are filled by linear interpolation between the two
    genuinely measured endpoints. No point is searched in the mask; no
    extrapolation is done beyond the first/last measurement.
    """
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64), [], 0
    vertical = orientation in ("land_below", "land_above")
    scan_values = points[:, 0] if vertical else points[:, 1]
    coordinate = points[:, 1] if vertical else points[:, 0]
    grouped: dict[int, list[float]] = {}
    for scan, value in zip(scan_values, coordinate, strict=True):
        grouped.setdefault(int(round(float(scan))), []).append(float(value))
    measured_scans = np.asarray(sorted(grouped), dtype=np.int64)
    measured_values = np.asarray([float(np.median(grouped[int(scan)])) for scan in measured_scans])
    first, last = int(measured_scans[0]), int(measured_scans[-1])
    full_scan = np.arange(first, last + 1, dtype=np.float64)
    full_value = np.interp(full_scan, measured_scans, measured_values)
    measured_mask = np.isin(full_scan.astype(np.int64), measured_scans)
    interpolated_ranges: list[list[int]] = []
    index = 0
    while index < len(full_scan):
        if measured_mask[index]:
            index += 1
            continue
        start = index
        while index < len(full_scan) and not measured_mask[index]:
            index += 1
        interpolated_ranges.append([max(0, start - 1), min(len(full_scan) - 1, index)])
    if vertical:  # land_below/above: scan x, boundary y
        line = np.column_stack([full_scan, full_value])
    else:  # land_left/right: scan y, boundary x
        line = np.column_stack([full_value, full_scan])
    return line.astype(np.float64), interpolated_ranges, len(measured_scans)


def _gap_bridgeable(sub: np.ndarray, orientation: str, first_scan: int, last_scan: int) -> bool:
    """False when the gap contains Ocean that leaves the frame (out of view)."""
    height, width = sub.shape
    for scan in range(int(first_scan) + 1, int(last_scan)):
        if orientation == "land_below":
            if 0 <= scan < width and sub[height - 1, scan]:
                return False
        elif orientation == "land_above":
            if 0 <= scan < width and sub[0, scan]:
                return False
        elif orientation == "land_right":
            if 0 <= scan < height and sub[scan, 0]:
                return False
        else:
            if 0 <= scan < height and sub[scan, width - 1]:
                return False
    return True


def _select_anchor_run(segments: list[np.ndarray], sub: np.ndarray, orientation: str) -> np.ndarray:
    """Pick the largest scan-contiguous run of one component."""
    points = np.concatenate(segments, axis=0)
    vertical = orientation in ("land_below", "land_above")
    order = np.argsort(points[:, 0] if vertical else points[:, 1])
    points = points[order]
    scan = points[:, 0] if vertical else points[:, 1]
    runs: list[np.ndarray] = []
    start = 0
    for index in range(1, len(points)):
        if int(scan[index]) == int(scan[index - 1]):
            continue
        if _gap_bridgeable(sub, orientation, int(scan[index - 1]), int(scan[index])):
            continue
        runs.append(points[start:index])
        start = index
    runs.append(points[start:])
    return max(
        runs,
        key=lambda run: len(np.unique((run[:, 0] if vertical else run[:, 1]).astype(int))),
    )


def _single_line_candidate(
    selected: np.ndarray,
    beach: np.ndarray,
    orientation: str,
    *,
    min_component_area: int,
    beach_depth: int,
    margin_px: int,
    max_gap_px: float,
    min_segment_px: int,
    smoothing_px: int,
    min_branch_depth_ratio: float,
    side_mask: np.ndarray,
) -> dict[str, Any] | None:
    """Build one coherent axis candidate from a selected Ocean mask."""
    height, width = selected.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8), connectivity=8
    )
    # Same Beach/orientation/depth for every component in this candidate.
    land_fraction = _land_fraction(beach, orientation, beach_depth)
    best: dict[str, Any] | None = None
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        component = labels == label_id
        sub = component & side_mask
        if int(sub.sum()) < max(200, int(area * 0.1)):
            continue
        points = _scanline_boundary(
            sub,
            land_fraction,
            orientation=orientation,
            margin=margin_px,
        )
        segments = _build_segments(
            points,
            orientation=orientation,
            max_gap=max_gap_px,
            min_segment_px=min_segment_px,
            smoothing_window=smoothing_px,
        )
        segments, _, branch_depths = _filter_thin_ocean_branches(
            segments, sub, orientation, min_branch_depth_ratio
        )
        if not segments:
            continue
        measured = _select_anchor_run(segments, component, orientation)
        vertical = orientation in ("land_below", "land_above")
        scan = measured[:, 0] if vertical else measured[:, 1]
        unique_scan = len(np.unique(scan.astype(np.int64)))
        measured_fraction = unique_scan / max(1, width if vertical else height)
        if measured_fraction <= 0:
            continue
        solidity = _component_solidity(sub)
        score = solidity * measured_fraction
        if best is not None and score <= float(best["score"]):
            continue
        best = {
            "orientation": orientation,
            "axis": "horizontal" if vertical else "vertical",
            "segments": segments,
            "run_points": measured,
            "solidity": solidity,
            "measured_fraction": measured_fraction,
            "score": score,
            "anchor_depth": max(branch_depths) if branch_depths else 0.0,
            "subregion_area": int(sub.sum()),
            "component_area": area,
            "component_bbox": [int(v) for v in stats[label_id, :4]],
        }
    return best


def _select_single_line(
    ocean: np.ndarray,
    beach: np.ndarray,
    sky: np.ndarray,
    *,
    min_component_area: int,
    beach_depth: int,
    margin_px: int,
    max_gap_px: float,
    min_segment_px: int,
    smoothing_px: int,
    min_branch_depth_ratio: float,
    orientations: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    """Evaluate both axes and return the best single coherent trace."""
    height, width = ocean.shape
    bcount, _, bstats, bcentroids = cv2.connectedComponentsWithStats(
        beach.astype(np.uint8), connectivity=8
    )
    if bcount <= 1:
        return None
    dominant = 1 + int(np.argmax(bstats[1:, cv2.CC_STAT_AREA]))
    beach_cx, beach_cy = bcentroids[dominant]
    yy, xx = np.mgrid[0:height, 0:width]
    side_masks = {
        "land_below": yy < beach_cy,
        "land_above": yy > beach_cy,
        "land_right": xx < beach_cx,
        "land_left": xx > beach_cx,
    }
    best: dict[str, Any] | None = None
    for orientation in orientations or SCAN_ORIENTATIONS:
        selected, _, _ = _select_coastline_components(ocean, beach, sky, orientation=orientation)
        candidate = _single_line_candidate(
            selected,
            beach,
            orientation,
            min_component_area=min_component_area,
            beach_depth=beach_depth,
            margin_px=margin_px,
            max_gap_px=max_gap_px,
            min_segment_px=min_segment_px,
            smoothing_px=smoothing_px,
            min_branch_depth_ratio=min_branch_depth_ratio,
            side_mask=side_masks[orientation],
        )
        if candidate is not None and (
            best is None or float(candidate["score"]) > float(best["score"])
        ):
            best = candidate
    if best is None:
        return None
    line, interpolated_ranges, measured_count = _bridge_scanline_points(
        np.asarray(best["run_points"], dtype=np.float64),
        str(best["orientation"]),
    )
    best["line"] = line
    best["interpolated_ranges"] = interpolated_ranges
    best["measured_count"] = measured_count
    best["measured_fraction"] = measured_count / max(
        1, width if best["axis"] == "horizontal" else height
    )
    return best


def _odd_kernel(value: float, *, minimum: int = 3, maximum: int | None = None) -> int:
    """Round a scalar to an odd kernel width, respecting optional bounds."""
    kernel = max(int(minimum), int(round(value)))
    if kernel % 2 == 0:
        kernel += 1
    if maximum is not None:
        kernel = min(kernel, int(maximum))
        if kernel % 2 == 0:
            kernel -= 1
    return max(3, kernel)


def _close_ocean(ocean: np.ndarray, kernel_px: int) -> np.ndarray:
    """Bridge thin occluders (trunks, poles, railings) in the Ocean mask only.

    This is applied to the *extraction* input. The stored class mask is never
    modified. A kernel sized to the frame closes gaps narrower than its
    diameter; wider objects still produce an honest segment break.
    """
    if kernel_px <= 1 or not ocean.any():
        return ocean.astype(bool, copy=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    closed = cv2.morphologyEx(ocean.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def _infer_orientation(ocean: np.ndarray, beach: np.ndarray, *, depth: int = 8) -> str:
    """Infer which side of the Ocean body the Beach lies on.

    The one-sided Ocean boundary is counted in each of the four directions and
    the foreground side with the most Beach-adjacent boundary pixels wins. This
    follows the actual waterline rather than the class centroids, which can be
    skewed by a large beach area at the top of the frame.

    ``land_above`` is usually the far shore across the bay. It is kept available
    for callers that pin it, but it is not chosen automatically because the near
    shore is normally below or to one side. No foreground side receives an
    automatic preference: on an oblique view the left/right boundary can
    legitimately outnumber the below boundary, and forcing ``land_below`` there
    discards real contact and truncates the line (audited 2026-11-26). It is
    still a heuristic for camera geometry, not a measurement; deployments whose
    geometry it gets wrong can pin ``shoreline_orientation`` in configuration.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(ocean.astype(np.uint8), kernel, iterations=1)
    boundary = ocean & (eroded == 0)
    if not boundary.any():
        return "land_below"
    counts: dict[str, int] = {}
    for candidate in SCAN_ORIENTATIONS:
        fraction = _land_fraction(beach, candidate, depth)
        counts[candidate] = int(np.count_nonzero(boundary & (fraction >= 0.5)))
    foreground = ("land_below", "land_left", "land_right")
    return max(foreground, key=lambda name: counts[name])


def _land_fraction(beach: np.ndarray, orientation: str, depth: int) -> np.ndarray:
    """Fraction of Beach pixels in the next ``depth`` pixels on the land side.

    ``fraction[y, x] > 0.5`` means an Ocean pixel at ``(y, x)`` is directly
    adjacent to Beach in the configured land direction. This excludes a
    Sky/Ocean horizon by construction: Sky is never Beach.

    The box sum is computed from an integral image of integer counts. That is
    the exact same count as adding ``depth`` shifted boolean masks; only the
    summation order changes.
    """
    height, width = beach.shape
    depth = max(1, int(depth))
    depth = min(depth, height) if orientation in ("land_below", "land_above") else min(depth, width)
    if depth <= 0:
        return np.zeros((height, width), dtype=np.float32)
    mask = beach.astype(np.int32)
    if orientation in ("land_below", "land_above"):
        integral = np.zeros((height + 1, width), dtype=np.int32)
        np.cumsum(mask, axis=0, dtype=np.int32, out=integral[1:])
        rows = np.arange(height)
        if orientation == "land_below":
            lower = integral[rows + 1]
            upper = integral[np.minimum(height, rows + depth + 1)]
        else:
            lower = integral[np.maximum(0, rows - depth)]
            upper = integral[rows]
        counts = upper - lower
    elif orientation in ("land_left", "land_right"):
        integral = np.zeros((height, width + 1), dtype=np.int32)
        np.cumsum(mask, axis=1, dtype=np.int32, out=integral[:, 1:])
        columns = np.arange(width)
        if orientation == "land_left":
            lower = integral[:, np.maximum(0, columns - depth)]
            upper = integral[:, columns]
        else:
            lower = integral[:, columns + 1]
            upper = integral[:, np.minimum(width, columns + depth + 1)]
        counts = upper - lower
    else:  # pragma: no cover - validated by the caller
        raise ShorelineExtractionError(f"unknown shoreline orientation {orientation!r}")
    return counts.astype(np.float32) / np.float32(depth)


def _scanline_boundary(
    ocean: np.ndarray,
    land_fraction: np.ndarray,
    *,
    orientation: str,
    margin: int,
    land_threshold: float = 0.5,
) -> np.ndarray:
    """One Ocean point per scanline that has Beach directly on the land side.

    For a typical beach camera (``land_below``) this is the lowest Ocean pixel
    in each column with Beach underneath. Exactly one y per x is possible by
    construction, so separate interface branches can never be joined by a
    median.

    A scanline is kept whenever at least one Ocean pixel has Beach on its land
    side within ``depth`` (enforced by ``land_fraction``). That adjacency test
    alone already rejects open water that exits the frame: such water has no
    Beach on its land side. The old whole-scanline "Ocean touches the edge"
    pre-skip was over-broad because a small foreground Ocean component touching
    the edge (for example a wet rock classified Ocean) discarded a scanline that
    still contained a valid contact. The frame margin still bounds where a point
    may be placed.
    """
    height, width = ocean.shape
    margin = max(0, int(margin))

    if orientation in ("land_below", "land_above"):
        if width - margin <= margin:
            return np.empty((0, 2), dtype=np.float64)
        sub_ocean = ocean[margin : height - margin, margin : width - margin]
        sub_fraction = land_fraction[margin : height - margin, margin : width - margin]
        valid = sub_ocean & (sub_fraction >= land_threshold)
        present = valid.any(axis=0)
        if not present.any():
            return np.empty((0, 2), dtype=np.float64)
        if orientation == "land_below":
            # Last valid row in each column.
            indices = valid.shape[0] - 1 - np.argmax(valid[::-1], axis=0)
        else:
            indices = np.argmax(valid, axis=0)
        xs = np.arange(margin, width - margin, dtype=np.float64)[present]
        ys = indices[present].astype(np.float64) + float(margin)
        return np.column_stack([xs, ys])

    if height - margin <= margin:
        return np.empty((0, 2), dtype=np.float64)
    sub_ocean = ocean[margin : height - margin, margin : width - margin]
    sub_fraction = land_fraction[margin : height - margin, margin : width - margin]
    valid = sub_ocean & (sub_fraction >= land_threshold)
    present = valid.any(axis=1)
    if not present.any():
        return np.empty((0, 2), dtype=np.float64)
    if orientation == "land_left":
        indices = np.argmax(valid, axis=1)
    else:
        indices = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    ys = np.arange(margin, height - margin, dtype=np.float64)[present]
    xs = indices[present].astype(np.float64) + float(margin)
    return np.column_stack([xs, ys])


def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Centred median filter that leaves short runs untouched."""
    if window <= 1 or values.size < 3:
        return values
    window = min(int(window), int(values.size))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    half = window // 2
    filtered = values.astype(np.float64, copy=True)
    for index in range(values.size):
        low = max(0, index - half)
        high = min(values.size, index + half + 1)
        filtered[index] = float(np.median(values[low:high]))
    return filtered


def _smooth_scanlines(points: np.ndarray, *, orientation: str, window: int) -> np.ndarray:
    """Median-filter the shoreline coordinate along each contiguous run.

    The primary value (y for column scans, x for row scans) is filtered. Runs
    are separated by missing scanlines, so smoothing never crosses a real gap.
    """
    if window <= 1 or len(points) < 3:
        return points
    vertical = orientation in ("land_below", "land_above")
    scan = points[:, 0] if vertical else points[:, 1]
    values = points[:, 1] if vertical else points[:, 0]
    smoothed = values.astype(np.float64, copy=True)
    start = 0
    for index in range(1, len(points) + 1):
        if index == len(points) or scan[index] - scan[index - 1] != 1.0:
            smoothed[start:index] = _median_filter_1d(values[start:index], window)
            start = index
    result = points.copy()
    result[:, 1 if vertical else 0] = smoothed
    return result


def _build_segments(
    points: np.ndarray,
    *,
    orientation: str,
    max_gap: float,
    min_segment_px: int,
    smoothing_window: int,
) -> list[np.ndarray]:
    """Smooth, split at gaps larger than ``max_gap``, drop short segments.

    Distance is Euclidean, so a segment can never contain a jump larger than
    ``max_gap``. Segments are returned in scan order; they are never joined.
    """
    if len(points) == 0:
        return []
    smoothed = _smooth_scanlines(points, orientation=orientation, window=smoothing_window)
    raw_segments: list[np.ndarray] = []
    start = 0
    for index in range(1, len(smoothed)):
        jump = float(np.linalg.norm(smoothed[index] - smoothed[index - 1]))
        if jump > max_gap:
            raw_segments.append(smoothed[start:index])
            start = index
    raw_segments.append(smoothed[start:])
    return [segment for segment in raw_segments if len(segment) >= max(2, int(min_segment_px))]


def _unavailable(
    *,
    reason: str,
    confidence: float,
    calibration_version: int | None,
    ocean_pixels: int = 0,
    beach_pixels: int = 0,
    rejected_ocean_components: int = 0,
    rejected_beach_components: int = 0,
    valid_columns: int = 0,
    orientation_used: str | None = None,
    shoreline_axis: str | None = None,
    measured_coverage: float | None = None,
    bridge_kernel_px: int | None = None,
    median_kernel_px: int | None = None,
    max_gap_px: float | None = None,
    margin_px: int | None = None,
) -> ShorelineObservationResult:
    return ShorelineObservationResult(
        points=[],
        calibrated_position_m=None,
        segmentation_confidence=confidence,
        calibration_version=calibration_version,
        segments=[],
        shoreline_available=False,
        reason=reason,
        valid_columns=valid_columns,
        ocean_pixels=ocean_pixels,
        beach_pixels=beach_pixels,
        rejected_ocean_components=rejected_ocean_components,
        rejected_beach_components=rejected_beach_components,
        orientation_used=orientation_used,
        shoreline_axis=shoreline_axis,
        measured_coverage=measured_coverage,
        bridge_kernel_px=bridge_kernel_px,
        median_kernel_px=median_kernel_px,
        max_gap_px=max_gap_px,
        margin_px=margin_px,
    )


def extract_shoreline_from_classes(
    class_mask: Any,
    *,
    calibration: CalibrationProfile | None = None,
    seaward_axis: tuple[float, float] | None = None,
    confidence: float | None = None,
    ocean_id: int = OCEAN,
    beach_id: int = BEACH,
    ignore_index: int = IGNORE,
    min_ocean_component_px: int | None = None,
    min_beach_component_px: int | None = None,
    close_kernel_px: int | None = None,
    median_kernel_px: int | None = None,
    max_gap_px: float | None = None,
    min_segment_px: int | None = None,
    margin_px: int | None = None,
    orientation: str | None = None,
    ocean_component_selection: str | None = None,
    min_branch_depth_ratio: float | None = None,
    single_line_mode: bool | None = None,
    min_measured_coverage_ratio: float | None = None,
    beach_check_depth_px: int | None = None,
    smoothing_window: int | None = None,
    min_valid_columns: int = 8,
    monitoring_polygon: Any | None = None,
) -> ShorelineObservationResult:
    """Extract the shared Ocean/Beach boundary from a 3-class class mask.

    Class ids are 0 Beach, 1 Ocean, 2 Sky, 255 ignore. The active algorithm is
    a **class-aware scanline** extractor, not a binary-mask contour:

    1. Morphologically close the Ocean mask (never the stored class mask) so
       thin occluders such as trunks, poles and railings do not cut the line.
    2. Keep large Ocean components and reject small artifacts. When more than
       one remains, select only the component with the longest shared border
       with the dominant Beach, plus any other *open* (Sky- or frame-border-
       touching) component on the seaward side. A false-Ocean patch on the land
       side is discarded before the scan, so it cannot be scanned or joined.
       Small Beach components are also rejected; isolated Beach patches inside
       the Ocean cannot create dots.
    3. Determine the Beach side of the Ocean (configurable, or inferred from
       the class centroids). For each scanline perpendicular to that axis, take
       the Ocean pixel that touches Beach on the land side. This yields at most
       one point per column (or row) and cannot join two interface branches.
    4. Ignore the frame margin. A scanline is kept only when an Ocean pixel has
       Beach on the land side; open water that exits the frame is rejected by
       that adjacency test rather than by a whole-scanline edge rule (which
       could discard valid contact when a small foreground component touched
       the edge).
    5. Median-filter along the scan direction, split at Euclidean gaps larger
       than ``max_gap_px`` and drop segments shorter than ``min_segment_px``.
       Secondary branches are then kept only when their local seaward water
       depth reaches ``SHORELINE_MIN_BRANCH_DEPTH_RATIO`` of the short frame
       side; the deepest branch is always kept.

    The Sky/Ocean horizon is never used: Sky is not Beach, so no contact exists
    there. ``shoreline_available=False`` with a reason is returned when no
    Ocean/Beach contact remains. The returned coordinates are original-frame
    pixels; calibration is applied exactly as in :func:`extract_shoreline`.

    All thresholds default to the ``SHORELINE_*`` application settings; the
    keyword arguments exist for tests and site-specific overrides.
    """
    array = np.asarray(class_mask)
    if array.ndim != 2:
        raise ShorelineExtractionError(f"class mask must be 2-D, got shape {array.shape}")
    # Operator constraint: enforce the calibration outline before any pixel
    # counts, component selection, horizon or scan. Outside becomes Sky, so it
    # can never contribute Beach/Ocean contact.
    array = apply_roi_to_class_mask(array, monitoring_polygon)
    settings = get_settings()
    height, width = array.shape
    frame_pixels = height * width
    confidence_value = 1.0 if confidence is None else float(confidence)
    calibration_version = calibration.version if calibration is not None else None

    ocean_raw = array == int(ocean_id)
    beach_raw = array == int(beach_id)
    if not ocean_raw.any():
        return _unavailable(
            reason="no ocean pixels in frame",
            confidence=confidence_value,
            calibration_version=calibration_version,
            beach_pixels=int(beach_raw.sum()),
        )
    if not beach_raw.any():
        return _unavailable(
            reason="no beach pixels in frame",
            confidence=confidence_value,
            calibration_version=calibration_version,
            ocean_pixels=int(ocean_raw.sum()),
        )

    # Thresholds scale with frame area; explicit pixel values act as a floor so
    # existing callers that pass absolute thresholds keep their intent.
    ocean_threshold = max(
        1, int(round(frame_pixels * settings.shoreline_min_ocean_component_ratio))
    )
    if min_ocean_component_px is not None:
        ocean_threshold = max(ocean_threshold, int(min_ocean_component_px))
    beach_threshold = max(
        1, int(round(frame_pixels * settings.shoreline_min_beach_component_ratio))
    )
    if min_beach_component_px is not None:
        beach_threshold = max(beach_threshold, int(min_beach_component_px))

    # Bridge thin occluders before component selection: otherwise a trunk that
    # severs the Ocean band would cause the right-hand half to be discarded.
    if close_kernel_px is None:
        close_kernel_px = _odd_kernel(
            min(height, width) * settings.shoreline_close_kernel_ratio,
            minimum=settings.shoreline_close_kernel_min_px,
            maximum=settings.shoreline_close_kernel_max_px,
        )
    ocean_closed = _close_ocean(ocean_raw, int(close_kernel_px))
    # Keep every large Ocean component, not just the single largest: a wide
    # occluder that closing cannot bridge must produce two honest segments, not
    # silently discard the water on the far side. Small marine artifacts are
    # still rejected by the area threshold.
    ocean, ocean_area, rejected_ocean = _keep_large_components(ocean_closed, ocean_threshold)
    if not ocean.any() or ocean_area < ocean_threshold:
        return _unavailable(
            reason="dominant ocean component too small",
            confidence=confidence_value,
            calibration_version=calibration_version,
            ocean_pixels=int(ocean_area),
            beach_pixels=int(beach_raw.sum()),
            rejected_ocean_components=rejected_ocean,
            bridge_kernel_px=int(close_kernel_px),
        )

    # Keep every large Beach component (the near shore and the far shore can be
    # separate); drop small patches such as people, foam, glare or reflections
    # that the model labelled Beach inside the Ocean.
    beach, beach_area, rejected_beach = _keep_large_components(beach_raw, beach_threshold)
    if not beach.any() or beach_area < beach_threshold:
        return _unavailable(
            reason="no usable beach component adjacent to ocean",
            confidence=confidence_value,
            calibration_version=calibration_version,
            ocean_pixels=int(ocean_area),
            beach_pixels=int(beach_area),
            rejected_ocean_components=rejected_ocean,
            rejected_beach_components=rejected_beach,
            bridge_kernel_px=int(close_kernel_px),
        )

    single_line_value = single_line_mode
    if single_line_value is None:
        single_line_value = bool(settings.shoreline_single_line_mode)
    coverage_ratio_value = min_measured_coverage_ratio
    if coverage_ratio_value is None:
        coverage_ratio_value = float(settings.shoreline_min_measured_coverage_ratio)
    coverage_ratio_value = float(coverage_ratio_value)
    if coverage_ratio_value < 0 or coverage_ratio_value > 1:
        raise ShorelineExtractionError(
            "shoreline_min_measured_coverage_ratio must be between 0 and 1; "
            f"got {coverage_ratio_value!r}"
        )
    if single_line_value:
        selection_signal = (
            str(ocean_component_selection or settings.shoreline_ocean_component_selection)
            .strip()
            .lower()
        )
        if selection_signal not in ("all", "coastline"):
            raise ShorelineExtractionError(
                "shoreline_ocean_component_selection must be 'coastline' or 'all'; "
                f"got {selection_signal!r}"
            )
        if min_branch_depth_ratio is not None and not 0 <= float(min_branch_depth_ratio) <= 1:
            raise ShorelineExtractionError(
                "shoreline_min_branch_depth_ratio must be between 0 and 1"
            )
        if orientation is not None:
            orientation_signal = str(orientation).strip().lower()
            if orientation_signal != "auto" and orientation_signal not in SCAN_ORIENTATIONS:
                raise ShorelineExtractionError(
                    f"shoreline_orientation must be 'auto' or one of {SCAN_ORIENTATIONS}; "
                    f"got {orientation_signal!r}"
                )
        depth_ratio_value = min_branch_depth_ratio
        if depth_ratio_value is None:
            depth_ratio_value = settings.shoreline_min_branch_depth_ratio
        depth_ratio_value = float(depth_ratio_value)
        short = min(height, width)
        depth_px = max(3, int(round(short * settings.shoreline_beach_check_depth_ratio)))
        margin_value = max(0, int(round(short * settings.shoreline_margin_ratio)))
        max_gap_value = max(8.0, max(height, width) * settings.shoreline_max_gap_ratio)
        min_segment_value = max(
            4, int(round(max(height, width) * settings.shoreline_min_segment_ratio))
        )
        smoothing_value = _odd_kernel(short * settings.shoreline_median_kernel_ratio)
        candidate = _select_single_line(
            ocean,
            beach,
            array == SKY,
            min_component_area=ocean_threshold,
            beach_depth=depth_px,
            margin_px=margin_value,
            max_gap_px=max_gap_value,
            min_segment_px=min_segment_value,
            smoothing_px=smoothing_value,
            min_branch_depth_ratio=depth_ratio_value,
            orientations=(
                (str(orientation).strip().lower(),)
                if orientation is not None and str(orientation).strip().lower() != "auto"
                else None
            ),
        )
        if candidate is None or int(candidate["measured_count"]) < int(min_valid_columns):
            return _unavailable(
                reason="insufficient_coastline_evidence",
                confidence=confidence_value,
                calibration_version=calibration_version,
                ocean_pixels=int(ocean_area),
                beach_pixels=int(beach_area),
                rejected_ocean_components=rejected_ocean,
                rejected_beach_components=rejected_beach,
                orientation_used=str(candidate["orientation"]) if candidate else None,
                shoreline_axis=str(candidate["axis"]) if candidate else None,
                measured_coverage=float(candidate["measured_fraction"]) if candidate else 0.0,
                bridge_kernel_px=int(close_kernel_px),
            )
        if float(candidate["measured_fraction"]) < coverage_ratio_value:
            return _unavailable(
                reason="insufficient_coastline_evidence",
                confidence=confidence_value,
                calibration_version=calibration_version,
                ocean_pixels=int(ocean_area),
                beach_pixels=int(beach_area),
                rejected_ocean_components=rejected_ocean,
                rejected_beach_components=rejected_beach,
                orientation_used=str(candidate["orientation"]),
                shoreline_axis=str(candidate["axis"]),
                measured_coverage=float(candidate["measured_fraction"]),
                bridge_kernel_px=int(close_kernel_px),
            )
        line = np.asarray(candidate["line"], dtype=np.float64)
        calibrated_position_m: float | None = None
        if (
            calibration is not None
            and calibration.homography is not None
            and seaward_axis is not None
        ):
            axis = np.asarray(seaward_axis, dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                raise ShorelineExtractionError("seaward_axis must be a non-zero vector")
            axis = axis / norm
            world = apply_homography(calibration.homography, [(p[0], p[1]) for p in line])
            calibrated_position_m = float(np.mean(world @ axis))
        return ShorelineObservationResult(
            points=line.tolist(),
            calibrated_position_m=calibrated_position_m,
            segmentation_confidence=confidence_value,
            calibration_version=calibration_version,
            segments=[line.tolist()],
            shoreline_available=True,
            valid_columns=int(candidate["measured_count"]),
            ocean_pixels=int(ocean_area),
            beach_pixels=int(beach_area),
            rejected_ocean_components=rejected_ocean,
            rejected_beach_components=rejected_beach,
            shoreline_axis=str(candidate["axis"]),
            interpolated_ranges=[
                [int(value) for value in item] for item in candidate["interpolated_ranges"]
            ],
            measured_coverage=float(candidate["measured_fraction"]),
            orientation_used=str(candidate["orientation"]),
            bridge_kernel_px=int(close_kernel_px),
            median_kernel_px=int(smoothing_value),
            max_gap_px=float(max_gap_value),
            margin_px=int(margin_value),
        )

    orientation_value = orientation
    if orientation_value is None:
        orientation_value = settings.shoreline_orientation
    orientation_value = str(orientation_value).strip().lower()
    auto_orientation = orientation_value == "auto"
    if beach_check_depth_px is None:
        beach_check_depth_px = max(
            3, int(round(min(height, width) * settings.shoreline_beach_check_depth_ratio))
        )
    if auto_orientation:
        orientation_value = _infer_orientation(ocean, beach, depth=int(beach_check_depth_px))
    if orientation_value not in SCAN_ORIENTATIONS:
        raise ShorelineExtractionError(
            f"shoreline_orientation must be 'auto' or one of {SCAN_ORIENTATIONS}; "
            f"got {orientation_value!r}"
        )

    selection_value = ocean_component_selection
    if selection_value is None:
        selection_value = settings.shoreline_ocean_component_selection
    selection_value = str(selection_value).strip().lower()
    if selection_value not in ("all", "coastline"):
        raise ShorelineExtractionError(
            "shoreline_ocean_component_selection must be 'coastline' or 'all'; "
            f"got {selection_value!r}"
        )
    depth_ratio_value = min_branch_depth_ratio
    if depth_ratio_value is None:
        depth_ratio_value = settings.shoreline_min_branch_depth_ratio
    depth_ratio_value = float(depth_ratio_value)
    if depth_ratio_value < 0 or depth_ratio_value > 1:
        raise ShorelineExtractionError(
            "shoreline_min_branch_depth_ratio must be between 0 and 1; "
            f"got {depth_ratio_value!r}"
        )
    ocean_components_kept: int | None = None
    ocean_components_rejected_geometry = 0
    if selection_value == "coastline":
        selected_ocean, ocean_components_kept, ocean_components_rejected_geometry = (
            _select_coastline_components(ocean, beach, array == SKY, orientation=orientation_value)
        )
        if ocean_components_rejected_geometry:
            ocean = selected_ocean
            ocean_area = int(np.count_nonzero(ocean))
            rejected_ocean += ocean_components_rejected_geometry
            if auto_orientation:
                orientation_value = _infer_orientation(
                    ocean, beach, depth=int(beach_check_depth_px)
                )

    scan_length = height if orientation_value in ("land_below", "land_above") else width
    if max_gap_px is None:
        max_gap_px = max(8.0, scan_length * settings.shoreline_max_gap_ratio)
    if min_segment_px is None:
        min_segment_px = max(4, int(round(scan_length * settings.shoreline_min_segment_ratio)))
    if margin_px is None:
        margin_px = max(0, int(round(min(height, width) * settings.shoreline_margin_ratio)))
    if median_kernel_px is None:
        median_kernel_px = _odd_kernel(min(height, width) * settings.shoreline_median_kernel_ratio)
    if smoothing_window is not None:
        # Legacy parameter: 0 preserves raw detail; >1 overrides the median
        # kernel width for callers that used the old extractor.
        median_kernel_px = int(smoothing_window)

    land_fraction = _land_fraction(beach, orientation_value, int(beach_check_depth_px))
    scanline_points = _scanline_boundary(
        ocean,
        land_fraction,
        orientation=orientation_value,
        margin=int(margin_px),
    )
    segments = _build_segments(
        scanline_points,
        orientation=orientation_value,
        max_gap=float(max_gap_px),
        min_segment_px=int(min_segment_px),
        smoothing_window=int(median_kernel_px),
    )
    branches_rejected_depth = 0
    if selection_value == "coastline":
        segments, branches_rejected_depth, _ = _filter_thin_ocean_branches(
            segments,
            ocean,
            orientation_value,
            depth_ratio_value,
        )
    valid_columns = int(sum(len(segment) for segment in segments))

    if len(scanline_points) == 0:
        return _unavailable(
            reason="no Ocean/Beach contact in any scanline",
            confidence=confidence_value,
            calibration_version=calibration_version,
            ocean_pixels=int(ocean_area),
            beach_pixels=int(beach_area),
            rejected_ocean_components=rejected_ocean,
            rejected_beach_components=rejected_beach,
            orientation_used=orientation_value,
            bridge_kernel_px=int(close_kernel_px),
            median_kernel_px=int(median_kernel_px),
            max_gap_px=float(max_gap_px),
            margin_px=int(margin_px),
        )
    if valid_columns < int(min_valid_columns) or not segments:
        return _unavailable(
            reason=(
                "too few columns contain an Ocean/Beach interface after " "gap and length filtering"
            ),
            confidence=confidence_value,
            calibration_version=calibration_version,
            ocean_pixels=int(ocean_area),
            beach_pixels=int(beach_area),
            rejected_ocean_components=rejected_ocean,
            rejected_beach_components=rejected_beach,
            valid_columns=valid_columns,
            orientation_used=orientation_value,
            bridge_kernel_px=int(close_kernel_px),
            median_kernel_px=int(median_kernel_px),
            max_gap_px=float(max_gap_px),
            margin_px=int(margin_px),
        )

    ordered = [segment.astype(np.float64) for segment in segments]
    all_points = np.concatenate(ordered, axis=0)

    segmented_position_m: float | None = None
    if calibration is not None and calibration.homography is not None and seaward_axis is not None:
        axis = np.asarray(seaward_axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ShorelineExtractionError("seaward_axis must be a non-zero vector")
        axis = axis / norm
        world = apply_homography(calibration.homography, [(p[0], p[1]) for p in all_points])
        segmented_position_m = float(np.mean(world @ axis))

    return ShorelineObservationResult(
        # Legacy flat field: the concatenation of the ordered segments. It is
        # no longer row-major; consumers should prefer ``segments``.
        points=all_points.tolist(),
        calibrated_position_m=segmented_position_m,
        segmentation_confidence=confidence_value,
        calibration_version=calibration_version,
        segments=[segment.tolist() for segment in ordered],
        shoreline_available=True,
        valid_columns=valid_columns,
        ocean_pixels=int(ocean_area),
        beach_pixels=int(beach_area),
        rejected_ocean_components=rejected_ocean,
        rejected_beach_components=rejected_beach,
        ocean_components_kept=ocean_components_kept,
        ocean_components_rejected_geometry=ocean_components_rejected_geometry,
        branches_rejected_depth=branches_rejected_depth,
        orientation_used=orientation_value,
        bridge_kernel_px=int(close_kernel_px),
        median_kernel_px=int(median_kernel_px),
        max_gap_px=float(max_gap_px),
        margin_px=int(margin_px),
    )
