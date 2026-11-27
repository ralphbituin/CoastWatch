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
* keep large Ocean and Beach components, rejecting small artifacts;
* when several Ocean components remain, keep only the component that shares
  the longest border with the dominant Beach and any other *open* (Sky- or
  frame-border-touching) component on its seaward side, so a false-Ocean patch
  on land cannot be scanned; a single component is never re-selected;
* pick one Ocean pixel per scanline that touches Beach on the land side, so
  separate interface branches can never be joined by a median;
* ignore the frame margin; keep a scanline when an Ocean pixel has Beach on the
  land side (open water exiting the frame fails that test);
* median-filter along the scanline, split at gaps, drop short segments;
* keep the deepest real branch and drop secondary branches whose local
  seaward water is thinner than ``SHORELINE_MIN_BRANCH_DEPTH_RATIO``.

The Sky/Ocean horizon is never used: Sky is not Beach, so no contact exists
there. Sky adjacency is used only for Ocean component selection (an open-sea
signal), never to draw the line on the horizon. Coordinates are original-frame
pixels. Calibration is a separate hook.
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
    # The foreground side with the most Beach-adjacent Ocean boundary wins.
    # land_above is kept available for an explicit override but is not chosen
    # automatically. No side gets an automatic preference: on an oblique view
    # the left/right boundary can outnumber the below boundary, and forcing
    # land_below truncates the line (audited 2026-11-26).
    foreground = ("land_below", "land_left", "land_right")
    return max(foreground, key=lambda name: counts[name])


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
      Beach and lies on its seaward side (using the already inferred scan
      orientation);
    * the eligible component with the longest shared Beach border is the anchor
      (the real coastline);
    * other eligible components are kept only when they are open water, i.e.
      they touch Sky or the frame border. A patch fully enclosed by Beach is a
      pool/rock/shadow, not the shoreline.

    Sky adjacency is used *only* to identify open sea for component selection;
    it is never used to draw the line on the Sky/Ocean horizon. When no
    eligible component exists (for example the two-Beach side-view geometry),
    the caller keeps the area-filtered mask, so single-component frames are
    unaffected.
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
    """
    points = np.asarray(segment, dtype=np.float64)
    height, width = ocean.shape
    seaward = np.asarray(_SEAWARD_VECTORS[orientation], dtype=np.float64)
    runs = np.zeros(len(points), dtype=np.float64)
    for index, point in enumerate(points):
        previous = points[max(0, index - 1)]
        following = points[min(len(points) - 1, index + 1)]
        tangent = following - previous
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-9:
            normal = seaward
        else:
            tangent /= norm
            normal = np.array([-tangent[1], tangent[0]])
            if float(normal @ seaward) < 0:
                normal = -normal
        x, y = point
        run = 0
        for step in range(1, max_steps + 1):
            xx = int(round(x + normal[0] * step))
            yy = int(round(y + normal[1] * step))
            if xx < 0 or xx >= width or yy < 0 or yy >= height or not ocean[yy, xx]:
                break
            run = step
        runs[index] = run
    return float(np.percentile(runs, 90))


def _filter_thin_ocean_branches(
    segments: list[np.ndarray],
    ocean: np.ndarray,
    orientation: str,
    min_depth_ratio: float,
) -> tuple[list[np.ndarray], int]:
    """Drop secondary branches whose seaward water body is too thin.

    The anchor branch (the one with the deepest local open water) is always
    kept. This is the branch that carries the real coastline. Other branches
    are kept only when their local seaward depth reaches ``min_depth_ratio`` of
    the short frame side. A pier-split real ocean has two deep halves and both
    survive; a thin landward strip between Beach and Sky does not.
    """
    if len(segments) < 2 or min_depth_ratio <= 0:
        return segments, 0
    short = min(ocean.shape)
    max_steps = max(8, int(round(short * max(0.12, min_depth_ratio * 2))))
    depths = [_branch_ocean_depth(segment, ocean, orientation, max_steps) for segment in segments]
    anchor_index = max(
        range(len(segments)), key=lambda index: (depths[index], len(segments[index]))
    )
    threshold = float(min_depth_ratio) * short
    kept = [
        segment
        for index, segment in enumerate(segments)
        if index == anchor_index or depths[index] >= threshold
    ]
    return kept, len(segments) - len(kept)


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
            # land_fraction already requires Beach on the land side, so open
            # water that exits the frame (no Beach beyond it) cannot qualify.
            # Do not pre-skip the whole scanline because some other Ocean
            # component touches an edge; that used to drop valid contact.
            valid = rows[
                (rows >= margin) & (rows < height - margin) & (land_fraction[rows, x] >= 0.5)
            ]
            if valid.size:
                y = valid.max() if orientation == "land_below" else valid.min()
                points.append((float(x), float(y)))
        return np.asarray(points, dtype=np.float64) if points else np.empty((0, 2))

    for y in range(margin, height - margin):
        cols = np.flatnonzero(ocean[y, :])
        if cols.size == 0:
            continue
        valid = cols[(cols >= margin) & (cols < width - margin) & (land_fraction[y, cols] >= 0.5)]
        if valid.size:
            x = valid.min() if orientation == "land_left" else valid.max()
            points.append((float(x), float(y)))
    return np.asarray(points, dtype=np.float64) if points else np.empty((0, 2))


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
        # Include the two measured endpoints so a renderer can style the whole
        # bridged span, not just its interior.
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
    """Pick the largest scan-contiguous run of one component.

    Gaps caused by an occluder remain in the run and are bridged later. Gaps
    where the Ocean leaves the frame are not bridged; the larger side becomes
    the anchor and the other side is discarded.
    """
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
) -> dict[str, object] | None:
    """Build one coherent axis candidate from a selected Ocean mask."""
    height, width = selected.shape
    short = min(height, width)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8), connectivity=8
    )
    best: dict[str, object] | None = None
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
            _land_fraction(beach, orientation, beach_depth),
            orientation=orientation,
            margin=margin_px,
        )
        segments = _build_segments(
            points,
            orientation=orientation,
            max_gap=max_gap_px,
            min_segment_px=min_segment_px,
            smoothing=smoothing_px,
        )
        segments, _ = _filter_thin_ocean_branches(
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
        max_steps = max(8, int(round(short * 0.12)))
        depths = [_branch_ocean_depth(segment, sub, orientation, max_steps) for segment in segments]
        best = {
            "orientation": orientation,
            "axis": "horizontal" if vertical else "vertical",
            "segments": segments,
            "run_points": measured,
            "solidity": solidity,
            "measured_fraction": measured_fraction,
            "score": score,
            "anchor_depth": max(depths),
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
) -> dict[str, object] | None:
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
    best: dict[str, object] | None = None
    for orientation in orientations or ORIENTATIONS:
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
    ocean_component_selection: str = "coastline",
    min_branch_depth_ratio: float = 0.06,
    single_line_mode: bool = True,
    min_measured_coverage_ratio: float = 0.20,
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
        ocean_closed = cv2.morphologyEx(ocean_raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(
            bool
        )
    else:
        ocean_closed = ocean_raw

    ocean, ocean_area, rejected_ocean = _keep_large_components(ocean_closed, ocean_min)
    beach, beach_area, rejected_beach = _keep_large_components(beach_raw, beach_min)
    if not ocean.any() or ocean_area < ocean_min or not beach.any() or beach_area < beach_min:
        reason = (
            "dominant ocean component too small" if ocean_area < ocean_min else "no usable beach"
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

    if single_line_mode:
        if str(ocean_component_selection).strip().lower() not in ("all", "coastline"):
            raise ValueError("ocean_component_selection must be 'coastline' or 'all'")
        if not 0 <= float(min_branch_depth_ratio) <= 1:
            raise ValueError("min_branch_depth_ratio must be between 0 and 1")
        orientation_signal = str(orientation).strip().lower()
        if orientation_signal != "auto" and orientation_signal not in ORIENTATIONS:
            raise ValueError(f"orientation must be 'auto' or one of {ORIENTATIONS}")
        if not 0 <= float(min_measured_coverage_ratio) <= 1:
            raise ValueError("min_measured_coverage_ratio must be between 0 and 1")
        short = min(height, width)
        depth = max(3, round(short * 0.02))
        margin_value = max(0, int(margin if margin is not None else round(short * 0.005)))
        max_gap_value = float(
            max_gap if max_gap is not None else max(8.0, max(height, width) * 0.03)
        )
        min_segment_value = int(
            min_segment_px
            if min_segment_px is not None
            else max(4, round(max(height, width) * 0.015))
        )
        smoothing_value = int(smoothing if smoothing is not None else max(3, round(short * 0.008)))
        if smoothing_value % 2 == 0:
            smoothing_value += 1
        candidate = _select_single_line(
            ocean,
            beach,
            array == SKY,
            min_component_area=ocean_min,
            beach_depth=depth,
            margin_px=margin_value,
            max_gap_px=max_gap_value,
            min_segment_px=min_segment_value,
            smoothing_px=smoothing_value,
            min_branch_depth_ratio=float(min_branch_depth_ratio),
            orientations=(
                (str(orientation).strip().lower(),)
                if str(orientation).strip().lower() != "auto"
                else None
            ),
        )
        if candidate is None or int(candidate["measured_count"]) < int(min_valid_columns):
            return ShorelineResult(
                points=np.empty((0, 2), dtype=np.float64),
                per_column=np.empty((0, 2), dtype=np.float64),
                boundary_mask=empty,
                valid_columns=0,
                ocean_pixels=int(ocean_area),
                beach_pixels=int(beach_area),
                rejected_ocean_components=rejected_ocean,
                rejected_beach_components=rejected_beach,
                metadata={
                    "available": False,
                    "reason": "insufficient_coastline_evidence",
                    "orientation": candidate["orientation"] if candidate else None,
                    "shoreline_axis": candidate["axis"] if candidate else None,
                    "measured_coverage": float(candidate["measured_fraction"])
                    if candidate
                    else 0.0,
                },
            )
        if float(candidate["measured_fraction"]) < float(min_measured_coverage_ratio):
            return ShorelineResult(
                points=np.empty((0, 2), dtype=np.float64),
                per_column=np.empty((0, 2), dtype=np.float64),
                boundary_mask=empty,
                valid_columns=0,
                ocean_pixels=int(ocean_area),
                beach_pixels=int(beach_area),
                rejected_ocean_components=rejected_ocean,
                rejected_beach_components=rejected_beach,
                metadata={
                    "available": False,
                    "reason": "insufficient_coastline_evidence",
                    "orientation": candidate["orientation"],
                    "shoreline_axis": candidate["axis"],
                    "measured_coverage": float(candidate["measured_fraction"]),
                },
            )
        line = np.asarray(candidate["line"], dtype=np.float64)
        boundary_mask = np.zeros((height, width), dtype=np.uint8)
        for x, y in line:
            if 0 <= int(y) < height and 0 <= int(x) < width:
                boundary_mask[int(y), int(x)] = 1
        return ShorelineResult(
            points=line,
            per_column=line,
            boundary_mask=boundary_mask,
            valid_columns=int(candidate["measured_count"]),
            ocean_pixels=int(ocean_area),
            beach_pixels=int(beach_area),
            rejected_ocean_components=rejected_ocean,
            rejected_beach_components=rejected_beach,
            segments=[line],
            metadata={
                "available": True,
                "image_size": [int(width), int(height)],
                "orientation": candidate["orientation"],
                "shoreline_axis": candidate["axis"],
                "interpolated_ranges": candidate["interpolated_ranges"],
                "measured_coverage": float(candidate["measured_fraction"]),
                "measured_count": int(candidate["measured_count"]),
                "solidity": float(candidate["solidity"]),
                "anchor_depth": float(candidate["anchor_depth"]),
                "subregion_area": int(candidate["subregion_area"]),
                "component_area": int(candidate["component_area"]),
                "component_bbox": candidate["component_bbox"],
                "single_line_mode": True,
                "close_kernel_px": int(kernel_px),
                "smoothing_px": int(smoothing_value),
                "max_gap_px": float(max_gap_value),
                "min_segment_px": int(min_segment_value),
                "margin_px": int(margin_value),
                "segments": 1,
            },
        )

    depth = max(3, round(min(height, width) * 0.02))
    orientation_value = str(orientation).strip().lower()
    auto_orientation = orientation_value == "auto"
    if auto_orientation:
        orientation_value = _infer_orientation(ocean, beach, depth=depth)
    if orientation_value not in ORIENTATIONS:
        raise ValueError(f"orientation must be 'auto' or one of {ORIENTATIONS}")
    selection_value = str(ocean_component_selection).strip().lower()
    if selection_value not in ("all", "coastline"):
        raise ValueError(
            "ocean_component_selection must be 'coastline' or 'all'; "
            f"got {ocean_component_selection!r}"
        )
    depth_ratio_value = float(min_branch_depth_ratio)
    if depth_ratio_value < 0 or depth_ratio_value > 1:
        raise ValueError(
            "min_branch_depth_ratio must be between 0 and 1; " f"got {min_branch_depth_ratio!r}"
        )
    if selection_value == "coastline":
        selected, kept, discarded = _select_coastline_components(
            ocean, beach, mask == 2, orientation=orientation_value
        )
        if discarded:
            ocean = selected
            ocean_area = int(np.count_nonzero(ocean))
            rejected_ocean += discarded
            if auto_orientation:
                orientation_value = _infer_orientation(ocean, beach, depth=depth)
    ocean_components_kept = int(kept) if selection_value == "coastline" else 0
    ocean_components_discarded = int(discarded) if selection_value == "coastline" else 0
    scan_length = height if orientation_value in ("land_below", "land_above") else width
    margin_value = max(0, int(margin if margin is not None else round(min(height, width) * 0.005)))
    gap_value = float(max_gap if max_gap is not None else max(8.0, scan_length * 0.03))
    min_segment = int(
        min_segment_px if min_segment_px is not None else max(4, round(scan_length * 0.015))
    )
    smoothing_value = int(
        smoothing if smoothing is not None else max(3, round(min(height, width) * 0.008))
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
    branches_rejected_depth = 0
    if selection_value == "coastline":
        segments, branches_rejected_depth = _filter_thin_ocean_branches(
            segments,
            ocean,
            orientation_value,
            depth_ratio_value,
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
            "ocean_component_selection": selection_value,
            "ocean_components_kept": ocean_components_kept,
            "ocean_components_rejected_geometry": ocean_components_discarded,
            "branches_rejected_depth": branches_rejected_depth,
            "min_branch_depth_px": float(depth_ratio_value * min(height, width)),
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
