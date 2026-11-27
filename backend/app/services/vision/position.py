"""Robust, segment-aware shoreline position measurement.

The extractor answers *where* the waterline is as one or more ordered polylines.
This module answers *what single number represents it over time* without
pretending that separate branches are the same line.

Conventions
-----------
* For a horizontal shoreline (``land_below`` / ``land_above``) the position is
  reported on the image **y** axis. For a vertical shoreline (``land_left`` /
  ``land_right``) it is reported on the image **x** axis.
* ``shoreline_position_px`` is the raw image coordinate.
* ``signed_position_px`` uses the repository-wide sign convention
  **positive = seaward**:

  ===========  ===========  =================================
  orientation  seaward dir  signed position
  ===========  ===========  =================================
  land_below   up (−y)      −y
  land_above   down (+y)    +y
  land_left    right (+x)   +x
  land_right   left (−x)    −x

* The representative value is the **median of the dominant (longest) segment**,
  not an average across segments. Per-segment medians and lengths are retained
  so a second branch is visible rather than silently blended.

Everything here is in pixels. Metres require a separate, valid homography; this
module never invents one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

HORIZONTAL_ORIENTATIONS = ("land_below", "land_above")
VERTICAL_ORIENTATIONS = ("land_left", "land_right")
POSITION_SIGN = "seaward_positive"


@dataclass(slots=True)
class SegmentMeasurement:
    index: int
    point_count: int
    axis: str
    median_position: float
    min_position: float
    max_position: float
    signed_median: float
    region_median_position: float | None = None
    region_point_count: int | None = None
    in_primary_region: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "point_count": self.point_count,
            "axis": self.axis,
            "median_position": round(self.median_position, 3),
            "min_position": round(self.min_position, 3),
            "max_position": round(self.max_position, 3),
            "signed_median": round(self.signed_median, 3),
            "region_median_position": (
                None
                if self.region_median_position is None
                else round(self.region_median_position, 3)
            ),
            "region_point_count": self.region_point_count,
            "in_primary_region": self.in_primary_region,
        }


@dataclass(slots=True)
class ShorelinePosition:
    available: bool
    position_px: float | None = None
    signed_position_px: float | None = None
    position_axis: str | None = None
    position_sign: str = POSITION_SIGN
    orientation: str | None = None
    segment_count: int = 0
    dominant_segment_index: int | None = None
    selection_method: str | None = None
    segments: list[SegmentMeasurement] = field(default_factory=list)
    reason: str | None = None

    def segment_dicts(self) -> list[dict[str, Any]]:
        return [segment.as_dict() for segment in self.segments]


def _axis_for(orientation: str) -> str:
    if orientation in HORIZONTAL_ORIENTATIONS:
        return "y"
    if orientation in VERTICAL_ORIENTATIONS:
        return "x"
    raise ValueError(
        "orientation must be one of land_below/land_above/land_left/land_right; "
        f"got {orientation!r}"
    )


def _signed(value: float, orientation: str) -> float:
    if orientation == "land_below":
        return -value
    if orientation == "land_above":
        return value
    if orientation == "land_left":
        return value
    if orientation == "land_right":
        return -value
    raise ValueError(f"unknown orientation {orientation!r}")


def measure_shoreline_position(
    segments: Any,
    orientation: str | None,
    *,
    min_segment_points: int = 2,
    previous_signed_position: float | None = None,
    continuity_min_length_fraction: float = 0.25,
    frame_shape: tuple[int, int] | None = None,
    primary_region_fraction: float | None = None,
) -> ShorelinePosition:
    """Measure the representative position of ordered shoreline segments.

    ``segments`` is the extractor's ``[[[x, y], ...], ...]``. Empty, malformed
    or short segments are rejected; if nothing usable remains the result is
    ``available=False`` with a reason. No value is fabricated.

    When the frame contains more than one substantial branch (for example a
    near-shore waterline plus the far shore across a bay), the longest branch is
    chosen for the first frame. On later frames, if a previous position is
    supplied, the substantial branch closest to it is tracked. This prevents a
    branch switch from masquerading as a huge shoreline jump; the untracked
    branch is still reported in ``segment_measurements``.

    ``frame_shape`` + ``primary_region_fraction`` additionally restrict the
    measurement to the **near-shore/foreground region** (lower half for
    ``land_below``, upper for ``land_above``, side for left/right). This is a
    declared region, not a cross-branch average: every segment keeps its full
    statistics, and the region median is stored separately. If no segment has
    points in the region the full-segment median is used and labelled as such.
    """
    if not segments:
        return ShorelinePosition(available=False, reason="no shoreline segments")
    if orientation not in (*HORIZONTAL_ORIENTATIONS, *VERTICAL_ORIENTATIONS):
        return ShorelinePosition(
            available=False,
            reason=f"unsupported shoreline orientation: {orientation!r}",
        )
    axis = _axis_for(orientation)
    column = 1 if axis == "y" else 0
    height = width = None
    fraction = None
    if frame_shape is not None and primary_region_fraction is not None:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        fraction = max(0.0, min(1.0, float(primary_region_fraction)))

    measurements: list[SegmentMeasurement] = []
    for index, segment in enumerate(segments):
        try:
            array = np.asarray(segment, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if array.ndim != 2 or array.shape[1] != 2:
            continue
        finite = array[np.isfinite(array).all(axis=1)]
        if len(finite) < max(2, int(min_segment_points)):
            continue
        values = finite[:, column]
        median = float(np.median(values))
        region_values: np.ndarray | None = None
        if fraction is not None and height is not None and width is not None:
            if axis == "y":
                if orientation == "land_below":
                    region_values = values[values >= fraction * height]
                else:
                    region_values = values[values <= (1.0 - fraction) * height]
            else:
                if orientation == "land_left":
                    region_values = values[values <= fraction * width]
                else:
                    region_values = values[values >= (1.0 - fraction) * width]
            if region_values.size < max(2, int(min_segment_points)):
                region_values = None
        region_median = None if region_values is None else float(np.median(region_values))
        measurements.append(
            SegmentMeasurement(
                index=index,
                point_count=int(len(finite)),
                axis=axis,
                median_position=median,
                min_position=float(values.min()),
                max_position=float(values.max()),
                signed_median=_signed(median, orientation),
                region_median_position=region_median,
                region_point_count=(None if region_values is None else int(region_values.size)),
                in_primary_region=region_median is not None,
            )
        )

    if not measurements:
        return ShorelinePosition(
            available=False,
            orientation=orientation,
            reason="no usable shoreline segment",
        )

    region_applied = fraction is not None and any(
        item.region_point_count is not None for item in measurements
    )

    def effective_count(item: SegmentMeasurement) -> int:
        if region_applied:
            return item.region_point_count or 0
        return item.point_count

    def effective_position(item: SegmentMeasurement) -> float:
        value = (
            item.region_median_position
            if item.region_median_position is not None
            else item.median_position
        )
        return _signed(value, orientation)

    longest = max(measurements, key=effective_count)
    chosen = longest
    selection_method = (
        "foreground_longest_segment" if longest.in_primary_region else "longest_segment"
    )
    if previous_signed_position is not None and len(measurements) > 1:
        minimum = max(
            2,
            int(
                round(effective_count(longest) * max(0.0, min(1.0, continuity_min_length_fraction)))
            ),
        )
        candidates = [item for item in measurements if effective_count(item) >= minimum]
        if candidates:
            chosen = min(
                candidates,
                key=lambda item: abs(effective_position(item) - float(previous_signed_position)),
            )
            selection_method = (
                "foreground_temporal_continuity"
                if chosen.in_primary_region
                else "temporal_continuity"
            )
    chosen_position = (
        chosen.region_median_position
        if chosen.region_median_position is not None
        else chosen.median_position
    )
    return ShorelinePosition(
        available=True,
        position_px=chosen_position,
        signed_position_px=_signed(chosen_position, orientation),
        position_axis=axis,
        orientation=orientation,
        segment_count=len(measurements),
        dominant_segment_index=chosen.index,
        selection_method=selection_method,
        segments=measurements,
    )
