"""Segment-aware shoreline position measurement and sign conventions."""

from __future__ import annotations

import pytest

from app.services.vision.position import measure_shoreline_position


def _horizontal(segment_y: float, start_x: int, count: int) -> list[list[float]]:
    return [[float(x), segment_y] for x in range(start_x, start_x + count)]


def test_land_below_is_seaward_negative_y() -> None:
    result = measure_shoreline_position([_horizontal(800, 0, 10)], "land_below")
    assert result.available
    assert result.position_axis == "y"
    assert result.position_px == 800
    assert result.signed_position_px == -800
    assert result.position_sign == "seaward_positive"


def test_land_above_is_seaward_positive_y() -> None:
    result = measure_shoreline_position([_horizontal(200, 0, 10)], "land_above")
    assert result.signed_position_px == 200


def test_vertical_orientations_have_opposite_signs() -> None:
    segment = [[900.0, float(y)] for y in range(0, 10)]
    left = measure_shoreline_position([segment], "land_left")
    right = measure_shoreline_position([segment], "land_right")
    assert left.position_axis == "x"
    assert left.signed_position_px == 900
    assert right.signed_position_px == -900


def test_dominant_segment_is_not_blended_with_short_branch() -> None:
    main = _horizontal(700, 0, 100)
    small_branch = _horizontal(100, 0, 5)
    result = measure_shoreline_position([main, small_branch], "land_below")
    assert result.segment_count == 2
    assert result.dominant_segment_index == 0
    assert result.position_px == 700
    medians = [segment.median_position for segment in result.segments]
    assert medians == [700.0, 100.0]


def test_foreground_region_excludes_the_far_shore_branch() -> None:
    # A far-shore branch at y~300 and a near-shore branch at y~900 in a 1080p frame.
    far = _horizontal(300, 0, 80)
    near = _horizontal(900, 0, 60)
    result = measure_shoreline_position(
        [far, near],
        "land_below",
        frame_shape=(1080, 1920),
        primary_region_fraction=0.5,
    )
    assert result.dominant_segment_index == 1
    assert result.signed_position_px == -900
    assert result.selection_method.startswith("foreground_")
    # Both branches remain visible in the per-segment diagnostics.
    assert result.segment_count == 2
    assert result.segments[0].in_primary_region is False
    assert result.segments[1].in_primary_region is True


def test_foreground_region_uses_the_lower_part_of_a_joined_segment() -> None:
    # One connected segment spanning far and near shore; the region median must
    # come from the lower (near-shore) points, not the full-line median.
    joined = [[float(x), 300.0 + x * (600.0 / 900.0)] for x in range(0, 900)]
    result = measure_shoreline_position(
        [joined],
        "land_below",
        frame_shape=(1080, 1920),
        primary_region_fraction=0.5,
    )
    assert result.available
    assert result.position_px is not None
    assert result.position_px >= 540
    assert result.position_px != result.segments[0].median_position


def test_region_falls_back_when_no_points_are_in_it() -> None:
    far = _horizontal(300, 0, 20)
    result = measure_shoreline_position(
        [far],
        "land_below",
        frame_shape=(1080, 1920),
        primary_region_fraction=0.5,
    )
    assert result.available
    assert result.position_px == 300
    assert result.selection_method == "longest_segment"


def test_temporal_continuity_tracks_the_same_branch() -> None:
    near = _horizontal(900, 0, 60)
    far = _horizontal(300, 0, 80)  # longer, but a different branch
    first = measure_shoreline_position([near, far], "land_below")
    assert first.dominant_segment_index == 1  # first frame picks the longest
    tracked = measure_shoreline_position(
        [near, far],
        "land_below",
        previous_signed_position=first.signed_position_px,
    )
    assert tracked.dominant_segment_index == 1
    switched = measure_shoreline_position(
        [near, far],
        "land_below",
        previous_signed_position=-900.0,
    )
    assert switched.dominant_segment_index == 0
    assert switched.selection_method == "temporal_continuity"
    assert switched.signed_position_px == -900


def test_median_is_robust_to_one_outlier() -> None:
    segment = _horizontal(500, 0, 9)
    segment[4][1] = 900.0
    result = measure_shoreline_position([segment], "land_below")
    assert result.position_px == 500


@pytest.mark.parametrize(
    "segments,orientation",
    [
        ([], "land_below"),
        ([[[1.0, 2.0]]], "land_below"),
        ([[[1.0, 2.0], [2.0, 3.0]]], "not_an_orientation"),
        ([["bad", "points"]], "land_below"),
    ],
)
def test_invalid_input_is_unavailable_with_reason(segments, orientation) -> None:
    result = measure_shoreline_position(segments, orientation)
    assert not result.available
    assert result.position_px is None
    assert result.reason
