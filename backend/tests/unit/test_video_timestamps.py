"""Video metadata sampling and visible-clock timestamp derivation."""

from __future__ import annotations

from datetime import timedelta

from app.services.video.sampling import VideoMetadata, target_samples
from app.services.video.timestamps import ClockAnchor, derive_timestamp

ANCHORS = [
    ClockAnchor(
        elapsed_seconds=5.0,
        timestamp="2026-11-19T21:30:56+07:00",
        confidence=0.95,
        verification="manual visual verification of burned-in clock",
    ),
    ClockAnchor(
        elapsed_seconds=7200.0,
        timestamp="2026-11-19T23:30:50+07:00",
        confidence=0.95,
        verification="manual visual verification of burned-in clock",
    ),
    ClockAnchor(
        elapsed_seconds=15500.0,
        timestamp="2026-11-20T01:49:07+07:00",
        confidence=0.9,
        verification="manual visual verification of burned-in clock",
    ),
]

METADATA = VideoMetadata(
    path="webcam_6h.mp4",
    file_size_bytes=7_790_924_244,
    duration_seconds=15502.942041,
    fps=30.0,
    average_fps=29.9988,
    frame_count=465088,
    width=1920,
    height=1080,
    video_codec="vp9",
    audio_codec="aac",
)


def test_known_anchor_is_reproduced() -> None:
    result = derive_timestamp(7200.0, ANCHORS, timezone="Asia/Bangkok")
    assert result.capture_timestamp is not None
    assert result.capture_timestamp.isoformat() == "2026-11-19T23:30:50+07:00"
    assert result.timestamp_source == "derived"
    assert result.timezone == "Asia/Bangkok"


def test_interpolation_between_anchors() -> None:
    result = derive_timestamp(3600.0, ANCHORS, timezone="Asia/Bangkok")
    assert result.capture_timestamp is not None
    # 21:30:56 + 3595/7195 of the 7194-second clock delta.
    assert result.capture_timestamp.replace(microsecond=0).isoformat() == (
        "2026-11-19T22:30:50+07:00"
    )
    assert result.method == "piecewise_linear_between_anchors"


def test_extrapolation_past_last_anchor_uses_nearest_pair() -> None:
    result = derive_timestamp(15502.942, ANCHORS, timezone="Asia/Bangkok")
    assert result.capture_timestamp is not None
    expected = ANCHORS[2].parsed("Asia/Bangkok") + timedelta(seconds=2.942)
    assert abs((result.capture_timestamp - expected).total_seconds()) < 0.01
    assert result.method == "linear_extrapolation_from_nearest_anchors"


def test_no_anchor_is_unavailable_not_fabricated() -> None:
    result = derive_timestamp(30.0, [], timezone="Asia/Bangkok")
    assert result.capture_timestamp is None
    assert result.timestamp_source == "unavailable"
    assert result.reason


def test_naive_anchor_gets_the_declared_timezone() -> None:
    result = derive_timestamp(
        0.0,
        [ClockAnchor(0.0, "2026-11-19T21:30:00")],
        timezone="Asia/Bangkok",
    )
    assert result.capture_timestamp is not None
    assert result.capture_timestamp.utcoffset() == timedelta(hours=7)


def test_target_samples_are_517_for_this_recording() -> None:
    samples = target_samples(METADATA, sample_interval_seconds=30.0)
    assert len(samples) == 517
    assert samples[0] == (0.0, 0)
    assert samples[1] == (30.0, 900)
    assert samples[-1] == (15480.0, 464400)
    indices = [index for _, index in samples]
    assert len(indices) == len(set(indices))


def test_target_samples_respect_range_and_limit() -> None:
    samples = target_samples(
        METADATA,
        sample_interval_seconds=60.0,
        start_elapsed=120.0,
        end_elapsed=300.0,
        max_samples=2,
    )
    assert samples == [(120.0, 3600), (180.0, 5400)]
