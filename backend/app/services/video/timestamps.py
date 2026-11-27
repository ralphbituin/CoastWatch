"""Timestamp derivation from verified visible-clock anchors.

The recording container has no capture date/time. The burned-in CCTV clock is
the only absolute-time source. This module deliberately distinguishes:

* ``visible_clock`` — the clock was read on that sampled frame;
* ``derived`` — the timestamp is interpolated/extrapolated from verified clock
  anchors and elapsed video time (the normal case);
* ``container`` — a container creation time existed (not used for this footage);
* ``unavailable`` — no absolute timestamp can be established.

It never invents a time zone or treats a naive anchor as UTC. For this footage
the caller must pass ``Asia/Bangkok``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True, slots=True)
class ClockAnchor:
    """One verified reading of the burned-in clock."""

    elapsed_seconds: float
    timestamp: str
    source: str = "visible_clock"
    confidence: float = 0.9
    verification: str | None = None

    def parsed(self, timezone: str) -> datetime:
        moment = datetime.fromisoformat(self.timestamp)
        if moment.tzinfo is None:
            try:
                moment = moment.replace(tzinfo=ZoneInfo(timezone))
            except ZoneInfoNotFoundError:  # pragma: no cover - platform data issue
                raise ValueError(f"unknown timezone {timezone!r}") from None
        return moment


@dataclass(slots=True)
class DerivedTimestamp:
    capture_timestamp: datetime | None
    timestamp_source: str
    timestamp_confidence: float | None
    method: str | None
    timezone: str
    reason: str | None = None


def _coerce_anchor(anchor: ClockAnchor | dict) -> ClockAnchor:
    if isinstance(anchor, ClockAnchor):
        return anchor
    return ClockAnchor(
        elapsed_seconds=float(anchor["elapsed_seconds"]),
        timestamp=str(anchor["timestamp"]),
        source=str(anchor.get("source", "visible_clock")),
        confidence=float(anchor.get("confidence", 0.9)),
        verification=anchor.get("verification"),
    )


def derive_timestamp(
    elapsed_seconds: float,
    anchors: Sequence[ClockAnchor | dict] | None,
    *,
    timezone: str,
) -> DerivedTimestamp:
    """Map an elapsed video time to absolute time using verified clock anchors.

    Piecewise-linear interpolation is used between anchors and the nearest pair
    is extrapolated outside them. With one anchor the clock is assumed to run at
    real time. With none, the result is explicitly unavailable.
    """
    cleaned = sorted(
        (_coerce_anchor(anchor) for anchor in (anchors or [])),
        key=lambda anchor: anchor.elapsed_seconds,
    )
    if not cleaned:
        return DerivedTimestamp(
            capture_timestamp=None,
            timestamp_source="unavailable",
            timestamp_confidence=None,
            method=None,
            timezone=timezone,
            reason="no verified visible-clock anchor",
        )

    try:
        moments = [anchor.parsed(timezone) for anchor in cleaned]
    except ValueError as exc:
        return DerivedTimestamp(
            capture_timestamp=None,
            timestamp_source="unavailable",
            timestamp_confidence=None,
            method=None,
            timezone=timezone,
            reason=f"invalid clock anchor: {exc}",
        )

    if len(cleaned) == 1:
        anchor, anchor_time = cleaned[0], moments[0]
        result = anchor_time + timedelta(seconds=elapsed_seconds - anchor.elapsed_seconds)
        return DerivedTimestamp(
            capture_timestamp=result,
            timestamp_source="derived",
            timestamp_confidence=anchor.confidence,
            method="single_anchor_plus_elapsed",
            timezone=timezone,
        )

    # Choose the bracketing pair, or the nearest two for extrapolation.
    low = 0
    for index in range(len(cleaned) - 1):
        if cleaned[index].elapsed_seconds <= elapsed_seconds <= cleaned[index + 1].elapsed_seconds:
            low = index
            break
    else:
        low = 0 if elapsed_seconds < cleaned[0].elapsed_seconds else len(cleaned) - 2

    first, second = cleaned[low], cleaned[low + 1]
    span = second.elapsed_seconds - first.elapsed_seconds
    if abs(span) < 1e-9:  # pragma: no cover - duplicate anchors rejected upstream
        return DerivedTimestamp(
            capture_timestamp=moments[low],
            timestamp_source="derived",
            timestamp_confidence=min(first.confidence, second.confidence),
            method="duplicate_anchor",
            timezone=timezone,
        )

    fraction = (elapsed_seconds - first.elapsed_seconds) / span
    clock_delta = moments[low + 1] - moments[low]
    result = moments[low] + clock_delta * fraction
    method = "piecewise_linear_between_anchors"
    if not (cleaned[low].elapsed_seconds <= elapsed_seconds <= cleaned[low + 1].elapsed_seconds):
        method = "linear_extrapolation_from_nearest_anchors"
    return DerivedTimestamp(
        capture_timestamp=result,
        timestamp_source="derived",
        timestamp_confidence=min(first.confidence, second.confidence),
        method=method,
        timezone=timezone,
    )


def anchor_dicts(anchors: list[ClockAnchor]) -> list[dict]:
    return [
        {
            "elapsed_seconds": anchor.elapsed_seconds,
            "timestamp": anchor.timestamp,
            "source": anchor.source,
            "confidence": anchor.confidence,
            "verification": anchor.verification,
        }
        for anchor in anchors
    ]


def utc_iso(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    return moment.astimezone(UTC).isoformat()
