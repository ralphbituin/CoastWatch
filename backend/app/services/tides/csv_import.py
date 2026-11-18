"""CSV import and validation (Phase 2).

Supports configurable column mapping (timestamp, tide height, optional station
and optional tidal event type) with a preview step before commit.

Validation covers:

* missing timestamps / invalid dates,
* duplicate records,
* missing or non-numeric heights,
* out-of-range numeric values,
* time-zone consistency (naive vs aware vs mixed rows; varying UTC offsets reported),
* unit and vertical-datum metadata (declared per import, kept per dataset),
* chronological ordering (sorted on import, reported as a warning).

Nothing is silently discarded: rejected records (bounded sample) and their
reasons are persisted in ``dataset_imports``.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from app.core.exceptions import ValidationError

# Keep large rejections bounded but still informative.
REJECTION_SAMPLES_CAP = 100
# Sanity bound for a tide height (metres). Real-world maxima are ~16 m.
MAX_REASONABLE_HEIGHT_M = 25.0


@dataclass(slots=True)
class ColumnMapping:
    """Maps logical fields onto the CSV's column names."""

    timestamp: str = "timestamp"
    height: str = "tide_height"
    station: str | None = "station"
    event_type: str | None = "event_type"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ColumnMapping:
        if not data:
            return cls()
        return cls(
            timestamp=(data.get("timestamp") or "timestamp").strip(),
            height=(data.get("height") or "tide_height").strip(),
            station=(data.get("station") or "station").strip() or None,
            event_type=(data.get("event_type") or None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "height": self.height,
            "station": self.station,
            "event_type": self.event_type,
        }


@dataclass(slots=True)
class RejectedRow:
    line: int
    reason: str
    raw: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedImport:
    records: list[dict[str, Any]]
    rejected: list[RejectedRow]
    columns_detected: list[str]
    total_records: int
    out_of_order: int
    warnings: list[str] = field(default_factory=list)

    @property
    def valid_records(self) -> int:
        return len(self.records)

    def rejection_summary(self) -> dict[str, Any]:
        counts = Counter(r.reason for r in self.rejected)
        return {
            "by_reason": dict(sorted(counts.items())),
            "out_of_order": self.out_of_order,
            "warnings": self.warnings,
        }

    def rejection_samples(
        self, cap: int = REJECTION_SAMPLES_CAP
    ) -> tuple[list[dict[str, Any]], bool]:
        samples = [{"line": r.line, "reason": r.reason, "raw": r.raw} for r in self.rejected[:cap]]
        return samples, len(self.rejected) > cap

    def coverage(self) -> tuple[datetime | None, datetime | None]:
        if not self.records:
            return None, None
        stamps = [r["timestamp"] for r in self.records]
        return min(stamps), max(stamps)


def _normalise_height(raw: str) -> float:
    """Parse a height cell, accepting common decimal representations."""
    text = raw.strip().replace(" ", "")
    if "," in text and "." in text:
        # Both separators present: comma is a thousands separator.
        text = text.replace(",", "")
    elif "," in text:
        # Only comma present: decimal comma.
        text = text.replace(",", ".")
    return float(text)


def validate_timezone(timezone: str) -> ZoneInfo:
    """Return a ``ZoneInfo`` for ``timezone`` or raise ``ValidationError``."""
    try:
        return ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValidationError(
            f"Unknown IANA time zone: {timezone!r}",
            details={"timezone": timezone},
        ) from exc


def parse_csv(
    content: bytes,
    *,
    mapping: ColumnMapping | None = None,
    timezone: str = "Asia/Manila",
) -> ParsedImport:
    """Parse and validate a NAMRIA-style tide CSV.

    Raises :class:`ValidationError` for structural problems (unreadable file,
    missing required columns, invalid time zone). Per-row problems are
    collected in the returned :class:`ParsedImport`.
    """
    mapping = mapping or ColumnMapping()
    tz = validate_timezone(timezone)

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError(
            "File is not valid UTF-8 text; convert the dataset to UTF-8 CSV first.",
            details={"error": str(exc)},
        ) from exc

    reader = csv.DictReader(io.StringIO(text))
    columns_detected = [c.strip() for c in (reader.fieldnames or []) if c and c.strip()]
    rows = [
        {(k.strip() if k else k): (v.strip() if v else "") for k, v in row.items()}
        for row in reader
    ]

    missing = [c for c in (mapping.timestamp, mapping.height) if c not in columns_detected]
    if missing:
        raise ValidationError(
            "Required column(s) not found in CSV header: " + ", ".join(missing),
            details={"missing_columns": missing, "columns_detected": columns_detected},
        )

    rejected: list[RejectedRow] = []
    candidates: list[dict[str, Any]] = []

    for line, raw in enumerate(rows, start=2):  # header = line 1
        ts_raw = raw.get(mapping.timestamp, "").strip()
        height_raw = raw.get(mapping.height, "").strip()

        if not ts_raw:
            rejected.append(RejectedRow(line, "missing_timestamp", raw))
            continue

        parsed = pd.to_datetime(ts_raw, errors="coerce")
        if pd.isna(parsed):
            rejected.append(RejectedRow(line, "invalid_timestamp", raw))
            continue

        pdt = parsed.to_pydatetime()
        offset_seconds = pdt.utcoffset().total_seconds() if pdt.tzinfo else None
        aware = pdt.tzinfo is not None
        timestamp = pdt.astimezone(UTC) if aware else pdt.replace(tzinfo=tz).astimezone(UTC)

        if not height_raw:
            rejected.append(RejectedRow(line, "missing_height", raw))
            continue
        try:
            height = _normalise_height(height_raw)
        except ValueError:
            rejected.append(RejectedRow(line, "invalid_height", raw))
            continue
        if abs(height) > MAX_REASONABLE_HEIGHT_M:
            rejected.append(RejectedRow(line, "out_of_range", raw))
            continue

        event_type = None
        if mapping.event_type:
            event_raw = (raw.get(mapping.event_type, "") or "").strip().lower()
            event_type = {"h": "high", "high": "high", "l": "low", "low": "low"}.get(event_raw)

        candidates.append(
            {
                "timestamp": timestamp,
                "height": height,
                "event_type": event_type,
                "line": line,
                "raw": raw,
                "aware": aware,
                "offset_seconds": offset_seconds,
            }
        )

    warnings: list[str] = []

    # Time-zone consistency: mix naive and aware rows cannot be reconciled.
    aware_flags = [c["aware"] for c in candidates]
    if any(aware_flags) and not all(aware_flags):
        kept: list[dict[str, Any]] = []
        for cand in candidates:
            if cand["aware"]:
                kept.append(cand)
            else:
                rejected.append(RejectedRow(cand["line"], "mixed_timezone", cand["raw"]))
        candidates = kept
        warnings.append("rejected timezone-naive rows in a mixed aware/naive file")

    distinct_offsets = {c["offset_seconds"] for c in candidates if c["aware"]}
    if len(distinct_offsets) > 1:
        warnings.append("timestamps carry varying UTC offsets (DST?); values normalised to UTC")

    # Duplicate records inside the file (first wins).
    seen: set[datetime] = set()
    unique: list[dict[str, Any]] = []
    for cand in candidates:
        if cand["timestamp"] in seen:
            rejected.append(RejectedRow(cand["line"], "duplicate", cand["raw"]))
        else:
            seen.add(cand["timestamp"])
            unique.append(cand)

    # Chronological ordering is validated, not fatal: sort and report.
    stamps = [c["timestamp"] for c in unique]
    out_of_order = sum(1 for i in range(1, len(stamps)) if stamps[i] < stamps[i - 1])
    unique.sort(key=lambda c: c["timestamp"])

    return ParsedImport(
        records=unique,
        rejected=rejected,
        columns_detected=columns_detected,
        total_records=len(rows),
        out_of_order=out_of_order,
        warnings=warnings,
    )


def file_hash(content: bytes) -> str:
    """SHA-256 of the raw upload, used for insert-only provenance."""
    return hashlib.sha256(content).hexdigest()
