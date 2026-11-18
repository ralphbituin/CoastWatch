"""Unit tests for the CSV tide importer (parsing + validation)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.exceptions import ValidationError
from app.services.tides.csv_import import ColumnMapping, parse_csv

VALID_ROWS = [
    "2026-11-19 00:00,0.50,high",
    "2026-11-19 01:00,0.40,",
    "2026-11-19 02:00,0.30,low",
]


def make_csv(rows: list[str]) -> bytes:
    return ("timestamp,tide_height,event_type\n" + "\n".join(rows) + "\n").encode()


def test_parse_valid_file() -> None:
    parsed = parse_csv(make_csv(VALID_ROWS), mapping=ColumnMapping(), timezone="Asia/Manila")
    assert parsed.valid_records == 3
    assert parsed.rejected == []
    assert parsed.total_records == 3


def test_coverage_normalised_to_utc() -> None:
    parsed = parse_csv(make_csv(VALID_ROWS), mapping=ColumnMapping(), timezone="Asia/Manila")
    start, end = parsed.coverage()
    assert start == datetime(2026, 11, 18, 16, 0, tzinfo=UTC)
    assert end == datetime(2026, 11, 18, 18, 0, tzinfo=UTC)


def test_missing_timestamp_rejected() -> None:
    parsed = parse_csv(
        make_csv([",0.50", "2026-11-19 01:00,0.40"]),
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    assert [r.reason for r in parsed.rejected] == ["missing_timestamp"]
    assert parsed.valid_records == 1


def test_invalid_date_rejected() -> None:
    parsed = parse_csv(
        make_csv(["2026-13-40 00:00,0.50"]),
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    assert [r.reason for r in parsed.rejected] == ["invalid_timestamp"]


def test_missing_height_rejected() -> None:
    parsed = parse_csv(
        make_csv(["2026-11-19 00:00,"]),
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    assert [r.reason for r in parsed.rejected] == ["missing_height"]


def test_non_numeric_height_rejected() -> None:
    parsed = parse_csv(
        make_csv(["2026-11-19 00:00,not-a-number"]),
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    assert [r.reason for r in parsed.rejected] == ["invalid_height"]


def test_height_comma_decimal_accepted() -> None:
    parsed = parse_csv(
        b'timestamp,tide_height,event_type\n2026-11-19 00:00,"0,75",high\n',
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    assert parsed.valid_records == 1
    assert parsed.records[0]["height"] == 0.75


def test_height_out_of_range_rejected() -> None:
    parsed = parse_csv(
        make_csv(["2026-11-19 00:00,999"]),
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    assert [r.reason for r in parsed.rejected] == ["out_of_range"]


def test_duplicate_rejected() -> None:
    parsed = parse_csv(
        make_csv(["2026-11-19 00:00,0.50", "2026-11-19 00:00,0.60"]),
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    assert [r.reason for r in parsed.rejected] == ["duplicate"]
    assert parsed.records[0]["height"] == 0.50  # first wins


def test_mixed_naive_and_aware_rows() -> None:
    parsed = parse_csv(
        make_csv(["2026-11-19T00:00:00+08:00,0.50", "2026-11-19 01:00,0.60"]),
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    reasons = [r.reason for r in parsed.rejected]
    assert "mixed_timezone" in reasons
    assert parsed.valid_records == 1  # the aware row survives


def test_aware_offset_normalised_to_utc() -> None:
    parsed = parse_csv(
        make_csv(["2026-11-19T00:00:00+08:00,0.50"]),
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    assert parsed.records[0]["timestamp"] == datetime(2026, 11, 18, 16, 0, tzinfo=UTC)


def test_out_of_order_rows_sorted_and_counted() -> None:
    parsed = parse_csv(
        make_csv(["2026-11-19 02:00,0.30", "2026-11-19 00:00,0.50", "2026-11-19 01:00,0.40"]),
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    stamps = [r["timestamp"] for r in parsed.records]
    assert stamps == sorted(stamps)
    assert parsed.out_of_order == 1  # one backward jump in file order


def test_event_type_mapping() -> None:
    parsed = parse_csv(
        make_csv(["2026-11-19 00:00,0.50,h", "2026-11-19 01:00,0.40,nonsense"]),
        mapping=ColumnMapping(),
        timezone="Asia/Manila",
    )
    assert parsed.records[0]["event_type"] == "high"
    assert parsed.records[1]["event_type"] is None  # unknown annotations dropped, not fatal


def test_unknown_timezone_raises() -> None:
    with pytest.raises(ValidationError):
        parse_csv(make_csv(VALID_ROWS), mapping=ColumnMapping(), timezone="Not/AZone")


def test_missing_required_column_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        parse_csv(b"when,tide_height\n" + b"2026-11-19 00:00,0.5\n", timezone="Asia/Manila")
    assert "missing_columns" in exc_info.value.details


def test_columns_detected_reported() -> None:
    parsed = parse_csv(make_csv(VALID_ROWS), mapping=ColumnMapping(), timezone="Asia/Manila")
    assert "timestamp" in parsed.columns_detected
    assert "tide_height" in parsed.columns_detected


def test_custom_column_mapping() -> None:
    content = b"tide_time,height_m\n" b"2026-11-19 00:00,0.50\n" b"2026-11-19 01:00,0.40\n"
    parsed = parse_csv(
        content,
        mapping=ColumnMapping(timestamp="tide_time", height="height_m", event_type=None),
        timezone="Asia/Manila",
    )
    assert parsed.valid_records == 2
