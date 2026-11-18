"""CSV import and validation (Phase 2).

Planned responsibilities:

* Configurable column mapping (timestamp, height, station, optional event).
* Validation of dates, duplicates, missing heights, numeric format, time-zone
  and unit consistency, vertical-datum metadata and chronological ordering.
* Import preview before commit.
* Persist an import-history entry (``dataset_imports``) and rejected records
  with reasons — nothing is silently discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ImportPreview:
    total_records: int
    valid_records: int
    rejected_records: int
    column_mapping: dict[str, str]
    rejection_reasons: dict[str, int]
    coverage_start: str | None
    coverage_end: str | None


def preview_import(*args: Any, **kwargs: Any) -> ImportPreview:
    raise NotImplementedError("CSV import preview is implemented in Phase 2.")


def commit_import(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("CSV import commit is implemented in Phase 2.")
