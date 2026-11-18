"""NAMRIA tide provider (Phase 2).

Design notes:

* CoastWatch does **not** assume an official, undocumented NAMRIA API exists.
  Manual CSV import of official station datasets is the primary path.
* If a downloadable dataset arrives as PDF (or any non-CSV format) it must go
  through an explicit parser/conversion step and be validated — it is never
  silently interpreted as CSV.
* Predicted heights are kept separately from measured sea levels; datums are
  kept per station and are never blended.
* Imported datasets are insert-only. Existing predictions are never
  overwritten — same-timestamp rows are rejected as ``existing_record``.

Official resources:

    https://www.namria.gov.ph/
    https://www.namria.gov.ph/kiosk/namria02.htm
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import insert, select, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import DataUnavailableError, NotFoundError, ValidationError
from app.models.tide import DatasetImport, TideDataset, TidePrediction, TideStation
from app.services.tides import csv_import, tide_phase
from app.services.tides.base import (
    HighLowWater,
    TidePhaseResult,
    TidePoint,
    TideProvider,
    TideStationInfo,
)
from app.services.tides.interpolation import InterpolationMethod, interpolate_height


class NAMRIATideProvider(TideProvider):
    """Tide provider backed by validated, imported NAMRIA-style datasets."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Station helpers
    # ------------------------------------------------------------------
    def get_station(self, station_code: str) -> TideStation:
        station = self._session.scalar(select(TideStation).where(TideStation.code == station_code))
        if station is None:
            raise NotFoundError(f"Tide station not found: {station_code}")
        return station

    def _station_info(self, station: TideStation) -> TideStationInfo:
        return TideStationInfo(
            code=station.code,
            name=station.name,
            country_iso=station.country.iso_code if station.country else "??",
            region=station.region,
            latitude=station.latitude,
            longitude=station.longitude,
            timezone=station.timezone,
            data_source=station.data_source,
            vertical_datum=station.vertical_datum,
            measurement_unit=station.measurement_unit,
        )

    def get_stations(self) -> list[TideStationInfo]:
        stations = self._session.scalars(select(TideStation).order_by(TideStation.code)).all()
        return [self._station_info(s) for s in stations]

    # ------------------------------------------------------------------
    # Series access
    # ------------------------------------------------------------------
    def _series(self, station: TideStation) -> tuple[list[datetime], list[float], list[int | None]]:
        """Chronologically ordered timestamps (aware UTC), heights, dataset_ids."""
        rows = self._session.execute(
            select(TidePrediction.timestamp, TidePrediction.height, TidePrediction.tide_dataset_id)
            .where(TidePrediction.tide_station_id == station.id)
            .order_by(TidePrediction.timestamp)
        ).all()
        timestamps = [r.timestamp for r in rows]
        heights = [float(r.height) for r in rows]
        dataset_ids = [r.tide_dataset_id for r in rows]
        return timestamps, heights, dataset_ids

    def station_coverage(self, station: TideStation) -> tuple[datetime | None, datetime | None]:
        """Coverage = union of all predictions for the station."""
        row = self._session.execute(
            text(
                "SELECT MIN(timestamp) AS start, MAX(timestamp) AS end "
                "FROM tide_predictions WHERE tide_station_id = :sid"
            ),
            {"sid": station.id},
        ).one()
        return row.start, row.end

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def get_predictions(self, station_code: str, start: datetime, end: datetime) -> list[TidePoint]:
        station = self.get_station(station_code)
        rows = self._session.execute(
            select(
                TidePrediction.timestamp,
                TidePrediction.height,
                TidePrediction.height_unit,
                TidePrediction.vertical_datum,
                TidePrediction.quality_status,
            )
            .where(TidePrediction.tide_station_id == station.id)
            .where(TidePrediction.timestamp >= start)
            .where(TidePrediction.timestamp <= end)
            .order_by(TidePrediction.timestamp)
        ).all()
        return [
            TidePoint(
                timestamp=r.timestamp,
                height=float(r.height),
                unit=r.height_unit,
                datum=r.vertical_datum,
                quality=self._quality(r.quality_status),
            )
            for r in rows
        ]

    @staticmethod
    def _quality(token: str | None):
        from app.services.tides.base import DataQuality

        try:
            return DataQuality(token or "unknown")
        except ValueError:
            return DataQuality.UNKNOWN

    def _series_or_unavailable(
        self, station: TideStation
    ) -> tuple[list[datetime], list[float], list[int | None]]:
        timestamps, heights, dataset_ids = self._series(station)
        if not timestamps:
            raise DataUnavailableError(
                f"Tide station {station.code} has no imported tide data.",
                details={"station_code": station.code},
            )
        return timestamps, heights, dataset_ids

    def interpolate_at(
        self,
        station: TideStation,
        at: datetime,
        *,
        method: InterpolationMethod = InterpolationMethod.PCHIP,
    ) -> tuple[float | None, str, int | None]:
        """Return (height_m, quality_token, dataset_id_of_nearest_sample)."""
        timestamps, heights, dataset_ids = self._series_or_unavailable(station)
        nearest_idx = min(
            range(len(timestamps)),
            key=lambda i: abs((timestamps[i] - at).total_seconds()),
        )
        nearest_dataset_id = dataset_ids[nearest_idx]
        epoch = [t.timestamp() for t in timestamps]
        # Exact sample?
        for t, h in zip(timestamps, heights, strict=False):
            if abs((t - at).total_seconds()) <= 1.0:
                return h, "ok", nearest_dataset_id
        height = interpolate_height(epoch, heights, at.timestamp(), method=method)
        if height is None:
            return None, "unavailable", None
        return height, "interpolated", nearest_dataset_id

    def get_high_low_waters(
        self, station_code: str, start: datetime, end: datetime
    ) -> list[HighLowWater]:
        station = self.get_station(station_code)
        timestamps, heights, _ = self._series_or_unavailable(station)
        extrema = tide_phase.find_extrema(
            timestamps, heights, prominence_m=settings.tide_extrema_prominence_m
        )
        events = [
            tide_phase.to_high_low_water(
                e, unit=station.measurement_unit, datum=station.vertical_datum
            )
            for e in extrema
            if start <= e.timestamp <= end
        ]
        return events

    def get_prediction_at(
        self,
        station_code: str,
        timestamp: datetime,
        *,
        method: InterpolationMethod = InterpolationMethod.PCHIP,
    ) -> TidePhaseResult:
        station = self.get_station(station_code)
        timestamps, heights, _ = self._series_or_unavailable(station)
        start, end = timestamps[0], timestamps[-1]
        at = timestamp.astimezone(UTC) if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
        if at < start or at > end:
            raise DataUnavailableError(
                f"Timestamp {at.isoformat()} is outside the supported coverage "
                f"for station {station.code}.",
                details={
                    "station_code": station.code,
                    "coverage_start": start.isoformat(),
                    "coverage_end": end.isoformat(),
                    "requested": at.isoformat(),
                },
            )

        height, quality, dataset_id = self.interpolate_at(station, at, method=method)
        assert height is not None  # coverage checked above

        extrema = tide_phase.find_extrema(
            timestamps, heights, prominence_m=settings.tide_extrema_prominence_m
        )
        slope = tide_phase.estimate_slope_per_hour(timestamps, heights, at)

        nearest_event = None
        if extrema:
            nearest = min(extrema, key=lambda e: abs((e.timestamp - at).total_seconds()))
            nearest_event = nearest.event

        direction = tide_phase.classify_direction(
            slope,
            slack_tolerance=settings.tide_slack_slope_tolerance_m_per_hour,
            nearest_event=nearest_event,
        )

        prev_extremum, next_extremum = tide_phase.events_around(extrema, at)
        previous = (
            tide_phase.to_high_low_water(
                prev_extremum, unit=station.measurement_unit, datum=station.vertical_datum
            )
            if prev_extremum
            else None
        )
        upcoming = (
            tide_phase.to_high_low_water(
                next_extremum, unit=station.measurement_unit, datum=station.vertical_datum
            )
            if next_extremum
            else None
        )

        return TidePhaseResult(
            timestamp=at,
            height=height,
            unit=station.measurement_unit,
            datum=station.vertical_datum,
            direction=direction,
            quality=self._quality(quality),
            previous_event=previous,
            next_event=upcoming,
            source=station.data_source,
            dataset_id=dataset_id,
        )

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------
    @staticmethod
    def _column_mapping(value: Any) -> csv_import.ColumnMapping:
        if value is None:
            return csv_import.ColumnMapping()
        if isinstance(value, csv_import.ColumnMapping):
            return value
        if isinstance(value, dict):
            return csv_import.ColumnMapping.from_dict(value)
        raise ValidationError(
            "column_mapping must be a JSON object with column names",
            details={"column_mapping": value},
        )

    def preview(
        self,
        content: bytes,
        *,
        mapping: csv_import.ColumnMapping | dict[str, Any] | None = None,
        timezone: str = "Asia/Manila",
    ) -> dict[str, Any]:
        """Dry-run parse; returns a preview without persisting anything."""
        mapping = self._column_mapping(mapping)
        parsed = csv_import.parse_csv(content, mapping=mapping, timezone=timezone)
        return self._preview_payload(
            parsed,
            station_code=None,
            timezone=timezone,
            mapping=mapping,
        )

    def import_data(
        self,
        content: Any,
        **options: Any,
    ) -> dict[str, Any]:
        """Import a NAMRIA-style CSV.

        ``content`` is the raw file bytes. ``options`` keys: ``station_code``,
        ``name``, ``source`` (default ``NAMRIA_CSV``), ``unit`` (default
        ``meters``), ``vertical_datum``, ``timezone``, ``description``,
        ``column_mapping`` (dict or ColumnMapping), ``source_filename``,
        ``commit`` (default True). Returns a summary dict for the API response.
        """
        content_bytes: bytes = content if isinstance(content, bytes) else content.read()
        station_code = options["station_code"]
        name = options.get("name") or options.get("source_filename") or "NAMRIA import"
        source_type = options.get("source") or "NAMRIA_CSV"
        unit = (options.get("unit") or "meters").lower()
        vertical_datum = options.get("vertical_datum")
        timezone = options.get("timezone") or settings.default_timezone
        description = options.get("description")
        source_filename = options.get("source_filename")
        mapping = self._column_mapping(options.get("column_mapping"))
        commit = options.get("commit", True)

        if unit not in {"meters", "feet", "centimeters"}:
            raise ValidationError(
                f"Unrecognised measurement unit: {unit!r}",
                details={"unit": unit, "allowed": ["meters", "feet", "centimeters"]},
            )

        station = self.get_station(station_code)
        csv_import.validate_timezone(timezone)

        parsed = csv_import.parse_csv(content_bytes, mapping=mapping, timezone=timezone)
        coverage_start, coverage_end = parsed.coverage()
        fhash = csv_import.file_hash(content_bytes)

        if not commit:
            return self._preview_payload(
                parsed,
                station_code=station.code,
                timezone=timezone,
                mapping=mapping,
                file_hash=fhash,
            )

        return self._commit(
            station=station,
            parsed=parsed,
            name=name,
            source=source_type,
            unit=unit,
            vertical_datum=vertical_datum,
            description=description,
            mapping=mapping,
            source_filename=source_filename,
            file_hash=fhash,
            timezone=timezone,
        )

    def _preview_payload(
        self,
        parsed: csv_import.ParsedImport,
        *,
        station_code: str | None,
        timezone: str,
        mapping: csv_import.ColumnMapping | None,
        file_hash: str | None = None,
    ) -> dict[str, Any]:
        coverage_start, coverage_end = parsed.coverage()
        samples, truncated = parsed.rejection_samples()
        summary = parsed.rejection_summary()
        if truncated:
            summary["rejected_records_truncated"] = True
        return {
            "station_code": station_code,
            "timezone": timezone,
            "total_records": parsed.total_records,
            "valid_records": parsed.valid_records,
            "rejected_records": len(parsed.rejected),
            "rejection_summary": summary,
            "rejection_samples": samples,
            "coverage_start": coverage_start,
            "coverage_end": coverage_end,
            "columns_detected": parsed.columns_detected,
            "column_mapping": (
                mapping.to_dict() if mapping else csv_import.ColumnMapping().to_dict()
            ),
            "file_hash": file_hash,
            "message": (
                f"{parsed.valid_records} valid records, {len(parsed.rejected)} rejected, "
                f"{parsed.out_of_order} out of order."
            ),
        }

    def _commit(
        self,
        *,
        station: TideStation,
        parsed: csv_import.ParsedImport,
        name: str,
        source: str,
        unit: str,
        vertical_datum: str | None,
        description: str | None,
        mapping: csv_import.ColumnMapping,
        source_filename: str | None,
        file_hash: str,
        timezone: str,
    ) -> dict[str, Any]:
        coverage_start, coverage_end = parsed.coverage()
        fhash = file_hash

        # Insert-only: same file already imported at this station?
        existing_dataset = self._session.scalar(
            select(TideDataset).where(
                TideDataset.tide_station_id == station.id,
                TideDataset.file_hash == fhash,
            )
        )
        if existing_dataset is not None:
            import_record = DatasetImport(
                tide_station_id=station.id,
                tide_dataset_id=existing_dataset.id,
                source_filename=source_filename,
                file_hash=fhash,
                column_mapping=mapping.to_dict(),
                total_records=parsed.total_records,
                imported_records=0,
                rejected_records=0,
                rejection_summary={"file": "duplicate_file_hash"},
                rejection_samples=None,
                status="duplicate",
                message=(
                    f"File already imported as dataset {existing_dataset.id}; "
                    "existing dataset left unchanged (insert-only)."
                ),
            )
            self._session.add(import_record)
            self._session.commit()
            return {
                **self._preview_payload(
                    parsed,
                    station_code=station.code,
                    timezone=timezone,
                    mapping=mapping,
                    file_hash=fhash,
                ),
                "dataset_id": existing_dataset.id,
                "import_id": import_record.id,
                "status": "duplicate",
                "imported_records": 0,
            }

        # Rows whose (station, timestamp) prediction already exists are rejected
        # as existing_record — never silently overwritten.
        new_records: list[dict[str, Any]] = []
        if parsed.records:
            existing_ts = self._existing_timestamps(
                station.id, [r["timestamp"] for r in parsed.records]
            )
            for record in parsed.records:
                if record["timestamp"] in existing_ts:
                    parsed.rejected.append(
                        csv_import.RejectedRow(record["line"], "existing_record", record["raw"])
                    )
                else:
                    new_records.append(record)

        dataset: TideDataset | None = None
        if new_records:
            timestamps = [r["timestamp"] for r in new_records]
            dataset = TideDataset(
                tide_station_id=station.id,
                name=name,
                source=source,
                description=description,
                unit=unit,
                vertical_datum=vertical_datum,
                coverage_start=min(timestamps),
                coverage_end=max(timestamps),
                record_count=len(new_records),
                source_filename=source_filename,
                file_hash=fhash,
            )
            self._session.add(dataset)
            self._session.flush()
            values = [
                {
                    "tide_station_id": station.id,
                    "timestamp": r["timestamp"],
                    "tide_dataset_id": dataset.id,
                    "height": r["height"],
                    "height_unit": unit,
                    "vertical_datum": vertical_datum,
                    "event_type": r["event_type"],
                    "quality_status": "ok",
                }
                for r in new_records
            ]
            self._session.execute(insert(TidePrediction), values)
            self._refresh_station_coverage(station)

        samples, truncated = parsed.rejection_samples()
        summary = parsed.rejection_summary()
        if truncated:
            summary["rejected_records_truncated"] = True
        if new_records and not parsed.rejected:
            status = "committed"
        elif new_records:
            status = "partial"
        else:
            status = "rejected"

        import_record = DatasetImport(
            tide_station_id=station.id,
            tide_dataset_id=dataset.id if dataset else None,
            source_filename=source_filename,
            file_hash=fhash,
            column_mapping=mapping.to_dict(),
            total_records=parsed.total_records,
            imported_records=len(new_records),
            rejected_records=len(parsed.rejected),
            rejection_summary=summary,
            rejection_samples=samples or None,
            status=status,
            message=(
                f"imported {len(new_records)} / {parsed.total_records} records "
                f"({len(parsed.rejected)} rejected)"
            ),
        )
        self._session.add(import_record)
        self._session.commit()

        payload = self._preview_payload(
            parsed,
            station_code=station.code,
            timezone=timezone,
            mapping=mapping,
            file_hash=fhash,
        )
        payload.update(
            dataset_id=dataset.id if dataset else None,
            import_id=import_record.id,
            status=status,
            imported_records=len(new_records),
        )
        return payload

    def _existing_timestamps(
        self, station_id: int, timestamps: Sequence[datetime]
    ) -> set[datetime]:
        existing: set[datetime] = set()
        chunk_size = 1000
        stamps = list(timestamps)
        for start_index in range(0, len(stamps), chunk_size):
            chunk = stamps[start_index : start_index + chunk_size]
            found = (
                self._session.execute(
                    text(
                        "SELECT timestamp FROM tide_predictions "
                        "WHERE tide_station_id = :sid AND timestamp = ANY(:ts)"
                    ),
                    {"sid": station_id, "ts": chunk},
                )
                .scalars()
                .all()
            )
            existing.update(found)
        return existing

    def _refresh_station_coverage(self, station: TideStation) -> None:
        row = self._session.execute(
            text(
                "SELECT MIN(timestamp) AS start, MAX(timestamp) AS end "
                "FROM tide_predictions WHERE tide_station_id = :sid"
            ),
            {"sid": station.id},
        ).one()
        station.prediction_start = row.start
        station.prediction_end = row.end
