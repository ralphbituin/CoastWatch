"""Read-only recording registry for DEMO MODE.

Only direct child directories of ``ml/reports/live_eval`` are considered, and
only the CSV tables inside them are read. A caller can never supply a path.
"""

from __future__ import annotations

import json
import math
import re
import threading
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import REPO_ROOT
from app.demo.replay import (
    DEFAULT_HOLD_SECONDS,
    MAX_RAMP_SECONDS,
    Sample,
    Segment,
    replay_unmodified,
)
from app.services.anomaly.rolling_baseline import (
    CLASS_DATA_UNRELIABLE,
    RollingBaselineConfig,
)

LIVE_EVAL_ROOT = REPO_ROOT / "ml" / "reports" / "live_eval"
PARITY_TOLERANCE = 1e-6
REFERENCE_MIN_REMAINING_S = (
    2.0 * MAX_RAMP_SECONDS + DEFAULT_HOLD_SECONDS + 600.0
)  # 2100 s, same as the offline primary arm
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _iso(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _config_from_payload(payload: dict[str, Any]) -> RollingBaselineConfig:
    allowed = set(RollingBaselineConfig.__dataclass_fields__)
    values = {key: value for key, value in payload.items() if key in allowed}
    config = RollingBaselineConfig(**values)
    config.validate()
    return config


def _find_config_file(run_dir: Path) -> Path | None:
    for relative in (
        "tables/config_used.json",
        "injection/tables/config_used.json",
        "config_used.json",
        "tables/detector_config.json",
    ):
        candidate = (run_dir / relative).resolve()
        if not candidate.is_file():
            continue
        if run_dir.resolve() not in candidate.parents:
            continue
        return candidate
    return None


@dataclass(slots=True)
class Recording:
    id: str
    run_dir: Path
    label: str
    camera: str | None = None
    source_type: str | None = None
    frame_height: float | None = None
    frame_width: float | None = None
    decoded_resolutions: list[str] = field(default_factory=list)
    nominal_interval_s: float | None = None
    duration_s: float | None = None
    row_counts: dict[str, int] = field(default_factory=dict)
    sessions: list[str] = field(default_factory=list)
    segment_count: int = 0
    scored_coverage: float | None = None
    longest_valid_window_s: float | None = None
    step_std_px: float | None = None
    move90_median_px: float | None = None
    move90_p95_px: float | None = None
    move90_max_px: float | None = None
    gate_median_px: float | None = None
    gate_p95_px: float | None = None
    polygon: bool = False
    notes: str = ""
    config: RollingBaselineConfig | None = None
    config_hash: str | None = None
    config_source: str | None = None
    status: str = "not_replayable"
    reason: str | None = None
    parity: dict[str, Any] | None = None
    samples: list[Sample] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    valid_starts: list[int] = field(default_factory=list)
    offline_dir: Path | None = None
    injected_min_px: float | None = None
    injected_max_px: float | None = None
    _snapshot_cache: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def replayable(self) -> bool:
        return self.status in {"replayable", "limited"}

    @property
    def valid_start_count(self) -> int:
        return len(self.valid_starts)

    def snapshot(self, start_index: int) -> dict[str, Any]:
        from app.demo.replay import snapshot_after_prefix

        with self._lock:
            cached = self._snapshot_cache.get(start_index)
        if cached is not None:
            return cached
        assert self.config is not None
        snapshot = snapshot_after_prefix(self.samples, self.segments, self.config, start_index)
        with self._lock:
            self._snapshot_cache[start_index] = snapshot
        return snapshot

    def start_time(self, start_index: int) -> datetime:
        return self.samples[start_index].time

    def to_dict(self) -> dict[str, Any]:
        samples: list[str] = []
        if self.valid_starts:
            if len(self.valid_starts) <= 12:
                chosen = self.valid_starts
            else:
                step = max(1, len(self.valid_starts) // 11)
                chosen = self.valid_starts[::step][:12]
            samples = [self.samples[index].time.isoformat() for index in chosen]
        return {
            "id": self.id,
            "label": self.label,
            "camera": self.camera,
            "source_type": self.source_type,
            "frame_height_px": self.frame_height,
            "frame_width_px": self.frame_width,
            "decoded_resolutions": self.decoded_resolutions,
            "nominal_interval_s": self.nominal_interval_s,
            "duration_s": self.duration_s,
            "row_counts": self.row_counts,
            "sessions": self.sessions,
            "segment_count": self.segment_count,
            "scored_coverage": self.scored_coverage,
            "longest_valid_window_s": self.longest_valid_window_s,
            "step_std_px": self.step_std_px,
            "move90_median_px": self.move90_median_px,
            "move90_p95_px": self.move90_p95_px,
            "move90_max_px": self.move90_max_px,
            "gate_median_px": self.gate_median_px,
            "gate_p95_px": self.gate_p95_px,
            "drawdown_floor_px": None if self.config is None else self.config.drawdown_floor_px,
            "scale_floor_px": None if self.config is None else self.config.scale_floor_px,
            "polygon": self.polygon,
            "notes": self.notes,
            "config_hash": self.config_hash,
            "config_source": self.config_source,
            "status": self.status,
            "reason": self.reason,
            "parity": self.parity,
            "valid_start_count": self.valid_start_count,
            "valid_start_samples": samples,
            "reference_min_remaining_s": REFERENCE_MIN_REMAINING_S,
        }


def _load_run_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_state.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _frame_dims(run_state: dict[str, Any]) -> tuple[float | None, float | None, list[str]]:
    height: float | None = None
    width: float | None = None
    regimes: list[str] = []

    calibration = run_state.get("calibration_profile") or {}
    if isinstance(calibration, dict):
        height = _num(calibration.get("frame_height")) or height
        width = _num(calibration.get("frame_width")) or width

    resolution = run_state.get("resolution")
    if isinstance(resolution, list | tuple) and len(resolution) >= 2:
        width = _num(resolution[0]) or width
        height = _num(resolution[1]) or height
        regimes.append(f"{int(width)}x{int(height)}" if width and height else "")

    verified = run_state.get("stream_verified") or {}
    if isinstance(verified, dict):
        width = _num(verified.get("width")) or width
        height = _num(verified.get("height")) or height

    regime = run_state.get("resolution_regime")
    if isinstance(regime, str):
        regimes.append(regime)
        match = re.search(r"(\d{3,5})\s*[x×]\s*(\d{3,5})", regime)
        if match:
            width = width or float(match.group(1))
            height = height or float(match.group(2))
            regimes.append(f"{match.group(1)}x{match.group(2)}")

    if height and width and not regimes:
        regimes.append(f"{int(width)}x{int(height)}")
    # Preserve order without duplicates.
    regimes = list(dict.fromkeys(item for item in regimes if item))
    return width, height, regimes


def _recorded_hashes(anomalies: pd.DataFrame) -> set[str]:
    if "config_hash" not in anomalies.columns:
        return set()
    values = anomalies["config_hash"].dropna().astype(str)
    return {value for value in values if value and value.lower() != "nan"}


def _resolve_config(
    run_dir: Path,
    run_state: dict[str, Any],
    recorded_hashes: set[str],
    frame_height: float | None,
) -> tuple[RollingBaselineConfig | None, str | None, str | None]:
    """Return (config, source, reason). Never guesses when a hash is available."""
    config_path = _find_config_file(run_dir)
    if config_path is not None:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
            config = _config_from_payload(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return None, None, f"saved detector config could not be read: {exc}"
        source = str(config_path.relative_to(run_dir)).replace("\\", "/")
        if not recorded_hashes:
            return (
                None,
                source,
                "no detector config hash is recorded in the anomaly rows",
            )
        if config.config_hash not in recorded_hashes:
            return (
                None,
                source,
                "saved detector config does not match the recorded run's config hash",
            )
        return config, source, None

    if not recorded_hashes:
        return None, None, "no detector config or config hash is recorded for this run"

    interval = _num(run_state.get("sample_interval_seconds"))
    candidates: list[RollingBaselineConfig] = []
    if frame_height is not None and interval is not None:
        candidates.append(
            RollingBaselineConfig(
                frame_height=float(frame_height),
                nominal_interval_seconds=float(interval),
            )
        )
    if frame_height is not None:
        candidates.append(RollingBaselineConfig(frame_height=float(frame_height)))
    if interval is not None:
        candidates.append(RollingBaselineConfig(nominal_interval_seconds=float(interval)))
    candidates.append(RollingBaselineConfig())
    for candidate in candidates:
        try:
            candidate.validate()
        except ValueError:
            continue
        if candidate.config_hash in recorded_hashes:
            return candidate, "reconstructed from the recorded run", None
    return (
        None,
        None,
        "the recorded config hash does not match any config reconstructed from the run",
    )


def _load_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_path = run_dir / "tables" / "observations.csv"
    ano_path = run_dir / "tables" / "anomalies.csv"
    observations = pd.read_csv(obs_path)
    anomalies = pd.read_csv(ano_path)
    return observations, anomalies


def _load_samples(observations: pd.DataFrame, anomalies: pd.DataFrame) -> list[Sample]:
    obs = observations.copy()
    ano = anomalies.copy()
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], errors="coerce", utc=True)
    ano["detected_at"] = pd.to_datetime(ano["detected_at"], errors="coerce", utc=True)
    obs = obs.dropna(subset=["observed_at"]).sort_values("observed_at", kind="stable")
    obs = obs.drop_duplicates("observed_at", keep="last").reset_index(drop=True)
    ano = ano.dropna(subset=["detected_at"]).sort_values("detected_at", kind="stable")
    ano = ano.drop_duplicates("detected_at", keep="last").reset_index(drop=True)

    ano_columns = [
        column
        for column in (
            "detected_at",
            "classification",
            "state",
            "quality_status",
            "reason",
            "rolling_baseline_px",
            "robust_z",
            "cumulative_drawdown_px",
            "sigma_k_px",
            "persistence_count",
        )
        if column in ano.columns
    ]
    merged = obs.merge(
        ano[ano_columns],
        left_on="observed_at",
        right_on="detected_at",
        how="left",
        validate="one_to_one",
    )

    position_column = (
        "signed_position_px" if "signed_position_px" in merged.columns else "shoreline_position_px"
    )
    samples: list[Sample] = []
    for index, row in enumerate(merged.itertuples(index=False)):
        record = row._asdict()
        position = _num(record.get(position_column))
        available = (
            str(record.get("extraction_status") or "") == "available" and position is not None
        )
        recorded_class = record.get("classification")
        recorded_class = (
            "MISSING_RECORDED_ROW"
            if recorded_class is None
            or (isinstance(recorded_class, float) and math.isnan(recorded_class))
            else str(recorded_class)
        )
        samples.append(
            Sample(
                index=index,
                time=record["observed_at"].to_pydatetime(),
                p=position,
                conf=_num(record.get("segmentation_confidence")),
                coverage=_num(record.get("measured_coverage")),
                frame_index=(
                    None if _num(record.get("frame_index")) is None else int(record["frame_index"])
                ),
                session=None
                if record.get("stream_session_id") is None
                else str(record.get("stream_session_id")),
                available=available,
                ts_conf=_num(record.get("capture_timestamp_confidence")),
                reason=(
                    None
                    if record.get("extraction_reason") is None
                    else str(record.get("extraction_reason"))
                ),
                horizon=_num(record.get("horizon_y_px")),
                recorded_class=recorded_class,
                recorded_state=str(record.get("state") or "UNKNOWN"),
                recorded_quality=None
                if record.get("quality_status") is None
                else str(record.get("quality_status")),
                recorded_reason=None if record.get("reason") is None else str(record.get("reason")),
                recorded_baseline=_num(record.get("rolling_baseline_px")),
                recorded_z=_num(record.get("robust_z")),
                recorded_ck=_num(record.get("cumulative_drawdown_px")),
                recorded_sigma_k=_num(record.get("sigma_k_px")),
                recorded_persistence=(
                    None
                    if _num(record.get("persistence_count")) is None
                    else int(record["persistence_count"])
                ),
            )
        )
    return samples


def _build_segments(
    samples: list[Sample], config: RollingBaselineConfig, run_state: dict[str, Any]
) -> list[Segment]:
    if not samples:
        return []
    boundaries = {0, len(samples)}
    max_gap = float(config.max_gap_seconds)
    for index in range(1, len(samples)):
        previous = samples[index - 1]
        current = samples[index]
        gap = (current.time - previous.time).total_seconds()
        if current.session != previous.session or gap > max_gap:
            boundaries.add(index)

    times = [sample.time for sample in samples]
    regime_sets = []
    for key in ("visual_regimes", "resolution_regimes", "decoded_regimes"):
        value = run_state.get(key)
        if isinstance(value, list):
            regime_sets.extend(value)
    for regime in regime_sets:
        if not isinstance(regime, dict):
            continue
        start = regime.get("start")
        if not isinstance(start, str):
            continue
        try:
            boundary_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            continue
        index = bisect_left(times, boundary_time)
        if 0 < index < len(samples):
            boundaries.add(index)

    ordered = sorted(boundaries)
    return [
        Segment(start=ordered[i], end=ordered[i + 1])
        for i in range(len(ordered) - 1)
        if ordered[i + 1] > ordered[i]
    ]


def _profile_stats(
    samples: list[Sample],
    segments: list[Segment],
    config: RollingBaselineConfig,
) -> dict[str, Any]:
    valid_pairs: list[tuple[datetime, float]] = []
    step_diffs: list[float] = []
    for segment in segments:
        previous: Sample | None = None
        for index in range(segment.start, segment.end):
            sample = samples[index]
            if sample.p is not None and sample.available and not sample.unreliable:
                valid_pairs.append((sample.time, float(sample.p)))
                if (
                    previous is not None
                    and (sample.time - previous.time).total_seconds() <= config.max_gap_seconds
                    and previous.p is not None
                ):
                    step_diffs.append(float(sample.p) - float(previous.p))
            previous = sample

    times = [item[0] for item in valid_pairs]
    values = [item[1] for item in valid_pairs]
    moves: list[float] = []
    for moment, value in valid_pairs:
        left = bisect_left(times, moment - timedelta(seconds=90.0))
        if left < len(times) and times[left] <= moment - timedelta(seconds=60.0):
            moves.append(abs(value - values[left]))

    gate_values = []
    for sample in samples:
        if sample.unreliable or sample.recorded_sigma_k is None:
            continue
        gate_values.append(
            max(5.0 * float(sample.recorded_sigma_k), float(config.drawdown_floor_px))
        )

    # Scored coverage after one full baseline window inside each segment.
    scored = 0
    eligible = 0
    for segment in segments:
        start_time = samples[segment.start].time + timedelta(seconds=config.window_seconds)
        for index in range(segment.start, segment.end):
            sample = samples[index]
            if sample.time < start_time:
                continue
            eligible += 1
            if sample.recorded_baseline is not None:
                scored += 1

    # Longest contiguous run of available, non-unreliable samples.
    longest = 0.0
    for segment in segments:
        run_start: datetime | None = None
        previous: Sample | None = None
        for index in range(segment.start, segment.end):
            sample = samples[index]
            usable = sample.available and sample.p is not None and not sample.unreliable
            if not usable:
                run_start = None
                previous = None
                continue
            if (
                previous is not None
                and run_start is not None
                and (sample.time - previous.time).total_seconds() <= config.max_gap_seconds
            ):
                longest = max(longest, (sample.time - run_start).total_seconds())
            else:
                run_start = sample.time
            previous = sample
        if run_start is not None and previous is not None:
            longest = max(longest, (previous.time - run_start).total_seconds())

    return {
        "scored_coverage": (scored / eligible) if eligible else None,
        "longest_valid_window_s": longest if longest else None,
        "step_std_px": float(np.std(step_diffs, ddof=0)) if step_diffs else None,
        "move90_median_px": float(np.median(moves)) if moves else None,
        "move90_p95_px": float(np.percentile(moves, 95)) if moves else None,
        "move90_max_px": float(np.max(moves)) if moves else None,
        "gate_median_px": float(np.median(gate_values)) if gate_values else None,
        "gate_p95_px": float(np.percentile(gate_values, 95)) if gate_values else None,
    }


def _parity_check(
    samples: list[Sample],
    segments: list[Segment],
    config: RollingBaselineConfig,
    rows: list[Any],
) -> dict[str, Any]:
    fields = (
        ("baseline", "recorded_baseline"),
        ("z", "recorded_z"),
        ("ck", "recorded_ck"),
    )
    max_abs = {name: 0.0 for name, _ in fields}
    counts = {name: 0 for name, _ in fields}
    class_mismatches = 0
    state_mismatches = 0
    checked = 0
    cutoff_by_segment = {
        segment.start: samples[segment.start].time + timedelta(seconds=config.window_seconds)
        for segment in segments
    }
    for segment in segments:
        cutoff = cutoff_by_segment[segment.start]
        for index in range(segment.start, segment.end):
            sample = samples[index]
            if sample.time < cutoff:
                continue
            row = rows[index]
            if row is None:
                continue
            checked += 1
            for name, recorded_attr in fields:
                recorded = getattr(sample, recorded_attr)
                if name == "z":
                    value = row.robust_z
                elif name == "baseline":
                    value = row.rolling_baseline_px
                else:
                    value = row.cumulative_drawdown_px
                if recorded is None or value is None:
                    continue
                counts[name] += 1
                max_abs[name] = max(max_abs[name], abs(float(recorded) - float(value)))
            if str(sample.recorded_class) != str(row.classification):
                class_mismatches += 1
            if str(sample.recorded_state) != str(row.state):
                state_mismatches += 1

    passed = (
        all(value <= PARITY_TOLERANCE for value in max_abs.values())
        and class_mismatches == 0
        and state_mismatches == 0
        and checked > 0
    )
    return {
        "pass": bool(passed),
        "tolerance": PARITY_TOLERANCE,
        "checked_samples_after_full_history": checked,
        "compared_samples": counts,
        "max_abs": max_abs,
        "classification_mismatches": class_mismatches,
        "state_mismatches": state_mismatches,
    }


def _valid_starts(
    samples: list[Sample],
    segments: list[Segment],
    config: RollingBaselineConfig,
) -> list[int]:
    if not samples:
        return []
    unreliable_prefix = [0]
    for sample in samples:
        unreliable_prefix.append(
            unreliable_prefix[-1] + int(sample.recorded_class == CLASS_DATA_UNRELIABLE)
        )
    times = [sample.time for sample in samples]
    window = float(config.window_seconds)
    valid: list[int] = []
    for segment in segments:
        first_time = samples[segment.start].time
        last_time = samples[segment.end - 1].time
        for index in range(segment.start, segment.end):
            sample = samples[index]
            if not sample.available or sample.p is None or sample.unreliable:
                continue
            if (sample.time - first_time).total_seconds() < window:
                continue
            if (last_time - sample.time).total_seconds() < REFERENCE_MIN_REMAINING_S:
                continue
            left = bisect_left(
                times,
                sample.time - timedelta(seconds=window),
                lo=segment.start,
                hi=index + 1,
            )
            if unreliable_prefix[index + 1] - unreliable_prefix[left] > 0:
                continue
            valid.append(index)
    return valid


def _polygon_bounds(
    run_state: dict[str, Any], width: float | None
) -> tuple[bool, float | None, float | None]:
    calibration = run_state.get("calibration_profile") or {}
    points = calibration.get("points") if isinstance(calibration, dict) else None
    if not isinstance(points, list) or len(points) < 3:
        return False, None, None
    xs = []
    for point in points:
        if isinstance(point, list | tuple) and len(point) >= 2:
            xs.append(_num(point[0]))
    xs = [value for value in xs if value is not None]
    if not xs or width is None:
        return True, None, None
    return True, min(xs) * width, max(xs) * width


class DemoRegistry:
    """Lazy, cached, read-only view of eligible recordings under live_eval."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or LIVE_EVAL_ROOT).resolve()
        self._recordings: dict[str, Recording] | None = None
        self._lock = threading.Lock()

    def _manifest(self) -> dict[str, Any]:
        path = (self.root / "demo" / "recordings.json").resolve()
        if self.root not in path.parents or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def scan(self, force: bool = False) -> dict[str, Recording]:
        with self._lock:
            if self._recordings is not None and not force:
                return self._recordings
            manifest = self._manifest()
            found: dict[str, Recording] = {}
            if self.root.exists():
                for child in sorted(self.root.iterdir(), key=lambda item: item.name):
                    if not child.is_dir() or child.name == "demo":
                        continue
                    if not (child / "tables" / "observations.csv").is_file():
                        continue
                    if not (child / "tables" / "anomalies.csv").is_file():
                        continue
                    recording = self._build(child, manifest.get(child.name, {}))
                    found[recording.id] = recording
            self._recordings = found
            return found

    def list(self) -> list[Recording]:
        return list(self.scan().values())

    def get(self, recording_id: str) -> Recording:
        if not _ID_RE.match(recording_id or ""):
            raise KeyError(recording_id)
        recordings = self.scan()
        if recording_id not in recordings:
            raise KeyError(recording_id)
        return recordings[recording_id]

    def _build(self, run_dir: Path, manifest_entry: Any) -> Recording:
        entry = manifest_entry if isinstance(manifest_entry, dict) else {}
        notes = str(entry.get("notes") or "")
        label = str(entry.get("label") or run_dir.name)
        run_state = _load_run_state(run_dir)
        camera = str(run_state.get("camera_code") or run_state.get("camera_name") or "") or None
        source_type = str(run_state.get("camera_type") or "") or None
        width, height, regimes = _frame_dims(run_state)
        nominal = _num(run_state.get("sample_interval_seconds"))

        recording = Recording(
            id=run_dir.name,
            run_dir=run_dir,
            label=label,
            camera=camera,
            source_type=source_type,
            frame_height=height,
            frame_width=width,
            decoded_resolutions=regimes,
            nominal_interval_s=nominal,
            notes=notes,
        )
        try:
            observations, anomalies = _load_tables(run_dir)
            recording.row_counts = {
                "observations": int(len(observations)),
                "anomalies": int(len(anomalies)),
            }
            samples = _load_samples(observations, anomalies)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            recording.reason = f"recording tables could not be read: {exc}"
            return recording

        if not samples:
            recording.reason = "recording has no usable observations"
            return recording
        if any(sample.recorded_class == "MISSING_RECORDED_ROW" for sample in samples):
            missing = sum(
                1 for sample in samples if sample.recorded_class == "MISSING_RECORDED_ROW"
            )
            recording.reason = f"{missing} observations have no matching recorded detector row"
            return recording

        recording.duration_s = (samples[-1].time - samples[0].time).total_seconds()
        recording.sessions = list(
            dict.fromkeys(sample.session for sample in samples if sample.session)
        )

        recorded_hashes = _recorded_hashes(anomalies)
        config, config_source, config_reason = _resolve_config(
            run_dir, run_state, recorded_hashes, height
        )
        recording.config_source = config_source
        if config is None:
            recording.reason = config_reason or "detector config cannot be verified"
            return recording
        recording.config = config
        recording.config_hash = config.config_hash
        if recording.frame_height is None:
            recording.frame_height = config.frame_height
        if recording.nominal_interval_s is None:
            recording.nominal_interval_s = config.nominal_interval_seconds
        if recording.nominal_interval_s is None:
            deltas = [
                (samples[i].time - samples[i - 1].time).total_seconds()
                for i in range(1, len(samples))
                if (samples[i].time - samples[i - 1].time).total_seconds() <= config.max_gap_seconds
            ]
            recording.nominal_interval_s = float(np.median(deltas)) if deltas else None

        recording.samples = samples
        recording.segments = _build_segments(samples, config, run_state)
        recording.segment_count = len(recording.segments)
        stats = _profile_stats(samples, recording.segments, config)
        for key, value in stats.items():
            setattr(recording, key, value)
        polygon, min_x, max_x = _polygon_bounds(run_state, recording.frame_width)
        recording.polygon = polygon
        recording.injected_min_px = min_x
        recording.injected_max_px = max_x
        recording.offline_dir = (
            run_dir / "injection" if (run_dir / "injection" / "tables").is_dir() else None
        )

        rows = replay_unmodified(samples, recording.segments, config)
        recording.parity = _parity_check(samples, recording.segments, config, rows)
        if not recording.parity["pass"]:
            recording.status = "not_replayable"
            recording.reason = "replay does not match the recorded run"
            return recording

        recording.valid_starts = _valid_starts(samples, recording.segments, config)
        if not recording.valid_starts:
            recording.status = "not_replayable"
            recording.reason = (
                "no valid start time has a full baseline history, clean recent data "
                "and enough remaining data inside one segment"
            )
        elif len(recording.valid_starts) < 10:
            recording.status = "limited"
            recording.reason = f"limited data: {len(recording.valid_starts)} valid starts"
        else:
            recording.status = "replayable"
            recording.reason = None
        return recording


_registry = DemoRegistry()


def get_registry() -> DemoRegistry:
    return _registry
