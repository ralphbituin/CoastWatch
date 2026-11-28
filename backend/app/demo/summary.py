"""Quick-estimate grid and comparison helpers for DEMO MODE.

A grid cell is a set of independent replays of one recording with the same
injected amplitude and ramp, each starting from a fresh detector state seeded
from unmodified same-segment history.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.demo.registry import Recording
from app.demo.replay import DEFAULT_HOLD_SECONDS, replay_injection

SEED = 20261128
STARTS_PER_CELL = 30
SUMMARY_AMPLITUDE_PCTS = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0]
SUMMARY_RAMPS_S = [30.0, 60.0, 120.0, 300.0, 600.0]


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _read_offline_summary(recording: Recording) -> pd.DataFrame | None:
    if recording.offline_dir is None:
        return None
    path = recording.offline_dir / "tables" / "injection_summary.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None


def _offline_starts(recording: Recording) -> list[int] | None:
    if recording.offline_dir is None:
        return None
    path = recording.offline_dir / "tables" / "start_times.csv"
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None
    if "selected_for_primary" not in frame.columns or "t0_utc" not in frame.columns:
        return None
    selected = frame[frame["selected_for_primary"].astype(str).str.lower() == "true"]
    by_iso = {sample.time.isoformat(): index for index, sample in enumerate(recording.samples)}
    times = [sample.time for sample in recording.samples]
    indices: list[int] = []
    for value in selected["t0_utc"].tolist():
        if value in by_iso:
            indices.append(by_iso[value])
            continue
        try:
            moment = pd.Timestamp(value).to_pydatetime()
        except (TypeError, ValueError):
            continue
        nearest = min(
            range(len(times)),
            key=lambda index: abs((times[index] - moment).total_seconds()),
        )
        if abs((times[nearest] - moment).total_seconds()) <= 1.0:
            indices.append(nearest)
    return sorted(set(indices)) if indices else None


def analysis_starts(recording: Recording) -> tuple[list[int], str]:
    offline = _offline_starts(recording)
    if offline:
        allowed = set(recording.valid_starts)
        usable = [index for index in offline if index in allowed]
        if usable:
            return usable, "offline start_times.csv (selected_for_primary)"
    rng = np.random.default_rng(SEED)
    candidates = recording.valid_starts
    take = min(STARTS_PER_CELL, len(candidates))
    chosen = sorted(rng.choice(candidates, size=take, replace=False).tolist()) if take else []
    return chosen, f"fixed seed {SEED}"


def _grid_definition(recording: Recording) -> tuple[list[float], list[float]]:
    offline = _read_offline_summary(recording)
    if offline is not None and not offline.empty:
        primary = offline[
            (offline["arm"] == "primary")
            & (offline["shape"] == "retreat_return")
            & (offline["sign"] == 1)
        ]
        if not primary.empty:
            amplitudes = sorted({float(value) for value in primary["amplitude_px"].tolist()})
            ramps = sorted({float(value) for value in primary["ramp_s"].tolist()})
            return amplitudes, ramps
    height = float(recording.frame_height or 720.0)
    return [round(height * pct / 100.0, 4) for pct in SUMMARY_AMPLITUDE_PCTS], list(SUMMARY_RAMPS_S)


def _cell(
    recording: Recording,
    *,
    amplitude_px: float,
    ramp_s: float,
    shape: str,
    sign: int,
    starts: list[int],
    hold_s: float = DEFAULT_HOLD_SECONDS,
) -> dict[str, Any]:
    failures = {
        "z_not_reached": 0,
        "movement_gate_not_reached": 0,
        "z_and_gate_not_simultaneous": 0,
        "persistence_not_reached": 0,
        "no_event_in_window": 0,
    }
    n_in_bounds = 0
    n_detected = 0
    false_events = 0
    gates: list[float] = []
    latencies: list[float] = []
    for start_index in starts:
        result = replay_injection(
            samples=recording.samples,
            segments=recording.segments,
            config=recording.config,  # type: ignore[arg-type]
            start_index=start_index,
            amplitude_px=amplitude_px,
            ramp_s=ramp_s,
            hold_s=hold_s,
            shape=shape,
            sign=sign,
            snapshot=recording.snapshot(start_index),
            injected_min_px=recording.injected_min_px,
            injected_max_px=recording.injected_max_px,
        )
        if not result["in_bounds"]:
            continue
        n_in_bounds += 1
        false_events += int(result["false_events"])
        if result["detected"]:
            n_detected += 1
            if result["latency_from_t0_s"] is not None:
                latencies.append(float(result["latency_from_t0_s"]))
        else:
            mode = result["failure_mode"]
            if mode in failures:
                failures[mode] += 1
        series = result["series"]
        cell_gates = [item["gate"] for item in series if item["gate"] is not None]
        if cell_gates:
            gates.append(float(np.median(cell_gates)))
    low, high = wilson_interval(n_detected, n_in_bounds)
    height = float(recording.frame_height or 720.0)
    return {
        "amplitude_px": amplitude_px,
        "amplitude_pct_frame": 100.0 * amplitude_px / height,
        "ramp_s": ramp_s,
        "hold_s": hold_s,
        "shape": shape,
        "sign": sign,
        "n_starts": len(starts),
        "n_in_bounds": n_in_bounds,
        "n_out_of_bounds": len(starts) - n_in_bounds,
        "n_detected": n_detected,
        "detection_rate": (n_detected / n_in_bounds) if n_in_bounds else 0.0,
        "wilson_lo": low,
        "wilson_hi": high,
        "median_latency_s": float(np.median(latencies)) if latencies else None,
        "gate_median_px": float(np.median(gates)) if gates else None,
        "failure_counts": failures,
        "false_events": false_events,
    }


def quick_cell(
    recording: Recording,
    *,
    amplitude_px: float,
    ramp_s: float,
    shape: str,
    sign: int,
    hold_s: float = DEFAULT_HOLD_SECONDS,
) -> dict[str, Any]:
    starts, source = analysis_starts(recording)
    cell = _cell(
        recording,
        amplitude_px=amplitude_px,
        ramp_s=ramp_s,
        shape=shape,
        sign=sign,
        starts=starts,
        hold_s=hold_s,
    )
    cell["start_source"] = source
    return cell


def _offline_comparison(recording: Recording, cells: list[dict[str, Any]]) -> dict[str, Any]:
    offline = _read_offline_summary(recording)
    if offline is None or offline.empty:
        return {"available": False, "mismatches": [], "checked_cells": 0}
    primary = offline[
        (offline["arm"] == "primary")
        & (offline["shape"] == "retreat_return")
        & (offline["sign"] == 1)
    ]
    lookup: dict[tuple[float, float], Any] = {}
    for row in primary.itertuples(index=False):
        lookup[(round(float(row.amplitude_px), 4), round(float(row.ramp_s), 4))] = row
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for cell in cells:
        key = (round(float(cell["amplitude_px"]), 4), round(float(cell["ramp_s"]), 4))
        row = lookup.get(key)
        if row is None:
            continue
        checked += 1
        diffs: dict[str, Any] = {}
        if int(cell["n_in_bounds"]) != int(row.n_in_bounds):
            diffs["n_in_bounds"] = [int(row.n_in_bounds), int(cell["n_in_bounds"])]
        if int(cell["n_detected"]) != int(row.n_detected):
            diffs["n_detected"] = [int(row.n_detected), int(cell["n_detected"])]
        if abs(float(cell["detection_rate"]) - float(row.detection_rate)) > 1e-9:
            diffs["detection_rate"] = [float(row.detection_rate), float(cell["detection_rate"])]
        if int(cell["false_events"]) != int(row.false_events):
            diffs["false_events"] = [int(row.false_events), int(cell["false_events"])]
        if diffs:
            mismatches.append(
                {
                    "amplitude_px": cell["amplitude_px"],
                    "ramp_s": cell["ramp_s"],
                    "offline": "saved injection_summary.csv",
                    "computed": "DEMO MODE replay",
                    "differences": diffs,
                }
            )
    return {
        "available": True,
        "source": str((recording.offline_dir or Path(".")) / "tables" / "injection_summary.csv"),
        "checked_cells": checked,
        "mismatches": mismatches,
        "matches": checked > 0 and not mismatches,
    }


@dataclass
class SummaryJob:
    recording_id: str
    status: str = "running"
    completed_cells: int = 0
    total_cells: int = 0
    completed_replays: int = 0
    total_replays: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    thread: threading.Thread | None = field(default=None, repr=False)

    def progress(self) -> dict[str, Any]:
        fraction = self.completed_cells / self.total_cells if self.total_cells else 0.0
        return {
            "recording_id": self.recording_id,
            "status": self.status,
            "completed_cells": self.completed_cells,
            "total_cells": self.total_cells,
            "completed_replays": self.completed_replays,
            "total_replays": self.total_replays,
            "fraction": fraction,
            "error": self.error,
        }


class SummaryManager:
    def __init__(self) -> None:
        self._jobs: dict[str, SummaryJob] = {}
        self._quick_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, recording: Recording) -> SummaryJob:
        with self._lock:
            existing = self._jobs.get(recording.id)
            if existing is not None and existing.status in {"running", "done"}:
                return existing
            job = SummaryJob(recording_id=recording.id)
            self._jobs[recording.id] = job
        thread = threading.Thread(target=self._run, args=(recording, job), daemon=True)
        job.thread = thread
        thread.start()
        return job

    def get(self, recording: Recording) -> SummaryJob:
        with self._lock:
            job = self._jobs.get(recording.id)
        if job is None:
            return self.start(recording)
        return job

    def _run(self, recording: Recording, job: SummaryJob) -> None:
        try:
            amplitudes, ramps = _grid_definition(recording)
            starts, start_source = analysis_starts(recording)
            job.total_cells = len(amplitudes) * len(ramps)
            job.total_replays = job.total_cells * len(starts)
            cells: list[dict[str, Any]] = []
            aggregate_failures: dict[str, int] = {}
            false_events = 0
            for amplitude_px in amplitudes:
                for ramp_s in ramps:
                    cell = _cell(
                        recording,
                        amplitude_px=amplitude_px,
                        ramp_s=ramp_s,
                        shape="retreat_return",
                        sign=1,
                        starts=starts,
                    )
                    cells.append(cell)
                    job.completed_cells += 1
                    job.completed_replays += len(starts)
                    false_events += int(cell["false_events"])
                    for key, value in cell["failure_counts"].items():
                        aggregate_failures[key] = aggregate_failures.get(key, 0) + int(value)

            minimums = []
            for ramp_s in ramps:
                rows = sorted(
                    [cell for cell in cells if cell["ramp_s"] == ramp_s],
                    key=lambda cell: cell["amplitude_px"],
                )
                hit = next(
                    (
                        cell
                        for cell in rows
                        if cell["n_in_bounds"] > 0 and cell["detection_rate"] >= 0.90
                    ),
                    None,
                )
                minimums.append(
                    {
                        "ramp_s": ramp_s,
                        "min_amplitude_px": None if hit is None else hit["amplitude_px"],
                        "min_amplitude_pct_frame": None
                        if hit is None
                        else hit["amplitude_pct_frame"],
                        "detection_rate": None if hit is None else hit["detection_rate"],
                        "n_in_bounds": None if hit is None else hit["n_in_bounds"],
                    }
                )

            result = {
                "recording_id": recording.id,
                "status": "done",
                "grid": {
                    "amplitudes_px": amplitudes,
                    "ramps_s": ramps,
                    "hold_s": DEFAULT_HOLD_SECONDS,
                    "shape": "retreat_return",
                    "sign": 1,
                },
                "cells": cells,
                "minimum_90": minimums,
                "failure_modes": aggregate_failures,
                "false_events_outside_injection": false_events,
                "analysis_starts": {
                    "source": start_source,
                    "count": len(starts),
                    "starts_utc": [recording.samples[index].time.isoformat() for index in starts],
                },
                "offline_comparison": _offline_comparison(recording, cells),
                "noise": {
                    "frame_height_px": recording.frame_height,
                    "step_std_px": recording.step_std_px,
                    "move90_median_px": recording.move90_median_px,
                    "move90_p95_px": recording.move90_p95_px,
                    "move90_max_px": recording.move90_max_px,
                    "gate_median_px": recording.gate_median_px,
                    "gate_p95_px": recording.gate_p95_px,
                    "drawdown_floor_px": None
                    if recording.config is None
                    else recording.config.drawdown_floor_px,
                },
                "caveat": (
                    "Quick estimate only: independent position-level injections on this "
                    "recording's own noise. It is not a real-world sensitivity or "
                    "false-alarm claim and does not test segmentation during a real retreat."
                ),
            }
            job.result = result
            job.status = "done"
        except Exception as exc:  # pragma: no cover - background guard
            job.status = "error"
            job.error = str(exc)

    def compare_cell(
        self,
        recording: Recording,
        *,
        amplitude_px: float,
        ramp_s: float,
        shape: str,
        sign: int,
        hold_s: float = DEFAULT_HOLD_SECONDS,
    ) -> dict[str, Any]:
        key = (
            recording.id,
            round(amplitude_px, 4),
            round(ramp_s, 4),
            shape,
            sign,
            round(hold_s, 4),
        )
        with self._lock:
            cached = self._quick_cache.get(key)
        if cached is not None:
            return cached
        cell = quick_cell(
            recording,
            amplitude_px=amplitude_px,
            ramp_s=ramp_s,
            shape=shape,
            sign=sign,
            hold_s=hold_s,
        )
        with self._lock:
            self._quick_cache[key] = cell
        return cell


_summary_manager = SummaryManager()


def get_summary_manager() -> SummaryManager:
    return _summary_manager
