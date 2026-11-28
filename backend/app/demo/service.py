"""In-memory simulation manager and presets for DEMO MODE.

No database session, model, camera client or live worker is imported here.
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from app.demo.registry import Recording
from app.demo.replay import (
    DEFAULT_HOLD_SECONDS,
    FAILURE_TEXT,
    replay_injection,
)
from app.demo.summary import (
    SEED,
    SummaryManager,
    _read_offline_summary,
    get_summary_manager,
)

SPEEDS = {1, 10, 30, 60}
SHAPES = {"retreat_return", "persistent"}
MIN_AMPLITUDE_PCT = 5.0
MAX_AMPLITUDE_PCT = 60.0
MAX_CONCURRENT_SIMULATIONS = 4
SIMULATION_TTL_SECONDS = 30 * 60.0
STOPPED_TTL_SECONDS = 120.0


class DemoValidationError(ValueError):
    """User-facing validation failure (HTTP 422/400)."""


class DemoCapacityError(RuntimeError):
    """Concurrency cap reached (HTTP 429)."""


@dataclass
class Simulation:
    id: str
    recording_id: str
    recording_label: str
    params: dict[str, Any]
    messages: list[dict[str, Any]]
    result: dict[str, Any]
    warnings: list[str]
    created_at: datetime
    status: str = "ready"
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    completed_at: datetime | None = None
    cleanup_at: datetime | None = None

    @property
    def active(self) -> bool:
        return self.status in {"ready", "streaming"}

    def status_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "recording_id": self.recording_id,
            "recording_label": self.recording_label,
            "status": self.status,
            "params": self.params,
            "warnings": self.warnings,
            "result": self.result,
            "banner": "SIMULATION - not real data",
        }


class SimulationManager:
    def __init__(self, summaries: SummaryManager | None = None) -> None:
        self._simulations: dict[str, Simulation] = {}
        self._lock = threading.Lock()
        self.summaries = summaries or get_summary_manager()

    def _cleanup(self) -> None:
        now = datetime.now(UTC)
        with self._lock:
            expired = [
                sim_id
                for sim_id, sim in self._simulations.items()
                if now - sim.created_at > timedelta(seconds=SIMULATION_TTL_SECONDS)
                or (sim.cleanup_at is not None and now >= sim.cleanup_at and not sim.active)
            ]
            for sim_id in expired:
                self._simulations.pop(sim_id, None)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for sim in self._simulations.values() if sim.active)

    def create(
        self,
        recording: Recording,
        body: dict[str, Any],
    ) -> Simulation:
        self._cleanup()
        if not recording.replayable:
            raise DemoValidationError(recording.reason or "this recording is not replayable")
        params, warnings, start_index = validate_simulation(recording, body)
        if recording.status == "limited" and self.active_count() > 0:
            raise DemoValidationError(
                "limited data: only one illustrative simulation can run at a time"
            )
        if self.active_count() >= MAX_CONCURRENT_SIMULATIONS:
            raise DemoCapacityError(
                f"at most {MAX_CONCURRENT_SIMULATIONS} demo simulations can run at once"
            )

        assert recording.config is not None
        result = replay_injection(
            samples=recording.samples,
            segments=recording.segments,
            config=recording.config,
            start_index=start_index,
            amplitude_px=params["amplitude_px"],
            ramp_s=params["ramp_s"],
            hold_s=params["hold_s"],
            shape=params["shape"],
            sign=params["sign"],
            snapshot=recording.snapshot(start_index),
            injected_min_px=recording.injected_min_px,
            injected_max_px=recording.injected_max_px,
        )
        sim_id = f"SIM-{uuid.uuid4().hex[:12]}"
        messages = build_messages(
            sim_id=sim_id,
            recording=recording,
            params=params,
            result=result,
            warnings=warnings,
        )
        summary = build_result_summary(sim_id, params, result)
        simulation = Simulation(
            id=sim_id,
            recording_id=recording.id,
            recording_label=recording.label,
            params=params,
            messages=messages,
            result=summary,
            warnings=warnings,
            created_at=datetime.now(UTC),
        )
        with self._lock:
            self._simulations[sim_id] = simulation
        return simulation

    def get(self, sim_id: str) -> Simulation | None:
        self._cleanup()
        with self._lock:
            return self._simulations.get(sim_id)

    def stop(self, sim_id: str) -> Simulation | None:
        with self._lock:
            simulation = self._simulations.get(sim_id)
        if simulation is None:
            return None
        simulation.stop_event.set()
        simulation.status = "stopped"
        simulation.cleanup_at = datetime.now(UTC) + timedelta(seconds=STOPPED_TTL_SECONDS)
        return simulation

    def delete(self, sim_id: str) -> bool:
        with self._lock:
            return self._simulations.pop(sim_id, None) is not None

    def disconnected(self, sim_id: str) -> None:
        with self._lock:
            simulation = self._simulations.get(sim_id)
        if simulation is None:
            return
        if simulation.status == "streaming":
            simulation.status = "stopped"
        simulation.cleanup_at = datetime.now(UTC) + timedelta(seconds=STOPPED_TTL_SECONDS)

    def mark_streaming(self, sim_id: str) -> None:
        with self._lock:
            simulation = self._simulations.get(sim_id)
        if simulation is not None and simulation.status == "ready":
            simulation.status = "streaming"

    def mark_done(self, sim_id: str) -> None:
        with self._lock:
            simulation = self._simulations.get(sim_id)
        if simulation is not None:
            simulation.status = "done"
            simulation.completed_at = datetime.now(UTC)
            simulation.cleanup_at = datetime.now(UTC) + timedelta(seconds=STOPPED_TTL_SECONDS)


_simulation_manager = SimulationManager()


def get_simulation_manager() -> SimulationManager:
    return _simulation_manager


def _parse_start(recording: Recording, value: Any, rng: np.random.Generator) -> int:
    candidates = list(recording.valid_starts)
    if not candidates:
        raise DemoValidationError("this recording has no valid start times")
    if value in (None, "", "random"):
        return int(rng.choice(candidates))
    if not isinstance(value, str):
        raise DemoValidationError("start must be 'random' or an ISO timestamp")
    requested = value.strip()
    if requested.lower() == "random":
        return int(rng.choice(candidates))
    try:
        moment = datetime.fromisoformat(requested.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DemoValidationError("start is not a valid ISO timestamp") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    nearest = min(
        candidates,
        key=lambda index: abs((recording.samples[index].time - moment).total_seconds()),
    )
    if abs((recording.samples[nearest].time - moment).total_seconds()) > 1.0:
        raise DemoValidationError("start is not one of this recording's valid start times")
    return nearest


def validate_simulation(
    recording: Recording, body: dict[str, Any]
) -> tuple[dict[str, Any], list[str], int]:
    if recording.config is None or recording.frame_height is None:
        raise DemoValidationError("recording profile is incomplete")
    height = float(recording.frame_height)
    interval = float(recording.nominal_interval_s or 0.0)
    warnings: list[str] = []

    try:
        amplitude_pct = float(body.get("amplitude_pct"))
    except (TypeError, ValueError) as exc:
        raise DemoValidationError("amplitude_pct must be a number") from exc
    if not math.isfinite(amplitude_pct) or not (
        amplitude_pct == 0.0 or MIN_AMPLITUDE_PCT <= amplitude_pct <= MAX_AMPLITUDE_PCT
    ):
        raise DemoValidationError(
            "amplitude_pct must be 0 (no injection) or between "
            f"{MIN_AMPLITUDE_PCT:g} and {MAX_AMPLITUDE_PCT:g} percent of frame height"
        )

    try:
        ramp_s = float(body.get("ramp_s"))
        hold_s = float(body.get("hold_s", DEFAULT_HOLD_SECONDS))
    except (TypeError, ValueError) as exc:
        raise DemoValidationError("ramp_s and hold_s must be numbers") from exc
    if not math.isfinite(ramp_s) or ramp_s <= 0:
        raise DemoValidationError("ramp_s must be positive")
    if not math.isfinite(hold_s) or hold_s <= 0:
        raise DemoValidationError("hold_s must be positive")
    if interval and ramp_s < interval and amplitude_pct > 0:
        raise DemoValidationError(
            f"ramp_s must be at least the recording's {interval:g} s sampling interval"
        )
    if interval and hold_s < interval:
        raise DemoValidationError(
            f"hold_s must be at least the recording's {interval:g} s sampling interval"
        )
    if interval and ramp_s < 3.0 * interval:
        samples_in_ramp = max(1, int(round(ramp_s / interval)))
        warnings.append(
            f"ramp is only about {samples_in_ramp} samples at this recording's "
            f"{interval:g} s interval"
        )

    shape = str(body.get("shape") or "retreat_return")
    if shape not in SHAPES:
        raise DemoValidationError("shape must be 'retreat_return' or 'persistent'")
    sign = int(body.get("sign", 1))
    if sign not in (-1, 1):
        raise DemoValidationError("sign must be +1 (seaward) or -1 (landward)")
    speed = int(body.get("speed", 30))
    if speed not in SPEEDS:
        raise DemoValidationError("speed must be one of 1, 10, 30 or 60")

    try:
        seed = int(body.get("seed", SEED))
    except (TypeError, ValueError) as exc:
        raise DemoValidationError("seed must be an integer") from exc
    rng = np.random.default_rng(seed)
    start_index = _parse_start(recording, body.get("start", "random"), rng)

    segment = next(
        (item for item in recording.segments if item.start <= start_index < item.end),
        None,
    )
    if segment is None:
        raise DemoValidationError("start is not inside a replayable segment")
    remaining = (
        recording.samples[segment.end - 1].time - recording.samples[start_index].time
    ).total_seconds()
    needed = 2.0 * ramp_s + hold_s + float(recording.config.cooldown_seconds)
    if remaining < needed:
        raise DemoValidationError(
            "this start does not have enough remaining data for the ramp, hold and cooldown"
        )

    amplitude_px = round(height * amplitude_pct / 100.0, 4)
    if recording.injected_min_px is not None and recording.injected_max_px is not None:
        min_position = min(
            sample.p
            for sample in recording.samples[segment.start : segment.end]
            if sample.p is not None
        )
        max_position = max(
            sample.p
            for sample in recording.samples[segment.start : segment.end]
            if sample.p is not None
        )
        if sign > 0 and max_position + amplitude_px > recording.injected_max_px:
            warnings.append("the injected seaward position can leave the recording's polygon/frame")
        if sign < 0 and min_position - amplitude_px < recording.injected_min_px:
            warnings.append(
                "the injected landward position can leave the recording's polygon/frame"
            )

    params = {
        "amplitude_pct": amplitude_pct,
        "amplitude_px": amplitude_px,
        "ramp_s": ramp_s,
        "hold_s": hold_s,
        "shape": shape,
        "sign": sign,
        "speed": speed,
        "start": recording.samples[start_index].time.isoformat(),
        "start_index": start_index,
        "seed": seed,
    }
    return params, warnings, start_index


def build_messages(
    *,
    sim_id: str,
    recording: Recording,
    params: dict[str, Any],
    result: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "type": "status",
            "sim_id": sim_id,
            "banner": "SIMULATION - not real data",
            "recording_id": recording.id,
            "recording_label": recording.label,
            "frame_height_px": recording.frame_height,
            "params": params,
            "warnings": warnings,
            "note": "Position-level injection on recorded noise; not a claim about real events.",
        }
    ]
    event_payload = None
    detection_time = result["detection_time"]
    for item in result["series"]:
        is_detection = (
            detection_time is not None and item["time"] == detection_time and event_payload is None
        )
        if is_detection:
            event_payload = {
                "id": f"{sim_id}-EVT",
                "classification": result["classification"],
                "detected_at": item["time"].isoformat(),
                "latency_s": result["latency_from_t0_s"],
                "gate_px": item["gate"],
                "z": item["z"],
                "ck": item["ck"],
            }
        sample_message = {
            "type": "sample",
            "time": item["time"].isoformat(),
            "real": item["real"],
            "injected": item["injected"],
            "baseline": item["baseline"],
            "z": item["z"],
            "ck": item["ck"],
            "gate": item["gate"],
            "state": item["state"],
            "classification": item["classification"],
            "persistence": item["persistence"],
            "event": event_payload if is_detection else None,
        }
        messages.append(sample_message)
        if is_detection and event_payload is not None:
            messages.append({"type": "event", "event": event_payload})
    messages.append(
        {
            "type": "done",
            "sim_id": sim_id,
            "result": build_result_summary(sim_id, params, result),
        }
    )
    return messages


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_result_summary(
    sim_id: str, params: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    detected = bool(result["detected"])
    if detected:
        outcome = "caught"
        latency = result["latency_from_t0_s"]
        explanation = "The detector emitted an event during the injection window" + (
            f" after {latency:.0f} s." if latency is not None else "."
        )
        if not result["in_bounds"]:
            explanation += " The injected position left the polygon/frame extent."
    else:
        outcome = "missed"
        explanation = result["failure_text"] or FAILURE_TEXT.get(
            result["failure_mode"], "No event was emitted during the injection window."
        )
    return {
        "sim_id": sim_id,
        "outcome": outcome,
        "detected": detected,
        "classification": result["classification"],
        "state_reached": result["state_reached"],
        "latency_from_t0_s": _finite(result["latency_from_t0_s"]),
        "latency_from_gate_s": _finite(result["latency_from_gate_s"]),
        "failure_mode": result["failure_mode"],
        "explanation": explanation,
        "peak_z": _finite(result["peak_z"]),
        "peak_ck_px": _finite(result["peak_ck_px"]),
        "false_events": result["false_events"],
        "in_bounds": result["in_bounds"],
        "params": params,
        "label": "SIMULATION - not real data",
    }


def _offline_row(
    recording: Recording, *, amplitude_px: float, ramp_s: float, sign: int
) -> dict[str, Any] | None:
    frame = _read_offline_summary(recording)
    if frame is None or frame.empty:
        return None
    arm = "primary" if sign > 0 else "landward"
    subset = frame[
        (frame["arm"] == arm)
        & (frame["shape"] == "retreat_return")
        & (frame["sign"] == sign)
        & (frame["ramp_s"] == ramp_s)
    ]
    if subset.empty:
        return None
    row = min(
        subset.itertuples(index=False),
        key=lambda item: abs(float(item.amplitude_px) - amplitude_px),
    )
    return {
        "detection_rate": float(row.detection_rate),
        "n_in_bounds": int(row.n_in_bounds),
        "n_detected": int(row.n_detected),
        "wilson_lo": float(row.wilson_lo),
        "wilson_hi": float(row.wilson_hi),
        "matched_amplitude_px": float(row.amplitude_px),
        "ramp_s": float(row.ramp_s),
        "source": "saved offline injection_summary.csv",
        "exact_amplitude": abs(float(row.amplitude_px) - amplitude_px) < 1e-6,
    }


def _expected(
    recording: Recording,
    *,
    amplitude_px: float,
    ramp_s: float,
    sign: int,
    no_injection: bool = False,
) -> dict[str, Any]:
    if no_injection:
        return {
            "available": True,
            "detection_rate": 0.0,
            "n_in_bounds": recording.valid_start_count,
            "n_detected": 0,
            "wilson_lo": 0.0,
            "wilson_hi": 0.0,
            "source": "no injected excursion",
            "note": "The recorded series is replayed unchanged.",
        }
    offline = _offline_row(recording, amplitude_px=amplitude_px, ramp_s=ramp_s, sign=sign)
    if offline is not None:
        if not offline["exact_amplitude"]:
            offline["note"] = (
                "Nearest tested amplitude in the saved offline grid "
                f"({offline['matched_amplitude_px']:.0f} px)."
            )
        return {"available": True, **offline}
    job = get_summary_manager()._jobs.get(recording.id)
    if job is not None and job.status == "done" and job.result is not None:
        cells = job.result["cells"]
        matching = [
            cell
            for cell in cells
            if cell["ramp_s"] == ramp_s
            and cell["sign"] == sign
            and cell["shape"] == "retreat_return"
        ]
        if matching:
            cell = min(matching, key=lambda item: abs(item["amplitude_px"] - amplitude_px))
            return {
                "available": True,
                "detection_rate": cell["detection_rate"],
                "n_in_bounds": cell["n_in_bounds"],
                "n_detected": cell["n_detected"],
                "wilson_lo": cell["wilson_lo"],
                "wilson_hi": cell["wilson_hi"],
                "matched_amplitude_px": cell["amplitude_px"],
                "ramp_s": cell["ramp_s"],
                "source": "computed quick-estimate grid",
                "exact_amplitude": abs(cell["amplitude_px"] - amplitude_px) < 1e-6,
            }
    return {
        "available": False,
        "source": "quick estimate still computing",
        "note": "Open the summary panel to compute this recording's grid.",
    }


def _pct(recording: Recording, amplitude_px: float) -> float:
    height = float(recording.frame_height or 720.0)
    return max(
        MIN_AMPLITUDE_PCT,
        min(MAX_AMPLITUDE_PCT, round(100.0 * amplitude_px / height, 2)),
    )


def build_presets(recording: Recording) -> list[dict[str, Any]]:
    if not recording.replayable:
        return []
    config = recording.config
    assert config is not None
    floor = float(config.drawdown_floor_px)
    gate_median = float(recording.gate_median_px or floor)
    gate_p95 = float(recording.gate_p95_px or gate_median)

    fast_px = gate_p95
    fast_pct = _pct(recording, fast_px)
    fast_px = round(float(recording.frame_height or 720.0) * fast_pct / 100.0, 4)

    border_pct = _pct(recording, gate_median)
    border_px = round(float(recording.frame_height or 720.0) * border_pct / 100.0, 4)

    below_pct = _pct(recording, gate_median / 2.0)
    below_px = round(float(recording.frame_height or 720.0) * below_pct / 100.0, 4)

    slow_pct = _pct(recording, max(gate_p95 * 1.5, 0.4 * float(recording.frame_height or 720.0)))
    slow_px = round(float(recording.frame_height or 720.0) * slow_pct / 100.0, 4)

    landward_pct = fast_pct
    landward_px = fast_px

    def preset(
        preset_id: str,
        label: str,
        description: str,
        *,
        amplitude_px: float,
        amplitude_pct: float,
        ramp_s: float,
        shape: str,
        sign: int,
        no_injection: bool = False,
    ) -> dict[str, Any]:
        expected = _expected(
            recording,
            amplitude_px=amplitude_px,
            ramp_s=ramp_s,
            sign=sign,
            no_injection=no_injection,
        )
        return {
            "id": preset_id,
            "label": label,
            "description": description,
            "params": {
                "amplitude_pct": amplitude_pct,
                "amplitude_px": amplitude_px,
                "ramp_s": ramp_s,
                "hold_s": DEFAULT_HOLD_SECONDS,
                "shape": shape,
                "sign": sign,
                "speed": 30,
                "start": "random",
            },
            "expected": expected,
            "banner": "SIMULATION - not real data",
        }

    return [
        preset(
            "no_injection",
            "No injection (real series)",
            "Replay the recorded series unchanged.",
            amplitude_px=0.0,
            amplitude_pct=0.0,
            ramp_s=0.0,
            shape="retreat_return",
            sign=1,
            no_injection=True,
        ),
        preset(
            "fast_retreat",
            "Fast retreat",
            "Amplitude at this recording's p95 effective gate, 60 s ramp.",
            amplitude_px=fast_px,
            amplitude_pct=fast_pct,
            ramp_s=60.0,
            shape="retreat_return",
            sign=1,
        ),
        preset(
            "borderline",
            "Borderline",
            "Amplitude near this recording's median effective gate, 60 s ramp.",
            amplitude_px=border_px,
            amplitude_pct=border_pct,
            ramp_s=60.0,
            shape="retreat_return",
            sign=1,
        ),
        preset(
            "below_gate",
            "Below the gate",
            "About half the median effective gate, 60 s ramp.",
            amplitude_px=below_px,
            amplitude_pct=below_pct,
            ramp_s=60.0,
            shape="retreat_return",
            sign=1,
        ),
        preset(
            "slow_retreat",
            "Slow retreat",
            "Large amplitude over a 600 s ramp.",
            amplitude_px=slow_px,
            amplitude_pct=slow_pct,
            ramp_s=600.0,
            shape="retreat_return",
            sign=1,
        ),
        preset(
            "landward",
            "Landward move",
            "The same excursion in the negative (landward) direction.",
            amplitude_px=landward_px,
            amplitude_pct=landward_pct,
            ramp_s=60.0,
            shape="retreat_return",
            sign=-1,
        ),
    ]


def compare_recordings(
    recordings: list[Recording],
    *,
    amplitude_pct: float,
    ramp_s: float,
    summaries: SummaryManager | None = None,
) -> dict[str, Any]:
    manager = summaries or get_summary_manager()
    rows = []
    for recording in recordings:
        base = {
            "id": recording.id,
            "label": recording.label,
            "status": recording.status,
            "reason": recording.reason,
            "valid_start_count": recording.valid_start_count,
            "frame_height_px": recording.frame_height,
            "step_std_px": recording.step_std_px,
            "move90_median_px": recording.move90_median_px,
            "move90_p95_px": recording.move90_p95_px,
            "gate_median_px": recording.gate_median_px,
            "gate_p95_px": recording.gate_p95_px,
        }
        if not recording.replayable or recording.frame_height is None:
            rows.append({**base, "available": False})
            continue
        amplitude_px = round(float(recording.frame_height) * amplitude_pct / 100.0, 4)
        cell = None
        job = manager._jobs.get(recording.id)
        if job is not None and job.status == "done" and job.result is not None:
            matching = [
                item
                for item in job.result["cells"]
                if item["ramp_s"] == ramp_s
                and item["sign"] == 1
                and item["shape"] == "retreat_return"
                and abs(item["amplitude_px"] - amplitude_px) <= 0.5
            ]
            if matching:
                cell = matching[0]
        if cell is None:
            cell = manager.compare_cell(
                recording,
                amplitude_px=amplitude_px,
                ramp_s=ramp_s,
                shape="retreat_return",
                sign=1,
            )
        rows.append(
            {
                **base,
                "available": True,
                "amplitude_px": amplitude_px,
                "detection_rate": cell["detection_rate"],
                "n_in_bounds": cell["n_in_bounds"],
                "n_starts": cell["n_starts"],
                "n_detected": cell["n_detected"],
                "wilson_lo": cell["wilson_lo"],
                "wilson_hi": cell["wilson_hi"],
                "median_latency_s": cell["median_latency_s"],
                "false_events": cell["false_events"],
                "limited_data": recording.status == "limited",
            }
        )
    return {
        "amplitude_pct": amplitude_pct,
        "ramp_s": ramp_s,
        "shape": "retreat_return",
        "hold_s": DEFAULT_HOLD_SECONDS,
        "recordings": rows,
        "note": (
            "Each bar applies to that recording's own scene and noise. Recordings are "
            "not combined, and this is not a tsunami validation or a real-world "
            "sensitivity/false-alarm claim."
        ),
        "banner": "SIMULATION - not real data",
    }
