from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.types import GymBoxAction, ValidationResult
from mllm_teleop_benchmark.logging.debug_session import DebugSession


class ControlDebugRecorder:
    def __init__(self, config: BenchmarkConfig, *, enabled: bool = False, session: DebugSession | None = None, filename: str = "control_events.jsonl") -> None:
        self.config = config
        self.enabled = bool(enabled)
        self.session = session
        self.fallback_path = Path(config.logging.output_dir) / "debug" / filename

    def reset(self) -> None:
        if not self.enabled:
            return
        if self.session is not None:
            self.session.write_manifest()
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def record(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(self._with_tags(record)), sort_keys=True) + "\n")

    def build_step_record(
        self,
        *,
        source: str,
        control_id: str,
        step_before: int,
        raw_action: dict[str, float],
        validation: ValidationResult,
        sim_action: Any,
        state_before: dict[str, Any],
        state_after: dict[str, Any],
    ) -> dict[str, Any]:
        expected = self.expected_eef_behavior(sim_action)
        return {
            "event": "step",
            "source": source,
            "control_id": control_id,
            "step_before": int(step_before),
            "step_after": int(state_after.get("step_id", step_before + 1)),
            "raw_action": raw_action,
            "valid_action": validation.is_valid,
            "validation_errors": list(validation.errors),
            "expected_behavior": expected,
            "actual_behavior": self.actual_eef_behavior(state_before, state_after),
            "state_before": state_before,
            "state_after": state_after,
        }

    def build_event_record(
        self,
        *,
        event: str,
        source: str,
        control_id: str,
        step_before: int,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": event,
            "source": source,
            "control_id": control_id,
            "step_before": int(step_before),
        }
        if state_before is not None:
            record["state_before"] = state_before
        if state_after is not None:
            record["state_after"] = state_after
        return record

    def _path(self) -> Path:
        if self.session is not None:
            return self.session.control_events_path()
        return self.fallback_path

    def _with_tags(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.session is None:
            return record
        return {**self.session.event_tags(), **record}

    def expected_eef_behavior(self, sim_action: Any) -> dict[str, Any]:
        if not isinstance(sim_action, GymBoxAction):
            return {"sim_action": to_jsonable(sim_action)}
        dimensions = tuple(str(v) for v in sim_action.metadata.get("dimensions", ()))
        values = tuple(float(v) for v in sim_action.values)
        eef_step_scale = float(self.config.simulator.env_kwargs.get("eef_step_scale", 1.0))
        dimension_values = {dim: values[idx] for idx, dim in enumerate(dimensions)}
        expected_delta = {
            dim: dimension_values[dim] * eef_step_scale
            for dim in ("dx", "dy", "dz")
            if dim in dimension_values
        }
        nonzero = {dim: value for dim, value in expected_delta.items() if abs(value) > 1e-12}
        return {
            "controller": sim_action.metadata.get("controller"),
            "dimensions": list(dimensions),
            "controller_output_values": list(values),
            "eef_step_scale": eef_step_scale,
            "expected_delta_eef": expected_delta,
            "description": self._expected_description(nonzero),
            "metadata": sim_action.metadata,
        }

    def actual_eef_behavior(self, state_before: dict[str, Any], state_after: dict[str, Any]) -> dict[str, Any]:
        before_eef = self._first_state_vector(state_before, ("fingertip_position", "ee_position", "end_effector_position"))
        after_eef = self._first_state_vector(state_after, ("fingertip_position", "ee_position", "end_effector_position"))
        before_target = self._first_state_vector(state_before, ("ee_target_xyz", "ee_target_xy", "target_eef_position"))
        after_target = self._first_state_vector(state_after, ("ee_target_xyz", "ee_target_xy", "target_eef_position"))
        return {
            "eef_position_before": before_eef,
            "eef_position_after": after_eef,
            "eef_delta": self._delta_vector(after_eef, before_eef),
            "ee_target_before": before_target,
            "ee_target_after": after_target,
            "ee_target_delta": self._delta_vector(after_target, before_target),
            "target_clamped": bool(state_after.get("ee_target_reachability_clamped", False)),
        }

    def _expected_description(self, expected_delta: dict[str, float]) -> str:
        if not expected_delta:
            return "no end-effector translation expected"
        parts = []
        for dim, value in expected_delta.items():
            direction = "+" if value > 0 else "-"
            parts.append(f"{direction}{dim[1:]} ({value:+.6f} m)")
        return "move EEF " + ", ".join(parts)

    def _first_state_vector(self, state: dict[str, Any], keys: tuple[str, ...]) -> list[float] | None:
        for key in keys:
            vector = self._state_vector(state, key)
            if vector is not None:
                return vector
        return None

    def _state_vector(self, state: dict[str, Any], key: str) -> list[float] | None:
        value = state.get(key)
        if value is None:
            return None
        try:
            return [float(v) for v in value]
        except TypeError:
            return None

    def _delta_vector(self, after: list[float] | None, before: list[float] | None) -> list[float] | None:
        if after is None or before is None or len(after) != len(before):
            return None
        return [float(a - b) for a, b in zip(after, before, strict=True)]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return to_jsonable(value.__dict__)
    return value
