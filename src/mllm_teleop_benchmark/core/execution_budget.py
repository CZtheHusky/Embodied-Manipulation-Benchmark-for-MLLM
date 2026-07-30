from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from mllm_teleop_benchmark.config.schema import BenchmarkConfig, SimulatorConfig


STEP_BUDGET_UNIT = "agent_command"


@dataclass(frozen=True)
class ExecutionBudget:
    max_agent_steps: int
    max_internal_substeps_per_agent_step: int
    required_simulator_steps: int
    configured_simulator_step_horizon: int

    def metadata(self) -> dict[str, Any]:
        return {
            "step_budget_unit": STEP_BUDGET_UNIT,
            "max_agent_steps": self.max_agent_steps,
            "max_internal_substeps_per_agent_step": (
                self.max_internal_substeps_per_agent_step
            ),
            "required_simulator_steps": self.required_simulator_steps,
            "configured_simulator_step_horizon": (
                self.configured_simulator_step_horizon
            ),
        }


def execution_budget(config: BenchmarkConfig) -> ExecutionBudget:
    max_agent_steps = int(config.task.max_steps)
    if max_agent_steps <= 0:
        raise ValueError("task.max_steps must be positive")

    max_substeps = _max_internal_substeps(config)
    required_simulator_steps = max_agent_steps * max_substeps
    requested_horizon = _configured_horizon(config.simulator)
    configured_horizon = max(required_simulator_steps, requested_horizon or 0)
    return ExecutionBudget(
        max_agent_steps=max_agent_steps,
        max_internal_substeps_per_agent_step=max_substeps,
        required_simulator_steps=required_simulator_steps,
        configured_simulator_step_horizon=configured_horizon,
    )


def simulator_config_for_execution_budget(
    config: BenchmarkConfig,
) -> SimulatorConfig:
    budget = execution_budget(config)
    params = dict(config.simulator.params)
    params["max_episode_steps"] = budget.configured_simulator_step_horizon
    return replace(config.simulator, params=params)


def _max_internal_substeps(config: BenchmarkConfig) -> int:
    if config.agent.type != "mllm":
        return 1

    motion_channel_count = len(config.control_interface.active_dimensions)
    motion_channels = config.control_interface.channel_names()[:motion_channel_count]
    maximum_magnitude = 0.0
    for name in motion_channels:
        channel = config.action_space.channels[name]
        if channel.kind == "continuous":
            if channel.minimum is None or channel.maximum is None:
                raise ValueError(
                    f"Continuous motion channel {name} requires min and max"
                )
            magnitude = max(abs(channel.minimum), abs(channel.maximum))
        elif channel.kind == "discrete":
            magnitude = max((abs(value) for value in channel.values), default=0.0)
        else:
            raise ValueError(
                f"Unsupported motion channel kind for execution budget: {channel.kind}"
            )
        if not math.isfinite(magnitude):
            raise ValueError(f"Motion channel {name} bounds must be finite")
        maximum_magnitude = max(maximum_magnitude, magnitude)
    return max(1, math.ceil(maximum_magnitude))


def _configured_horizon(config: SimulatorConfig) -> int | None:
    raw_horizon = config.params.get("max_episode_steps")
    if raw_horizon is None:
        return None
    if isinstance(raw_horizon, bool):
        raise ValueError("simulator.params.max_episode_steps must be a positive integer")
    horizon = int(raw_horizon)
    if horizon <= 0:
        raise ValueError("simulator.params.max_episode_steps must be a positive integer")
    return horizon
