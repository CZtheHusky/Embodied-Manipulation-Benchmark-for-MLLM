from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Pose:
    position: Vec3
    rotation_rpy: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Observation:
    step_id: int
    images: dict[str, Any]
    state: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulatorAction:
    """Base class for actions after benchmark-level controller translation."""


@dataclass(frozen=True)
class SimAction(SimulatorAction):
    delta_position: Vec3
    delta_rotation_rpy: Vec3
    gripper: float


@dataclass(frozen=True)
class GymBoxAction(SimulatorAction):
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManiSkillAction(SimulatorAction):
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetaworldAction(SimulatorAction):
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobosuiteAction(SimulatorAction):
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalvinAction(SimulatorAction):
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DMControlAction(SimulatorAction):
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepTransition:
    step_id: int
    observation: Observation
    raw_action: Any
    parsed_action: dict[str, float] | None
    validation: ValidationResult
    sim_action: SimulatorAction
    next_observation: Observation
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: int
    seed: int
    success: bool
    steps: int
    metrics: dict[str, Any]
    log_dir: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    episodes: list[EpisodeResult]
    aggregate_metrics: dict[str, Any]
