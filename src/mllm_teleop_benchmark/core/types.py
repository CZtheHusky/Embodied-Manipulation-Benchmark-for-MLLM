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
class SimplerEnvAction(SimulatorAction):
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RLBenchAction(SimulatorAction):
    delta_position: Vec3
    delta_rotation_rpy: Vec3 = (0.0, 0.0, 0.0)
    gripper: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobosuiteAction(SimulatorAction):
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoboCasaAction(SimulatorAction):
    values: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiberoAction(SimulatorAction):
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
class DMControlPlanarEEFAction(SimulatorAction):
    delta_xy: tuple[float, float]
    gripper: float
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
class AgentTrace:
    actor_type: str
    status: str = "ok"
    provider: str | None = None
    api: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    request_id: str | None = None
    previous_response_id: str | None = None
    response_id: str | None = None
    context_recovery_count: int = 0
    latency_ms: float | None = None
    attempts: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_hit_ratio: float | None = None
    cache_metrics_status: str = "unavailable"
    common_prefix_bytes: int = 0
    request_bytes: int = 0
    image_count: int = 0
    error: str | None = None
    prompt_version: str | None = None
    artifact_dir: str | None = None


@dataclass(frozen=True)
class AgentArtifact:
    relative_path: str
    content: bytes


@dataclass(frozen=True)
class AgentDecision:
    action: Any
    trace: AgentTrace
    artifacts: tuple[AgentArtifact, ...] = ()


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
    agent_trace: AgentTrace | None = None
    reflection: str | None = None


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: int
    seed: int
    success: bool
    steps: int
    metrics: dict[str, Any]
    log_dir: str | None = None
    termination_reason: str = "completed"


@dataclass(frozen=True)
class BenchmarkResult:
    episodes: list[EpisodeResult]
    aggregate_metrics: dict[str, Any]
