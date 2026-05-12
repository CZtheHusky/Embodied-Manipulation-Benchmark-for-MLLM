from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required config field: {key}")
    return data[key]


@dataclass(frozen=True)
class BenchmarkSection:
    name: str
    num_episodes: int
    seed: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSection":
        num_episodes = int(_require(data, "num_episodes"))
        if num_episodes <= 0:
            raise ValueError("benchmark.num_episodes must be positive")
        return cls(name=str(_require(data, "name")), num_episodes=num_episodes, seed=int(_require(data, "seed")))


@dataclass(frozen=True)
class SimulatorConfig:
    type: str
    control_dt: float = 0.1
    workspace_low: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    workspace_high: tuple[float, float, float] = (1.0, 1.0, 1.0)
    env_id: str | None = None
    render_mode: str | None = None
    frame_skip: int | None = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    wrappers: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulatorConfig":
        bounds = data.get("workspace_bounds", {})
        low = tuple(float(v) for v in bounds.get("low", (-1.0, -1.0, 0.0)))
        high = tuple(float(v) for v in bounds.get("high", (1.0, 1.0, 1.0)))
        if len(low) != 3 or len(high) != 3:
            raise ValueError("simulator.workspace_bounds low/high must be 3D")
        return cls(
            type=str(_require(data, "type")),
            control_dt=float(data.get("control_dt", 0.1)),
            workspace_low=low,
            workspace_high=high,
            env_id=str(data["env_id"]) if "env_id" in data else None,
            render_mode=str(data["render_mode"]) if data.get("render_mode") is not None else None,
            frame_skip=int(data["frame_skip"]) if "frame_skip" in data else None,
            env_kwargs=dict(data.get("env_kwargs", {})),
            wrappers=tuple(str(v) for v in data.get("wrappers", ())),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class TaskConfig:
    type: str
    max_steps: int
    success_threshold: float
    target_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_ee_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_hold_steps: int = 1
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        target = tuple(float(v) for v in data.get("target_position", (0.0, 0.0, 0.0)))
        initial = tuple(float(v) for v in data.get("initial_ee_position", (0.0, 0.0, 0.0)))
        if len(target) != 3 or len(initial) != 3:
            raise ValueError("task target_position and initial_ee_position must be 3D")
        return cls(
            type=str(_require(data, "type")),
            max_steps=int(_require(data, "max_steps")),
            success_threshold=float(_require(data, "success_threshold")),
            target_position=target,
            initial_ee_position=initial,
            goal_hold_steps=int(data.get("goal_hold_steps", 1)),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class CameraSpec:
    name: str
    type: str
    width: int
    height: int
    pose: tuple[float, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraSpec":
        return cls(
            name=str(_require(data, "name")),
            type=str(data.get("type", "dummy_rgb")),
            width=int(data.get("width", 256)),
            height=int(data.get("height", 256)),
            pose=tuple(float(v) for v in data.get("pose", ())),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class CameraConfig:
    manager: str
    visible_metadata: bool
    cameras: tuple[CameraSpec, ...]
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraConfig":
        cameras = tuple(CameraSpec.from_dict(item) for item in _require(data, "cameras"))
        if not cameras:
            raise ValueError("camera.cameras must contain at least one camera")
        return cls(
            manager=str(data.get("manager", "fixed")),
            visible_metadata=bool(data.get("visible_metadata", True)),
            cameras=cameras,
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class ChannelConfig:
    kind: str
    values: tuple[float, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelConfig":
        return cls(
            kind=str(_require(data, "kind")),
            values=tuple(float(v) for v in data.get("values", ())),
            minimum=float(data["min"]) if "min" in data else None,
            maximum=float(data["max"]) if "max" in data else None,
        )


@dataclass(frozen=True)
class ActionSpaceConfig:
    type: str
    channels: dict[str, ChannelConfig]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSpaceConfig":
        channels = {name: ChannelConfig.from_dict(value) for name, value in _require(data, "channels").items()}
        return cls(type=str(data.get("type", "joystick")), channels=channels)


@dataclass(frozen=True)
class MappingConfig:
    type: str
    visible_to_agent: bool
    channels: dict[str, str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MappingConfig":
        return cls(type=str(data.get("type", "fixed")), visible_to_agent=bool(data.get("visible_to_agent", True)), channels=dict(_require(data, "channels")))


@dataclass(frozen=True)
class ControllerConfig:
    type: str
    frame: str
    scales: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControllerConfig":
        return cls(
            type=str(_require(data, "type")),
            frame=str(data.get("frame", "base")),
            scales={k: float(v) for k, v in data.get("scales", {}).items()},
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class FrameTransformConfig:
    type: str
    visible_to_agent: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameTransformConfig":
        return cls(type=str(data.get("type", "identity")), visible_to_agent=bool(data.get("visible_to_agent", True)))


@dataclass(frozen=True)
class AgentConfig:
    type: str
    invalid_action_recovery: str = "noop"
    history_window: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        return cls(type=str(_require(data, "type")), invalid_action_recovery=str(data.get("invalid_action_recovery", "noop")), history_window=int(data.get("history_window", 5)))


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: Path
    save_images: bool = True
    save_video: bool = False
    save_trajectory: bool = True
    save_config_snapshot: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        return cls(
            output_dir=Path(str(_require(data, "output_dir"))),
            save_images=bool(data.get("save_images", True)),
            save_video=bool(data.get("save_video", False)),
            save_trajectory=bool(data.get("save_trajectory", True)),
            save_config_snapshot=bool(data.get("save_config_snapshot", True)),
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark: BenchmarkSection
    simulator: SimulatorConfig
    task: TaskConfig
    camera: CameraConfig
    action_space: ActionSpaceConfig
    mapping: MappingConfig
    controller: ControllerConfig
    frame_transform: FrameTransformConfig
    agent: AgentConfig
    logging: LoggingConfig
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        return cls(
            benchmark=BenchmarkSection.from_dict(_require(data, "benchmark")),
            simulator=SimulatorConfig.from_dict(_require(data, "simulator")),
            task=TaskConfig.from_dict(_require(data, "task")),
            camera=CameraConfig.from_dict(_require(data, "camera")),
            action_space=ActionSpaceConfig.from_dict(_require(data, "action_space")),
            mapping=MappingConfig.from_dict(_require(data, "mapping")),
            controller=ControllerConfig.from_dict(_require(data, "controller")),
            frame_transform=FrameTransformConfig.from_dict(_require(data, "frame_transform")),
            agent=AgentConfig.from_dict(_require(data, "agent")),
            logging=LoggingConfig.from_dict(_require(data, "logging")),
            raw=data,
        )
