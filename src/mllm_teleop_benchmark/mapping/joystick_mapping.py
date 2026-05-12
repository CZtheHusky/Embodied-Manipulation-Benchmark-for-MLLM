from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import MappingConfig

CONTROL_DIMS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


@dataclass(frozen=True)
class ControlCommand:
    values: dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = 0.0) -> float:
        return float(self.values.get(name, default))


class JoystickMapping(ABC):
    visible_to_agent: bool

    @abstractmethod
    def map_action(self, action: JoystickAction) -> ControlCommand:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError


class FixedJoystickMapping(JoystickMapping):
    def __init__(self, channels: dict[str, str], visible_to_agent: bool = True) -> None:
        for dim in channels.values():
            if dim not in CONTROL_DIMS:
                raise ValueError(f"Unknown control dimension: {dim}")
        self.channels = dict(channels)
        self.visible_to_agent = visible_to_agent

    @classmethod
    def from_config(cls, config: MappingConfig) -> "FixedJoystickMapping":
        if config.type != "fixed":
            raise ValueError(f"Unsupported mapping type: {config.type}")
        return cls(config.channels, config.visible_to_agent)

    def map_action(self, action: JoystickAction) -> ControlCommand:
        values: dict[str, float] = {}
        for channel, dim in self.channels.items():
            if channel not in action.channels:
                raise ValueError(f"Action is missing mapped channel: {channel}")
            values[dim] = float(action.channels[channel])
        return ControlCommand(values)

    def metadata(self) -> dict[str, object]:
        if not self.visible_to_agent:
            return {"type": "fixed", "visible_to_agent": False}
        return {"type": "fixed", "visible_to_agent": True, "channels": dict(self.channels)}
