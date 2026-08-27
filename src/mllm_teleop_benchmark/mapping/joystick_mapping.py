from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import MappingConfig

@dataclass(frozen=True)
class ControlCommand:
    values: dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = 0.0) -> float:
        return self.values.get(name, default)


class JoystickMapping(ABC):
    visible_to_agent: bool

    @abstractmethod
    def map_action(self, action: JoystickAction) -> ControlCommand:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError


class FixedJoystickMapping(JoystickMapping):
    def __init__(self, channels: dict[str, str], visible_to_agent: bool = True, signs: dict[str, float] | None = None) -> None:
        self.channels = dict(channels)
        self.signs = {channel: float(value) for channel, value in (signs or {}).items()}
        self.visible_to_agent = visible_to_agent

    @classmethod
    def from_config(cls, config: MappingConfig) -> "FixedJoystickMapping":
        return cls(config.channels, config.visible_to_agent, config.signs)

    def map_action(self, action: JoystickAction) -> ControlCommand:
        values: dict[str, float] = {}
        for channel, dim in self.channels.items():
            values[dim] = float(action.channels[channel]) * self.signs.get(channel, 1.0)
        return ControlCommand(values)

    def metadata(self) -> dict[str, object]:
        if not self.visible_to_agent:
            return {"type": "fixed", "visible_to_agent": False}
        return {"type": "fixed", "visible_to_agent": True, "channels": dict(self.channels), "signs": dict(self.signs)}
