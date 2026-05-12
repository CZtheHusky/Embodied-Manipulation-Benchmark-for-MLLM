from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ActionSpaceConfig, ChannelConfig


@dataclass(frozen=True)
class ActionSpace:
    channels: dict[str, ChannelConfig]

    @classmethod
    def from_config(cls, config: ActionSpaceConfig) -> "ActionSpace":
        if config.type != "joystick":
            raise ValueError(f"Unsupported action space type: {config.type}")
        return cls(channels=dict(config.channels))

    def schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {}
        for name, channel in self.channels.items():
            if channel.kind == "discrete":
                schema[name] = {"kind": "discrete", "values": list(channel.values)}
            elif channel.kind == "continuous":
                schema[name] = {"kind": "continuous", "min": channel.minimum, "max": channel.maximum}
            else:
                schema[name] = {"kind": channel.kind}
        return schema

    def sample(self, rng: random.Random) -> JoystickAction:
        values: dict[str, float] = {}
        for name, channel in self.channels.items():
            if channel.kind == "discrete":
                values[name] = float(rng.choice(channel.values))
            elif channel.kind == "continuous":
                if channel.minimum is None or channel.maximum is None:
                    raise ValueError(f"Continuous channel {name} requires min and max")
                values[name] = rng.uniform(channel.minimum, channel.maximum)
            else:
                raise ValueError(f"Unsupported channel kind for sampling: {channel.kind}")
        return JoystickAction(values)
