from __future__ import annotations

import math
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
            else:
                values[name] = rng.uniform(channel.minimum, channel.maximum)
        return JoystickAction(values)

    def clip_continuous_channels(
        self,
        action: JoystickAction,
        channel_names: tuple[str, ...],
    ) -> JoystickAction:
        values = dict(action.channels)
        for name in channel_names:
            channel = self.channels.get(name)
            if (
                channel is None
                or channel.kind != "continuous"
                or channel.minimum is None
                or channel.maximum is None
                or name not in values
                or not math.isfinite(values[name])
            ):
                continue
            values[name] = min(
                max(values[name], channel.minimum),
                channel.maximum,
            )
        return JoystickAction(values)
