from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JoystickAction:
    channels: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        return dict(self.channels)
