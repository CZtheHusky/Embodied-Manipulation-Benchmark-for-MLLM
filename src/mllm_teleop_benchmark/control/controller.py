from __future__ import annotations

from abc import ABC, abstractmethod

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.core.types import SimAction


class Controller(ABC):
    @abstractmethod
    def translate(self, action: JoystickAction) -> SimAction:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError
