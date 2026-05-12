from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import SimAction


class JointDeltaController(Controller):
    """Placeholder for future joint-space control modes."""

    def translate(self, action: JoystickAction) -> SimAction:
        raise NotImplementedError("JointDeltaController is reserved for a later benchmark version")

    def metadata(self) -> dict[str, object]:
        return {"type": "joint_delta", "implemented": False}
