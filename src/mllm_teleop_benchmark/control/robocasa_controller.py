from __future__ import annotations

import numpy as np

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import RoboCasaAction
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping
from mllm_teleop_benchmark.utils.math import clamp


class RoboCasaOSCController(Controller):
    """Controller for RoboCasa's Gym Dict action wrapper."""

    def __init__(
        self,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform,
        scales: dict[str, float],
        control_mode: float = 1.0,
    ) -> None:
        self.mapping = mapping
        self.frame_transform = frame_transform
        self.scales = dict(scales)
        self.control_mode = float(control_mode)

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform,
    ) -> "RoboCasaOSCController":
        return cls(
            mapping=mapping,
            frame_transform=frame_transform,
            scales=config.scales,
            control_mode=float(config.params.get("control_mode", 1.0)),
        )

    def translate(self, action: JoystickAction) -> RoboCasaAction:
        command = self.mapping.map_action(action)

        dx = command.get("dx", 0.0) * self.scales.get("dx", 0.05)
        dy = command.get("dy", 0.0) * self.scales.get("dy", 0.05)
        dz = command.get("dz", 0.0) * self.scales.get("dz", 0.05)

        droll = command.get("droll", 0.0) * self.scales.get("droll", 0.1)
        dpitch = command.get("dpitch", 0.0) * self.scales.get("dpitch", 0.1)
        dyaw = command.get("dyaw", 0.0) * self.scales.get("dyaw", 0.1)

        delta_pos = _fit_normalized_vector(
            self.frame_transform.transform_delta_position((dx, dy, dz))
        )
        delta_rot = _fit_normalized_vector(
            self.frame_transform.transform_delta_rotation((droll, dpitch, dyaw))
        )
        gripper_close = clamp(command.get("gripper", 0.0), 0.0, 1.0)

        values = {
            "action.end_effector_position": np.asarray(delta_pos, dtype=np.float32),
            "action.end_effector_rotation": np.asarray(delta_rot, dtype=np.float32),
            "action.gripper_close": np.asarray([gripper_close], dtype=np.float32),
            "action.base_motion": np.zeros(4, dtype=np.float32),
            "action.control_mode": np.asarray([self.control_mode], dtype=np.float32),
        }
        return RoboCasaAction(
            values=values,
            metadata={
                "controller": "robocasa_osc",
                "dimensions": (
                    "dx",
                    "dy",
                    "dz",
                    "droll",
                    "dpitch",
                    "dyaw",
                    "gripper",
                    "base_motion",
                    "control_mode",
                ),
                "frame_transform": self.frame_transform.metadata(),
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "robocasa_osc",
            "scales": dict(self.scales),
            "control_mode": self.control_mode,
            "mapping": self.mapping.metadata(),
            "frame_transform": self.frame_transform.metadata(),
        }


def _fit_normalized_vector(
    values: tuple[float, float, float],
) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    max_abs = float(np.max(np.abs(vector)))
    if max_abs > 1.0:
        vector = vector / max_abs
    return tuple(float(value) for value in vector)
