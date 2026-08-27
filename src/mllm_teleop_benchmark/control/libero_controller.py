from __future__ import annotations

import numpy as np

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import LiberoAction
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping
from mllm_teleop_benchmark.utils.math import clamp


class LiberoDeltaEEFController(Controller):
    """Maps benchmark joystick commands to LIBERO's 7D delta pose action."""

    ordered_dims = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")

    def __init__(
        self,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform,
        scales: dict[str, float],
        gripper_open_value: float = -1.0,
        gripper_closed_value: float = 1.0,
    ) -> None:
        self.mapping = mapping
        self.frame_transform = frame_transform
        self.scales = dict(scales)
        self.gripper_open_value = float(gripper_open_value)
        self.gripper_closed_value = float(gripper_closed_value)

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform,
    ) -> "LiberoDeltaEEFController":
        return cls(
            mapping=mapping,
            frame_transform=frame_transform,
            scales=config.scales,
            gripper_open_value=float(config.params.get("gripper_open_value", -1.0)),
            gripper_closed_value=float(config.params.get("gripper_closed_value", 1.0)),
        )

    def translate(self, action: JoystickAction) -> LiberoAction:
        command = self.mapping.map_action(action)

        delta_pos = _fit_normalized_vector(
            self.frame_transform.transform_delta_position(
                (
                    command.get("dx", 0.0) * self.scales.get("dx", 0.05),
                    command.get("dy", 0.0) * self.scales.get("dy", 0.05),
                    command.get("dz", 0.0) * self.scales.get("dz", 0.05),
                )
            )
        )
        delta_rot = _fit_normalized_vector(
            self.frame_transform.transform_delta_rotation(
                (
                    command.get("droll", 0.0) * self.scales.get("droll", 0.1),
                    command.get("dpitch", 0.0) * self.scales.get("dpitch", 0.1),
                    command.get("dyaw", 0.0) * self.scales.get("dyaw", 0.1),
                )
            )
        )
        gripper = clamp(command.get("gripper", 0.0), 0.0, 1.0)
        gripper = self.gripper_open_value + gripper * (
            self.gripper_closed_value - self.gripper_open_value
        )
        values = np.asarray((*delta_pos, *delta_rot, gripper), dtype=np.float32)
        return LiberoAction(
            values=tuple(float(value) for value in values),
            metadata={
                "controller": "libero_delta_eef",
                "dimensions": (*self.ordered_dims, "gripper"),
                "frame_transform": self.frame_transform.metadata(),
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "libero_delta_eef",
            "dimensions": (*self.ordered_dims, "gripper"),
            "scales": dict(self.scales),
            "mapping": self.mapping.metadata(),
            "frame_transform": self.frame_transform.metadata(),
            "gripper_open_value": self.gripper_open_value,
            "gripper_closed_value": self.gripper_closed_value,
        }

def _fit_normalized_vector(
    values: tuple[float, float, float],
) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    max_abs = float(np.max(np.abs(vector)))
    if max_abs > 1.0:
        vector = vector / max_abs
    return tuple(float(value) for value in vector)
