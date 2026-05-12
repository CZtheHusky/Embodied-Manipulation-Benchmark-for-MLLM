from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import SimAction
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping
from mllm_teleop_benchmark.utils.math import clamp


class EEDeltaPoseController(Controller):
    def __init__(
        self,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform,
        scales: dict[str, float],
        frame: str = "base",
    ) -> None:
        self.mapping = mapping
        self.frame_transform = frame_transform
        self.scales = dict(scales)
        self.frame = frame

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform,
    ) -> "EEDeltaPoseController":
        if config.type != "ee_delta_pose":
            raise ValueError(f"Unsupported controller type: {config.type}")
        return cls(mapping=mapping, frame_transform=frame_transform, scales=config.scales, frame=config.frame)

    def translate(self, action: JoystickAction) -> SimAction:
        command = self.mapping.map_action(action)
        delta_position = (
            command.get("dx") * self.scales.get("dx", 1.0),
            command.get("dy") * self.scales.get("dy", 1.0),
            command.get("dz") * self.scales.get("dz", 1.0),
        )
        delta_rotation = (
            command.get("droll") * self.scales.get("droll", 1.0),
            command.get("dpitch") * self.scales.get("dpitch", 1.0),
            command.get("dyaw") * self.scales.get("dyaw", 1.0),
        )
        return SimAction(
            delta_position=self.frame_transform.transform_delta_position(delta_position),
            delta_rotation_rpy=self.frame_transform.transform_delta_rotation(delta_rotation),
            gripper=clamp(command.get("gripper", 0.0), 0.0, 1.0),
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "ee_delta_pose",
            "frame": self.frame,
            "scales": dict(self.scales),
            "mapping": self.mapping.metadata(),
            "frame_transform": self.frame_transform.metadata(),
        }
