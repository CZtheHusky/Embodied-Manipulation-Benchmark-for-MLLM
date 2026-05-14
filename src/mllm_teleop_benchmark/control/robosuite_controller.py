from __future__ import annotations

import numpy as np

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import SimulatorAction
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping
from mllm_teleop_benchmark.utils.math import clamp


class RobosuiteOSCController(Controller):
    """
    Controller for robosuite's OSC_POSE (Operational Space Control) mode.
    Expects actions as [dx, dy, dz, droll, dpitch, dyaw, gripper].
    """

    def __init__(
        self,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform,
        scales: dict[str, float],
    ) -> None:
        self.mapping = mapping
        self.frame_transform = frame_transform
        self.scales = dict(scales)

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform,
    ) -> "RobosuiteOSCController":
        if config.type != "robosuite_osc":
            raise ValueError(f"Unsupported controller type: {config.type}")
        return cls(mapping=mapping, frame_transform=frame_transform, scales=config.scales)

    def translate(self, action: JoystickAction) -> SimulatorAction:
        # 1. Map virtual joysticks to control-frame dimensions (e.g. dx, dy)
        command = self.mapping.map_action(action)
        
        # 2. Scale the commands
        dx = command.get("dx", 0.0) * self.scales.get("dx", 0.05)
        dy = command.get("dy", 0.0) * self.scales.get("dy", 0.05)
        dz = command.get("dz", 0.0) * self.scales.get("dz", 0.05)
        
        droll = command.get("droll", 0.0) * self.scales.get("droll", 0.1)
        dpitch = command.get("dpitch", 0.0) * self.scales.get("dpitch", 0.1)
        dyaw = command.get("dyaw", 0.0) * self.scales.get("dyaw", 0.1)
        
        # 3. Apply coordinate frame transform (e.g. rotation)
        delta_pos = self.frame_transform.transform_delta_position((dx, dy, dz))
        delta_rot = self.frame_transform.transform_delta_rotation((droll, dpitch, dyaw))
        
        # 4. Handle gripper (robosuite gripper is usually 1 for closed, -1 for open, 
        # but the OSC controller often maps [0, 1] or [-1, 1] depending on config.
        # Standard robosuite behavior is 1.0 (closed) and -1.0 (open)).
        # Our benchmark uses [0, 1] for gripper state, we map it to robosuite's [-1, 1].
        gripper_raw = clamp(command.get("gripper", 0.0), 0.0, 1.0)
        robosuite_gripper = 2.0 * gripper_raw - 1.0
        
        # 5. Package as robosuite-compatible numpy array
        return np.array([
            delta_pos[0], delta_pos[1], delta_pos[2],
            delta_rot[0], delta_rot[1], delta_rot[2],
            robosuite_gripper
        ], dtype=np.float32)

    def metadata(self) -> dict[str, object]:
        return {
            "type": "robosuite_osc",
            "scales": dict(self.scales),
            "mapping": self.mapping.metadata(),
            "frame_transform": self.frame_transform.metadata(),
        }
