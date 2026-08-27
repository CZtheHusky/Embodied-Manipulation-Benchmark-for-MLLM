from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import RLBenchAction
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping
from mllm_teleop_benchmark.utils.math import clamp


class RLBenchDeltaEEFController(Controller):
    def __init__(
        self,
        mapping: JoystickMapping,
        scales: dict[str, float],
        frame_transform: ControlFrameTransform | None = None,
        dimensions: tuple[str, ...] = ("dx", "dy", "dz", "droll", "dpitch", "dyaw"),
        default_gripper: float = 1.0,
    ) -> None:
        self.mapping = mapping
        self.scales = dict(scales)
        self.frame_transform = frame_transform
        self.dimensions = dimensions
        self.default_gripper = float(default_gripper)

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform | None = None,
    ) -> "RLBenchDeltaEEFController":
        dimensions = tuple(str(v) for v in config.params.get("dimensions", ("dx", "dy", "dz", "droll", "dpitch", "dyaw")))
        return cls(
            mapping=mapping,
            scales=config.scales,
            frame_transform=frame_transform,
            dimensions=dimensions,
            default_gripper=float(config.params.get("default_gripper", 1.0)),
        )

    def translate(self, action: JoystickAction) -> RLBenchAction:
        command = self.mapping.map_action(action)
        values_by_dim = {
            dim: command.get(dim) * self.scales.get(dim, 1.0)
            for dim in self.dimensions
            if dim != "gripper"
        }

        delta_position = (
            values_by_dim.get("dx", 0.0),
            values_by_dim.get("dy", 0.0),
            values_by_dim.get("dz", 0.0),
        )
        delta_rotation = (
            values_by_dim.get("droll", 0.0),
            values_by_dim.get("dpitch", 0.0),
            values_by_dim.get("dyaw", 0.0),
        )

        if self.frame_transform is not None:
            if any(dim in values_by_dim for dim in ("dx", "dy", "dz")):
                delta_position = self.frame_transform.transform_delta_position(delta_position)
            if any(dim in values_by_dim for dim in ("droll", "dpitch", "dyaw")):
                delta_rotation = self.frame_transform.transform_delta_rotation(delta_rotation)

        gripper = self.default_gripper
        if "gripper" in command.values:
            gripper = clamp(command.get("gripper", self.default_gripper), 0.0, 1.0)

        return RLBenchAction(
            delta_position=tuple(float(v) for v in delta_position),
            delta_rotation_rpy=tuple(float(v) for v in delta_rotation),
            gripper=float(gripper),
            metadata={
                "controller": "rlbench_delta_eef",
                "dimensions": self.dimensions,
                "frame_transform": self.frame_transform.metadata() if self.frame_transform else None,
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "rlbench_delta_eef",
            "dimensions": self.dimensions,
            "scales": dict(self.scales),
            "mapping": self.mapping.metadata(),
            "frame_transform": self.frame_transform.metadata() if self.frame_transform else None,
            "default_gripper": self.default_gripper,
        }
