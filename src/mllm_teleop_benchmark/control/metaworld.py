from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import MetaworldAction
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping


class MetaworldBoxActionController(Controller):
    def __init__(
        self,
        mapping: JoystickMapping,
        dimensions: tuple[str, ...],
        scales: dict[str, float],
        frame_transform: ControlFrameTransform | None = None,
        gripper_open_value: float = -1.0,
        gripper_closed_value: float = 1.0,
        default_gripper_value: float = 1.0,
    ) -> None:
        self.mapping = mapping
        self.dimensions = dimensions
        self.scales = dict(scales)
        self.frame_transform = frame_transform
        self.gripper_open_value = float(gripper_open_value)
        self.gripper_closed_value = float(gripper_closed_value)
        self.default_gripper_value = float(default_gripper_value)

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform | None = None,
    ) -> "MetaworldBoxActionController":
        dimensions = tuple(str(value) for value in config.params.get("dimensions", ("dx", "dy", "dz", "gripper")))
        return cls(
            mapping=mapping,
            dimensions=dimensions,
            scales=config.scales,
            frame_transform=frame_transform,
            gripper_open_value=float(config.params.get("gripper_open_value", -1.0)),
            gripper_closed_value=float(config.params.get("gripper_closed_value", 1.0)),
            default_gripper_value=float(config.params.get("default_gripper_value", 1.0)),
        )

    def translate(self, action: JoystickAction) -> MetaworldAction:
        command = self.mapping.map_action(action)
        values_by_dim = {dim: command.get(dim) * self.scales.get(dim, 1.0) for dim in self.dimensions if dim != "gripper"}

        if self.frame_transform is not None and any(dim in values_by_dim for dim in ("dx", "dy", "dz")):
            transformed = self.frame_transform.transform_delta_position(
                (
                    values_by_dim.get("dx", 0.0),
                    values_by_dim.get("dy", 0.0),
                    values_by_dim.get("dz", 0.0),
                )
            )
            for dim, value in zip(("dx", "dy", "dz"), transformed, strict=True):
                if dim in values_by_dim:
                    values_by_dim[dim] = value

        values = [values_by_dim.get(dim, 0.0) for dim in self.dimensions if dim != "gripper"]
        if "gripper" in self.dimensions:
            values.append(self._gripper_value(command.values))

        return MetaworldAction(
            values=tuple(values),
            metadata={
                "controller": "metaworld_box",
                "dimensions": self.dimensions,
                "frame_transform": self.frame_transform.metadata() if self.frame_transform else None,
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "metaworld_box",
            "dimensions": self.dimensions,
            "scales": self.scales,
            "mapping": self.mapping.metadata(),
            "frame_transform": self.frame_transform.metadata() if self.frame_transform else None,
            "gripper_open_value": self.gripper_open_value,
            "gripper_closed_value": self.gripper_closed_value,
            "default_gripper_value": self.default_gripper_value,
        }

    def _gripper_value(self, command_values: dict[str, float]) -> float:
        if "gripper" not in command_values:
            return self.default_gripper_value
        clipped = min(max(float(command_values["gripper"]), 0.0), 1.0)
        return self.gripper_open_value + clipped * (self.gripper_closed_value - self.gripper_open_value)
