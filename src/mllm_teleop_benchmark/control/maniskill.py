from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import ManiSkillAction
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping


class ManiSkillDeltaEEFController(Controller):
    def __init__(
        self,
        mapping: JoystickMapping,
        dimensions: tuple[str, ...],
        scales: dict[str, float],
        frame_transform: ControlFrameTransform | None = None,
        gripper_open_value: float = -1.0,
        gripper_closed_value: float = 1.0,
        rotation_input_sign: float = 1.0,
    ) -> None:
        self.mapping = mapping
        self.dimensions = dimensions
        self.scales = dict(scales)
        self.frame_transform = frame_transform
        self.gripper_open_value = float(gripper_open_value)
        self.gripper_closed_value = float(gripper_closed_value)
        self.rotation_input_sign = float(rotation_input_sign)
        if self.rotation_input_sign not in {-1.0, 1.0}:
            raise ValueError("rotation_input_sign must be -1 or 1")

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform | None = None,
    ) -> "ManiSkillDeltaEEFController":
        dimensions = tuple(str(v) for v in config.params.get("dimensions", ("dx", "dy", "dz", "droll", "dpitch", "dyaw")))
        if not dimensions:
            raise ValueError("maniskill_delta_eef controller requires at least one dimension")
        return cls(
            mapping=mapping,
            dimensions=dimensions,
            scales=config.scales,
            frame_transform=frame_transform,
            gripper_open_value=float(config.params.get("gripper_open_value", -1.0)),
            gripper_closed_value=float(config.params.get("gripper_closed_value", 1.0)),
            rotation_input_sign=float(
                config.params.get("rotation_input_sign", 1.0)
            ),
        )

    def translate(self, action: JoystickAction) -> ManiSkillAction:
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

        if self.frame_transform is not None and any(dim in values_by_dim for dim in ("droll", "dpitch", "dyaw")):
            transformed = self.frame_transform.transform_delta_rotation(
                (
                    values_by_dim.get("droll", 0.0),
                    values_by_dim.get("dpitch", 0.0),
                    values_by_dim.get("dyaw", 0.0),
                )
            )
            for dim, value in zip(("droll", "dpitch", "dyaw"), transformed, strict=True):
                if dim in values_by_dim:
                    values_by_dim[dim] = value

        for dim in ("droll", "dpitch", "dyaw"):
            if dim in values_by_dim:
                values_by_dim[dim] *= self.rotation_input_sign

        values = [values_by_dim.get(dim, 0.0) for dim in self.dimensions if dim != "gripper"]
        if "gripper" in self.dimensions:
            values.append(self._map_gripper(command.get("gripper", 0.0)))

        return ManiSkillAction(
            values=tuple(float(v) for v in values),
            metadata={
                "controller": "maniskill_delta_eef",
                "dimensions": self.dimensions,
                "frame_transform": self.frame_transform.metadata() if self.frame_transform else None,
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "maniskill_delta_eef",
            "dimensions": self.dimensions,
            "scales": self.scales,
            "mapping": self.mapping.metadata(),
            "frame_transform": self.frame_transform.metadata() if self.frame_transform else None,
            "gripper_open_value": self.gripper_open_value,
            "gripper_closed_value": self.gripper_closed_value,
            "rotation_input_sign": self.rotation_input_sign,
        }

    def _map_gripper(self, value: float) -> float:
        clipped = min(max(float(value), 0.0), 1.0)
        return self.gripper_open_value + clipped * (self.gripper_closed_value - self.gripper_open_value)
