from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import CalvinAction
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping
from mllm_teleop_benchmark.utils.math import fit_vector_to_component_limit


class CalvinDeltaEEFController(Controller):
    def __init__(
        self,
        mapping: JoystickMapping,
        dimensions: tuple[str, ...],
        scales: dict[str, float],
        frame_transform: ControlFrameTransform | None = None,
        gripper_open_value: float = 1.0,
        gripper_closed_value: float = -1.0,
        transform_rotation: bool = True,
    ) -> None:
        self.mapping = mapping
        self.dimensions = dimensions
        self.scales = dict(scales)
        self.frame_transform = frame_transform
        self.gripper_open_value = float(gripper_open_value)
        self.gripper_closed_value = float(gripper_closed_value)
        self.transform_rotation = bool(transform_rotation)

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform | None = None,
    ) -> "CalvinDeltaEEFController":
        dimensions = tuple(str(v) for v in config.params.get("dimensions", ("dx", "dy", "dz", "droll", "dpitch", "dyaw")))
        if not dimensions:
            raise ValueError("calvin_delta_eef controller requires at least one dimension")
        return cls(
            mapping=mapping,
            dimensions=dimensions,
            scales=config.scales,
            frame_transform=frame_transform,
            gripper_open_value=float(config.params.get("gripper_open_value", 1.0)),
            gripper_closed_value=float(config.params.get("gripper_closed_value", -1.0)),
            transform_rotation=bool(config.params.get("transform_rotation", True)),
        )

    def translate(self, action: JoystickAction) -> CalvinAction:
        command = self.mapping.map_action(action)
        values_by_dim = {dim: command.get(dim) * self.scales.get(dim, 1.0) for dim in self.dimensions if dim != "gripper"}

        if self.frame_transform is not None and any(dim in values_by_dim for dim in ("dx", "dy", "dz")):
            transformed = fit_vector_to_component_limit(
                self.frame_transform.transform_delta_position(
                    (
                        values_by_dim.get("dx", 0.0),
                        values_by_dim.get("dy", 0.0),
                        values_by_dim.get("dz", 0.0),
                    )
                )
            )
            for dim, value in zip(("dx", "dy", "dz"), transformed, strict=True):
                if dim in values_by_dim:
                    values_by_dim[dim] = value

        if (
            self.transform_rotation
            and self.frame_transform is not None
            and any(dim in values_by_dim for dim in ("droll", "dpitch", "dyaw"))
        ):
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

        ordered_dims = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
        values = [values_by_dim.get(dim, 0.0) for dim in ordered_dims]
        if "gripper" in self.dimensions:
            values.append(self._map_gripper(command.get("gripper", 0.0)))
        else:
            values.append(self.gripper_open_value)

        return CalvinAction(
            values=tuple(float(v) for v in values),
            metadata={
                "controller": "calvin_delta_eef",
                "dimensions": self.dimensions,
                "frame_transform": self.frame_transform.metadata() if self.frame_transform else None,
                "transform_rotation": self.transform_rotation,
                "translation_bound_policy": "uniform_component_fit",
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "calvin_delta_eef",
            "dimensions": self.dimensions,
            "scales": self.scales,
            "mapping": self.mapping.metadata(),
            "frame_transform": self.frame_transform.metadata() if self.frame_transform else None,
            "transform_rotation": self.transform_rotation,
            "translation_bound_policy": "uniform_component_fit",
            "gripper_open_value": self.gripper_open_value,
            "gripper_closed_value": self.gripper_closed_value,
        }

    def _map_gripper(self, value: float) -> float:
        clipped = min(max(float(value), 0.0), 1.0)
        return self.gripper_open_value if clipped >= 0.5 else self.gripper_closed_value
