from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import DMControlAction
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping


class DMControlController(Controller):
    def __init__(
        self,
        mapping: JoystickMapping,
        dimensions: tuple[str, ...],
        scales: dict[str, float],
        frame_transform: ControlFrameTransform | None = None,
    ) -> None:
        self.mapping = mapping
        self.dimensions = dimensions
        self.scales = dict(scales)
        self.frame_transform = frame_transform

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform | None = None,
    ) -> "DMControlController":
        dimensions = tuple(str(value) for value in config.params.get("dimensions", ()))
        if not dimensions:
            raise ValueError("dmcontrol controller requires params.dimensions")
        return cls(mapping=mapping, dimensions=dimensions, scales=config.scales, frame_transform=frame_transform)

    def translate(self, action: JoystickAction) -> DMControlAction:
        command = self.mapping.map_action(action)
        values_by_dim = {dim: command.get(dim) * self.scales.get(dim, 1.0) for dim in self.dimensions}

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

        return DMControlAction(
            values=tuple(float(values_by_dim.get(dim, 0.0)) for dim in self.dimensions),
            metadata={
                "controller": "dmcontrol",
                "dimensions": self.dimensions,
                "frame_transform": self.frame_transform.metadata() if self.frame_transform else None,
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "dmcontrol",
            "dimensions": self.dimensions,
            "scales": self.scales,
            "mapping": self.mapping.metadata(),
            "frame_transform": self.frame_transform.metadata() if self.frame_transform else None,
        }
