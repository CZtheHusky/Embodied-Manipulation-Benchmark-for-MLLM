from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import LanguageTableAction
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping


class LanguageTablePlanarController(Controller):
    def __init__(
        self,
        mapping: JoystickMapping,
        scales: dict[str, float],
        frame_transform: ControlFrameTransform,
    ) -> None:
        self.mapping = mapping
        self.scales = dict(scales)
        self.frame_transform = frame_transform

    @classmethod
    def from_config(
        cls,
        config: ControllerConfig,
        mapping: JoystickMapping,
        frame_transform: ControlFrameTransform,
    ) -> "LanguageTablePlanarController":
        return cls(mapping, config.scales, frame_transform)

    def translate(self, action: JoystickAction) -> LanguageTableAction:
        command = self.mapping.map_action(action)
        delta = self.frame_transform.transform_delta_position(
            (
                command.get("dx") * self.scales.get("dx", 1.0),
                command.get("dy") * self.scales.get("dy", 1.0),
                0.0,
            )
        )
        return LanguageTableAction(
            delta_xy=(float(delta[0]), float(delta[1])),
            metadata={
                "controller": "language_table_planar",
                "dimensions": ("dx", "dy"),
                "frame_transform": self.frame_transform.metadata(),
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "language_table_planar",
            "dimensions": ("dx", "dy"),
            "scales": self.scales,
            "mapping": self.mapping.metadata(),
            "frame_transform": self.frame_transform.metadata(),
        }
