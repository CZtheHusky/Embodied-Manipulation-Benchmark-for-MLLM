from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import ControllerConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import GymBoxAction
from mllm_teleop_benchmark.mapping.joystick_mapping import JoystickMapping


class GymnasiumBoxActionController(Controller):
    def __init__(self, mapping: JoystickMapping, dimensions: tuple[str, ...], scales: dict[str, float]) -> None:
        self.mapping = mapping
        self.dimensions = dimensions
        self.scales = dict(scales)

    @classmethod
    def from_config(cls, config: ControllerConfig, mapping: JoystickMapping) -> "GymnasiumBoxActionController":
        dimensions = tuple(str(v) for v in config.params.get("dimensions", ()))
        if not dimensions:
            raise ValueError("gymnasium_box controller requires params.dimensions")
        return cls(mapping=mapping, dimensions=dimensions, scales=config.scales)

    def translate(self, action: JoystickAction) -> GymBoxAction:
        command = self.mapping.map_action(action)
        values = tuple(command.get(dim) * self.scales.get(dim, 1.0) for dim in self.dimensions)
        return GymBoxAction(values=values, metadata={"controller": "gymnasium_box", "dimensions": self.dimensions})

    def metadata(self) -> dict[str, object]:
        return {"type": "gymnasium_box", "dimensions": self.dimensions, "scales": self.scales, "mapping": self.mapping.metadata()}


class ReacherDeltaEEFJoystickController(GymnasiumBoxActionController):
    @classmethod
    def from_config(cls, config: ControllerConfig, mapping: JoystickMapping) -> "ReacherDeltaEEFJoystickController":
        dimensions = tuple(str(v) for v in config.params.get("dimensions", ("dx", "dy")))
        return cls(mapping=mapping, dimensions=dimensions, scales=config.scales)

    def translate(self, action: JoystickAction) -> GymBoxAction:
        command = self.mapping.map_action(action)
        values = tuple(command.get(dim) * self.scales.get(dim, 1.0) for dim in self.dimensions)
        return GymBoxAction(
            values=values,
            metadata={
                "controller": "reacher_delta_eef",
                "dimensions": self.dimensions,
                "unused_dimensions": ("dz", "droll", "dpitch", "dyaw", "gripper"),
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "type": "reacher_delta_eef",
            "dimensions": self.dimensions,
            "scales": self.scales,
            "mapping": self.mapping.metadata(),
        }
