from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ControlInstruction:
    text: str
    panel_lines: tuple[str, ...]


class ControlInstructionBuilder:
    def build(self, interface: dict[str, Any], action_schema: dict[str, Any]) -> ControlInstruction:
        if interface.get("source") != "control_interface":
            text = f"You control the robot with {len(action_schema)} virtual joystick channels. Return a JSON action with exactly the listed channels."
            return ControlInstruction(text=text, panel_lines=("control source: legacy action_space/mapping", f"joysticks: {len(action_schema)}"))

        channels = list(interface.get("joystick_channels", ()))
        dimensions = list(interface.get("active_dimensions", ()))
        gripper = str(interface.get("gripper", "none"))
        mode = str(interface.get("mode", "unknown"))
        scale_mode = str(interface.get("scale_mode", "configured_by_controller"))
        visible = bool(interface.get("visible_metadata", True))
        mapping_complexity = bool(interface.get("mapping_complexity_present", False))
        frame_complexity = bool(interface.get("frame_complexity_present", False))
        mapping_summary = _mapping_summary(interface.get("mapping"))

        text_lines = [
            f"You control the robot with {len(channels)} virtual joystick channel(s).",
            f"The control mode is {mode}.",
            "Each motion joystick value must be one of -1, 0, or 1.",
        ]
        panel_lines = [
            f"control mode: {mode}",
            f"joysticks: {len(channels)}",
            f"scales: {scale_mode}",
            f"mapping complexity: {'present' if mapping_complexity else 'none'}",
            f"frame complexity: {'present' if frame_complexity else 'none'}",
        ]

        if dimensions and visible:
            text_lines.append(f"The active controllable dimensions are: {', '.join(str(v) for v in dimensions)}.")
            dimension_semantics = _dimension_semantics(dimensions)
            if dimension_semantics:
                text_lines.append(dimension_semantics)
            panel_lines.append(f"active dims: {', '.join(str(v) for v in dimensions)}")
            if mapping_summary:
                text_lines.append(f"The visible joystick mapping is: {mapping_summary}.")
                panel_lines.append(f"mapping: {mapping_summary}")
            else:
                text_lines.append("The exact joystick-to-dimension mapping is not disclosed.")
                panel_lines.append("mapping: hidden")
            if frame_complexity:
                text_lines.append("Frame complexity is present, so the exact physical-frame transform is not disclosed.")
        else:
            text_lines.append("The joystick-to-dimension mapping is not visible.")
            panel_lines.append("active dims: hidden")
            panel_lines.append("mapping: hidden")

        if gripper != "none":
            text_lines.append(f"A gripper channel is available with {gripper} control. The gripper is always the final virtual joystick.")
            panel_lines.append(f"gripper: {gripper}, final joystick")
        else:
            panel_lines.append("gripper: none")

        return ControlInstruction(text=" ".join(text_lines), panel_lines=tuple(panel_lines))


def _mapping_summary(mapping: object) -> str:
    if not isinstance(mapping, dict):
        return ""
    channels = mapping.get("channels")
    signs = mapping.get("signs", {})
    if not isinstance(channels, dict):
        return ""
    if not isinstance(signs, dict):
        signs = {}
    parts: list[str] = []
    for channel in sorted(channels):
        dim = str(channels[channel])
        sign = float(signs.get(channel, 1.0))
        target = f"-{dim}" if sign < 0.0 and dim != "gripper" else dim
        parts.append(f"{channel} -> {target}")
    return "; ".join(parts)


def _dimension_semantics(dimensions: list[object]) -> str:
    names = {str(value) for value in dimensions}
    statements: list[str] = []
    if names.intersection({"dx", "dy", "dz"}):
        statements.append(
            "dx, dy, and dz denote end-effector translation along the control "
            "frame's x, y, and z axes"
        )
    if names.intersection({"droll", "dpitch", "dyaw"}):
        statements.append(
            "droll, dpitch, and dyaw denote end-effector rotation about the "
            "control frame's x, y, and z axes"
        )
    if not statements:
        return ""
    return (
        "; ".join(statements)
        + ". A positive mapped value commands the positive direction, a negative "
        "value commands the opposite direction, and zero commands no change."
    )
