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
        control_frame_description = str(interface.get("control_frame_description", "")).strip()

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
            dimension_semantics = str(
                interface.get("dimension_semantics_description", "")
            ).strip()
            if not dimension_semantics:
                dimension_semantics = _dimension_semantics(
                    dimensions,
                    str(interface.get("rotation_representation", "")),
                )
            if dimension_semantics:
                text_lines.append(_sentence(dimension_semantics))
            if control_frame_description:
                text_lines.append(f"The physical controller frame is {control_frame_description}.")
                panel_lines.append(f"controller frame: {control_frame_description}")
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
                text_lines.append("No additional control-frame rotation is applied.")
            hidden_components = []
            if mapping_complexity:
                hidden_components.append("joystick mapping")
            if frame_complexity:
                hidden_components.append("control-frame transform")
            if hidden_components:
                noun = " and ".join(hidden_components)
                verb = "remain" if len(hidden_components) > 1 else "remains"
                text_lines.append(f"The hidden {noun} {verb} fixed for the entire episode.")
                text_lines.append(
                    "Infer the hidden controls from visual observations and your "
                    "own action history; no worked example or sampled hidden-control "
                    "value is provided."
                )
        else:
            text_lines.append("The joystick-to-dimension mapping is not visible.")
            panel_lines.append("active dims: hidden")
            panel_lines.append("mapping: hidden")

        if gripper != "none":
            text_lines.append(f"A gripper channel is available with {gripper} control. The gripper is always the final virtual joystick.")
            gripper_semantics = str(
                interface.get("gripper_semantics_description", "")
            ).strip()
            if not gripper_semantics:
                gripper_semantics = _gripper_semantics(
                    interface.get("gripper_commands")
                )
            if gripper_semantics:
                text_lines.append(_sentence(gripper_semantics))
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


def _sentence(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _dimension_semantics(
    dimensions: list[object],
    rotation_representation: str = "",
) -> str:
    names = {str(value) for value in dimensions}
    statements: list[str] = []
    translation = [
        (dimension, axis)
        for dimension, axis in (("dx", "x"), ("dy", "y"), ("dz", "z"))
        if dimension in names
    ]
    if translation:
        dimension_names = _join_terms([item[0] for item in translation])
        axes = _join_terms([item[1] for item in translation])
        verb = "denotes" if len(translation) == 1 else "denote"
        axis_noun = "axis" if len(translation) == 1 else "axes"
        statements.append(
            f"{dimension_names} {verb} end-effector translation along the "
            f"control frame's {axes} {axis_noun}"
        )
    rotation = [
        (dimension, axis)
        for dimension, axis in (
            ("droll", "x"),
            ("dpitch", "y"),
            ("dyaw", "z"),
        )
        if dimension in names
    ]
    if rotation:
        dimension_names = _join_terms([item[0] for item in rotation])
        axes = _join_terms([item[1] for item in rotation])
        verb = "denotes" if len(rotation) == 1 else "denote"
        if rotation_representation == "xyz_euler_parameters":
            statements.append(
                f"{dimension_names} {verb} increments of the target pose's "
                f"{axes} Euler parameters"
            )
        else:
            axis_noun = "axis" if len(rotation) == 1 else "axes"
            statements.append(
                f"{dimension_names} {verb} end-effector rotation about the "
                f"control frame's {axes} {axis_noun}"
            )
    if not statements:
        return ""
    return (
        "; ".join(statements)
        + ". A positive mapped value commands the positive direction, a negative "
        "value commands the opposite direction, and zero commands no change."
    )


def _join_terms(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _gripper_semantics(commands: object) -> str:
    if not isinstance(commands, dict):
        return ""
    parts = [
        f"{_format_number(value)} means {name}"
        for name, value in commands.items()
        if name in {"open", "closed", "hold", "neutral"}
    ]
    if not parts:
        return ""
    return "For the final gripper joystick, " + ", ".join(parts) + "."


def _format_number(value: object) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)
