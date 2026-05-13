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
        mapping = interface.get("mapping")
        signs = interface.get("mapping_signs", {})
        gripper = str(interface.get("gripper", "none"))
        mode = str(interface.get("mode", "unknown"))
        strategy = str(interface.get("mapping_strategy", "fixed"))
        frame = str(interface.get("frame_transform", "identity"))
        frame_metadata = interface.get("frame_transform_metadata", {})
        scale_mode = str(interface.get("scale_mode", "configured_by_controller"))
        visible = bool(interface.get("visible_metadata", True))

        text_lines = [
            f"You control the robot with {len(channels)} virtual joystick channel(s).",
            f"The control mode is {mode}.",
            "Each motion joystick value must be one of -1, 0, or 1.",
        ]
        panel_lines = [
            f"control mode: {mode}",
            f"joysticks: {len(channels)}",
            f"scales: {scale_mode}",
            f"mapping: {strategy} ({'visible' if visible and mapping else 'hidden'})",
            f"frame: {frame}{' (visible)' if isinstance(frame_metadata, dict) and frame_metadata.get('visible_to_agent') else ' (hidden)' if isinstance(frame_metadata, dict) and frame_metadata.get('visible_to_agent') is False else ''}",
        ]

        if mapping:
            pairs = []
            for channel, dim in dict(mapping).items():
                sign = float(dict(signs).get(channel, 1.0))
                prefix = "-" if sign < 0 else ""
                pairs.append(f"{channel} -> {prefix}{dim}")
            mapping_text = ", ".join(pairs)
            text_lines.append(f"Visible joystick mapping: {mapping_text}.")
            panel_lines.append(f"axes: {mapping_text}")
        elif dimensions:
            text_lines.append(f"The active controllable dimensions are: {', '.join(str(v) for v in dimensions)}.")
            panel_lines.append(f"axes: {', '.join(str(v) for v in dimensions)}")
        else:
            text_lines.append("The joystick-to-dimension mapping is not visible.")
            panel_lines.append("axes: hidden")

        if gripper != "none":
            text_lines.append(f"A gripper channel is available with {gripper} control. The gripper is always the final virtual joystick and is not permuted or sign-flipped.")
            panel_lines.append(f"gripper: {gripper}, final joystick")
        else:
            panel_lines.append("gripper: none")

        return ControlInstruction(text=" ".join(text_lines), panel_lines=tuple(panel_lines))
