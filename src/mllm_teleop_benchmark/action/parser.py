from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from mllm_teleop_benchmark.action.joystick import JoystickAction


class ActionParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedModelOutput:
    action: JoystickAction
    reflection: str | None = None


class ActionParser:
    def parse(self, raw_action: Any) -> JoystickAction:
        if isinstance(raw_action, JoystickAction):
            return raw_action
        decoded = self._decode_mapping(raw_action)
        payload = decoded.get("channels", decoded)
        return self._parse_channels(payload)

    def parse_model_output(
        self,
        raw_action: Any,
        output_mode: str,
    ) -> ParsedModelOutput:
        decoded = self._decode_mapping(raw_action)
        if "channels" not in decoded:
            raise ActionParseError("Model output must contain a top-level 'channels' key")
        if output_mode == "action_only":
            allowed_keys = {"channels"}
        elif output_mode == "action_with_reflection":
            allowed_keys = {"channels", "Reflection"}
        else:
            raise ActionParseError(f"Unsupported model action output mode: {output_mode}")
        extra_keys = sorted(str(key) for key in decoded if key not in allowed_keys)
        if extra_keys:
            raise ActionParseError(f"Unknown top-level model output keys: {extra_keys}")
        reflection = decoded.get("Reflection")
        if reflection is not None and not isinstance(reflection, str):
            raise ActionParseError("Model output Reflection must be a string")
        return ParsedModelOutput(
            action=self._parse_channels(decoded["channels"]),
            reflection=reflection,
        )

    @staticmethod
    def _decode_mapping(raw_action: Any) -> dict[str, Any]:
        if isinstance(raw_action, str):
            try:
                raw_action = json.loads(raw_action)
            except json.JSONDecodeError as exc:
                raise ActionParseError(f"Invalid JSON action: {exc}") from exc
        if not isinstance(raw_action, dict):
            raise ActionParseError("Action must be a dict, JSON string, or JoystickAction")
        return raw_action

    @staticmethod
    def _parse_channels(payload: Any) -> JoystickAction:
        if not isinstance(payload, dict):
            raise ActionParseError("Action channels must be a mapping")
        try:
            channels = {str(name): float(value) for name, value in payload.items()}
        except (TypeError, ValueError) as exc:
            raise ActionParseError("Action channel values must be numeric") from exc
        return JoystickAction(channels)
