from __future__ import annotations

import json
from typing import Any

from mllm_teleop_benchmark.action.joystick import JoystickAction


class ActionParseError(ValueError):
    pass


class ActionParser:
    def parse(self, raw_action: Any) -> JoystickAction:
        if isinstance(raw_action, JoystickAction):
            return raw_action
        if isinstance(raw_action, str):
            try:
                raw_action = json.loads(raw_action)
            except json.JSONDecodeError as exc:
                raise ActionParseError(f"Invalid JSON action: {exc}") from exc
        if not isinstance(raw_action, dict):
            raise ActionParseError("Action must be a dict, JSON string, or JoystickAction")
        payload = raw_action.get("channels", raw_action)
        if not isinstance(payload, dict):
            raise ActionParseError("Action channels must be a mapping")
        try:
            channels = {str(name): float(value) for name, value in payload.items()}
        except (TypeError, ValueError) as exc:
            raise ActionParseError("Action channel values must be numeric") from exc
        return JoystickAction(channels)
