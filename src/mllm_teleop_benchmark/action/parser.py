from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.core.types import MagnifierUpdate


class ActionParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedModelOutput:
    action: JoystickAction
    reflection: str | None = None
    magnifier_update: MagnifierUpdate = field(default_factory=MagnifierUpdate)


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
    ) -> ParsedModelOutput:
        decoded = self._decode_mapping(raw_action)
        if "channels" not in decoded:
            raise ActionParseError("Model output must contain a top-level 'channels' key")
        allowed_keys = {"channels", "Reflection", "magnifier"}
        extra_keys = sorted(str(key) for key in decoded if key not in allowed_keys)
        if extra_keys:
            raise ActionParseError(f"Unknown top-level model output keys: {extra_keys}")
        reflection = decoded.get("Reflection")
        if reflection is not None and not isinstance(reflection, str):
            raise ActionParseError("Model output Reflection must be a string")
        return ParsedModelOutput(
            action=self._parse_channels(decoded["channels"]),
            reflection=reflection,
            magnifier_update=self._parse_magnifier(decoded),
        )

    @staticmethod
    def _parse_magnifier(decoded: dict[str, Any]) -> MagnifierUpdate:
        if "magnifier" not in decoded:
            return MagnifierUpdate()
        payload = decoded["magnifier"]
        if payload is None:
            return MagnifierUpdate(requested=True, clear=True)
        if not isinstance(payload, dict):
            return MagnifierUpdate(
                requested=True,
                error="magnifier must be an object or null",
            )
        extra_keys = sorted(
            str(key)
            for key in payload
            if key not in {"source_rgb_key", "bbox_xyxy"}
        )
        if extra_keys:
            return MagnifierUpdate(
                requested=True,
                error=f"unknown magnifier keys: {extra_keys}",
            )
        source_rgb_key = payload.get("source_rgb_key")
        if not isinstance(source_rgb_key, str) or not source_rgb_key:
            return MagnifierUpdate(
                requested=True,
                error="magnifier.source_rgb_key must be a non-empty string",
            )
        bbox = payload.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return MagnifierUpdate(
                requested=True,
                error="magnifier.bbox_xyxy must contain four normalized numbers",
            )
        if any(isinstance(value, bool) for value in bbox):
            return MagnifierUpdate(
                requested=True,
                error="magnifier.bbox_xyxy values must be numeric",
            )
        try:
            values = tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            return MagnifierUpdate(
                requested=True,
                error="magnifier.bbox_xyxy values must be numeric",
            )
        x_min, y_min, x_max, y_max = values
        if not (
            all(math.isfinite(value) for value in values)
            and 0.0 <= x_min < x_max <= 1.0
            and 0.0 <= y_min < y_max <= 1.0
        ):
            return MagnifierUpdate(
                requested=True,
                error="magnifier.bbox_xyxy must satisfy 0 <= min < max <= 1",
            )
        return MagnifierUpdate(
            requested=True,
            source_rgb_key=source_rgb_key,
            bbox_xyxy=values,
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
