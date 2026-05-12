from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.action.space import ActionSpace
from mllm_teleop_benchmark.core.types import ValidationResult


class ActionValidator:
    def __init__(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def validate(self, action: JoystickAction) -> ValidationResult:
        errors: list[str] = []
        expected = set(self._action_space.channels)
        actual = set(action.channels)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f"Missing channels: {missing}")
        if extra:
            errors.append(f"Unknown channels: {extra}")

        for name, channel in self._action_space.channels.items():
            if name not in action.channels:
                continue
            value = action.channels[name]
            if channel.kind == "discrete":
                allowed = set(channel.values)
                if value not in allowed:
                    errors.append(f"{name} must be one of {sorted(allowed)}, got {value}")
            elif channel.kind == "continuous":
                if channel.minimum is None or channel.maximum is None:
                    errors.append(f"{name} continuous range is not configured")
                elif not channel.minimum <= value <= channel.maximum:
                    errors.append(f"{name} must be in [{channel.minimum}, {channel.maximum}], got {value}")
            else:
                errors.append(f"Unsupported channel kind for {name}: {channel.kind}")
        return ValidationResult(is_valid=not errors, errors=tuple(errors))
