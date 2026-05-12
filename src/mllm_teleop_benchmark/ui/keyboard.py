from __future__ import annotations


DEFAULT_KEY_BINDINGS = {
    "w": ("joystick_0", -1.0),
    "s": ("joystick_0", 1.0),
    "a": ("joystick_1", -1.0),
    "d": ("joystick_1", 1.0),
    "q": ("joystick_2", -1.0),
    "e": ("joystick_2", 1.0),
    "j": ("joystick_3", -1.0),
    "l": ("joystick_3", 1.0),
    "k": ("joystick_4", -1.0),
    "i": ("joystick_4", 1.0),
    "u": ("joystick_5", -1.0),
    "o": ("joystick_5", 1.0),
}


class KeyboardJoystickMapper:
    def __init__(self, channels: tuple[str, ...], key_bindings: dict[str, tuple[str, float]] | None = None) -> None:
        self.channels = channels
        self.key_bindings = dict(key_bindings or DEFAULT_KEY_BINDINGS)
        self.gripper_closed = False

    def toggle_gripper(self) -> None:
        self.gripper_closed = not self.gripper_closed

    def action_from_pressed(self, pressed_keys: set[str]) -> dict[str, float]:
        action = {channel: 0.0 for channel in self.channels}
        for key, (channel, value) in self.key_bindings.items():
            if key in pressed_keys and channel in action:
                action[channel] = value
        if "joystick_6" in action:
            action["joystick_6"] = 1.0 if self.gripper_closed else 0.0
        return action

    def action_from_key(self, key_name: str) -> dict[str, float] | None:
        action = {channel: 0.0 for channel in self.channels}
        diagonal = {
            "q": {"joystick_0": -0.707, "joystick_1": -0.707},
            "e": {"joystick_0": -0.707, "joystick_1": 0.707},
            "z": {"joystick_0": 0.707, "joystick_1": -0.707},
            "x": {"joystick_0": 0.707, "joystick_1": 0.707},
        }
        if key_name in diagonal:
            for channel, value in diagonal[key_name].items():
                if channel in action:
                    action[channel] = value
        elif key_name in self.key_bindings:
            channel, value = self.key_bindings[key_name]
            if channel in action:
                action[channel] = value
        else:
            return None
        if "joystick_6" in action:
            action["joystick_6"] = 1.0 if self.gripper_closed else 0.0
        return action
