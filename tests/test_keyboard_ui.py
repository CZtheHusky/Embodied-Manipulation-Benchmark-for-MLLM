from mllm_teleop_benchmark.ui.keyboard import KeyboardJoystickMapper


def test_keyboard_mapper_generates_standard_joystick_action():
    mapper = KeyboardJoystickMapper(("joystick_0", "joystick_1"))
    action = mapper.action_from_pressed({"a", "w"})
    assert action["joystick_0"] == -1.0
    assert action["joystick_1"] == -1.0
    assert "joystick_6" not in action
    assert mapper.action_from_key("x") is None
    mapper.toggle_gripper()
    assert "joystick_6" not in mapper.action_from_pressed(set())
