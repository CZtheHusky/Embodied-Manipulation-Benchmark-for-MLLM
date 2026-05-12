from mllm_teleop_benchmark.ui.keyboard import KeyboardJoystickMapper


def test_keyboard_mapper_generates_standard_joystick_action():
    mapper = KeyboardJoystickMapper(tuple(f"joystick_{idx}" for idx in range(7)))
    action = mapper.action_from_pressed({"a", "w"})
    assert action["joystick_0"] == -1.0
    assert action["joystick_1"] == -1.0
    assert action["joystick_6"] == 0.0
    diagonal = mapper.action_from_key("x")
    assert diagonal is not None
    assert diagonal["joystick_0"] == 0.707
    assert diagonal["joystick_1"] == 0.707
    mapper.toggle_gripper()
    assert mapper.action_from_pressed(set())["joystick_6"] == 1.0
