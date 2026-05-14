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


def test_keyboard_mapper_toggles_configured_final_gripper_channel():
    mapper = KeyboardJoystickMapper(("joystick_0", "joystick_1", "joystick_2"), gripper_channel="joystick_2")

    assert mapper.action_from_pressed({"q"}) == {"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0}
    assert mapper.action_from_key("q") is None

    closed = mapper.action_from_key("g")
    assert closed == {"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 1.0}

    motion_with_closed_gripper = mapper.action_from_key("w")
    assert motion_with_closed_gripper == {"joystick_0": -1.0, "joystick_1": 0.0, "joystick_2": 1.0}

    opened = mapper.action_from_key("space")
    assert opened == {"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0}
