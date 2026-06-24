from mllm_teleop_benchmark.action.parser import ActionParser
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.core.registry import build_action_space
from mllm_teleop_benchmark.ui.human_player import PygameHumanPlayer
from mllm_teleop_benchmark.ui.keyboard import KeyboardJoystickMapper
from mllm_teleop_benchmark.ui.runtime import SimAdapterWorkerRuntime


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


def test_pygame_backend_uses_control_interface_gripper_channel():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    app = PygameHumanPlayer(config, SimAdapterWorkerRuntime(config))

    assert app._gripper_channel() == "joystick_6"


def test_pygame_backend_invalid_action_recovers_to_noop():
    config = load_config("configs/reacher_delta_eef.yaml")
    app = PygameHumanPlayer(config, SimAdapterWorkerRuntime(config))
    action_space = build_action_space(config)

    parsed, validation = app._parse_and_validate(
        ActionParser(),
        ActionValidator(action_space),
        action_space,
        {"joystick_0": 1.0},
    )

    assert not validation.is_valid
    assert parsed.as_dict() == {"joystick_0": 0.0, "joystick_1": 0.0}
