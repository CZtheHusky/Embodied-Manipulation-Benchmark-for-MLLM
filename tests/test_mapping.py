from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.core.registry import build_mapping


def test_fixed_mapping_correctness():
    config = load_config("configs/dummy_reach.yaml")
    mapping = build_mapping(config, config.benchmark.seed)
    command = mapping.map_action(JoystickAction({"joystick_0": 1, "joystick_1": -1, "joystick_2": 0, "joystick_3": 0, "joystick_4": 0, "joystick_5": 0, "joystick_6": 0.25}))
    assert command.get("dx") == 1
    assert command.get("dy") == -1
    assert command.get("gripper") == 0.25
