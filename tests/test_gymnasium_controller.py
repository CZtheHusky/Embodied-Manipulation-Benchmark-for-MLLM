from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.gymnasium_box import ReacherDeltaEEFJoystickController
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping


def test_reacher_delta_eef_controller_maps_xy_only():
    config = load_config("configs/reacher_delta_eef.yaml")
    controller = ReacherDeltaEEFJoystickController.from_config(config.controller, FixedJoystickMapping.from_config(config.mapping))
    action = JoystickAction(
        {
            "joystick_0": 1,
            "joystick_1": -1,
            "joystick_2": 1,
            "joystick_3": 1,
            "joystick_4": 1,
            "joystick_5": 1,
            "joystick_6": 0.5,
        }
    )
    sim_action = controller.translate(action)
    assert sim_action.values == (1.0, -1.0)
    assert sim_action.metadata["controller"] == "reacher_delta_eef"
