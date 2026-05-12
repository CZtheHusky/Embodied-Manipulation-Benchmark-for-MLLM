from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.ee_delta_pose import EEDeltaPoseController
from mllm_teleop_benchmark.mapping.frame_transform import IdentityControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping


def test_scale_application_and_direct_gripper():
    config = load_config("configs/dummy_reach.yaml")
    controller = EEDeltaPoseController.from_config(
        config.controller,
        FixedJoystickMapping.from_config(config.mapping),
        IdentityControlFrameTransform.from_config(config.frame_transform),
    )
    sim_action = controller.translate(
        JoystickAction(
            {
                "joystick_0": 1,
                "joystick_1": -1,
                "joystick_2": 0,
                "joystick_3": 1,
                "joystick_4": 0,
                "joystick_5": -1,
                "joystick_6": 0.7,
            }
        )
    )
    assert sim_action.delta_position == (0.02, -0.02, 0.0)
    assert sim_action.delta_rotation_rpy == (0.05, 0.0, -0.05)
    assert sim_action.gripper == 0.7
