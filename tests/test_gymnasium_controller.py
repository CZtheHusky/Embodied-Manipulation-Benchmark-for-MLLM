from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import FrameTransformConfig
from mllm_teleop_benchmark.control.gymnasium_box import PusherDeltaEEFJoystickController, ReacherDeltaEEFJoystickController
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping


def test_reacher_delta_eef_controller_maps_xy_only():
    config = load_config("configs/reacher_delta_eef.yaml")
    controller = ReacherDeltaEEFJoystickController.from_config(config.controller, FixedJoystickMapping.from_config(config.mapping))
    action = JoystickAction(
        {
            "joystick_0": 1,
            "joystick_1": -1,
        }
    )
    sim_action = controller.translate(action)
    assert sim_action.values == (1.0, -1.0)
    assert sim_action.metadata["controller"] == "reacher_delta_eef"


def test_reacher_delta_eef_controller_applies_2d_frame_rotation():
    config = load_config("configs/reacher_delta_eef.yaml")
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_2d", visible_to_agent=True, params={"angle_degrees": 90.0})
    )
    controller = ReacherDeltaEEFJoystickController.from_config(config.controller, FixedJoystickMapping.from_config(config.mapping), transform)
    sim_action = controller.translate(JoystickAction({"joystick_0": 1, "joystick_1": 0}))
    assert abs(sim_action.values[0]) < 1e-9
    assert abs(sim_action.values[1] - 1.0) < 1e-9


def test_pusher_delta_eef_controller_maps_xyz():
    config = load_config("configs/pusher_delta_eef.yaml")
    controller = PusherDeltaEEFJoystickController.from_config(config.controller, FixedJoystickMapping.from_config(config.mapping))
    action = JoystickAction({"joystick_0": 1, "joystick_1": -1, "joystick_2": 0})
    sim_action = controller.translate(action)
    assert sim_action.values == (1.0, -1.0, 0.0)
    assert sim_action.metadata["controller"] == "pusher_delta_eef"


def test_pusher_delta_eef_controller_applies_3d_frame_rotation():
    config = load_config("configs/pusher_delta_eef.yaml")
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0, 0, 90]})
    )
    controller = PusherDeltaEEFJoystickController.from_config(config.controller, FixedJoystickMapping.from_config(config.mapping), transform)
    sim_action = controller.translate(JoystickAction({"joystick_0": 1, "joystick_1": 0, "joystick_2": 0}))
    assert abs(sim_action.values[0]) < 1e-9
    assert abs(sim_action.values[1] - 1.0) < 1e-9
    assert abs(sim_action.values[2]) < 1e-9
