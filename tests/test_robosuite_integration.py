import numpy as np
import pytest
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.robosuite_controller import RobosuiteOSCController
from mllm_teleop_benchmark.mapping.frame_transform import IdentityControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping
from mllm_teleop_benchmark.sim.robosuite_adapter import RobosuiteSimAdapter


def test_robosuite_adapter_init():
    config = load_config("configs/robosuite_lift.yaml")
    adapter = RobosuiteSimAdapter()
    obs = adapter.reset(config.task, seed=42)
    
    assert "agentview" in obs.images
    assert "robot0_eye_in_hand" in obs.images
    assert obs.images["agentview"].shape == (240, 320, 3)
    
    state = adapter.get_state()
    assert "eef_pos" in state
    assert "is_success" in state


def test_robosuite_controller_translation():
    mapping = FixedJoystickMapping(
        channels={"joystick_0": "dx", "joystick_1": "dy", "joystick_2": "dz", "joystick_3": "gripper"},
        signs={"joystick_0": 1.0, "joystick_1": 1.0, "joystick_2": 1.0, "joystick_3": 1.0}
    )
    transform = IdentityControlFrameTransform()
    scales = {"dx": 0.1, "dy": 0.1, "dz": 0.1, "gripper": 1.0}
    controller = RobosuiteOSCController(mapping, transform, scales)
    
    # Test movement - include all mapped channels
    action = JoystickAction(channels={"joystick_0": 1.0, "joystick_1": -1.0, "joystick_2": 0.5, "joystick_3": 0.0})
    sim_action = controller.translate(action)
    
    assert isinstance(sim_action, np.ndarray)
    assert sim_action.shape == (7,)
    # dx = 1.0 * 0.1 = 0.1
    # dy = -1.0 * 0.1 = -0.1
    # dz = 0.5 * 0.1 = 0.05
    np.testing.assert_allclose(sim_action[:3], [0.1, -0.1, 0.05])
    
    # Test gripper: benchmark 0.0 -> robosuite -1.0
    action_open = JoystickAction(channels={"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0, "joystick_3": 0.0})
    sim_action_open = controller.translate(action_open)
    assert sim_action_open[6] == -1.0
    
    # Test gripper: benchmark 1.0 -> robosuite 1.0
    action_close = JoystickAction(channels={"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0, "joystick_3": 1.0})
    sim_action_close = controller.translate(action_close)
    assert sim_action_close[6] == 1.0
