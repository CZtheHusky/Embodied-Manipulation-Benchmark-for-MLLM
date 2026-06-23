import os

import numpy as np
import pytest
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.robosuite_controller import RobosuiteOSCController
from mllm_teleop_benchmark.core.types import RobosuiteAction
from mllm_teleop_benchmark.mapping.frame_transform import IdentityControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping
from mllm_teleop_benchmark.sim.robosuite_adapter import RobosuiteSimAdapter, sample_camera_randomization


def _require_robosuite() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    try:
        __import__("robosuite")
    except ImportError as exc:
        pytest.skip(f"robosuite is not available in this environment: {exc}")
    except Exception as exc:
        if os.environ.get("MLLM_TELEOP_REQUIRE_ROBOSUITE") == "1" or "robosuite" in os.environ.get("CONDA_DEFAULT_ENV", ""):
            raise
        pytest.skip(f"robosuite failed to import in this optional-backend environment: {exc}")


def test_robosuite_adapter_init():
    _require_robosuite()
    config = load_config("configs/robosuite_lift.yaml")
    adapter = RobosuiteSimAdapter(config.simulator)
    try:
        obs = adapter.reset(config.task, seed=42)

        assert "agentview" in obs.images
        assert "robot0_eye_in_hand" in obs.images
        assert obs.images["agentview"].shape == (240, 320, 3)

        state = adapter.get_state()
        assert "eef_pos" in state
        assert "is_success" in state

        step = adapter.step(RobosuiteAction((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)))
        assert step.observation.step_id == 1
    finally:
        adapter.close()


def test_robosuite_camera_randomization_sampling_is_seeded():
    cfg = {
        "agentview": {
            "pos_range": [[0.9, 1.1], [-0.1, 0.1], [1.4, 1.6]],
            "fovy_range": [40.0, 50.0],
        }
    }

    first = sample_camera_randomization(cfg, seed=5)
    second = sample_camera_randomization(cfg, seed=5)
    third = sample_camera_randomization(cfg, seed=6)

    assert first == second
    assert first != third
    assert len(first["agentview"]["pos"]) == 3
    assert 40.0 <= first["agentview"]["fovy"] <= 50.0


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
    
    assert isinstance(sim_action, RobosuiteAction)
    values = np.asarray(sim_action.values)
    assert values.shape == (7,)
    # dx = 1.0 * 0.1 = 0.1
    # dy = -1.0 * 0.1 = -0.1
    # dz = 0.5 * 0.1 = 0.05
    np.testing.assert_allclose(values[:3], [0.1, -0.1, 0.05])
    
    # Test gripper: benchmark 0.0 -> robosuite -1.0
    action_open = JoystickAction(channels={"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0, "joystick_3": 0.0})
    sim_action_open = controller.translate(action_open)
    assert sim_action_open.values[6] == -1.0
    
    # Test gripper: benchmark 1.0 -> robosuite 1.0
    action_close = JoystickAction(channels={"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0, "joystick_3": 1.0})
    sim_action_close = controller.translate(action_close)
    assert sim_action_close.values[6] == 1.0
