import pytest
from unittest.mock import MagicMock
import numpy as np

from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.core.registry import build_camera_manager, build_sim, build_task, build_controller, build_observation_provider
from mllm_teleop_benchmark.core.types import CalvinAction
from mllm_teleop_benchmark.action.joystick import JoystickAction

def test_calvin_config_load():
    """Verify that the CALVIN config can be loaded even if calvin-env is missing."""
    config = load_config("configs/calvin_delta_eef.yaml")
    assert config.simulator.type == "calvin"
    assert config.task.type == "calvin_generic"
    assert config.controller.type == "calvin_delta_eef"

def test_calvin_controller_translation():
    """Verify that the CALVIN controller correctly translates joystick actions."""
    config = load_config("configs/calvin_delta_eef.yaml")
    # Mock mapping and frame transform
    mapping = MagicMock()
    mapping.map_action.return_value = {
        "dx": 1.0, "dy": 0.0, "dz": -1.0, 
        "droll": 0.0, "dpitch": 0.5, "dyaw": 0.0, 
        "gripper": 1.0
    }
    mapping.metadata.return_value = {}
    
    from mllm_teleop_benchmark.control.calvin_controller import CalvinDeltaEEFController
    controller = CalvinDeltaEEFController.from_config(config.controller, mapping)
    
    action = JoystickAction(channels={"joystick_0": 1.0}) # Values don't matter as we mock mapping
    sim_action = controller.translate(action)
    
    assert isinstance(sim_action, CalvinAction)
    # Expected: [dx*scale, dy*scale, dz*scale, dr*scale, dp*scale, dy*scale, gripper_mapped]
    # Scales are 0.1 in the config
    expected_values = [0.1, 0.0, -0.1, 0.0, 0.05, 0.0, 1.0] # gripper 1.0 maps to open_value=1.0
    np.testing.assert_allclose(sim_action.values, expected_values)

    mapping.map_action.return_value = {
        "dx": 0.0, "dy": 0.0, "dz": 0.0,
        "droll": 0.0, "dpitch": 0.0, "dyaw": 0.0,
        "gripper": 0.25,
    }
    sim_action = controller.translate(action)
    assert sim_action.values[-1] == -1.0

def test_calvin_sim_adapter_reset_step_smoke():
    """Run reset+step when real calvin_env is installed; otherwise skip cleanly."""
    pytest.importorskip("calvin_env")
    config = load_config("configs/calvin_delta_eef.yaml")
    sim = build_sim(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    observation_provider = build_observation_provider(config, camera_manager)
    observation_provider.configure_sim(sim)
    try:
        obs = sim.reset(config.task, seed=42)
        assert obs.step_id == 0
        assert obs.state["step_id"] == 0
        obs = observation_provider.get_observation(sim)
        assert {"static", "gripper"}.issubset(obs.images)
        assert obs.images["static"].shape[-1] == 3
        assert obs.images["gripper"].shape[-1] == 3

        result = sim.step(CalvinAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))
        assert result.observation.step_id == 1
        assert result.info["step_id"] == 1
    finally:
        sim.close()

def test_calvin_randomized_camera_manager():
    """Verify that RandomizedCalvinCameraManager samples positions."""
    from mllm_teleop_benchmark.config.schema import CameraConfig, CameraSpec
    from mllm_teleop_benchmark.observation.camera import RandomizedCalvinCameraManager
    
    config = CameraConfig(
        manager="calvin_randomized",
        visible_metadata=True,
        cameras=(
            CameraSpec(
                name="static",
                type="calvin_rgb",
                width=200,
                height=200,
                pose=(),
                params={
                    "position_ranges": ([0.5, 1.0], [-0.2, 0.2], [0.4, 0.8]),
                    "look_at_ranges": ([-0.1, 0.1], [-0.1, 0.1], [0.0, 0.2])
                }
            ),
        )
    )
    
    manager = RandomizedCalvinCameraManager(config)
    
    # Reset with a seed
    manager.reset(seed=42)
    active = manager.get_active_cameras()
    assert len(active) == 1
    cam = active[0]
    assert "position" in cam.params
    assert "look_at" in cam.params
    pos1 = cam.params["position"]
    look1 = cam.params["look_at"]
    
    # Same seed should give same results
    manager.reset(seed=42)
    pos2 = manager.get_active_cameras()[0].params["position"]
    assert pos1 == pos2
    
    # Different seed should give different results
    manager.reset(seed=43)
    pos3 = manager.get_active_cameras()[0].params["position"]
    assert pos1 != pos3
