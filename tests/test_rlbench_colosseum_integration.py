import os
import re
from pathlib import Path

import numpy as np
import pytest

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import FrameTransformConfig
from mllm_teleop_benchmark.control.rlbench import RLBenchDeltaEEFController
from mllm_teleop_benchmark.core.registry import build_action_space, build_camera_manager, build_controller, build_mapping, build_observation_provider, build_sim
from mllm_teleop_benchmark.core.types import RLBenchAction
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.sim.rlbench.adapter import RLBenchSimAdapter
from mllm_teleop_benchmark.sim.rlbench.observation import RLBenchObservationProvider


ROOT = Path(__file__).resolve().parents[1]


def test_rlbench_uv_input_pins_python_source_and_excludes_coppeliasim():
    requirement = ROOT.joinpath("envs/uv/rlbench.in").read_text()

    assert re.search(r"^rlbench @ git\+https://github\.com/stepjam/RLBench\.git@[0-9a-f]{40}$", requirement, re.MULTILINE)
    assert "coppeliasim" not in requirement.lower()


def test_rlbench_reach_target_config_builds_without_rlbench_install():
    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    action_space = build_action_space(config)
    controller = build_controller(config, build_mapping(config, seed=7), seed=7)
    sim = build_sim(config)

    assert config.simulator.type == "rlbench"
    assert config.task.type == "rlbench_generic"
    assert config.controller.type == "rlbench_delta_eef"
    assert tuple(action_space.channels) == ("joystick_0", "joystick_1", "joystick_2")
    assert controller.metadata()["type"] == "rlbench_delta_eef"
    assert isinstance(sim, RLBenchSimAdapter)


def test_rlbench_delta_eef_controller_maps_delta_position_and_default_gripper():
    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    controller = RLBenchDeltaEEFController.from_config(config.controller, build_mapping(config, config.benchmark.seed))

    sim_action = controller.translate(JoystickAction({"joystick_0": 1.0, "joystick_1": -1.0, "joystick_2": 0.0}))

    assert isinstance(sim_action, RLBenchAction)
    assert sim_action.delta_position == (0.02, -0.02, 0.0)
    assert sim_action.delta_rotation_rpy == (0.0, 0.0, 0.0)
    assert sim_action.gripper == 1.0


def test_rlbench_delta_eef_controller_applies_frame_rotation():
    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0, 0, 90]})
    )
    controller = RLBenchDeltaEEFController.from_config(config.controller, build_mapping(config, config.benchmark.seed), transform)

    sim_action = controller.translate(JoystickAction({"joystick_0": 1.0, "joystick_1": 0.0, "joystick_2": 0.0}))

    np.testing.assert_allclose(sim_action.delta_position, [0.0, 0.02, 0.0], atol=1e-12)


def test_rlbench_adapter_converts_delta_to_absolute_pose_action():
    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    adapter = RLBenchSimAdapter(config.simulator)
    adapter.last_native_obs = type(
        "FakeRLBenchObs",
        (),
        {"gripper_pose": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)},
    )()

    native = adapter._native_action(RLBenchAction(delta_position=(0.1, -0.1, 1.0), gripper=2.0))

    np.testing.assert_allclose(native, [0.1, -0.1, 1.35, 0.0, 0.0, 0.0, 1.0, 1.0])


def test_rlbench_observation_provider_extracts_configured_rgb_camera():
    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    provider = RLBenchObservationProvider(build_camera_manager(config.camera, config.benchmark.seed))
    frame = np.zeros((4, 5, 4), dtype=np.float32)
    frame[..., 0] = 1.0
    obs = type("FakeRLBenchObs", (), {"front_rgb": frame})()

    images = provider._extract_images(obs)

    assert images["front"].shape == (4, 5, 3)
    assert images["front"].dtype == np.uint8
    assert images["front"][0, 0, 0] == 255


def test_rlbench_runtime_reset_step_render_smoke_when_installed():
    try:
        __import__("rlbench")
        __import__("pyrep")
    except ImportError as exc:
        pytest.skip(f"RLBench/PyRep are not installed in this environment: {exc}")

    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    sim = build_sim(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    observation_provider = build_observation_provider(config, camera_manager)
    observation_provider.configure_sim(sim)
    try:
        try:
            obs = sim.reset(config.task, seed=42)
        except RuntimeError as exc:
            if os.environ.get("MLLM_TELEOP_REQUIRE_RLBENCH") == "1":
                raise
            pytest.skip(f"RLBench/CoppeliaSim runtime unavailable: {exc}")
        assert obs.step_id == 0

        frame = sim.render(config.camera.cameras[0])
        assert np.asarray(frame).shape[-1] == 3

        result = sim.step(RLBenchAction(delta_position=(0.0, 0.0, 0.0), gripper=1.0))
        assert result.observation.step_id == 1
        assert result.info["step_id"] == 1
    finally:
        sim.close()


def test_colosseum_route_is_blocked_until_rlbench_smoke_is_verified():
    pytest.skip(
        "Colosseum is intentionally not wired yet: first verify RLBench reset/step/render against "
        "CoppeliaSim/PyRep in the rlbench conda backend; see docs/rlbench_colosseum_integration.md."
    )
