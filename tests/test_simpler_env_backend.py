import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import FrameTransformConfig
from mllm_teleop_benchmark.control.simpler_env import SimplerEnvDeltaEEFController
from mllm_teleop_benchmark.core.registry import build_action_space, build_camera_manager, build_controller, build_mapping, build_observation_provider, build_sim
from mllm_teleop_benchmark.core.types import SimplerEnvAction
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.sim.simpler_env.adapter import SimplerEnvSimAdapter
from mllm_teleop_benchmark.sim.simpler_env.observation import SimplerEnvObservationProvider


def test_simpler_env_smoke_imports_runtime_dependencies_before_pytest():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "manage_uv_env.py"
    spec = importlib.util.spec_from_file_location("manage_uv_env", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    command = module.BACKENDS["simpler-env"]["smoke"][0]
    assert command[:2] == ["python", "-c"]
    preflight = command[2]
    assert preflight.index("import simpler_env") < preflight.index("import pytest")
    assert preflight.index("import mani_skill2_real2sim") < preflight.index(
        "import pytest"
    )


def test_simpler_env_config_builds_without_simpler_env_install():
    config = load_config("configs/simpler_env_google_robot_pick_coke_can.yaml")
    action_space = build_action_space(config)
    controller = build_controller(config, build_mapping(config, seed=7), seed=7)
    sim = build_sim(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    observation_provider = build_observation_provider(config, camera_manager)

    assert config.simulator.type == "simpler_env"
    assert config.simulator.env_id == "google_robot_pick_coke_can"
    assert config.task.type == "simpler_env_generic"
    assert tuple(action_space.channels) == tuple(f"joystick_{idx}" for idx in range(7))
    assert controller.metadata()["type"] == "simpler_env_delta_eef"
    assert isinstance(sim, SimplerEnvSimAdapter)
    assert isinstance(observation_provider, SimplerEnvObservationProvider)


def test_simpler_env_config_uses_worker_play_runtime():
    config = load_config("configs/simpler_env_google_robot_pick_coke_can.yaml")

    assert config.ui.backend == "pygame"
    assert config.ui.runtime == "sim_adapter_worker"
    assert config.ui.frame_camera == "default"


def test_simpler_env_delta_eef_controller_maps_pose_and_gripper_to_7d_action():
    config = load_config("configs/simpler_env_google_robot_pick_coke_can.yaml")
    controller = SimplerEnvDeltaEEFController.from_config(config.controller, build_mapping(config, config.benchmark.seed))
    action = JoystickAction(
        {
            "joystick_0": 1,
            "joystick_1": -1,
            "joystick_2": 0,
            "joystick_3": 1,
            "joystick_4": 0,
            "joystick_5": -1,
            "joystick_6": 1.0,
        }
    )

    sim_action = controller.translate(action)

    assert isinstance(sim_action, SimplerEnvAction)
    assert sim_action.values == (
        0.01,
        -0.01,
        0.0,
        0.0314159265358979,
        0.0,
        -0.0314159265358979,
        1.0,
    )
    assert sim_action.metadata["action_order"] == ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def test_simpler_env_mllm_control_output_canary_preserves_scaled_signs():
    config = load_config(
        "configs/simpler_env_google_robot_pick_coke_can_mllm_reflection.yaml"
    )
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )
    sim_action = controller.translate(
        JoystickAction(
            {
                "joystick_0": 1.0,
                "joystick_1": -1.0,
                "joystick_2": 1.0,
                "joystick_3": -1.0,
                "joystick_4": 1.0,
                "joystick_5": -1.0,
                "joystick_6": 0.5,
            }
        )
    )

    np.testing.assert_allclose(
        sim_action.values,
        [
            0.01,
            -0.01,
            0.01,
            -0.0314159265358979,
            0.0314159265358979,
            -0.0314159265358979,
            0.0,
        ],
    )

    adapter = SimplerEnvSimAdapter(config.simulator)
    adapter.env = type(
        "FakeEnv",
        (),
        {
            "action_space": type(
                "FakeBox",
                (),
                {"shape": (7,), "low": -np.ones(7), "high": np.ones(7)},
            )()
        },
    )()
    native, metadata = adapter._native_action(sim_action)
    np.testing.assert_allclose(
        native,
        [
            0.01,
            -0.01,
            0.01,
            -0.0314159265358979,
            0.0314159265358979,
            -0.0314159265358979,
            0.0,
        ],
    )
    assert metadata["clipped"] is False


def test_simpler_env_delta_eef_controller_applies_frame_rotation():
    config = load_config("configs/simpler_env_google_robot_pick_coke_can.yaml")
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0, 0, 90]})
    )
    controller = SimplerEnvDeltaEEFController.from_config(config.controller, build_mapping(config, config.benchmark.seed), transform)

    sim_action = controller.translate(
        JoystickAction(
            {
                "joystick_0": 1,
                "joystick_1": 0,
                "joystick_2": 0,
                "joystick_3": 0,
                "joystick_4": 0,
                "joystick_5": 0,
                "joystick_6": 0.0,
            }
        )
    )

    np.testing.assert_allclose(sim_action.values[:3], [0.0, 0.01, 0.0], atol=1e-12)
    assert sim_action.values[-1] == -1.0


def test_simpler_env_observation_provider_extracts_rgb_images():
    config = load_config("configs/simpler_env_google_robot_pick_coke_can.yaml")
    provider = SimplerEnvObservationProvider(build_camera_manager(config.camera, config.benchmark.seed))
    frame = np.zeros((1, 4, 5, 4), dtype=np.uint8)
    frame[..., 2] = 255

    images = provider._extract_images({"agent": {"camera": {"rgb": frame}}})

    assert images["agent/camera"].shape == (4, 5, 3)
    assert images["agent/camera"].dtype == np.uint8
    assert images["agent/camera"][0, 0, 2] == 255


def test_simpler_env_adapter_clips_native_action_against_box_space():
    config = load_config("configs/simpler_env_google_robot_pick_coke_can.yaml")
    adapter = SimplerEnvSimAdapter(config.simulator)
    adapter.env = type(
        "FakeEnv",
        (),
        {
            "action_space": type(
                "FakeBox",
                (),
                {"shape": (7,), "low": -np.ones(7), "high": np.ones(7)},
            )()
        },
    )()

    native, metadata = adapter._native_action(SimplerEnvAction(values=(2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.5)))

    np.testing.assert_allclose(native, [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    assert metadata["clipped"] is True


def test_simpler_env_runtime_smoke_when_installed():
    """When SimplerEnv is installed, reset/render/step must work instead of skipping setup failures."""
    if importlib.util.find_spec("simpler_env") is None:
        pytest.skip("simpler_env is not installed")
    try:
        import simpler_env  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"simpler_env is installed but failed to import: {exc}")
    config = load_config("configs/simpler_env_google_robot_pick_coke_can.yaml")
    sim = build_sim(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    observation_provider = build_observation_provider(config, camera_manager)
    try:
        try:
            sim.reset(config.task, seed=42)
            observation = observation_provider.get_observation(sim)
            assert observation.images
            first_image = np.asarray(next(iter(observation.images.values())))
            assert first_image.ndim == 3
            assert first_image.shape[-1] == 3

            step = sim.step(SimplerEnvAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
            assert step.observation.step_id == 1
            assert step.info["step_id"] == 1
            next_observation = observation_provider.get_observation(sim)
            assert next_observation.images
        except Exception as exc:
            pytest.fail(f"SimplerEnv runtime is installed but reset/render/step failed: {exc}")
    finally:
        sim.close()
