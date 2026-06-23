import numpy as np
import multiprocessing as mp
import pytest

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import FrameTransformConfig
from mllm_teleop_benchmark.control.maniskill import ManiSkillDeltaEEFController
from mllm_teleop_benchmark.core.registry import build_action_space, build_controller, build_mapping
from mllm_teleop_benchmark.core.types import ManiSkillAction
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.observation.camera import RandomizedManiSkillCameraManager
from mllm_teleop_benchmark.sim.maniskill.adapter import ManiSkillSimAdapter
from mllm_teleop_benchmark.sim.maniskill.observation import ManiSkillObservationProvider
from mllm_teleop_benchmark.ui.direct_maniskill_player import DirectManiSkillPlayer
from mllm_teleop_benchmark.ui.direct_maniskill_player import env_worker as maniskill_env_worker
from mllm_teleop_benchmark.ui.direct_player import WorkerCommand, WorkerResult


def test_maniskill_pickcube_config_builds_without_maniskill_install():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    action_space = build_action_space(config)
    controller = build_controller(config, build_mapping(config, seed=7), seed=7)

    assert config.simulator.type == "maniskill"
    assert config.raw["ui"]["backend"] == "direct_maniskill"
    assert tuple(action_space.channels) == tuple(f"joystick_{idx}" for idx in range(7))
    assert controller.metadata()["type"] == "maniskill_delta_eef"


def test_direct_maniskill_player_builds_without_starting_env():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    player = DirectManiSkillPlayer(config)

    assert player.runtime_tag == "direct_maniskill_player"
    assert player.max_steps == 100


def test_maniskill_delta_eef_controller_maps_pose_and_gripper():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    controller = ManiSkillDeltaEEFController.from_config(config.controller, build_mapping(config, config.benchmark.seed))
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

    assert isinstance(sim_action, ManiSkillAction)
    assert sim_action.values == (1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0)


def test_maniskill_delta_eef_controller_applies_frame_rotation():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0, 0, 90]})
    )
    controller = ManiSkillDeltaEEFController.from_config(config.controller, build_mapping(config, config.benchmark.seed), transform)

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

    assert abs(sim_action.values[0]) < 1e-9
    assert abs(sim_action.values[1] - 1.0) < 1e-9
    assert abs(sim_action.values[2]) < 1e-9


def test_maniskill_camera_randomization_is_episode_level_and_deterministic():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    manager = RandomizedManiSkillCameraManager(config.camera)

    manager.reset(5)
    first = manager.get_active_cameras()[0].pose
    repeated_without_reset = manager.get_active_cameras()[0].pose
    manager.reset(5)
    second = manager.get_active_cameras()[0].pose
    manager.reset(6)
    third = manager.get_active_cameras()[0].pose

    assert first == repeated_without_reset
    assert first == second
    assert first != third


def test_maniskill_observation_provider_extracts_rgb_images():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    provider = ManiSkillObservationProvider(RandomizedManiSkillCameraManager(config.camera))
    frame = np.zeros((1, 4, 5, 4), dtype=np.uint8)
    frame[..., 1] = 255

    images = provider._extract_images({"sensor_data": {"base_camera": {"rgb": frame}}})

    assert images["base_camera"].shape == (4, 5, 3)
    assert images["base_camera"].dtype == np.uint8
    assert images["base_camera"][0, 0, 1] == 255


def test_maniskill_adapter_clips_native_action_against_box_space():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    adapter = ManiSkillSimAdapter(config.simulator)
    adapter.env = type(
        "FakeEnv",
        (),
        {
            "action_space": type(
                "FakeBox",
                (),
                {"shape": (3,), "low": np.asarray([-1.0, -0.5, -1.0]), "high": np.asarray([1.0, 0.5, 1.0])},
            )()
        },
    )()

    native, metadata = adapter._native_action(ManiSkillAction(values=(2.0, -1.0, 0.25)))

    assert native.tolist() == [1.0, -0.5, 0.25]
    assert metadata["clipped"] is True


def test_direct_maniskill_worker_reset_step_shutdown_when_installed():
    pytest.importorskip("mani_skill")
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    action_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    worker = mp.Process(target=maniskill_env_worker, args=(config, action_queue, result_queue))
    worker.start()
    try:
        initial = result_queue.get(timeout=30)
        assert isinstance(initial, WorkerResult)
        if initial.error:
            pytest.skip(f"ManiSkill runtime unavailable: {initial.error}")
        assert np.asarray(initial.frame).ndim == 3

        controller = build_controller(config, build_mapping(config), config.benchmark.seed)
        sim_action = controller.translate(
            JoystickAction(
                {
                    "joystick_0": 0.0,
                    "joystick_1": 0.0,
                    "joystick_2": 0.0,
                    "joystick_3": 0.0,
                    "joystick_4": 0.0,
                    "joystick_5": 0.0,
                    "joystick_6": 0.0,
                }
            )
        )
        action_queue.put(WorkerCommand("step", sim_action))
        stepped = result_queue.get(timeout=30)
        assert isinstance(stepped, WorkerResult)
        if stepped.error:
            pytest.skip(f"ManiSkill runtime step unavailable: {stepped.error}")
        assert np.asarray(stepped.frame).shape[-1] == 3
        assert stepped.step_result is not None
        assert stepped.step_result.info["step_id"] == 1
    finally:
        action_queue.put(WorkerCommand("close"))
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
