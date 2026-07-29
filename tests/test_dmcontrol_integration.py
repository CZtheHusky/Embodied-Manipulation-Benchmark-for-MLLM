from __future__ import annotations

from dataclasses import replace
import multiprocessing as mp

import numpy as np
import pytest

from mllm_teleop_benchmark import BenchmarkRunner, load_config
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.registry import build_controller, build_mapping
from mllm_teleop_benchmark.core.types import DMControlAction
from mllm_teleop_benchmark.ui.runtime import (
    PlayerCommand,
    PlayerRuntimeResult,
    sim_adapter_worker_entry,
)


def test_dmcontrol_short_episode(tmp_path):
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")
    config = load_config("configs/dmcontrol_example.yaml")
    config = replace(
        config,
        benchmark=replace(config.benchmark, name="test_dmcontrol", num_episodes=1),
        task=replace(config.task, max_steps=2),
        logging=replace(config.logging, output_dir=str(tmp_path), save_images=False),
    )

    result = BenchmarkRunner(config).run()

    assert result.aggregate_metrics["num_episodes"] == 1
    assert 1 <= result.episodes[0].steps <= 2
    assert "reward" in result.episodes[0].metrics
    assert "success" in result.episodes[0].metrics


def test_dmcontrol_manipulator_native_action_is_actuator_space():
    suite = pytest.importorskip("dm_control.suite")
    env = suite.load("manipulator", "bring_ball")

    spec = env.action_spec()
    actuator_names = [
        env.physics.model.id2name(index, "actuator")
        for index in range(env.physics.model.nu)
    ]

    assert spec.shape == (5,)
    assert actuator_names == ["root", "shoulder", "elbow", "wrist", "grasp"]
    assert actuator_names != ["dx", "dy", "dz", "droll", "gripper"]


def test_dmcontrol_legacy_controller_translation_is_shape_compatible():
    config = load_config("configs/dmcontrol_example.yaml")
    controller = build_controller(config, build_mapping(config))
    action = JoystickAction(
        channels={
            "joystick_0": 1.0,
            "joystick_1": 0.0,
            "joystick_2": -1.0,
            "joystick_3": 0.5,
            "joystick_4": 1.0,
        }
    )

    sim_action = controller.translate(action)

    assert isinstance(sim_action, DMControlAction)
    assert sim_action.values == (0.1, 0.0, -0.1, 0.05, 1.0)


def test_dmcontrol_reward_success_key_uses_threshold():
    from mllm_teleop_benchmark.tasks.dmcontrol_task import DMControlTask

    task = DMControlTask(
        TaskConfig(
            type="dmcontrol_generic",
            max_steps=10,
            success_threshold=0.95,
            params={"success_key": "reward"},
        )
    )

    assert task.is_success(None, {"reward": 0.5}) is False
    assert task.is_success(None, {"reward": 0.95}) is True


def test_dmcontrol_action_to_native_clips_and_validates_shape():
    pytest.importorskip("dm_control")
    from dm_env import specs

    from mllm_teleop_benchmark.sim.dmcontrol.actions import action_to_native

    action_spec = specs.BoundedArray(
        shape=(2,),
        dtype=np.float64,
        minimum=[-1.0, -0.5],
        maximum=[1.0, 0.5],
    )

    clipped = action_to_native(DMControlAction(values=(2.0, -2.0)), action_spec)
    np.testing.assert_allclose(clipped, np.array([1.0, -0.5]))
    with pytest.raises(ValueError, match="shape"):
        action_to_native(DMControlAction(values=(0.0, 0.0, 0.0)), action_spec)


def test_dmcontrol_named_camera_config_applies_to_model():
    from mllm_teleop_benchmark.sim.dmcontrol.observation import (
        DMControlObservationProvider,
    )

    class FakeModel:
        def __init__(self):
            self.cam_pos = np.zeros((1, 3), dtype=np.float64)
            self.cam_quat = np.zeros((1, 4), dtype=np.float64)
            self.cam_fovy = np.zeros((1,), dtype=np.float64)

        def name2id(self, name, kind):
            assert name == "fixed"
            assert kind == "camera"
            return 0

    class FakePhysics:
        def __init__(self):
            self.model = FakeModel()

    physics = FakePhysics()
    provider = DMControlObservationProvider(camera_manager=None)
    provider._apply_mujoco_camera_config(
        physics,
        {
            "name": "fixed",
            "pose": (1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5),
            "fovy": 45.0,
        },
    )

    np.testing.assert_allclose(
        physics.model.cam_pos[0],
        np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(
        physics.model.cam_quat[0],
        np.array([0.5, 0.5, 0.5, 0.5]),
    )
    assert physics.model.cam_fovy[0] == 45.0


def test_dmcontrol_free_camera_randomization_fails_loudly():
    from mllm_teleop_benchmark.sim.dmcontrol.observation import (
        DMControlObservationProvider,
    )

    provider = DMControlObservationProvider(camera_manager=None)
    with pytest.raises(NotImplementedError, match="free-camera orbit"):
        provider._apply_mujoco_camera_config(
            object(),
            {
                "distance": 1.0,
                "azimuth": 10.0,
                "elevation": -30.0,
                "lookat": (0.0, 0.0, 0.0),
            },
        )


def test_dmcontrol_sim_adapter_worker_reset_step_shutdown():
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")
    config = load_config("configs/dmcontrol_example.yaml")
    context = mp.get_context("spawn")
    action_queue: mp.Queue = context.Queue()
    result_queue: mp.Queue = context.Queue()
    worker = context.Process(
        target=sim_adapter_worker_entry,
        args=(config, action_queue, result_queue),
    )
    worker.start()
    try:
        action_queue.put(PlayerCommand("reset"))
        initial = result_queue.get(timeout=15)
        assert isinstance(initial, PlayerRuntimeResult)
        assert initial.error is None
        assert initial.observation is not None
        assert np.asarray(initial.frame).ndim == 3

        controller = build_controller(
            config,
            build_mapping(config),
            config.benchmark.seed,
        )
        sim_action = controller.translate(
            JoystickAction(
                {
                    f"joystick_{index}": 0.0
                    for index in range(5)
                }
            )
        )
        action_queue.put(PlayerCommand("step", sim_action))
        stepped = result_queue.get(timeout=15)
        assert isinstance(stepped, PlayerRuntimeResult)
        assert stepped.error is None
        assert stepped.observation is not None
        assert np.asarray(stepped.frame).shape[-1] == 3
        assert stepped.step_result is not None
        assert stepped.step_result.info["step_id"] == 1
    finally:
        action_queue.put(PlayerCommand("close"))
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
