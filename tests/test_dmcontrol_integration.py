import pytest

import numpy as np
from mllm_teleop_benchmark import BenchmarkRunner, load_config
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.config.schema import FrameTransformConfig
from mllm_teleop_benchmark.control.dmcontrol_controller import DMControlController
from mllm_teleop_benchmark.core.registry import build_controller, build_mapping
from mllm_teleop_benchmark.core.types import DMControlAction
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform

def test_dmcontrol_short_episode(tmp_path):
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")

    # Use the example config but override for a very short run
    config = load_config("configs/dmcontrol_example.yaml")

    # Simple patching of the config for testing
    patched_logging = config.logging.__class__(
        output_dir=str(tmp_path),
        save_images=False,
        save_trajectory=True,
    )
    patched_benchmark = config.benchmark.__class__(
        name="test_dmcontrol",
        num_episodes=1,
        seed=42,
    )
    patched_task = config.task.__class__(
        type=config.task.type,
        max_steps=2,
        success_threshold=config.task.success_threshold,
        params=config.task.params,
    )

    patched = config.__class__(
        **{
            **config.__dict__,
            "logging": patched_logging,
            "benchmark": patched_benchmark,
            "task": patched_task,
        }
    )

    runner = BenchmarkRunner(patched)
    result = runner.run()

    assert result.aggregate_metrics["num_episodes"] == 1
    assert 1 <= result.episodes[0].steps <= 2
    assert "reward" in result.episodes[0].metrics
    assert "success" in result.episodes[0].metrics

def test_dmcontrol_controller_translation():
    config = load_config("configs/dmcontrol_example.yaml")
    mapping = build_mapping(config)
    controller = build_controller(config, mapping)

    # Test mapping a simple joystick action
    action = JoystickAction(
        channels={
            "joystick_0": 1.0,
            "joystick_1": 0.0,
            "joystick_2": -1.0,
            "joystick_3": 0.5,
            "joystick_4": 1.0,
        }
    )  # dx, dy, dz, droll, gripper
    sim_action = controller.translate(action)

    assert isinstance(sim_action, DMControlAction)
    # Check scales application: dx: 0.1, dy: 0.1, dz: 0.1, droll: 0.1, gripper: 1.0
    assert sim_action.values == (0.1, 0.0, -0.1, 0.05, 1.0)


def test_dmcontrol_controller_applies_frame_rotation_to_position_and_rotation():
    config = load_config("configs/dmcontrol_example.yaml")
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0, 0, 90]})
    )
    controller = DMControlController.from_config(config.controller, build_mapping(config, config.benchmark.seed), transform)

    sim_action = controller.translate(
        JoystickAction(
            {
                "joystick_0": 1.0,
                "joystick_1": 0.0,
                "joystick_2": 0.0,
                "joystick_3": 1.0,
                "joystick_4": 0.0,
            }
        )
    )

    np.testing.assert_allclose(sim_action.values[:3], [0.0, 0.1, 0.0], atol=1e-12)
    assert abs(sim_action.values[3]) < 1e-12


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

    action_spec = specs.BoundedArray(shape=(2,), dtype=np.float64, minimum=[-1.0, -0.5], maximum=[1.0, 0.5])

    clipped = action_to_native(DMControlAction(values=(2.0, -2.0)), action_spec)
    np.testing.assert_allclose(clipped, np.array([1.0, -0.5]))

    with pytest.raises(ValueError, match="shape"):
        action_to_native(DMControlAction(values=(0.0, 0.0, 0.0)), action_spec)


def test_dmcontrol_named_camera_config_applies_to_model():
    from mllm_teleop_benchmark.sim.dmcontrol.observation import DMControlObservationProvider

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
        {"name": "fixed", "pose": (1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5), "fovy": 45.0},
    )

    np.testing.assert_allclose(physics.model.cam_pos[0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(physics.model.cam_quat[0], np.array([0.5, 0.5, 0.5, 0.5]))
    assert physics.model.cam_fovy[0] == 45.0


def test_dmcontrol_free_camera_randomization_fails_loudly():
    from mllm_teleop_benchmark.sim.dmcontrol.observation import DMControlObservationProvider

    provider = DMControlObservationProvider(camera_manager=None)

    with pytest.raises(NotImplementedError, match="free-camera orbit"):
        provider._apply_mujoco_camera_config(
            object(),
            {"distance": 1.0, "azimuth": 10.0, "elevation": -30.0, "lookat": (0.0, 0.0, 0.0)},
        )
