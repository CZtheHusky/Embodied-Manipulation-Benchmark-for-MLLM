import numpy as np
import pytest

from mllm_teleop_benchmark import load_config
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.gymnasium_envs.registry import REGISTERED_PUSHER_DELTA_EEF_ID, register_gymnasium_envs
from mllm_teleop_benchmark.tasks.gymnasium_pusher import GymnasiumPusherTask


def test_pusher_success_requires_configured_goal_hold_counter():
    config = load_config("configs/pusher_delta_eef.yaml")
    task = GymnasiumPusherTask(config.task)
    observation = Observation(step_id=0, images={}, state={})
    threshold = config.task.success_threshold

    assert config.task.goal_hold_steps == 3
    assert not task.is_success(
        observation,
        {"object_goal_distance": threshold, "goal_hold_counter": 2},
    )
    assert task.is_success(
        observation,
        {"object_goal_distance": threshold, "goal_hold_counter": 3},
    )
    assert not task.is_success(
        observation,
        {"object_goal_distance": threshold + 0.001, "goal_hold_counter": 3},
    )
    assert not task.is_success(
        observation,
        {"object_goal_distance": threshold},
    )


def test_pusher_delta_eef_env_reset_and_step():
    gym = pytest.importorskip("gymnasium")

    register_gymnasium_envs()
    env = gym.make(REGISTERED_PUSHER_DELTA_EEF_ID, render_mode="rgb_array")
    try:
        obs, info = env.reset(seed=3)
        assert env.unwrapped.eef_step_scale == pytest.approx(0.01)
        assert obs.shape == (23,)
        assert "object_goal_distance" in info
        assert "fingertip_object_distance" in info
        assert np.asarray(env.render()).shape == (512, 512, 3)
        next_obs, reward, terminated, truncated, step_info = env.step(np.zeros(3, dtype=np.float32))
        assert next_obs.shape == (23,)
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "ee_target_xyz" in step_info
    finally:
        env.close()


def test_pusher_delta_eef_env_rejects_wrong_action_shape():
    gym = pytest.importorskip("gymnasium")

    register_gymnasium_envs()
    env = gym.make(REGISTERED_PUSHER_DELTA_EEF_ID)
    try:
        env.reset(seed=3)
        with pytest.raises(ValueError, match="must be 3D"):
            env.step(np.zeros(2, dtype=np.float32))
    finally:
        env.close()
