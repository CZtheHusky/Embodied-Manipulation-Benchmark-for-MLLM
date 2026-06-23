import numpy as np
import pytest

from mllm_teleop_benchmark.sim.gymnasium_envs.registry import REGISTERED_PUSHER_DELTA_EEF_ID, register_gymnasium_envs


def test_pusher_delta_eef_env_reset_and_step():
    gym = pytest.importorskip("gymnasium")

    register_gymnasium_envs()
    env = gym.make(REGISTERED_PUSHER_DELTA_EEF_ID)
    try:
        obs, info = env.reset(seed=3)
        assert obs.shape == (23,)
        assert "object_goal_distance" in info
        assert "fingertip_object_distance" in info
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
