from __future__ import annotations

REGISTERED_REACHER_DELTA_EEF_ID = "MLLMTeleop/ReacherDeltaEEF-v0"


def register_gymnasium_envs() -> None:
    import gymnasium as gym
    from gymnasium.envs.registration import register

    try:
        gym.spec(REGISTERED_REACHER_DELTA_EEF_ID)
    except Exception:
        register(
            id=REGISTERED_REACHER_DELTA_EEF_ID,
            entry_point="mllm_teleop_benchmark.sim.gymnasium_envs.envs.reacher_delta_eef:ReacherDeltaEEFEnv",
            max_episode_steps=1000,
        )
