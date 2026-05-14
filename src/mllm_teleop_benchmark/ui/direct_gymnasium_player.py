from __future__ import annotations

import os
import multiprocessing as mp
from typing import Any

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.types import GymBoxAction, Observation, StepResult
from mllm_teleop_benchmark.ui.direct_player import DirectBenchmarkPlayer, WorkerCommand, WorkerResult


def env_worker(config: BenchmarkConfig, action_queue: mp.Queue, res_queue: mp.Queue) -> None:
    """Isolated process for MuJoCo to avoid OpenGL context conflicts."""
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    import gymnasium as gym
    from mllm_teleop_benchmark.sim.gymnasium_envs.actions import action_to_native
    from mllm_teleop_benchmark.sim.gymnasium_envs.registry import REGISTERED_REACHER_DELTA_EEF_ID, register_gymnasium_envs

    def jsonable_array(value: Any) -> Any:
        try:
            import numpy as worker_np

            if isinstance(value, worker_np.ndarray):
                return value.tolist()
            if isinstance(value, worker_np.generic):
                return value.item()
        except Exception:
            pass
        return value

    def make_state(step_id: int, native_obs: Any, reward: float | None, info: dict[str, Any]) -> dict[str, Any]:
        state = {
            "step_id": step_id,
            "env_id": config.simulator.env_id or REGISTERED_REACHER_DELTA_EEF_ID,
            "native_observation": jsonable_array(native_obs),
            "reward": reward,
        }
        state.update({key: jsonable_array(value) for key, value in info.items()})
        if "goal_distance" in state:
            state["distance_to_target"] = float(state["goal_distance"])
        if "object_goal_distance" in state:
            state["distance_to_target"] = float(state["object_goal_distance"])
        return state

    def render_result(
        step_id: int,
        native_obs: Any,
        reward: float | None,
        info: dict[str, Any],
        done: bool,
        terminated: bool = False,
        truncated: bool = False,
    ) -> WorkerResult:
        state = make_state(step_id, native_obs, reward, info)
        observation = Observation(step_id=step_id, images={}, state=state, metadata={"simulator": "gymnasium", "env_id": state["env_id"]})
        step_result = StepResult(observation=observation, terminated=terminated, truncated=truncated, info=state)
        return WorkerResult(frame=env.render(), info=info, done=done, state=state, step_result=step_result)

    register_gymnasium_envs()
    env_kwargs = dict(config.simulator.env_kwargs)
    if config.simulator.frame_skip is not None:
        env_kwargs["frame_skip"] = config.simulator.frame_skip
    ui_config = config.raw.get("ui", {}) if isinstance(config.raw.get("ui", {}), dict) else {}
    max_episode_steps = int(ui_config.get("max_steps", config.task.max_steps))

    env = gym.make(
        config.simulator.env_id or REGISTERED_REACHER_DELTA_EEF_ID,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        **env_kwargs,
    )

    step_id = 0
    native_obs, info = env.reset(seed=config.benchmark.seed)
    res_queue.put(render_result(step_id, native_obs, None, dict(info), False))

    try:
        while True:
            cmd = action_queue.get()
            if cmd is None:
                break

            if isinstance(cmd, WorkerCommand):
                kind = cmd.kind
                action = cmd.action
            else:
                action, reset_req = cmd
                kind = "reset" if reset_req else "step"

            try:
                if kind == "reset":
                    step_id = 0
                    native_obs, info = env.reset(seed=config.benchmark.seed)
                    res_queue.put(render_result(step_id, native_obs, None, dict(info), False))
                elif kind == "step":
                    native_action = action_to_native(action) if isinstance(action, GymBoxAction) else action
                    native_obs, reward, terminated, truncated, info = env.step(native_action)
                    step_id += 1
                    res_queue.put(render_result(step_id, native_obs, float(reward), dict(info), terminated or truncated, bool(terminated), bool(truncated)))
                elif kind == "close":
                    break
                else:
                    raise ValueError(f"Unknown worker command kind: {kind}")
            except Exception as exc:
                res_queue.put(WorkerResult(frame=None, info={}, done=True, state=None, step_result=None, error=str(exc)))
    finally:
        env.close()


class DirectGymnasiumPlayer(DirectBenchmarkPlayer):
    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config, worker_target=env_worker, runtime_tag="direct_gymnasium_player")


__all__ = ["DirectGymnasiumPlayer", "WorkerCommand", "WorkerResult", "env_worker"]
