from __future__ import annotations

import os
import multiprocessing as mp
from typing import Any

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.types import DMControlAction, Observation, StepResult
from mllm_teleop_benchmark.ui.direct_player import DirectBenchmarkPlayer, WorkerCommand, WorkerResult


def env_worker(config: BenchmarkConfig, action_queue: mp.Queue, res_queue: mp.Queue) -> None:
    """Isolated process for DMControl to avoid OpenGL context conflicts."""
    ui_config = config.raw.get("ui", {}) if isinstance(config.raw.get("ui", {}), dict) else {}
    os.environ["MUJOCO_GL"] = str(ui_config.get("mujoco_gl", config.simulator.params.get("mujoco_gl", "egl")))
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    from dm_control import suite
    import numpy as worker_np
    from mllm_teleop_benchmark.sim.dmcontrol.actions import action_to_native

    def jsonable_array(value: Any) -> Any:
        try:
            if isinstance(value, worker_np.ndarray):
                return value.tolist()
            if isinstance(value, worker_np.generic):
                return value.item()
        except Exception:
            pass
        return value

    def make_state(step_id: int, timestep: Any, domain: str, task: str) -> dict[str, Any]:
        state = {
            "step_id": step_id,
            "reward": float(timestep.reward) if timestep.reward is not None else None,
            "step_type": int(timestep.step_type),
            "domain": domain,
            "task": task,
        }
        if timestep.observation:
            for key, val in timestep.observation.items():
                state[f"observation/{key}"] = jsonable_array(val)
        return state

    def render_result(
        step_id: int,
        timestep: Any,
        domain: str,
        task: str,
    ) -> WorkerResult:
        state = make_state(step_id, timestep, domain, task)
        # Using the first camera from config for rendering in the UI
        camera_name = None
        width = 640
        height = 480
        if config.camera.cameras:
            camera = config.camera.cameras[0]
            camera_name = camera.params.get("camera_id", camera.name)
            width = camera.width
            height = camera.height

        frame = env.physics.render(height=height, width=width, camera_id=camera_name)

        observation = Observation(
            step_id=step_id,
            images={},
            state=state,
            metadata={"simulator": "dmcontrol", "domain": domain, "task": task},
        )
        step_result = StepResult(
            observation=observation,
            terminated=timestep.last(),
            truncated=False,
            info=state,
        )
        return WorkerResult(
            frame=frame,
            info={},
            done=timestep.last(),
            state=state,
            step_result=step_result,
        )

    domain_name = str(config.simulator.params.get("domain_name", ""))
    task_name = str(config.simulator.params.get("task_name", ""))
    if not domain_name or not task_name:
        res_queue.put(
            WorkerResult(
                frame=None,
                info={},
                done=True,
                state=None,
                step_result=None,
                error="DMControl requires simulator.params.domain_name and simulator.params.task_name",
            )
        )
        return

    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs=config.simulator.env_kwargs,
        visualize_reward=bool(config.simulator.params.get("visualize_reward", False)),
    )

    step_id = 0
    if hasattr(env.task, "random"):
        env.task.random.seed(config.benchmark.seed)

    timestep = env.reset()
    res_queue.put(render_result(step_id, timestep, domain_name, task_name))

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
                    if hasattr(env.task, "random"):
                        env.task.random.seed(config.benchmark.seed)
                    timestep = env.reset()
                    res_queue.put(render_result(step_id, timestep, domain_name, task_name))
                elif kind == "step":
                    native_action = action_to_native(action, env.action_spec(), clip=True) if isinstance(action, DMControlAction) else action
                    timestep = env.step(native_action)
                    step_id += 1
                    res_queue.put(render_result(step_id, timestep, domain_name, task_name))
                elif kind == "close":
                    break
                else:
                    raise ValueError(f"Unknown worker command kind: {kind}")
            except Exception as exc:
                res_queue.put(WorkerResult(frame=None, info={}, done=True, state=None, step_result=None, error=str(exc)))
    finally:
        env.close()


class DirectDMControlPlayer(DirectBenchmarkPlayer):
    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config, worker_target=env_worker, runtime_tag="direct_dmcontrol_player")


__all__ = ["DirectDMControlPlayer", "WorkerCommand", "WorkerResult", "env_worker"]
