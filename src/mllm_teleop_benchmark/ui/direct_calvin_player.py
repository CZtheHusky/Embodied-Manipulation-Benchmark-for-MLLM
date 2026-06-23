from __future__ import annotations

import multiprocessing as mp
from typing import Any

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.registry import build_camera_manager, build_observation_provider
from mllm_teleop_benchmark.core.types import Observation, StepResult
from mllm_teleop_benchmark.ui.direct_player import DirectBenchmarkPlayer, WorkerCommand, WorkerResult


def env_worker(config: BenchmarkConfig, action_queue: mp.Queue, res_queue: mp.Queue) -> None:
    """Isolated process for CALVIN rendering and stepping."""
    from mllm_teleop_benchmark.core.types import CalvinAction
    from mllm_teleop_benchmark.sim.calvin.adapter import CalvinSimAdapter

    sim = CalvinSimAdapter(config.simulator)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    observation_provider = build_observation_provider(config, camera_manager)
    observation_provider.configure_sim(sim)

    def result_from_observation(
        observation: Observation,
        *,
        terminated: bool = False,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> WorkerResult:
        frame_name, frame = first_frame(observation)
        state = dict(observation.state)
        worker_info = dict(info or state)
        worker_info.setdefault("frame_name", frame_name)
        step_result = StepResult(observation=observation, terminated=terminated, truncated=truncated, info=state)
        return WorkerResult(
            frame=frame,
            info=worker_info,
            done=terminated or truncated,
            state=state,
            step_result=step_result,
        )

    def first_frame(observation: Observation) -> tuple[str, Any]:
        # Try to find 'static' camera first as it's usually the main view in CALVIN
        if "static" in observation.images:
            return "static", observation.images["static"]
        for name, frame in observation.images.items():
            return str(name), frame
        for camera in camera_manager.get_active_cameras():
            return camera.name, sim.render(camera)
        raise RuntimeError("CALVIN observation did not contain an RGB frame and render fallback failed")

    try:
        observation = sim.reset(config.task, config.benchmark.seed)
        observation = observation_provider.get_observation(sim)
        res_queue.put(result_from_observation(observation))

        while True:
            cmd = action_queue.get()
            if cmd is None:
                break
            kind = cmd.kind if isinstance(cmd, WorkerCommand) else ("reset" if cmd[1] else "step")
            action = cmd.action if isinstance(cmd, WorkerCommand) else cmd[0]
            try:
                if kind == "reset":
                    camera_manager.reset(config.benchmark.seed)
                    observation_provider.configure_sim(sim)
                    observation = sim.reset(config.task, config.benchmark.seed)
                    observation = observation_provider.get_observation(sim)
                    res_queue.put(result_from_observation(observation))
                elif kind == "step":
                    if not isinstance(action, CalvinAction):
                        raise TypeError(f"CALVIN direct player requires CalvinAction, got {type(action).__name__}")
                    step_result = sim.step(action)
                    observation = observation_provider.get_observation(sim)
                    res_queue.put(
                        result_from_observation(
                            observation,
                            terminated=step_result.terminated,
                            truncated=step_result.truncated,
                            info=step_result.info,
                        )
                    )
                elif kind == "close":
                    break
                else:
                    raise ValueError(f"Unknown worker command kind: {kind}")
            except Exception as exc:
                res_queue.put(WorkerResult(frame=None, info={}, done=True, state=None, step_result=None, error=str(exc)))
    except Exception as exc:
        res_queue.put(WorkerResult(frame=None, info={}, done=True, state=None, step_result=None, error=str(exc)))
    finally:
        sim.close()


class DirectCalvinPlayer(DirectBenchmarkPlayer):
    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config, worker_target=env_worker, runtime_tag="direct_calvin_player")


__all__ = ["DirectCalvinPlayer", "env_worker"]
