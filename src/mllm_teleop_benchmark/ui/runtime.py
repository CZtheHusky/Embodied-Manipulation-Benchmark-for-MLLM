from __future__ import annotations

import multiprocessing as mp
import os
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.types import Observation, SimulatorAction, StepResult


@dataclass(frozen=True)
class PlayerCommand:
    kind: str
    action: SimulatorAction | None = None


@dataclass(frozen=True)
class PlayerRuntimeResult:
    frame: Any
    observation: Observation | None
    state: dict[str, Any] | None
    step_result: StepResult | None
    terminated: bool = False
    truncated: bool = False
    error: str | None = None


class PlayerRuntime(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> PlayerRuntimeResult:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: SimulatorAction) -> PlayerRuntimeResult:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class PlayerRuntimeHooks:
    def prepare_worker_env(self, config: BenchmarkConfig) -> None:
        if config.ui.display:
            os.environ["DISPLAY"] = config.ui.display
        if config.ui.mujoco_gl:
            os.environ["MUJOCO_GL"] = config.ui.mujoco_gl
        if os.environ.get("MUJOCO_GL") == "egl":
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    def select_frame(self, config: BenchmarkConfig, observation: Observation, sim: Any, camera_manager: Any) -> tuple[str, Any]:
        if config.ui.frame_camera and config.ui.frame_camera in observation.images:
            return config.ui.frame_camera, observation.images[config.ui.frame_camera]
        for name, frame in observation.images.items():
            return str(name), frame
        for camera in camera_manager.get_active_cameras():
            return camera.name, sim.render(camera)
        raise RuntimeError("Player runtime could not find an RGB frame in observation or render fallback")


class MujocoPlayerRuntimeHooks(PlayerRuntimeHooks):
    def prepare_worker_env(self, config: BenchmarkConfig) -> None:
        if not config.ui.mujoco_gl:
            os.environ.setdefault("MUJOCO_GL", str(config.simulator.params.get("mujoco_gl", "egl")))
        super().prepare_worker_env(config)


def hooks_for_config(config: BenchmarkConfig) -> PlayerRuntimeHooks:
    if config.simulator.type in {"gymnasium", "dmcontrol", "metaworld", "robosuite", "robocasa", "libero"}:
        return MujocoPlayerRuntimeHooks()
    return PlayerRuntimeHooks()


class SimAdapterWorkerRuntime(PlayerRuntime):
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.context = mp.get_context("spawn")
        self.action_queue: mp.Queue | None = None
        self.result_queue: mp.Queue | None = None
        self.worker: mp.Process | None = None

    def start(self) -> None:
        if self.worker is not None:
            return
        self.action_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
        self.worker = self.context.Process(target=sim_adapter_worker_entry, args=(self.config, self.action_queue, self.result_queue))
        self.worker.start()

    def reset(self) -> PlayerRuntimeResult:
        self._ensure_started()
        self._put(PlayerCommand("reset"))
        return self._get_result()

    def step(self, action: SimulatorAction) -> PlayerRuntimeResult:
        self._ensure_started()
        self._put(PlayerCommand("step", action))
        return self._get_result()

    def close(self) -> None:
        if self.action_queue is not None and self.worker is not None and self.worker.is_alive():
            self.action_queue.put(PlayerCommand("close"))
        if self.worker is not None:
            self.worker.join(timeout=1.0)
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join(timeout=1.0)
        self.action_queue = None
        self.result_queue = None
        self.worker = None

    def _ensure_started(self) -> None:
        if self.worker is None:
            self.start()
        if self.worker is None or self.action_queue is None or self.result_queue is None:
            raise RuntimeError("Player runtime failed to start worker process")
        if not self.worker.is_alive():
            raise RuntimeError(f"Player runtime worker is not alive; exitcode={self.worker.exitcode}")

    def _put(self, command: PlayerCommand) -> None:
        if self.action_queue is None:
            raise RuntimeError("Player runtime command queue is unavailable")
        self.action_queue.put(command)

    def _get_result(self) -> PlayerRuntimeResult:
        if self.result_queue is None:
            raise RuntimeError("Player runtime result queue is unavailable")
        try:
            result = self.result_queue.get(timeout=self.config.ui.worker_timeout_seconds)
        except queue.Empty as exc:
            exitcode = self.worker.exitcode if self.worker is not None else None
            raise RuntimeError(f"Timed out waiting for player runtime worker result; exitcode={exitcode}") from exc
        if not isinstance(result, PlayerRuntimeResult):
            raise RuntimeError(f"Player runtime returned unsupported result: {type(result).__name__}")
        return result


def sim_adapter_worker_entry(config: BenchmarkConfig, command_queue: mp.Queue, result_queue: mp.Queue) -> None:
    hooks = hooks_for_config(config)
    try:
        hooks.prepare_worker_env(config)

        from mllm_teleop_benchmark.core.registry import build_camera_manager, build_observation_provider, build_sim, build_task

        sim = build_sim(config)
        task = build_task(config)
        camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
        observation_provider = build_observation_provider(config, camera_manager)
    except Exception as exc:
        result_queue.put(_error_result(exc))
        return

    try:
        while True:
            command = command_queue.get()
            if command is None:
                break
            if not isinstance(command, PlayerCommand):
                result_queue.put(_error_result(TypeError(f"Unsupported player command: {type(command).__name__}")))
                continue
            try:
                if command.kind == "reset":
                    camera_manager.reset(config.benchmark.seed)
                    observation_provider.configure_sim(sim)
                    task.reset(sim, config.benchmark.seed)
                    observation = observation_provider.get_observation(sim)
                    result_queue.put(_result_from_observation(config, hooks, observation, sim, camera_manager))
                elif command.kind == "step":
                    if command.action is None:
                        raise ValueError("Player step command requires a simulator action")
                    step_result = sim.step(command.action)
                    observation = observation_provider.get_observation(sim)
                    result_queue.put(_result_from_observation(config, hooks, observation, sim, camera_manager, step_result=step_result))
                elif command.kind == "close":
                    break
                else:
                    raise ValueError(f"Unknown player command kind: {command.kind}")
            except Exception as exc:
                result_queue.put(_error_result(exc))
    finally:
        if hasattr(sim, "close"):
            sim.close()


def _result_from_observation(
    config: BenchmarkConfig,
    hooks: PlayerRuntimeHooks,
    observation: Observation,
    sim: Any,
    camera_manager: Any,
    *,
    step_result: StepResult | None = None,
) -> PlayerRuntimeResult:
    frame_name, frame = hooks.select_frame(config, observation, sim, camera_manager)
    if frame_name not in observation.images:
        images = dict(observation.images)
        images[frame_name] = frame
        observation = Observation(
            step_id=observation.step_id,
            images=images,
            state=observation.state,
            metadata=observation.metadata,
        )
    state = dict(observation.state)
    info = dict(step_result.info if step_result is not None else state)
    info.setdefault("frame_name", frame_name)
    normalized_step_result = StepResult(
        observation=observation,
        terminated=bool(step_result.terminated) if step_result is not None else False,
        truncated=bool(step_result.truncated) if step_result is not None else False,
        info=info,
    )
    return PlayerRuntimeResult(
        frame=frame,
        observation=observation,
        state=state,
        step_result=normalized_step_result,
        terminated=normalized_step_result.terminated,
        truncated=normalized_step_result.truncated,
    )


def _error_result(exc: Exception) -> PlayerRuntimeResult:
    return PlayerRuntimeResult(
        frame=None,
        observation=None,
        state=None,
        step_result=None,
        terminated=True,
        truncated=False,
        error=f"{type(exc).__name__}: {exc}",
    )


__all__ = [
    "PlayerCommand",
    "PlayerRuntime",
    "PlayerRuntimeHooks",
    "PlayerRuntimeResult",
    "SimAdapterWorkerRuntime",
    "hooks_for_config",
    "sim_adapter_worker_entry",
]
