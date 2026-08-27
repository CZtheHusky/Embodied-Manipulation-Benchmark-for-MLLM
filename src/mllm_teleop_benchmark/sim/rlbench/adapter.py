from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import Observation, RLBenchAction, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.coppeliasim_lock import (
    CoppeliaSimRuntimeBusyError,
    CoppeliaSimRuntimeLock,
)
from mllm_teleop_benchmark.utils.math import clamp


class RLBenchSimAdapter(SimAdapter):
    simulator_type = "rlbench"

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.task_class_name = str(config.env_id or config.params.get("task_class", "ReachTarget"))
        self.headless = bool(config.params.get("headless", True))
        self.env: Any | None = None
        self.task: Any | None = None
        self.step_id = 0
        self.last_native_obs: Any = None
        self.last_descriptions: tuple[str, ...] = ()
        self.last_reward: float | None = None
        self.last_terminate = False
        self.last_success = False
        self.last_action_rejected = False
        self.variation_index: int | None = None
        self._camera_specs: tuple[CameraSpec, ...] = ()
        self._invalid_action_error_type: type[Exception] | None = None
        self._runtime_lock = CoppeliaSimRuntimeLock(self.simulator_type)

    def set_camera_configuration(self, camera_specs: tuple[CameraSpec, ...]) -> None:
        # ObservationProvider configures cameras before task.reset launches RLBench.
        assert self.env is None
        self._camera_specs = tuple(camera_specs)

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        numpy_random_state = np.random.get_state()
        try:
            np.random.seed(seed)
            self._ensure_task()
            self.step_id = 0
            self.last_reward = None
            self.last_terminate = False
            self.last_success = False
            self.last_action_rejected = False
            self.variation_index = self._sample_variation(seed)
            descriptions, obs = self.task.reset()
            native_success, native_terminate = self._native_task_success()
        finally:
            np.random.set_state(numpy_random_state)
        self.last_descriptions = tuple(str(item) for item in descriptions)
        self.last_native_obs = obs
        self.last_success = native_success
        self.last_terminate = native_terminate
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        self._ensure_task()
        native_action = self._native_action(action)
        try:
            obs, reward, terminate = self.task.step(native_action)
        except Exception as exc:
            if self._invalid_action_error_type is None or not isinstance(
                exc, self._invalid_action_error_type
            ):
                raise
            self.step_id += 1
            self.last_reward = 0.0
            self.last_terminate = False
            self.last_action_rejected = True
            return StepResult(
                observation=self._observation(),
                terminated=False,
                truncated=False,
                info=self.get_state(),
            )
        self.step_id += 1
        self.last_native_obs = obs
        self.last_reward = float(reward)
        self.last_terminate = bool(terminate)
        self.last_success = bool(reward)
        self.last_action_rejected = False
        return StepResult(
            observation=self._observation(),
            terminated=self.last_terminate,
            truncated=False,
            info=self.get_state(),
        )

    def render(self, camera: CameraSpec) -> Any:
        return self._extract_rgb(self.last_native_obs, camera.name)

    def get_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "step_id": self.step_id,
            "env_id": self.task_class_name,
            "task_class": self.task_class_name,
            "variation_index": self.variation_index,
            "reward": self.last_reward,
            "terminate": self.last_terminate,
            "success": self.last_success,
            "action_rejected": self.last_action_rejected,
            "descriptions": list(self.last_descriptions),
        }
        gripper_pose = self._last_gripper_pose()
        if gripper_pose is not None:
            state["gripper_pose"] = gripper_pose
            state["ee_position"] = gripper_pose[:3]
            state["tcp_position"] = gripper_pose[:3]
        for key in ("gripper_open", "joint_positions", "task_low_dim_state"):
            value = getattr(self.last_native_obs, key, None)
            converted = self._small_array(value)
            if converted is not None:
                state[key] = converted
        return state

    def close(self) -> None:
        try:
            if self.env is not None:
                self.env.shutdown()
        finally:
            self.env = None
            self.task = None
            self._runtime_lock.release()

    def _launch_environment(self, task_class: type[Any]) -> None:
        # _ensure_task constructs the environment immediately before launch.
        assert self.env is not None
        try:
            self._runtime_lock.acquire()
            self.env.launch()
            self.task = self.env.get_task(task_class)
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _ensure_task(self) -> None:
        if self.task is not None:
            return
        try:
            import pyrep  # noqa: F401
            from rlbench.environment import Environment
            invalid_action_error_type = _load_invalid_action_error_type()
        except ImportError as exc:
            raise ImportError(
                "RLBench backend requires rlbench, pyrep, and an external CoppeliaSim install. "
                "Create the conda backend with `python3 scripts/manage_conda_env.py create rlbench`, "
                "then set or inherit COPPELIASIM_ROOT, LD_LIBRARY_PATH, QT_QPA_PLATFORM_PLUGIN_PATH, "
                "and DISPLAY/headless settings before using simulator.type=rlbench."
            ) from exc

        aliases = {
            "ReachTarget": "rlbench.tasks.reach_target:ReachTarget",
            "reach_target": "rlbench.tasks.reach_target:ReachTarget",
        }
        target = aliases.get(self.task_class_name, self.task_class_name)
        module_name, class_name = target.split(":", 1)
        task_class = getattr(importlib.import_module(module_name), class_name)
        self._invalid_action_error_type = invalid_action_error_type
        action_mode = self._action_mode()
        obs_config = self._observation_config()
        kwargs = dict(self.config.env_kwargs)
        kwargs.setdefault("headless", self.headless)
        kwargs.setdefault("obs_config", obs_config)
        kwargs.setdefault("static_positions", bool(self.config.params.get("static_positions", False)))
        if "dataset_root" in self.config.params:
            kwargs.setdefault("dataset_root", self.config.params["dataset_root"])
        if "robot_setup" in self.config.params:
            kwargs.setdefault("robot_setup", self.config.params["robot_setup"])

        try:
            self.env = Environment(action_mode=action_mode, **kwargs)
            self._launch_environment(task_class)
        except CoppeliaSimRuntimeBusyError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Failed to launch RLBench/CoppeliaSim. Verify CoppeliaSim v4.1.0, PyRep, "
                "COPPELIASIM_ROOT, LD_LIBRARY_PATH, QT_QPA_PLATFORM_PLUGIN_PATH, and DISPLAY/headless GPU setup."
            ) from exc

    def _action_mode(self) -> Any:
        try:
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
            from rlbench.action_modes.gripper_action_modes import Discrete

            return MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(),
                gripper_action_mode=Discrete(),
            )
        except ImportError:
            from rlbench.action_modes import ActionMode, ArmActionMode, GripperActionMode

            return ActionMode(
                arm=ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME,
                gripper=GripperActionMode.OPEN_AMOUNT,
            )

    def _observation_config(self) -> Any:
        from rlbench.observation_config import ObservationConfig

        config = ObservationConfig()
        if hasattr(config, "set_all"):
            config.set_all(False)
        for attr in ("joint_positions", "gripper_open", "gripper_pose", "task_low_dim_state"):
            if hasattr(config, attr):
                setattr(config, attr, True)

        camera_specs = self._camera_specs or (
            CameraSpec(name="front", type="rlbench_rgb", width=512, height=512),
        )
        for camera in camera_specs:
            camera_config = getattr(config, f"{camera.name}_camera", None)
            if camera_config is None:
                continue
            if hasattr(camera_config, "set_all"):
                camera_config.set_all(False)
            if hasattr(camera_config, "rgb"):
                camera_config.rgb = True
            if hasattr(camera_config, "depth"):
                camera_config.depth = False
            if hasattr(camera_config, "mask"):
                camera_config.mask = False
            if hasattr(camera_config, "image_size"):
                camera_config.image_size = (int(camera.width), int(camera.height))
        return config

    def _sample_variation(self, seed: int) -> int | None:
        # reset() calls this only after _ensure_task has created the task.
        assert self.task is not None
        variation = self.config.params.get("variation_index")
        if variation is not None:
            self.task.set_variation(int(variation))
            return int(variation)
        if hasattr(self.task, "variation_count") and hasattr(
            self.task, "set_variation"
        ):
            variation_count = int(self.task.variation_count())
            sampled = int(seed) % variation_count
            self.task.set_variation(sampled)
            return sampled
        if hasattr(self.task, "sample_variation"):
            sampled = self.task.sample_variation()
            return int(sampled) if sampled is not None else None
        return None

    def _native_task_success(self) -> tuple[bool, bool]:
        native_task = getattr(self.task, "_task", None)
        success = getattr(native_task, "success", None)
        result = success()
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "RLBench native Task.success() must return (success, terminate)"
            )
        return bool(result[0]), bool(result[1])

    def _native_action(self, action: RLBenchAction) -> np.ndarray:
        pose = self._last_gripper_pose()
        position = [
            clamp(float(pose[idx]) + float(action.delta_position[idx]), self.config.workspace_low[idx], self.config.workspace_high[idx])
            for idx in range(3)
        ]
        quat = _normalize_quat(tuple(float(v) for v in pose[3:7]))
        if any(abs(value) > 0.0 for value in action.delta_rotation_rpy):
            quat = _normalize_quat(_quat_multiply(quat, _quat_from_rpy(action.delta_rotation_rpy)))
        values = [*position, *quat, clamp(float(action.gripper), 0.0, 1.0)]
        return np.asarray(values, dtype=np.float32)

    def _observation(self) -> Observation:
        return Observation(
            step_id=self.step_id,
            images={},
            state=self.get_state(),
            metadata={
                "simulator": self.simulator_type,
                "env_id": self.task_class_name,
            },
        )

    def _last_gripper_pose(self) -> list[float] | None:
        value = getattr(self.last_native_obs, "gripper_pose", None)
        converted = self._small_array(value)
        if not isinstance(converted, list) or len(converted) < 7:
            return None
        return [float(v) for v in converted[:7]]

    def _extract_rgb(self, observation: Any, camera_name: str) -> np.ndarray | None:
        candidates = (camera_name, f"{camera_name}_rgb")
        for attr in candidates:
            value = getattr(observation, attr, None)
            frame = self._as_hwc_uint8(value)
            if frame is not None:
                return frame
        return None

    def _as_hwc_uint8(self, value: Any) -> np.ndarray | None:
        array = self._to_numpy(value)
        if array is None or array.ndim != 3:
            return None
        if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.shape[-1] != 3:
            return None
        if array.dtype != np.uint8:
            if float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _small_array(self, value: Any) -> Any | None:
        array = self._to_numpy(value)
        if array is None:
            return value if isinstance(value, int | float | bool | str) else None
        if array.size <= 32:
            return array.reshape(-1).tolist()
        return {"array_shape": list(array.shape), "dtype": str(array.dtype)}

    def _to_numpy(self, value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        if hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.cpu().numpy()
        if isinstance(value, np.generic):
            return np.asarray(value)
        return None


def _quat_from_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _load_invalid_action_error_type() -> type[Exception]:
    try:
        from rlbench.backend.exceptions import InvalidActionError
    except ImportError:
        from rlbench.task_environment import InvalidActionError
    return InvalidActionError


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _normalize_quat(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quat))
    return tuple(float(value / norm) for value in quat)
