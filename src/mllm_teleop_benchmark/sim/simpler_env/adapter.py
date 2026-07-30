from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import Observation, SimplerEnvAction, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class SimplerEnvSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        if config.type != "simpler_env":
            raise ValueError(f"Unsupported simulator type: {config.type}")
        self.config = config
        self.env_id = config.env_id
        if self.env_id is None:
            raise ValueError("SimplerEnv simulator requires simulator.env_id")
        raw_absolute_gripper = config.params.get("absolute_gripper_target", False)
        if not isinstance(raw_absolute_gripper, bool):
            raise ValueError(
                "SimplerEnv simulator.params.absolute_gripper_target must be boolean"
            )
        self.absolute_gripper_target = raw_absolute_gripper
        self.env: Any | None = None
        self.step_id = 0
        self.native_step_id = 0
        self.last_native_steps = 0
        self.last_native_obs: Any = None
        self.last_info: dict[str, Any] = {}
        self.last_reward: float | None = None
        self.last_success: bool | None = None
        self.last_truncated = False
        self.last_render: Any = None
        self._task_max_steps: int | None = None

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        self._task_max_steps = task_config.max_steps
        self._ensure_env()
        self.step_id = 0
        self.native_step_id = 0
        self.last_native_steps = 0
        self.last_truncated = False
        native_obs, reset_info = self._reset_native(seed)
        evaluator_info = self._reset_evaluator_info()
        reset_info.update(evaluator_info)
        self.last_native_obs = native_obs
        self.last_info = reset_info
        self.last_reward = None
        self.last_success = bool(evaluator_info["success"])
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        if not isinstance(action, SimplerEnvAction):
            raise TypeError(f"SimplerEnvSimAdapter requires SimplerEnvAction, got {type(action).__name__}")
        self._ensure_env()
        native_action, clip_metadata = self._native_action(action)
        gripper_application: dict[str, Any] | None = None
        if self.absolute_gripper_target:
            native_action, gripper_application = self._set_absolute_gripper_target(
                native_action
            )
        native_obs, reward, success, truncated, info = self._normalize_step_result(
            self.env.step(native_action)
        )

        self.step_id += 1
        self.native_step_id += 1
        self.last_native_steps = 1
        self.last_native_obs = native_obs
        self.last_reward = self._float_or_none(reward)
        self.last_success = bool(success)
        self.last_truncated = bool(truncated)
        self.last_info = self._jsonable_mapping(info)
        self.last_info["benchmark_applied_native_action"] = {
            "shape": list(native_action.shape),
            "values": self._jsonable(native_action),
        }
        if gripper_application is not None:
            self.last_info["benchmark_absolute_gripper"] = gripper_application
        if clip_metadata["clipped"]:
            self.last_info["action_clip"] = clip_metadata
        return StepResult(
            observation=self._observation(),
            terminated=self.last_success,
            truncated=self.last_truncated,
            info=self.get_state(),
        )

    def render(self, camera: CameraSpec) -> Any:
        self._ensure_env()
        frame = self.extract_image()
        if frame is not None:
            self.last_render = frame
            return frame
        if hasattr(self.env, "render"):
            try:
                rendered = self.env.render(mode=self.config.render_mode or "rgb_array")
            except TypeError:
                rendered = self.env.render()
            frame = self._as_hwc_uint8(rendered)
            self.last_render = frame if frame is not None else rendered
            return self.last_render
        raise RuntimeError("SimplerEnv environment does not expose render() and no RGB observation was available")

    def get_state(self) -> dict[str, Any]:
        state = {
            "step_id": self.step_id,
            "native_step_id": self.native_step_id,
            "native_steps_this_action": self.last_native_steps,
            "absolute_gripper_target": self.absolute_gripper_target,
            "env_id": self.env_id,
            "reward": self.last_reward,
            "success": self.last_success,
            "truncated": self.last_truncated,
        }
        instruction = self._language_instruction()
        if instruction is not None:
            state["language_instruction"] = instruction
        state.update(self.last_info)
        return state

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def extract_image(self) -> np.ndarray | None:
        if self.last_native_obs is None:
            return None
        try:
            from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        except ImportError:
            return None
        return self._as_hwc_uint8(get_image_from_maniskill2_obs_dict(self.env, self.last_native_obs))

    def _ensure_env(self) -> None:
        if self.env is not None:
            return
        try:
            import simpler_env
        except ImportError as exc:
            raise ImportError(
                "SimplerEnv backend requires the optional simpler_env package and mani_skill2_real2sim. "
                "Create the uv backend with: python3 scripts/manage_uv_env.py create simpler-env"
            ) from exc

        kwargs = dict(self.config.env_kwargs)
        kwargs.update(dict(self.config.params.get("make_kwargs", {})))
        configured_horizon = self.config.params.get("max_episode_steps")
        if configured_horizon is None:
            configured_horizon = self._task_max_steps
        if configured_horizon is not None:
            kwargs.setdefault("max_episode_steps", int(configured_horizon))
        self.env = simpler_env.make(self.env_id, **kwargs)
        expected_horizon = configured_horizon
        actual_horizon = getattr(self.env, "_max_episode_steps", None)
        if (
            expected_horizon is not None
            and actual_horizon is not None
            and int(actual_horizon) < int(expected_horizon)
        ):
            self.close()
            raise RuntimeError(
                "SimplerEnv horizon is smaller than the required simulator "
                f"execution budget: {actual_horizon} < {expected_horizon}"
            )

    def _reset_native(self, seed: int) -> tuple[Any, dict[str, Any]]:
        reset_kwargs = dict(self.config.params.get("reset_kwargs", {}))
        try:
            reset_result = self.env.reset(seed=seed, **reset_kwargs)
        except TypeError:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
            reset_result = self.env.reset(**reset_kwargs)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            return reset_result[0], self._jsonable_mapping(reset_result[1])
        return reset_result, {}

    def _evaluator_info(self) -> dict[str, Any]:
        get_info = None
        get_wrapper_attr = getattr(self.env, "get_wrapper_attr", None)
        if callable(get_wrapper_attr):
            try:
                get_info = get_wrapper_attr("get_info")
            except AttributeError:
                pass
        if get_info is None:
            get_info = getattr(self.env, "get_info", None)
        if not callable(get_info):
            raise RuntimeError(
                "SimplerEnv environment does not expose the public get_info() evaluator path"
            )

        info = self._jsonable_mapping(get_info())
        if "success" not in info:
            raise RuntimeError("SimplerEnv get_info() did not return required success")
        return info

    def _reset_evaluator_info(self) -> dict[str, Any]:
        raw_fields = self.config.params.get("reset_evaluator_state_fields")
        if not isinstance(raw_fields, (list, tuple)) or not raw_fields:
            raise RuntimeError(
                "SimplerEnv simulator.params.reset_evaluator_state_fields must "
                "declare the evaluator bookkeeping restored after reset inspection"
            )
        fields = tuple(str(field) for field in raw_fields)
        owner = self._evaluator_state_owner(fields)
        snapshots = {field: deepcopy(getattr(owner, field)) for field in fields}
        try:
            return self._evaluator_info()
        finally:
            for field, value in snapshots.items():
                setattr(owner, field, value)

    def _evaluator_state_owner(self, fields: tuple[str, ...]) -> Any:
        pending = [getattr(self.env, "unwrapped", None), self.env]
        visited: set[int] = set()
        while pending:
            candidate = pending.pop(0)
            if candidate is None or id(candidate) in visited:
                continue
            visited.add(id(candidate))
            if all(hasattr(candidate, field) for field in fields):
                return candidate
            pending.extend(
                (
                    getattr(candidate, "unwrapped", None),
                    getattr(candidate, "env", None),
                )
            )
        raise RuntimeError(
            "SimplerEnv evaluator bookkeeping fields are unavailable: "
            + ", ".join(fields)
        )

    def _native_action(self, action: SimplerEnvAction) -> tuple[np.ndarray, dict[str, Any]]:
        values = np.asarray(action.values, dtype=np.float32)
        if values.shape != (7,):
            raise ValueError(f"SimplerEnv action must contain 7 values, got shape {values.shape}")

        action_space = getattr(self.env, "action_space", None)
        metadata: dict[str, Any] = {"clipped": False}
        if action_space is None or not hasattr(action_space, "shape"):
            return values, metadata
        target_shape = tuple(int(value) for value in action_space.shape)
        if values.shape != target_shape:
            if len(target_shape) == 2 and target_shape[0] == 1 and values.shape == target_shape[1:]:
                values = values.reshape(target_shape)
            else:
                raise ValueError(f"SimplerEnv action shape mismatch: controller produced {values.shape}, env expects {target_shape}")
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            before = values.copy()
            values = np.clip(values, action_space.low, action_space.high)
            clipped = bool(np.any(before != values))
            metadata = {
                "clipped": clipped,
                "before": before.tolist() if clipped else None,
                "after": values.tolist() if clipped else None,
            }
        return values, metadata

    def _set_absolute_gripper_target(
        self,
        native_action: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        values = np.array(native_action, dtype=np.float32, copy=True)
        flat_values = values.reshape(-1)
        gripper_command = float(flat_values[-1])
        if not (
            np.isclose(gripper_command, 0.0, atol=1e-7)
            or np.isclose(abs(gripper_command), 1.0, atol=1e-7)
        ):
            raise ValueError(
                "SimplerEnv absolute gripper target requires native command -1, 0, or 1"
            )

        gripper = self._native_gripper_controller()
        required_flags = (
            "use_target",
            "clip_target",
            "small_action_repeat_last_target",
        )
        if any(not bool(getattr(gripper.config, name, False)) for name in required_flags):
            raise RuntimeError(
                "SimplerEnv absolute gripper target requires the pinned target-delta "
                "controller with target clipping and small-action target retention"
            )

        target_update = "preserved"
        if not np.isclose(gripper_command, 0.0, atol=1e-7):
            joint_limits = np.asarray(
                gripper.articulation.get_qlimits()[gripper.joint_indices],
                dtype=np.float64,
            )
            clip_threshold = float(gripper.config.clip_target_thres)
            if gripper_command < 0.0:
                target = joint_limits[:, 0] - clip_threshold
                target_update = "open"
            else:
                target = joint_limits[:, 1] + clip_threshold
                target_update = "closed"
            gripper.set_state({"target_qpos": target})

        get_state = getattr(gripper, "get_state", None)
        if not callable(get_state):
            raise RuntimeError(
                "SimplerEnv absolute gripper target requires controller.get_state()"
            )
        gripper_state = self._jsonable_mapping(get_state())
        if "target_qpos" not in gripper_state:
            raise RuntimeError(
                "SimplerEnv absolute gripper controller state is missing target_qpos"
            )
        flat_values[-1] = 0.0
        return values, {
            "controller_command": gripper_command,
            "applied_env_command": 0.0,
            "target_update": target_update,
            "target_qpos": gripper_state["target_qpos"],
        }

    def _native_gripper_controller(self) -> Any:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        agent = getattr(unwrapped, "agent", None)
        combined_controller = getattr(agent, "controller", None)
        controllers = getattr(combined_controller, "controllers", None)
        if not isinstance(controllers, dict) or "gripper" not in controllers:
            raise RuntimeError(
                "SimplerEnv absolute gripper target requires a named gripper controller"
            )
        gripper = controllers["gripper"]
        if not callable(getattr(gripper, "set_state", None)):
            raise RuntimeError(
                "SimplerEnv absolute gripper target requires controller.set_state()"
            )
        return gripper

    def _normalize_step_result(self, step_result: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        if not isinstance(step_result, tuple):
            raise TypeError(f"SimplerEnv env.step returned unsupported result type: {type(step_result).__name__}")
        if len(step_result) == 5:
            native_obs, reward, success, truncated, info = step_result
            return native_obs, reward, bool(success), bool(truncated), self._jsonable_mapping(info)
        if len(step_result) == 4:
            native_obs, reward, done, info = step_result
            max_steps = int(self.config.params.get("max_episode_steps", 0) or 0)
            truncated = bool(max_steps and self.step_id + 1 >= max_steps)
            return native_obs, reward, bool(done), truncated, self._jsonable_mapping(info)
        raise ValueError(f"SimplerEnv env.step returned {len(step_result)} values; expected 4 or 5")

    def _observation(self) -> Observation:
        return Observation(
            step_id=self.step_id,
            images={},
            state=self.get_state(),
            metadata={"simulator": "simpler_env", "env_id": self.env_id},
        )

    def _language_instruction(self) -> str | None:
        if self.env is None:
            return None
        get_instruction = None
        get_wrapper_attr = getattr(self.env, "get_wrapper_attr", None)
        if callable(get_wrapper_attr):
            try:
                get_instruction = get_wrapper_attr("get_language_instruction")
            except AttributeError:
                pass
        if get_instruction is None:
            get_instruction = getattr(
                getattr(self.env, "unwrapped", self.env),
                "get_language_instruction",
                None,
            )
        if not callable(get_instruction):
            return None
        try:
            return str(get_instruction())
        except Exception:
            return None

    def _jsonable_mapping(self, value: Any) -> dict[str, Any]:
        converted = self._jsonable(value)
        return converted if isinstance(converted, dict) else {"info": converted}

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        pose = self._pose_mapping(value)
        if pose is not None:
            return pose
        array = self._to_numpy(value)
        if array is not None:
            if array.size == 1:
                return array.reshape(-1)[0].item()
            if array.size <= 32:
                return array.tolist()
            return {"array_shape": list(array.shape), "dtype": str(array.dtype)}
        return value

    def _as_hwc_uint8(self, value: Any) -> np.ndarray | None:
        array = self._to_numpy(value)
        if array is None:
            return None
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            return None
        if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            if float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

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

    def _pose_mapping(self, value: Any) -> dict[str, Any] | None:
        if not hasattr(value, "p") or not hasattr(value, "q"):
            return None
        position = self._to_numpy(value.p)
        quaternion = self._to_numpy(value.q)
        if position is None or quaternion is None:
            return None
        if position.size != 3 or quaternion.size != 4:
            return None
        return {
            "position": position.reshape(3).tolist(),
            "quaternion_wxyz": quaternion.reshape(4).tolist(),
        }

    def _float_or_none(self, value: Any) -> float | None:
        if value is None:
            return None
        array = self._to_numpy(value)
        if array is not None:
            return float(array.reshape(-1)[0])
        return float(value)
