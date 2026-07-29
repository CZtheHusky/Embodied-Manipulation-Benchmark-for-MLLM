from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import CalvinAction, Observation, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class CalvinSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        if config.type != "calvin":
            raise ValueError(f"Unsupported simulator type: {config.type}")
        self.config = config
        self.env_id = None if config.env_id in {None, "", "None", "null"} else config.env_id
        self.env: Any | None = None
        self.step_id = 0
        self.last_native_obs: Any = None
        self.last_info: dict[str, Any] = {}
        self.last_reward: float | None = None
        self.last_render: Any = None
        self._camera_config: dict[str, Any] | None = None

    def set_camera_configuration(self, camera_config: dict[str, Any] | None) -> None:
        self._camera_config = dict(camera_config) if camera_config is not None else None

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        self._ensure_env(task_config)
        self.step_id = 0
        if self._camera_config:
            self._apply_camera_configuration()

        native_obs, info = self._reset_native(seed)

        self.last_native_obs = native_obs
        self.last_info = self._jsonable_mapping(info)
        self.last_reward = None
        return self._observation()

    def _apply_camera_configuration(self) -> None:
        """Apply sampled camera parameters to PyBullet cameras."""
        if not self.env or not self._camera_config:
            return

        try:
            import pybullet
        except ImportError:
            return

        unwrapped = getattr(self.env, "unwrapped", self.env)
        cameras = getattr(unwrapped, "cameras", [])
        if isinstance(cameras, dict):
            cameras = cameras.values()

        for cam_config in self._camera_config.values():
            cam_name = cam_config.get("name")
            pos = cam_config.get("position")
            look_at = cam_config.get("look_at", [0.0, 0.0, 0.0])

            if pos is None:
                continue

            view_matrix = pybullet.computeViewMatrix(
                cameraEyePosition=pos,
                cameraTargetPosition=look_at,
                cameraUpVector=[0, 0, 1],
            )

            for cam in cameras:
                if cam_name == "static" and (not hasattr(cam, "name") or cam.name == "static"):
                    if hasattr(cam, "view_matrix"):
                        cam.view_matrix = view_matrix
                    elif hasattr(cam, "viewMatrix"):
                        cam.viewMatrix = view_matrix
                    break

    def step(self, action: SimulatorAction) -> StepResult:
        if not isinstance(action, CalvinAction):
            raise TypeError(f"CalvinSimAdapter requires CalvinAction, got {type(action).__name__}")
        self._ensure_env(None)

        native_action = np.asarray(action.values, dtype=np.float32)
        if native_action.shape != (7,):
            raise ValueError(f"CALVIN action must contain 7 values, got shape {native_action.shape}")

        step_result = self.env.step(native_action)
        native_obs, reward, terminated, truncated, info = self._normalize_step_result(step_result)
        self.step_id += 1
        self.last_native_obs = native_obs
        self.last_reward = float(reward) if reward is not None else None
        self.last_info = self._jsonable_mapping(info)

        return StepResult(
            observation=self._observation(),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=self.get_state(),
        )

    def render(self, camera: CameraSpec) -> Any:
        self._ensure_env(None)
        # CALVIN usually renders multiple cameras. For manual play display, we take the default.
        if hasattr(self.env, "render"):
            try:
                frame = self.env.render(mode="rgb_array")
            except TypeError:
                frame = self.env.render()
            self.last_render = frame
            return frame
        raise RuntimeError("CALVIN environment does not expose render()")

    def get_state(self) -> dict[str, Any]:
        state = {
            "step_id": self.step_id,
            "env_id": self.env_id,
            "reward": self.last_reward,
        }
        state.update(self.last_info)

        tcp_pos = self._extract_tcp_pos()
        if tcp_pos is not None:
            state["ee_position"] = tcp_pos
            state["tcp_position"] = tcp_pos

        return state

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            if hasattr(self.env, "ownsPhysicsClient"):
                self.env.ownsPhysicsClient = False
            if hasattr(self.env, "cid"):
                self.env.cid = -1
            self.env = None

    def _ensure_env(self, task_config: TaskConfig | None) -> None:
        if self.env is not None:
            return
        try:
            # CALVIN env registration usually happens on import or via custom entry points
            import calvin_env  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "CALVIN backend requires calvin_env from https://github.com/mees/calvin_env. "
                "Install it in the mllm-teleop-calvin conda env."
            ) from exc

        kwargs = dict(self.config.env_kwargs)
        if self.env_id:
            self.env = self._make_registered_env(kwargs)
            return

        self.env = self._make_hydra_env(kwargs)

    def _make_registered_env(self, kwargs: dict[str, Any]) -> Any:
        for module_name in ("gymnasium", "gym"):
            try:
                gym = __import__(module_name)
            except ImportError:
                continue
            try:
                return gym.make(self.env_id, **kwargs)
            except Exception:
                if module_name == "gym":
                    raise
        raise ImportError("CALVIN registered env creation requires gym or gymnasium")

    def _make_hydra_env(self, kwargs: dict[str, Any]) -> Any:
        try:
            import hydra
            from omegaconf import open_dict
        except ImportError as exc:
            raise ImportError("CALVIN direct PlayTableSimEnv creation requires hydra-core and omegaconf") from exc

        import calvin_env

        config_name = str(self.config.params.get("hydra_config_name", "config_data_collection"))
        camera_config = str(self.config.params.get("camera_config", "static_and_gripper"))
        config_dir = Path(str(self.config.params.get("hydra_config_dir", Path(calvin_env.__file__).parents[1] / "conf")))
        overrides = [
            f"cameras={camera_config}",
            f"env.show_gui={str(bool(self.config.params.get('show_gui', False))).lower()}",
            f"env.use_egl={str(bool(self.config.params.get('use_egl', True))).lower()}",
        ]
        overrides.extend(str(item) for item in self.config.params.get("hydra_overrides", ()))

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize_config_dir(config_dir=str(config_dir), job_name="mllm_teleop_calvin", version_base=None):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with open_dict(cfg):
            cfg.use_vr = bool(kwargs.pop("use_vr", False))
            cfg.env.use_vr = cfg.use_vr
            cfg.env.use_scene_info = bool(kwargs.pop("use_scene_info", True))
            for key, value in kwargs.items():
                cfg.env[key] = value
        return hydra.utils.instantiate(cfg.env)

    def _observation(self) -> Observation:
        return Observation(
            step_id=self.step_id,
            images={},
            state=self.get_state(),
            metadata={"simulator": "calvin", "env_id": self.env_id},
        )

    def _jsonable_mapping(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"info": str(value)}
        return {str(k): self._jsonable(v) for k, v in value.items()}

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        return value

    def _extract_tcp_pos(self) -> list[float] | None:
        if self.last_native_obs is None:
            return None
        if isinstance(self.last_native_obs, dict) and "robot_obs" in self.last_native_obs:
            robot_obs = np.asarray(self.last_native_obs["robot_obs"])
            return robot_obs[:3].astype(float).tolist()
        return None

    def _reset_native(self, seed: int) -> tuple[Any, dict[str, Any]]:
        try:
            reset_result = self.env.reset(seed=seed)
        except TypeError:
            self._seed_legacy_env(seed)
            reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            return reset_result[0], self._jsonable_mapping(reset_result[1])
        info = self.env.get_info() if hasattr(self.env, "get_info") else {}
        return reset_result, self._jsonable_mapping(info)

    def _seed_legacy_env(self, seed: int) -> None:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        seed_target = self.env if callable(getattr(self.env, "seed", None)) else unwrapped
        seed_method = getattr(seed_target, "seed", None)
        if not callable(seed_method):
            raise RuntimeError("CALVIN environment does not expose a seed() fallback")

        seed_method(seed)
        np_random = getattr(seed_target, "np_random", None)
        if np_random is None:
            np_random = getattr(unwrapped, "np_random", None)
        scene = getattr(unwrapped, "scene", None)
        if np_random is None or scene is None:
            return

        scene.np_random = np_random
        for movable_object in getattr(scene, "movable_objects", ()):
            movable_object.np_random = np_random

    def _normalize_step_result(self, step_result: Any) -> tuple[Any, float | None, bool, bool, dict[str, Any]]:
        if not isinstance(step_result, tuple):
            raise TypeError(f"CALVIN env.step returned unsupported result type: {type(step_result).__name__}")
        if len(step_result) == 5:
            native_obs, reward, terminated, truncated, info = step_result
            return native_obs, reward, bool(terminated), bool(truncated), self._jsonable_mapping(info)
        if len(step_result) == 4:
            native_obs, reward, done, info = step_result
            max_steps = int(self.config.params.get("max_episode_steps", 0) or 0)
            truncated = bool(max_steps and self.step_id + 1 >= max_steps)
            return native_obs, reward, bool(done), truncated, self._jsonable_mapping(info)
        raise ValueError(f"CALVIN env.step returned {len(step_result)} values; expected 4 or 5")
