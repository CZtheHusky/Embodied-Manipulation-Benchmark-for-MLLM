from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.observation.camera import CameraManager, RandomizedMujocoCameraManager
from mllm_teleop_benchmark.observation.provider import ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class DMControlObservationProvider(ObservationProvider):
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

    def get_observation(self, sim: SimAdapter) -> Observation:
        if isinstance(self.camera_manager, RandomizedMujocoCameraManager):
            config = self.camera_manager.mujoco_camera_config()
            if config:
                self._apply_mujoco_camera_config(sim.env.physics, config)

        state = sim.get_state()
        images: dict[str, Any] = {}
        for camera in self.camera_manager.get_active_cameras():
            images[camera.name] = sim.render(camera)

        return Observation(
            step_id=state.get("step_id", 0),
            images=images,
            state=state,
            metadata={
                "camera": self.camera_manager.metadata(),
                "simulator": "dmcontrol",
                "domain": state.get("domain"),
                "task": state.get("task"),
            },
        )

    def _apply_mujoco_camera_config(self, physics: Any, config: dict[str, Any]) -> None:
        """Apply named fixed-camera config to a DMControl MuJoCo model.

        The shared RandomizedMujocoCameraManager emits free-camera orbit
        parameters. DMControl's physics.render does not expose a stable hook for
        applying those parameters before rendering, so this backend only applies
        explicit fixed-camera model edits.
        """
        camera_configs = config.get("cameras")
        if camera_configs is None:
            camera_configs = (config,)
        if isinstance(camera_configs, dict):
            camera_configs = tuple(camera_configs.values())

        applied = False
        for camera_config in camera_configs:
            if not isinstance(camera_config, dict):
                continue
            camera_id_value = camera_config.get("camera_id", camera_config.get("name"))
            if camera_id_value is None:
                continue
            camera_id = self._camera_id(physics, camera_id_value)
            pose = camera_config.get("pose")
            pos = camera_config.get("pos", camera_config.get("position"))
            quat = camera_config.get("quat", camera_config.get("quaternion"))
            fovy = camera_config.get("fovy")

            if pose is not None:
                pose_values = tuple(float(v) for v in pose)
                if len(pose_values) >= 3:
                    physics.model.cam_pos[camera_id] = pose_values[:3]
                    applied = True
                if len(pose_values) == 7:
                    physics.model.cam_quat[camera_id] = pose_values[3:]
                    applied = True
            if pos is not None:
                pos_values = tuple(float(v) for v in pos)
                physics.model.cam_pos[camera_id] = pos_values
                applied = True
            if quat is not None:
                quat_values = tuple(float(v) for v in quat)
                physics.model.cam_quat[camera_id] = quat_values
                applied = True
            if fovy is not None:
                physics.model.cam_fovy[camera_id] = float(fovy)
                applied = True

    def _camera_id(self, physics: Any, camera_id: object) -> int:
        if isinstance(camera_id, int):
            return camera_id
        return int(physics.model.name2id(str(camera_id), "camera"))

    def configure_sim(self, sim: SimAdapter) -> None:
        for camera in self.camera_manager.get_active_cameras():
            if camera.pose or "fovy" in camera.params:
                self._apply_mujoco_camera_config(
                    sim.env.physics,
                    {
                        "name": camera.params.get("camera_id", camera.name),
                        "pose": camera.pose,
                        "fovy": camera.params.get("fovy"),
                    },
                )
