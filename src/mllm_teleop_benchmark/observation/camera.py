from __future__ import annotations

from mllm_teleop_benchmark.config.schema import CameraConfig, CameraSpec


class CameraManager:
    def __init__(self, config: CameraConfig) -> None:
        if config.manager != "fixed":
            raise ValueError(f"Unsupported camera manager: {config.manager}")
        self.config = config

    def reset(self, seed: int) -> None:
        return None

    def get_active_cameras(self) -> tuple[CameraSpec, ...]:
        return self.config.cameras

    def metadata(self) -> dict[str, object]:
        if not self.config.visible_metadata:
            return {"manager": self.config.manager, "visible_metadata": False}
        return {
            "manager": self.config.manager,
            "visible_metadata": True,
            "cameras": [
                {"name": camera.name, "type": camera.type, "width": camera.width, "height": camera.height, "pose": camera.pose}
                for camera in self.config.cameras
            ],
        }
