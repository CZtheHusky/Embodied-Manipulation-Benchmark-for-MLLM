from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

VECTOR_CAMERA_KEYS = {"lookat", "fixedcampos", "fixedcamlookat"}
INT_CAMERA_KEYS = {"trackbodyid"}


class MujocoCameraConfigurator:
    def prepare(self, camera_configuration: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if camera_configuration is None:
            return None
        prepared: dict[str, Any] = {}
        for key, value in camera_configuration.items():
            if key in VECTOR_CAMERA_KEYS:
                prepared[key] = np.asarray(value, dtype=np.float64)
            elif key in INT_CAMERA_KEYS:
                prepared[key] = int(value)
            else:
                prepared[key] = float(value)
        return prepared

    def apply(self, mujoco_renderer: Any, camera_configuration: Mapping[str, Any] | None) -> None:
        if camera_configuration is None:
            return
        mujoco_renderer.default_cam_config = {
            key: np.asarray(value, dtype=np.float64) if key in VECTOR_CAMERA_KEYS else value
            for key, value in camera_configuration.items()
        }
        for viewer in mujoco_renderer._viewers.values():
            cam = viewer.cam
            for key, value in mujoco_renderer.default_cam_config.items():
                if key in VECTOR_CAMERA_KEYS:
                    getattr(cam, key)[:] = value
                else:
                    setattr(cam, key, value)
