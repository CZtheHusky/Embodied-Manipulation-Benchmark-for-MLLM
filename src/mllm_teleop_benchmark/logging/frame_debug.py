from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.logging.debug_session import DebugSession


class FrameDebugRecorder:
    def __init__(self, config: BenchmarkConfig, *, enabled: bool = False, session: DebugSession | None = None) -> None:
        self.enabled = bool(enabled)
        self.session = session
        self.fallback_path = Path(config.logging.output_dir) / "debug" / "raw_frames"

    def record_frame(
        self,
        frame: Any,
        *,
        step_id: int,
        prefix: str = "step",
        source: str = "env_render",
        camera_name: str = "front",
    ) -> Path | None:
        if not self.enabled:
            return None
        array = self._rgb_array(frame)
        if array is None:
            return None
        file_path = self._frame_path(step_id=step_id, prefix=prefix, source=source, camera_name=camera_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_png(file_path, array)
        return file_path

    def _frame_path(self, *, step_id: int, prefix: str, source: str, camera_name: str) -> Path:
        if self.session is not None:
            return self.session.frame_path(step_id=step_id, prefix=prefix, source=source, camera_name=camera_name)
        return self.fallback_path / f"{source}_{camera_name}_{prefix}_{int(step_id):04d}.png"

    def _rgb_array(self, frame: Any) -> np.ndarray | None:
        try:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] < 3:
                return None
            return np.ascontiguousarray(array[:, :, :3], dtype=np.uint8)
        except Exception:
            return None

    def _write_png(self, path: Path, array: np.ndarray) -> None:
        import imageio.v2 as imageio

        imageio.imwrite(path, array)
