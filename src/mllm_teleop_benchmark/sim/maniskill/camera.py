from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.config.schema import CameraSpec


def maniskill_sensor_config(cameras: tuple[CameraSpec, ...], sampled_pose_config: dict[str, Any] | None = None) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    pose_config = sampled_pose_config or {}
    for camera in cameras:
        params = dict(camera.params)
        source = str(params.pop("source", "sensor"))
        if source == "human_render":
            continue
        if source != "sensor":
            raise ValueError(
                f"Unsupported ManiSkill camera source for {camera.name}: {source}"
            )
        entry = {
            "width": int(camera.width),
            "height": int(camera.height),
            **params,
        }
        if camera.pose:
            if len(camera.pose) == 7:
                entry["pose"] = {"p": list(camera.pose[:3]), "q": list(camera.pose[3:])}
            else:
                entry["pose"] = list(camera.pose)
        if "p" in entry and "q" in entry:
            entry["pose"] = {"p": entry.pop("p"), "q": entry.pop("q")}
        if camera.name in pose_config and isinstance(pose_config[camera.name], dict):
            entry.update(pose_config[camera.name])
        configs[camera.name] = entry
    return {"sensor_configs": configs} if configs else {}
