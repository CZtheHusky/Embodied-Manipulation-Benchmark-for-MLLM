from __future__ import annotations

from pathlib import Path

import pytest

from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import CameraSpec


EXPECTED_CAMERAS = {
    "calvin": {"static": (512, 512), "gripper": (84, 84)},
    "colosseum": {"front": (512, 512)},
    "dmcontrol": {"fixed": (512, 384), "hand": (320, 240)},
    "dummy": {"front": (512, 512)},
    "gymnasium": {"front": (512, 512)},
    "libero": {"agentview": (512, 512), "robot0_eye_in_hand": (128, 128)},
    "language_table": {"rgb": (320, 180)},
    "maniskill": {"render_camera": (512, 512)},
    "metaworld": {"topview": (512, 512), "behindGripper": (480, 480)},
    "rlbench": {"front": (512, 512)},
    "robocasa": {
        "robot0_agentview_left": (512, 512),
        "robot0_agentview_right": (512, 512),
        "robot0_eye_in_hand": (256, 256),
    },
    "robosuite": {"agentview": (512, 512), "robot0_eye_in_hand": (256, 256)},
    "simpler_env": {"image/overhead_camera": (512, 410)},
}


CONFIG_PATHS = tuple(sorted(Path("configs").glob("*.yaml")))


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_every_config_uses_the_audited_native_camera_resolution(config_path: Path) -> None:
    config = load_config(config_path)

    actual = {
        camera.name: (camera.width, camera.height)
        for camera in config.camera.cameras
    }

    assert actual == EXPECTED_CAMERAS[config.simulator.type]


def test_unspecified_camera_resolution_defaults_to_a_512_square() -> None:
    camera = CameraSpec.from_dict({"name": "front"})

    assert (camera.width, camera.height) == (512, 512)


def test_a_single_hand_camera_without_an_upstream_size_defaults_to_512() -> None:
    camera = CameraSpec.from_dict({"name": "wrist_rgb"})

    assert (camera.width, camera.height) == (512, 512)


def test_resized_rectangular_cameras_preserve_the_upstream_aspect_ratio() -> None:
    dmcontrol = load_config(
        "configs/dmcontrol_manipulator_bring_ball_mllm_chat_completions.yaml"
    )
    simpler = load_config(
        "configs/simpler_env_google_robot_pick_coke_can_mllm_chat_completions.yaml"
    )

    fixed = next(camera for camera in dmcontrol.camera.cameras if camera.name == "fixed")
    overhead = simpler.camera.cameras[0]

    assert fixed.width / fixed.height == pytest.approx(320 / 240)
    assert overhead.width / overhead.height == pytest.approx(640 / 512, rel=0.002)
