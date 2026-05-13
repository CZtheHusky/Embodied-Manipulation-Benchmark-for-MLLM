import numpy as np

from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.logging.debug_session import DebugSession, slugify
from mllm_teleop_benchmark.logging.frame_debug import FrameDebugRecorder


def test_frame_debug_recorder_saves_rgb_frame(tmp_path):
    config = load_config("configs/pusher_delta_eef.yaml", (f"logging.output_dir={tmp_path}",))
    session = DebugSession.from_config(config, actor="human_keyboard", episode_id=0, seed=7, run_id="20260513-120000-test")
    recorder = FrameDebugRecorder(config, enabled=True, session=session)
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    frame[:, :, 0] = 255

    path = recorder.record_frame(frame, step_id=2, prefix="step", source="test", camera_name="front")

    assert path is not None
    assert path.name == "step_0002.png"
    assert path.parent.name == "front"
    assert path.parent.parent.name == "test"
    assert "mllmteleop_pusherdeltaeef_v0" in path.parts
    assert "eef_delta_pose_dx_dy_dz" in path.parts
    assert "human_keyboard" in path.parts
    assert path.exists()


def test_frame_debug_recorder_noops_when_disabled(tmp_path):
    config = load_config("configs/pusher_delta_eef.yaml", (f"logging.output_dir={tmp_path}",))
    recorder = FrameDebugRecorder(config, enabled=False)

    path = recorder.record_frame(np.zeros((4, 5, 3), dtype=np.uint8), step_id=2)

    assert path is None
    assert not (tmp_path / "debug").exists()


def test_debug_session_manifest_and_slug(tmp_path):
    config = load_config("configs/pusher_delta_eef.yaml", (f"logging.output_dir={tmp_path}",))
    session = DebugSession.from_config(config, actor="human keyboard", episode_id=3, seed=11, run_id="run-1")

    manifest = session.write_manifest()

    assert slugify("MLLMTeleop/PusherDeltaEEF-v0") == "mllmteleop_pusherdeltaeef_v0"
    assert session.episode_dir().name == "episode_0003"
    assert manifest.exists()
    assert session.root_dir.name == "run-1"
