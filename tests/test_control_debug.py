import json

from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.core.types import GymBoxAction, ManiSkillAction, ValidationResult
from mllm_teleop_benchmark.logging.control_debug import ControlDebugRecorder
from mllm_teleop_benchmark.logging.debug_session import DebugSession


def test_control_debug_recorder_uses_session_path_and_tags(tmp_path):
    config = load_config("configs/pusher_delta_eef.yaml", (f"logging.output_dir={tmp_path}",))
    session = DebugSession.from_config(config, actor="human_keyboard", episode_id=0, seed=7, run_id="debug-run")
    recorder = ControlDebugRecorder(config, enabled=True, session=session)
    recorder.reset()
    record = recorder.build_step_record(
        source="human_keyboard",
        control_id="w",
        step_before=0,
        raw_action={"joystick_0": -1.0, "joystick_1": 0.0, "joystick_2": 0.0},
        validation=ValidationResult(True),
        sim_action=GymBoxAction(values=(-1.0, 0.0, 0.0), metadata={"controller": "pusher_delta_eef", "dimensions": ("dx", "dy", "dz")}),
        state_before={"step_id": 0, "fingertip_position": [1.0, 2.0, 3.0], "ee_target_xyz": [1.0, 2.0, 3.0]},
        state_after={"step_id": 1, "fingertip_position": [0.9, 2.0, 3.0], "ee_target_xyz": [0.99, 2.0, 3.0]},
    )
    recorder.record(record)

    path = session.control_events_path()
    data = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert path.name == "control_events.jsonl"
    assert path.parent.name == "episode_0000"
    assert data["debug_run_id"] == "debug-run"
    assert data["actor"] == "human_keyboard"
    assert data["env_slug"] == "mllmteleop_pusherdeltaeef_v0"
    assert data["control_slug"] == "eef_delta_pose_dx_dy_dz"
    assert data["expected_behavior"]["expected_delta_eef"]["dx"] == -0.01
    assert (session.root_dir / "manifest.json").exists()


def test_control_debug_recorder_describes_maniskill_delta_eef_action():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    recorder = ControlDebugRecorder(config, enabled=False)

    expected = recorder.expected_eef_behavior(
        ManiSkillAction(
            values=(1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            metadata={"controller": "maniskill_delta_eef", "dimensions": ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")},
        )
    )

    assert expected["controller"] == "maniskill_delta_eef"
    assert expected["expected_delta_eef"] == {"dx": 1.0, "dy": -1.0, "dz": 0.0}
