from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from scripts import audit_mllm_run


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _png(path: Path, *, black: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (4, 4))
    if black:
        image.putdata([(0, 0, 0)] * 16)
    else:
        image.putdata(
            [
                (10 + index * 7, 30 + (index % 4) * 20, 60 + (index % 3) * 30)
                for index in range(16)
            ]
        )
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_fixture(
    tmp_path: Path,
    *,
    requested_episodes: int = 1,
    valid: bool = True,
    trace_status: str = "ok",
    attempts: int = 2,
) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    episode_dir = run_dir / "episode_0000"
    image_path = episode_dir / "model_calls/images/step_0000/front.png"
    digest = _png(image_path)
    _write_json(
        run_dir / "run_manifest.json",
        {"benchmark": {"num_episodes": requested_episodes}},
    )
    episode_record = {
        "episode_id": 0,
        "success": True,
        "steps": 1,
        "termination_reason": "success",
    }
    _write_json(run_dir / "run_summary.json", {"episodes": [episode_record]})
    _write_json(
        episode_dir / "summary.json",
        {
            **episode_record,
            "review_video": {"status": "completed"},
        },
    )
    (episode_dir / "trajectory.jsonl").write_text(
        json.dumps(
            {
                "step_id": 0,
                "validation": {"is_valid": valid},
                "agent_trace": {
                    "actor_type": "mllm",
                    "status": trace_status,
                    "attempts": attempts,
                    "artifact_dir": "model_calls/step_0000",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        episode_dir / "model_calls/step_0000/request.json",
        {
            "request": {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Move the object toward the target. "
                            "The target is visible in the public image."
                        ),
                    }
                ]
            },
            "images": [
                {
                    "artifact": "model_calls/images/step_0000/front.png",
                    "sha256": digest,
                }
            ],
        },
    )
    _write_json(
        episode_dir / "model_calls/step_0000/response.json",
        {"model": "test-model"},
    )
    _write_json(episode_dir / "review/manifest.json", {"status": "completed"})
    (episode_dir / "review/model_view.mp4").write_bytes(b"video")
    return run_dir, image_path


def _mock_ffprobe(monkeypatch, streams: list[dict[str, object]] | None = None) -> None:
    streams = streams or [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "2/1",
        }
    ]
    monkeypatch.setattr(audit_mllm_run.shutil, "which", lambda name: f"/mock/{name}")
    monkeypatch.setattr(
        audit_mllm_run.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"streams": streams}), stderr=""
        ),
    )


def _make_formal_run(
    run_dir: Path,
    *,
    requested_model: str | None = "test-model",
) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest["source"] = {
        "status": "reported",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "config_path": "configs/test.yaml",
    }
    manifest["model"] = (
        {"requested_model": requested_model}
        if requested_model is not None
        else {}
    )
    _write_json(manifest_path, manifest)


def _append_completed_step(
    run_dir: Path,
    *,
    step_id: int,
    response_model: str,
) -> None:
    episode_dir = run_dir / "episode_0000"
    trajectory_path = episode_dir / "trajectory.jsonl"
    transition = json.loads(
        trajectory_path.read_text(encoding="utf-8").splitlines()[0]
    )
    transition["step_id"] = step_id
    transition["agent_trace"]["attempts"] = 1
    transition["agent_trace"]["artifact_dir"] = f"model_calls/step_{step_id:04d}"
    with trajectory_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(transition) + "\n")

    request = json.loads(
        (episode_dir / "model_calls/step_0000/request.json").read_text(
            encoding="utf-8"
        )
    )
    request["images"] = []
    call_dir = episode_dir / f"model_calls/step_{step_id:04d}"
    _write_json(call_dir / "request.json", request)
    _write_json(call_dir / "response.json", {"model": response_model})

    episode_summary_path = episode_dir / "summary.json"
    episode_summary = json.loads(episode_summary_path.read_text(encoding="utf-8"))
    episode_summary["steps"] = step_id + 1
    _write_json(episode_summary_path, episode_summary)
    run_summary_path = run_dir / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    run_summary["episodes"][0]["steps"] = step_id + 1
    _write_json(run_summary_path, run_summary)


def _set_simulator_type(run_dir: Path, simulator_type: str) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["simulator"] = {"type": simulator_type}
    _write_json(manifest_path, manifest)


def _set_rlbench_action_evidence(
    run_dir: Path,
    *,
    step_id: int = 0,
    rejected: object = False,
    sim_action: object | None = None,
) -> None:
    trajectory_path = run_dir / "episode_0000/trajectory.jsonl"
    transitions = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    transition = transitions[step_id]
    transition["sim_action"] = sim_action or {
        "delta_position": [0.01, -0.02, 0.03],
        "delta_rotation_rpy": [0.0, 0.0, 0.0],
        "gripper": 1.0,
        "metadata": {"controller": "rlbench_delta_eef"},
    }
    transition["info"] = {"action_rejected": rejected}
    trajectory_path.write_text(
        "".join(json.dumps(item) + "\n" for item in transitions),
        encoding="utf-8",
    )


def test_summary_report_passes_and_counts_retries(monkeypatch, tmp_path):
    run_dir, _image_path = _run_fixture(tmp_path, attempts=3)
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "pass"
    assert report["counts"]["steps"] == 1
    assert report["counts"]["attempts"] == 3
    assert report["counts"]["retries"] == 2
    assert report["counts"]["agent_trace_statuses"] == {"ok": 1}
    assert "steps" not in report["episodes"][0]
    assert report["episodes"][0]["counts"]["valid_steps"] == 1
    assert report["checks"]["prompt_privacy"] == {
        "status": "pass",
        "scanned_requests": 1,
        "findings": [],
    }
    assert report["checks"]["response_model"] == {
        "status": "legacy_unreported",
        "requested": None,
        "observed": [],
        "missing_count": 0,
        "mismatch_paths": [],
    }
    assert report["anomalies"] == []


def test_invalid_validation_and_non_ok_trace_fail_but_retries_are_counted(
    monkeypatch,
    tmp_path,
):
    run_dir, _image_path = _run_fixture(
        tmp_path,
        valid=False,
        trace_status="error",
        attempts=4,
    )
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "fail"
    assert report["counts"]["invalid_steps"] == 1
    assert report["counts"]["retries"] == 3
    assert report["counts"]["agent_trace_statuses"] == {"error": 1}
    assert {"validation_not_valid", "agent_trace_not_ok"} <= {
        item["code"] for item in report["anomalies"]
    }


def test_requested_episode_count_mismatch_is_incomplete(monkeypatch, tmp_path):
    run_dir, _image_path = _run_fixture(tmp_path, requested_episodes=2)
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "incomplete"
    assert report["complete"] is False
    assert report["counts"]["requested_episodes"] == 2
    assert report["counts"]["summarized_episodes"] == 1
    assert "requested_episode_count_mismatch" in {
        item["code"] for item in report["anomalies"]
    }


def test_zero_step_episode_is_not_counted_as_model_trajectory(monkeypatch, tmp_path):
    run_dir, _image_path = _run_fixture(tmp_path)
    episode_dir = run_dir / "episode_0000"
    episode_summary = json.loads(
        (episode_dir / "summary.json").read_text(encoding="utf-8")
    )
    episode_summary["steps"] = 0
    _write_json(episode_dir / "summary.json", episode_summary)
    run_summary = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    run_summary["episodes"][0]["steps"] = 0
    _write_json(run_dir / "run_summary.json", run_summary)
    (episode_dir / "trajectory.jsonl").write_text("", encoding="utf-8")
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "fail"
    assert report["counts"]["steps"] == 0
    assert "episode_has_no_model_steps" in {
        item["code"] for item in report["anomalies"]
    }


def test_prompt_privacy_checks_exact_keys_and_hidden_control_names(
    monkeypatch,
    tmp_path,
):
    run_dir, _image_path = _run_fixture(tmp_path)
    request_path = run_dir / "episode_0000/model_calls/step_0000/request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    hidden_names = {
        "permuted_sign_flipped",
        "random_rotation_2d",
        "random_rotation_3d",
        "rotation_matrix",
        "quaternion_xyzw",
    }
    privileged_keys = {
        "reward",
        "success",
        "seed",
        "sim_action",
        "simulator_action",
        "distance_to_target",
        "eef_pos",
        "target_position",
        "robot_proprio",
        "native_action",
        "benchmark_applied_native_action",
        "benchmark_absolute_gripper",
        "target_qpos",
    }
    request["request"]["messages"].append(
        {
            "role": "system",
            "content": (
                "Internal names: "
                + " ".join(sorted(hidden_names))
                + ". Ordinary target text is public."
            ),
        }
    )
    request["request"]["metadata"] = {
        key: [0, 0, 0] for key in privileged_keys
    }
    _write_json(request_path, request)
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "fail"
    findings = report["checks"]["prompt_privacy"]["findings"]
    assert {(item["kind"], item["name"]) for item in findings} == (
        {("hidden_control_name", name) for name in hidden_names}
        | {("privileged_key", key) for key in privileged_keys}
    )
    assert all(item["name"] != "target" for item in findings)


def test_simpler_native_action_audit_is_required_and_fail_loud(
    monkeypatch,
    tmp_path,
):
    run_dir, _image_path = _run_fixture(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark"]["name"] = "simpler_env_fixture"
    _write_json(manifest_path, manifest)

    trajectory_path = run_dir / "episode_0000/trajectory.jsonl"
    transition = json.loads(trajectory_path.read_text(encoding="utf-8"))
    transition["sim_action"] = {
        "values": [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    }
    transition["info"] = {
        "absolute_gripper_target": True,
        "benchmark_applied_native_action": {
            "shape": [7],
            "values": [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "benchmark_absolute_gripper": {
            "controller_command": -1.0,
            "applied_env_command": 0.0,
            "target_update": "open",
            "target_qpos": [0.0, 0.0],
        },
    }
    trajectory_path.write_text(
        json.dumps(transition) + "\n",
        encoding="utf-8",
    )
    _mock_ffprobe(monkeypatch)

    clean = audit_mllm_run.audit_run(run_dir)

    assert clean["status"] == "pass"
    assert clean["checks"]["native_actions"] == {
        "required": True,
        "status": "pass",
        "checked_steps": 1,
        "rejected_steps": 0,
        "invalid_steps": 0,
    }

    transition["info"]["benchmark_applied_native_action"]["values"][6] = 1.0
    trajectory_path.write_text(
        json.dumps(transition) + "\n",
        encoding="utf-8",
    )

    invalid = audit_mllm_run.audit_run(run_dir)

    assert invalid["status"] == "fail"
    assert invalid["checks"]["native_actions"]["invalid_steps"] == 1
    assert "native_gripper_command_not_zero" in {
        item["code"] for item in invalid["anomalies"]
    }


@pytest.mark.parametrize("simulator_type", ["rlbench", "colosseum"])
def test_rlbench_format_action_audit_reports_accepted_and_rejected_steps(
    monkeypatch,
    tmp_path,
    simulator_type,
):
    run_dir, _image_path = _run_fixture(tmp_path)
    _set_simulator_type(run_dir, simulator_type)
    _set_rlbench_action_evidence(run_dir)
    _append_completed_step(run_dir, step_id=1, response_model="test-model")
    _set_rlbench_action_evidence(run_dir, step_id=1, rejected=True)
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    # One accepted action proves the episode exercised the simulator motion path.
    expected = {
        "required": True,
        "status": "pass",
        "checked_steps": 2,
        "rejected_steps": 1,
        "invalid_steps": 0,
    }
    assert report["status"] == "pass"
    assert report["checks"]["native_actions"] == expected
    assert report["episodes"][0]["native_action"] == expected


def test_rlbench_action_audit_fails_loud_when_evidence_is_missing(
    monkeypatch,
    tmp_path,
):
    run_dir, _image_path = _run_fixture(tmp_path)
    _set_simulator_type(run_dir, "rlbench")
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "fail"
    assert report["checks"]["native_actions"] == {
        "required": True,
        "status": "fail",
        "checked_steps": 1,
        "rejected_steps": 0,
        "invalid_steps": 1,
    }
    assert {"sim_action_missing", "sim_action_info_missing"} <= {
        item["code"] for item in report["anomalies"]
    }


def test_colosseum_action_audit_rejects_non_replayable_values(
    monkeypatch,
    tmp_path,
):
    run_dir, _image_path = _run_fixture(tmp_path)
    _set_simulator_type(run_dir, "colosseum")
    _set_rlbench_action_evidence(
        run_dir,
        rejected="false",
        sim_action={
            "delta_position": [0.0, 0.0],
            "delta_rotation_rpy": [0.0, float("inf"), 0.0],
            "gripper": 1.1,
            "metadata": {"controller": "other_controller"},
        },
    )
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    # Offline audit evidence must be numeric and unambiguous enough to replay.
    assert report["status"] == "fail"
    assert report["checks"]["native_actions"]["invalid_steps"] == 1
    assert {
        "sim_action_delta_position_invalid",
        "sim_action_delta_rotation_rpy_invalid",
        "sim_action_gripper_invalid",
        "sim_action_controller_invalid",
        "sim_action_rejection_invalid",
    } <= {item["code"] for item in report["anomalies"]}


def test_all_rejected_rlbench_actions_fail_without_counting_invalid_evidence(
    monkeypatch,
    tmp_path,
):
    run_dir, _image_path = _run_fixture(tmp_path)
    _set_simulator_type(run_dir, "rlbench")
    _set_rlbench_action_evidence(run_dir, rejected=True)
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "fail"
    assert report["checks"]["native_actions"] == {
        "required": True,
        "status": "fail",
        "checked_steps": 1,
        "rejected_steps": 1,
        "invalid_steps": 0,
    }
    assert "episode_all_simulator_actions_rejected" in {
        item["code"] for item in report["anomalies"]
    }


def test_image_tampering_black_screen_and_secret_leaks_fail(
    monkeypatch,
    tmp_path,
):
    run_dir, image_path = _run_fixture(tmp_path)
    _png(image_path, black=True)
    _write_json(
        run_dir / "episode_0000/model_calls/step_0000/response.json",
        {
            "Authorization": "Bearer SECRET_AUTHORIZATION_123",
            "api_key": "SECRET_API_KEY_123",
            "image": "data:image/png;base64,AAAA",
        },
    )
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "fail"
    assert len(report["checks"]["images"]["hash_mismatch_paths"]) == 1
    assert len(report["checks"]["images"]["black_screens"]) == 1
    assert report["checks"]["secrets"]["counts"] == {
        "authorization": 1,
        "api_key": 1,
        "base64_data_url": 1,
    }
    serialized = json.dumps(report)
    assert "SECRET_AUTHORIZATION_123" not in serialized
    assert "SECRET_API_KEY_123" not in serialized


def test_running_directory_is_incomplete(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "run_manifest.json",
        {"benchmark": {"num_episodes": 2}},
    )
    _make_formal_run(run_dir)
    (run_dir / "episode_0000").mkdir(parents=True)
    monkeypatch.setattr(audit_mllm_run.shutil, "which", lambda _name: None)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "incomplete"
    assert report["complete"] is False
    assert {"run_summary_missing", "summary_missing", "trajectory_missing"} <= {
        item["code"] for item in report["anomalies"]
    }
    assert report["checks"]["response_model"] == {
        "status": "pass",
        "requested": "test-model",
        "observed": [],
        "missing_count": 0,
        "mismatch_paths": [],
    }
    assert not {
        "response_model_missing",
        "response_model_mismatch",
        "response_model_mixed",
    } & {item["code"] for item in report["anomalies"]}


def test_in_progress_unwritten_response_is_not_a_model_mismatch(
    monkeypatch,
    tmp_path,
):
    run_dir, _image_path = _run_fixture(tmp_path)
    _make_formal_run(run_dir)
    (run_dir / "run_summary.json").unlink()
    (run_dir / "episode_0000/summary.json").unlink()
    (run_dir / "episode_0000/model_calls/step_0000/response.json").unlink()
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "incomplete"
    assert report["checks"]["response_model"] == {
        "status": "pass",
        "requested": "test-model",
        "observed": [],
        "missing_count": 0,
        "mismatch_paths": [],
    }
    assert "model_response_missing" in {
        item["code"] for item in report["anomalies"]
    }
    assert not {
        "response_model_missing",
        "response_model_mismatch",
        "response_model_mixed",
    } & {item["code"] for item in report["anomalies"]}


def test_schema_v2_requires_clean_source_revision(monkeypatch, tmp_path):
    run_dir, _image_path = _run_fixture(tmp_path)
    _mock_ffprobe(monkeypatch)
    _make_formal_run(run_dir)

    clean = audit_mllm_run.audit_run(run_dir)

    assert clean["status"] == "pass"
    assert clean["checks"]["source_revision"] == {
        "status": "pass",
        "capture_status": "reported",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "config_path": "configs/test.yaml",
    }
    assert clean["checks"]["response_model"] == {
        "status": "pass",
        "requested": "test-model",
        "observed": ["test-model"],
        "missing_count": 0,
        "mismatch_paths": [],
    }

    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest["source"]["git_dirty"] = True
    _write_json(run_dir / "run_manifest.json", manifest)

    dirty = audit_mllm_run.audit_run(run_dir)

    assert dirty["status"] == "fail"
    assert "source_tree_dirty" in {
        item["code"] for item in dirty["anomalies"]
    }


def test_formal_run_requires_requested_model(monkeypatch, tmp_path):
    run_dir, _image_path = _run_fixture(tmp_path)
    _make_formal_run(run_dir, requested_model=None)
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "fail"
    assert report["checks"]["response_model"] == {
        "status": "fail",
        "requested": None,
        "observed": ["test-model"],
        "missing_count": 0,
        "mismatch_paths": [],
    }
    assert "requested_model_missing" in {
        item["code"] for item in report["anomalies"]
    }


def test_response_model_field_is_required_for_formal_runs(monkeypatch, tmp_path):
    run_dir, _image_path = _run_fixture(tmp_path)
    _make_formal_run(run_dir)
    _write_json(
        run_dir / "episode_0000/model_calls/step_0000/response.json",
        {"id": "sanitized-response"},
    )
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "fail"
    assert report["checks"]["response_model"] == {
        "status": "fail",
        "requested": "test-model",
        "observed": [],
        "missing_count": 1,
        "mismatch_paths": [],
    }
    assert "response_model_missing" in {
        item["code"] for item in report["anomalies"]
    }


def test_response_models_prevent_compatible_endpoint_routing_drift(
    monkeypatch,
    tmp_path,
):
    run_dir, _image_path = _run_fixture(tmp_path)
    _make_formal_run(run_dir)
    response_path = (
        run_dir / "episode_0000/model_calls/step_0000/response.json"
    )
    _write_json(response_path, {"model": "compat-endpoint-routed-model"})
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    # The endpoint response, not a compatible alias, proves which model served the run.
    mismatch_path = "episode_0000/model_calls/step_0000/response.json"
    assert report["status"] == "fail"
    assert report["checks"]["response_model"] == {
        "status": "fail",
        "requested": "test-model",
        "observed": ["compat-endpoint-routed-model"],
        "missing_count": 0,
        "mismatch_paths": [mismatch_path],
    }
    assert "response_model_mismatch" in {
        item["code"] for item in report["anomalies"]
    }
    assert "response_model_mixed" not in {
        item["code"] for item in report["anomalies"]
    }


def test_mixed_response_models_fail_formal_run(monkeypatch, tmp_path):
    run_dir, _image_path = _run_fixture(tmp_path)
    _make_formal_run(run_dir)
    _append_completed_step(
        run_dir,
        step_id=1,
        response_model="compat-endpoint-routed-model",
    )
    _mock_ffprobe(monkeypatch)

    report = audit_mllm_run.audit_run(run_dir)

    mismatch_path = "episode_0000/model_calls/step_0001/response.json"
    assert report["status"] == "fail"
    assert report["checks"]["response_model"] == {
        "status": "fail",
        "requested": "test-model",
        "observed": ["compat-endpoint-routed-model", "test-model"],
        "missing_count": 0,
        "mismatch_paths": [mismatch_path],
    }
    assert {"response_model_mismatch", "response_model_mixed"} <= {
        item["code"] for item in report["anomalies"]
    }


def test_ffprobe_unavailable_and_wrong_streams_are_not_passed(
    monkeypatch,
    tmp_path,
):
    run_dir, _image_path = _run_fixture(tmp_path)
    monkeypatch.setattr(audit_mllm_run.shutil, "which", lambda _name: None)
    unavailable = audit_mllm_run.audit_run(run_dir)
    assert unavailable["status"] == "unavailable"
    assert unavailable["checks"]["model_view"]["status"] == "unavailable"

    _mock_ffprobe(
        monkeypatch,
        [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 640,
                "height": 360,
                "avg_frame_rate": "30/1",
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    )
    invalid = audit_mllm_run.audit_run(run_dir)
    assert invalid["status"] == "fail"
    assert invalid["episodes"][0]["review"]["observed"]["audio_streams"] == 1
