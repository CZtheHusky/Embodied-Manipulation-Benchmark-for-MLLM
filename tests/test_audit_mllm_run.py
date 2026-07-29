from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image

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
    (run_dir / "episode_0000").mkdir(parents=True)
    monkeypatch.setattr(audit_mllm_run.shutil, "which", lambda _name: None)

    report = audit_mllm_run.audit_run(run_dir)

    assert report["status"] == "incomplete"
    assert report["complete"] is False
    assert {"run_summary_missing", "summary_missing", "trajectory_missing"} <= {
        item["code"] for item in report["anomalies"]
    }


def test_schema_v2_requires_clean_source_revision(monkeypatch, tmp_path):
    run_dir, _image_path = _run_fixture(tmp_path)
    _mock_ffprobe(monkeypatch)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "benchmark": {"num_episodes": 1},
            "source": {
                "status": "reported",
                "git_commit": "a" * 40,
                "git_dirty": False,
                "config_path": "configs/test.yaml",
            },
        },
    )

    clean = audit_mllm_run.audit_run(run_dir)

    assert clean["status"] == "pass"
    assert clean["checks"]["source_revision"] == {
        "status": "pass",
        "capture_status": "reported",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "config_path": "configs/test.yaml",
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
