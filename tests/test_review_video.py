from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mllm_teleop_benchmark import BenchmarkRunner, load_config
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.schema import (
    AgentConfig,
    AgentHistoryConfig,
    AgentObservationConfig,
    ModelConfig,
    VideoConfig,
)
from mllm_teleop_benchmark.core.types import (
    AgentTrace,
    EpisodeResult,
    GymBoxAction,
    MagnifierView,
    Observation,
    StepTransition,
    ValidationResult,
)
from mllm_teleop_benchmark.logging.review_video import (
    CANVAS_SIZE,
    ReviewVideoError,
    ReviewVideoRenderer,
    ReviewVideoResult,
    evaluator_debug_record,
    model_review_record,
)
from mllm_teleop_benchmark.logging.episode_logger import EpisodeLogger
from mllm_teleop_benchmark.utils.images import encode_rgb_png


def _config(tmp_path: Path, *, debug: bool = False, cameras=("front",)):
    config = load_config("configs/reacher_delta_eef_mllm_responses.yaml")
    return replace(
        config,
        agent=replace(
            config.agent,
            observation=replace(config.agent.observation, cameras=tuple(cameras), image_max_side=32),
        ),
        logging=replace(
            config.logging,
            output_dir=tmp_path,
            save_video=True,
            video=VideoConfig(include_evaluator_debug=debug),
        ),
    )


def _observation(step_id: int, cameras=("front",)) -> Observation:
    images = {
        camera: np.full((18, 24, 3), 30 + index * 100 + step_id, dtype=np.uint8)
        for index, camera in enumerate(cameras)
    }
    return Observation(
        step_id=step_id,
        images=images,
        state={
            "reward": -0.25,
            "distance_to_target": 0.25,
            "target_position": [1.0, 2.0, 3.0],
            "fingertip_position": [0.1, 0.2, 0.3],
            "PRIVATE_STATE": "PRIVATE_ORACLE_SENTINEL",
        },
        metadata={"PRIVATE_METADATA": "PRIVATE_METADATA_SENTINEL"},
    )


def _transition(step_id: int, cameras=("front",), *, valid: bool = True) -> StepTransition:
    return StepTransition(
        step_id=step_id,
        observation=_observation(step_id, cameras),
        raw_action='{"channels":{"joystick_0":1,"joystick_1":0}}',
        parsed_action={"joystick_0": 1.0, "joystick_1": 0.0} if valid else None,
        validation=ValidationResult(valid, () if valid else ("invalid model JSON",)),
        sim_action=GymBoxAction(values=(1.0, 0.0), metadata={"controller": "test"}),
        next_observation=_observation(step_id + 1, cameras),
        terminated=False,
        truncated=False,
        info={
            "reward": -0.25,
            "success": False,
            "PRIVATE_INFO": "PRIVATE_INFO_SENTINEL",
        },
        agent_trace=AgentTrace(
            actor_type="mllm",
            status="ok",
            model="fake-model",
            latency_ms=4.5,
            attempts=1,
            input_tokens=100,
            output_tokens=8,
            cached_input_tokens=20,
            cache_hit_ratio=0.2,
            cache_metrics_status="reported",
            artifact_dir=f"model_calls/step_{step_id:04d}",
        ),
    )


def _episode_result(tmp_path: Path, steps: int = 1) -> EpisodeResult:
    return EpisodeResult(
        episode_id=0,
        seed=7,
        success=False,
        steps=steps,
        metrics={"success": False},
        log_dir=str(tmp_path),
        termination_reason="max_steps",
    )


def _metadata():
    return {
        "task_instruction": "Move to the visible target.",
        "control_interface": {
            "mode": "eef_delta_pose",
            "active_dimensions": ["dx", "dy"],
            "gripper": "none",
            "mapping": {"channels": {"joystick_0": "dx", "joystick_1": "dy"}},
        },
    }


def _write_request_artifacts(
    episode_dir: Path,
    transition: StepTransition,
    cameras=("front",),
) -> dict[str, str]:
    images = []
    hashes = {}
    for camera in cameras:
        encoded = encode_rgb_png(transition.observation.images[camera], 32)
        relative = f"model_calls/images/step_{transition.step_id:04d}/{camera}.png"
        target = episode_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded.content)
        hashes[camera] = encoded.sha256
        images.append(
            {
                "step_id": transition.step_id,
                "camera": camera,
                "width": encoded.width,
                "height": encoded.height,
                "sha256": encoded.sha256,
                "artifact": relative,
                "kind": "rgb",
            }
        )
    if transition.observation.magnifier is not None:
        view = transition.observation.magnifier
        encoded = encode_rgb_png(view.image, 32)
        relative = (
            f"model_calls/images/step_{transition.step_id:04d}/magnifier/"
            f"{view.source_rgb_key}.png"
        )
        target = episode_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded.content)
        hashes["magnifier"] = encoded.sha256
        images.append(
            {
                "step_id": transition.step_id,
                "camera": f"magnifier:{view.source_rgb_key}",
                "width": encoded.width,
                "height": encoded.height,
                "sha256": encoded.sha256,
                "artifact": relative,
                "kind": "magnifier",
                "source_rgb_key": view.source_rgb_key,
                "bbox_xyxy": list(view.bbox_xyxy),
            }
        )
    request_path = episode_dir / f"model_calls/step_{transition.step_id:04d}/request.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps({"images": images}), encoding="utf-8")
    return hashes


def _fake_encoder(captured):
    def encode(_self, target, frames):
        rendered = list(frames)
        captured[target.name] = rendered
        target.write_bytes(b"fake-mp4")

    return encode


def test_model_review_projection_excludes_oracle_and_debug_is_whitelisted():
    transition = _transition(0)
    transition = replace(
        transition,
        agent_trace=replace(
            transition.agent_trace,
            error="Authorization: Bearer PRIVATE_API_KEY_SENTINEL",
        ),
    )

    model_record = model_review_record(transition, _metadata())
    debug_record = evaluator_debug_record(transition)
    model_text = json.dumps(model_record, sort_keys=True)

    assert "PRIVATE_ORACLE_SENTINEL" not in model_text
    assert "PRIVATE_METADATA_SENTINEL" not in model_text
    assert "PRIVATE_INFO_SENTINEL" not in model_text
    assert "PRIVATE_API_KEY_SENTINEL" not in model_text
    assert "simulator_action" not in model_text
    assert set(debug_record) == {
        "reward",
        "distance",
        "success",
        "target_position",
        "eef_position",
        "simulator_action",
        "simulator_substeps",
    }
    assert "PRIVATE_INFO_SENTINEL" not in json.dumps(debug_record, sort_keys=True)


def test_renderer_pairs_current_images_in_camera_order_and_writes_manifest(
    monkeypatch,
    tmp_path,
):
    cameras = ("wrist", "front")
    transition = _transition(0, cameras)
    original_hashes = _write_request_artifacts(tmp_path, transition, cameras)
    config = _config(tmp_path, debug=True, cameras=cameras)
    renderer = ReviewVideoRenderer(config, tmp_path)
    captured = {}
    monkeypatch.setattr(ReviewVideoRenderer, "_ffmpeg_version", lambda _self: "ffmpeg test")
    monkeypatch.setattr(ReviewVideoRenderer, "_encode_video", _fake_encoder(captured))

    result = renderer.render(
        (transition,),
        transition.next_observation,
        _metadata(),
        _episode_result(tmp_path),
    )

    manifest = json.loads((tmp_path / "review/manifest.json").read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert result.model_view == "review/model_view.mp4"
    assert result.evaluator_debug == "review/evaluator_debug.mp4"
    assert [item["camera"] for item in manifest["steps"][0]["images"]] == list(cameras)
    assert manifest["step_budget_unit"] == "agent_command"
    assert manifest["max_agent_steps"] == config.task.max_steps
    assert manifest["agent_steps"] == 1
    assert manifest["simulator_substeps"] == 1
    assert {
        item["camera"]: item["sha256"] for item in manifest["steps"][0]["images"]
    } == original_hashes
    assert manifest["terminal"]["step_id"] == 1
    assert set(captured) == {"model_view.mp4", "evaluator_debug.mp4"}
    assert all(frame.size == CANVAS_SIZE for frame, _duration in captured["model_view.mp4"])
    assert [duration for _frame, duration in captured["model_view.mp4"]] == [1.0, 2.0]


def test_renderer_includes_model_visible_magnifier_and_terminal_lens(
    monkeypatch,
    tmp_path,
):
    view = MagnifierView(
        source_rgb_key="front",
        bbox_xyxy=(0.25, 0.25, 0.75, 0.75),
        image=np.full((18, 18, 3), 200, dtype=np.uint8),
    )
    transition = _transition(0)
    transition = replace(
        transition,
        observation=replace(transition.observation, magnifier=view),
        next_observation=replace(transition.next_observation, magnifier=view),
    )
    _write_request_artifacts(tmp_path, transition)
    renderer = ReviewVideoRenderer(_config(tmp_path), tmp_path)
    captured = {}
    monkeypatch.setattr(ReviewVideoRenderer, "_ffmpeg_version", lambda _self: "ffmpeg test")
    monkeypatch.setattr(ReviewVideoRenderer, "_encode_video", _fake_encoder(captured))

    result = renderer.render(
        (transition,),
        transition.next_observation,
        _metadata(),
        _episode_result(tmp_path),
    )

    manifest = json.loads((tmp_path / "review/manifest.json").read_text())
    assert result.status == "completed"
    assert [item["kind"] for item in manifest["steps"][0]["images"]] == [
        "rgb",
        "magnifier",
    ]
    assert manifest["steps"][0]["images"][1]["source_rgb_key"] == "front"
    assert [item["kind"] for item in manifest["terminal"]["images"]] == [
        "rgb",
        "magnifier",
    ]
    assert (tmp_path / "review/terminal/magnifier/front.png").is_file()


@pytest.mark.parametrize("failure", ("missing", "hash"))
def test_source_integrity_failure_is_recorded_without_losing_episode_result(
    monkeypatch,
    tmp_path,
    failure,
):
    transition = _transition(0)
    _write_request_artifacts(tmp_path, transition)
    source = tmp_path / "model_calls/images/step_0000/front.png"
    if failure == "missing":
        source.unlink()
    else:
        source.write_bytes(b"tampered")
    renderer = ReviewVideoRenderer(_config(tmp_path), tmp_path)
    monkeypatch.setattr(ReviewVideoRenderer, "_ffmpeg_version", lambda _self: "ffmpeg test")
    episode = _episode_result(tmp_path)

    result = renderer.render((transition,), transition.next_observation, _metadata(), episode)

    assert result.status == "failed"
    assert episode.steps == 1
    assert (tmp_path / "review/video_error.json").exists()
    assert json.loads((tmp_path / "review/manifest.json").read_text())["status"] == "failed"


def test_missing_ffmpeg_and_encoding_failure_are_explicit(monkeypatch, tmp_path):
    transition = _transition(0)
    _write_request_artifacts(tmp_path, transition)
    missing = ReviewVideoRenderer(
        _config(tmp_path / "missing"),
        tmp_path,
        ffmpeg_binary="definitely-not-an-ffmpeg-binary",
    )

    missing_result = missing.render(
        (transition,),
        transition.next_observation,
        _metadata(),
        _episode_result(tmp_path),
    )

    assert missing_result.status == "failed"
    assert "not found" in json.dumps(missing_result.error).lower()

    monkeypatch.setattr(ReviewVideoRenderer, "_ffmpeg_version", lambda _self: "ffmpeg test")

    def fail_encode(_self, _target, _frames):
        raise ReviewVideoError("synthetic encoder failure")

    monkeypatch.setattr(ReviewVideoRenderer, "_encode_video", fail_encode)
    failed = ReviewVideoRenderer(_config(tmp_path), tmp_path).render(
        (transition,),
        transition.next_observation,
        _metadata(),
        _episode_result(tmp_path),
    )
    assert failed.status == "failed"
    assert "synthetic encoder failure" in json.dumps(failed.error)


def test_episode_logger_preserves_summary_when_review_video_fails(monkeypatch, tmp_path):
    config = _config(tmp_path)
    logger = EpisodeLogger(config, episode_id=0, seed=7, run_id="failed-video")
    logger.start(_metadata())
    episode = _episode_result(logger.episode_dir, steps=0)

    monkeypatch.setattr(
        ReviewVideoRenderer,
        "render",
        lambda *_args, **_kwargs: ReviewVideoResult(
            status="failed",
            model_view=None,
            evaluator_debug=None,
            manifest="review/manifest.json",
            error={"errors": [{"stage": "test", "message": "synthetic"}]},
        ),
    )

    with pytest.warns(RuntimeWarning, match="results and raw artifacts were preserved"):
        logger.finish(episode, terminal_observation=_observation(0))

    summary = json.loads((logger.episode_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["steps"] == 0
    assert summary["termination_reason"] == "max_steps"
    assert summary["review_video"]["status"] == "failed"


class _FakeClient:
    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        channels = ",".join(f'"joystick_{index}":0' for index in range(7))
        text = '{"channels":{' + channels + "}}"
        return ModelResponse(
            text=text,
            request_id=f"fake-{len(self.requests)}",
            model="fake-model",
            usage=ModelUsage(input_tokens=100, output_tokens=8, cached_input_tokens=20),
            raw={"id": f"fake-{len(self.requests)}", "choices": [{"message": {"content": text}}]},
            latency_ms=5.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


def test_fake_mllm_episode_generates_two_real_review_videos(monkeypatch, tmp_path):
    config = load_config("configs/dummy_reach.yaml")
    agent = AgentConfig(
        type="mllm",
        model=ModelConfig(
            name="fake-model",
            base_url="http://fake.test/v1",
            api_key_env="TEST_MLLM_API_KEY",
        ),
        observation=AgentObservationConfig(cameras=("front",), image_max_side=32),
        history=AgentHistoryConfig(action_window=0, image_window=0),
    )
    config = replace(
        config,
        benchmark=replace(config.benchmark, num_episodes=1),
        task=replace(config.task, max_steps=3),
        agent=agent,
        logging=replace(
            config.logging,
            output_dir=tmp_path,
            save_video=True,
            video=VideoConfig(include_evaluator_debug=True),
        ),
    )
    client = _FakeClient()
    monkeypatch.setenv("TEST_MLLM_API_KEY", "PRIVATE_KEY_SENTINEL")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )
    monkeypatch.setattr(
        "mllm_teleop_benchmark.sim.dummy.DummySimAdapter.render",
        lambda self, _camera: np.full((32, 32, 3), self.step_id, dtype=np.uint8),
    )

    episode = BenchmarkRunner(config).run().episodes[0]
    episode_dir = Path(episode.log_dir)
    summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))

    assert episode.steps == 3
    assert len(client.requests) == 3
    assert summary["review_video"]["status"] == "completed"
    assert (episode_dir / "review/model_view.mp4").stat().st_size > 0
    assert (episode_dir / "review/evaluator_debug.mp4").stat().st_size > 0
    terminal = episode_dir / "review/terminal/front.png"
    assert terminal.exists()
    terminal_manifest = json.loads(
        (episode_dir / "review/manifest.json").read_text(encoding="utf-8")
    )["terminal"]["images"][0]
    assert hashlib.sha256(terminal.read_bytes()).hexdigest() == terminal_manifest["sha256"]
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            str(episode_dir / "review/model_view.mp4"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    assert streams == [
        {
            "avg_frame_rate": "2/1",
            "codec_name": "h264",
            "codec_type": "video",
            "height": 720,
            "width": 1280,
        }
    ]
    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            assert b"PRIVATE_KEY_SENTINEL" not in artifact.read_bytes()
