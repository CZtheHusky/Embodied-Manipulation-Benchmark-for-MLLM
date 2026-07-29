from __future__ import annotations

import json
from pathlib import Path
import shutil
import threading
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import pytest

from mllm_teleop_benchmark.review.server import STATIC_DIR, build_server
from mllm_teleop_benchmark.review.store import (
    EpisodeReviewStore,
    ReviewCatalogStore,
    ReviewDataError,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _episode(root: Path, name: str = "episode_0000") -> Path:
    episode = root / name
    _write_json(
        episode / "summary.json",
        {
            "episode_id": 0,
            "seed": 7,
            "success": False,
            "steps": 2,
            "termination_reason": "max_steps",
            "review_video": {"status": "completed"},
        },
    )
    transitions = []
    for step_id in range(2):
        artifact = f"model_calls/images/step_{step_id:04d}/front.png"
        image_path = episode / artifact
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"png-{step_id}".encode())
        messages = [
            {"role": "system", "content": "SYSTEM_PROMPT"},
            {"role": "user", "content": "STATIC_TASK"},
        ]
        if step_id:
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Observation step 0."},
                            {
                                "type": "text",
                                "text": "[image omitted by history policy]",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": '{"channels":{"joystick_0":0}}',
                    },
                ]
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Observation step {step_id}."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;sha256=hash-{step_id}"},
                    },
                ],
            }
        )
        artifact_dir = episode / "model_calls" / f"step_{step_id:04d}"
        _write_json(
            artifact_dir / "request.json",
            {
                "prompt_version": "mllm_joystick_v1",
                "common_prefix_bytes": step_id * 100,
                "request": {
                    "model": "fake-model",
                    "messages": messages,
                },
                "images": [
                    {
                        "step_id": step_id,
                        "camera": "front",
                        "width": 16,
                        "height": 12,
                        "sha256": f"hash-{step_id}",
                        "artifact": artifact,
                    }
                ],
            },
        )
        response_content = f'{{"channels":{{"joystick_0":{step_id}}}}}'
        _write_json(
            artifact_dir / "response.json",
            {
                "id": f"response-{step_id}",
                "choices": [{"message": {"content": response_content}}],
            },
        )
        transitions.append(
            {
                "step_id": step_id,
                "raw_action": response_content,
                "parsed_action": {"joystick_0": float(step_id)},
                "reflection": f"reflection-{step_id}",
                "validation": {"is_valid": True, "errors": []},
                "agent_trace": {
                    "actor_type": "mllm",
                    "status": "ok",
                    "model": "fake-model",
                    "latency_ms": 5.0,
                },
                "state": {"PRIVATE_STATE": "MUST_NOT_LEAK"},
                "info": {"PRIVATE_INFO": "MUST_NOT_LEAK"},
            }
        )
    with (episode / "trajectory.jsonl").open("w", encoding="utf-8") as handle:
        for transition in transitions:
            handle.write(json.dumps(transition) + "\n")
    return episode


def _responses_episode(root: Path) -> Path:
    episode = _episode(root)
    transitions = []
    for step_id in range(2):
        artifact_dir = episode / "model_calls" / f"step_{step_id:04d}"
        request_path = artifact_dir / "request.json"
        wrapper = json.loads(request_path.read_text(encoding="utf-8"))
        wrapper.update(
            {
                "api": "responses",
                "prompt_version": "mllm_joystick_responses_v1",
                "current_input_index": 0,
                "request": {
                    "model": "fake-model",
                    "instructions": "RESPONSES_INSTRUCTIONS",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"Observation step {step_id}.",
                                },
                                {
                                    "type": "input_image",
                                    "image_url": (
                                        f"data:image/png;sha256=hash-{step_id}"
                                    ),
                                },
                            ],
                        }
                    ],
                    "previous_response_id": (
                        f"response-{step_id - 1}" if step_id else None
                    ),
                    "store": True,
                },
            }
        )
        _write_json(request_path, wrapper)
        response_content = f'{{"channels":{{"joystick_0":{step_id}}}}}'
        _write_json(
            artifact_dir / "response.json",
            {
                "id": f"response-{step_id}",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": response_content}
                        ],
                    }
                ],
            },
        )
        transitions.append(
            {
                "step_id": step_id,
                "raw_action": response_content,
                "parsed_action": {"joystick_0": float(step_id)},
                "validation": {"is_valid": True, "errors": []},
                "agent_trace": {
                    "actor_type": "mllm",
                    "status": "ok",
                    "api": "responses",
                    "model": "fake-model",
                    "previous_response_id": (
                        f"response-{step_id - 1}" if step_id else None
                    ),
                    "response_id": f"response-{step_id}",
                },
            }
        )
    with (episode / "trajectory.jsonl").open("w", encoding="utf-8") as handle:
        for transition in transitions:
            handle.write(json.dumps(transition) + "\n")
    return episode


def test_store_indexes_run_and_preserves_prompt_contract(tmp_path):
    _episode(tmp_path)

    index = EpisodeReviewStore(tmp_path).index()

    assert index["name"] == tmp_path.name
    assert len(index["episodes"]) == 1
    episode = index["episodes"][0]
    assert episode["id"] == "episode_0000"
    assert episode["prompt_version"] == "mllm_joystick_v1"
    assert episode["step_ids"] == [0, 1]
    assert "seed" not in episode
    assert episode["step_entries"] == [
        {"id": 0, "status": "ok", "valid": True},
        {"id": 1, "status": "ok", "valid": True},
    ]


def test_catalog_discovers_nested_runs_and_keeps_run_artifacts_isolated(tmp_path):
    current_run = tmp_path / "model-current" / "20260728-current"
    archived_run = (
        tmp_path
        / "archive"
        / "gpt-5.4"
        / "model-legacy"
        / "20260727-legacy"
    )
    partial_run = tmp_path / "interrupted" / "user-paused" / "20260728-partial"
    _episode(current_run)
    _episode(archived_run)
    partial_episode = _episode(partial_run)
    (partial_episode / "summary.json").unlink()
    _write_json(
        current_run / "run_manifest.json",
        {
            "model": {
                "requested_model": "gpt-5.6-sol",
                "reasoning_effort": "low",
                "api": "responses",
            }
        },
    )
    _write_json(
        current_run / "run_summary.json",
        {"aggregate_metrics": {"success_rate": 0.8}},
    )
    _write_json(
        tmp_path / "not-reviewable" / "episode_0000" / "summary.json",
        {"episode_id": 0},
    )

    catalog = ReviewCatalogStore(tmp_path)
    index = catalog.index()

    assert [group["id"] for group in index["groups"]] == [
        "model-current",
        "archive/gpt-5.4/model-legacy",
        "interrupted/user-paused",
    ]
    assert index["default_group_id"] == "model-current"
    current = index["groups"][0]["runs"][0]
    assert current == {
        "id": "model-current/20260728-current",
        "name": "20260728-current",
        "relative_path": "model-current/20260728-current",
        "episode_count": 1,
        "aggregate_metrics": {"success_rate": 0.8},
        "model": "gpt-5.6-sol",
        "reasoning_effort": "low",
        "api": "responses",
    }
    run_index = catalog.run_index(current["id"])
    assert run_index["episodes"][0]["id"] == "episode_0000"
    step = catalog.step(current["id"], "episode_0000", 0)
    assert step["images"][0]["url"].startswith(
        "/artifacts/runs/model-current%2F20260728-current/episode_0000/"
    )
    image = catalog.artifact(
        current["id"],
        "episode_0000",
        "model_calls/images/step_0000/front.png",
    )
    assert image.read_bytes() == b"png-0"
    with pytest.raises(ReviewDataError, match="Unknown review run"):
        catalog.run_index("../outside")
    partial_index = catalog.run_index(
        "interrupted/user-paused/20260728-partial"
    )
    assert partial_index["episodes"][0]["termination_reason"] == "incomplete"
    assert partial_index["episodes"][0]["step_entries"][0] == {
        "id": 0,
        "status": "ok",
        "valid": True,
    }


def test_catalog_refreshes_when_runs_are_added_or_removed(tmp_path):
    original = tmp_path / "model-original" / "20260728-original"
    replacement = tmp_path / "model-replacement" / "20260728-replacement"
    _episode(original)
    catalog = ReviewCatalogStore(tmp_path)

    _episode(replacement)
    shutil.rmtree(original)
    index = catalog.index()

    assert [group["id"] for group in index["groups"]] == ["model-replacement"]
    assert index["default_run_id"] == (
        "model-replacement/20260728-replacement"
    )
    assert catalog.run_index(index["default_run_id"])["episodes"][0]["id"] == (
        "episode_0000"
    )


def test_catalog_http_routes_select_run_episode_step_and_allowlisted_image(tmp_path):
    run = tmp_path / "model-current" / "20260728-current"
    _episode(run)
    catalog = ReviewCatalogStore(tmp_path)
    server = build_server(catalog, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    run_id = quote("model-current/20260728-current", safe="")
    try:
        with urlopen(f"{base_url}/api/index") as response:
            index = json.load(response)
        with urlopen(f"{base_url}/api/runs/{run_id}") as response:
            run_index = json.load(response)
        with urlopen(
            f"{base_url}/api/runs/{run_id}/episodes/episode_0000/steps/0"
        ) as response:
            step = json.load(response)
        image_url = step["images"][0]["url"]
        with urlopen(f"{base_url}{image_url}") as response:
            image = response.read()
        with pytest.raises(HTTPError) as error:
            urlopen(f"{base_url}/api/runs/not-a-run")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert index["default_group_id"] == "model-current"
    assert run_index["episodes"][0]["id"] == "episode_0000"
    assert step["step_id"] == 0
    assert image == b"png-0"
    assert error.value.code == 404


def test_step_returns_exact_messages_images_and_raw_response_without_private_state(tmp_path):
    _episode(tmp_path)
    store = EpisodeReviewStore(tmp_path)

    step = store.step("episode_0000", 1)

    assert step["request"]["messages"][0] == {
        "role": "system",
        "content": "SYSTEM_PROMPT",
    }
    assert step["request"]["messages"][-1]["content"][0]["text"] == "Observation step 1."
    assert step["images"][0]["url"].endswith(
        "/model_calls/images/step_0001/front.png"
    )
    assert step["response_content"] == '{"channels":{"joystick_0":1}}'
    assert step["action"]["reflection"] == "reflection-1"
    serialized = json.dumps(step)
    assert "MUST_NOT_LEAK" not in serialized
    assert "state" not in step
    assert "info" not in step


def test_conversation_uses_each_current_observation_and_following_raw_response(tmp_path):
    _episode(tmp_path)

    conversation = EpisodeReviewStore(tmp_path).conversation("episode_0000")

    assert len(conversation["turns"]) == 2
    assert conversation["turns"][0]["observation"]["content"][0]["text"] == (
        "Observation step 0."
    )
    assert conversation["turns"][1]["response_content"] == (
        '{"channels":{"joystick_0":1}}'
    )


def test_responses_review_reconstructs_incremental_chain_without_rewriting_raw_request(
    tmp_path,
):
    _responses_episode(tmp_path)
    store = EpisodeReviewStore(tmp_path)

    step = store.step("episode_0000", 1)
    conversation = store.conversation("episode_0000")

    assert step["api"] == "responses"
    assert step["request"]["previous_response_id"] == "response-0"
    assert step["display_messages"][0] == {
        "role": "system",
        "content": "RESPONSES_INSTRUCTIONS",
    }
    assert step["current_observation"]["content"][0]["text"] == "Observation step 1."
    assert step["response_content"] == '{"channels":{"joystick_0":1}}'
    assert step["public_trace"]["response_id"] == "response-1"
    assert conversation["static_messages"] == step["display_messages"][:1]
    assert [turn["step_id"] for turn in conversation["turns"]] == [0, 1]
    assert conversation["turns"][0]["response_content"] == (
        '{"channels":{"joystick_0":0}}'
    )


def test_artifact_access_is_limited_to_model_visible_request_images(tmp_path):
    episode = _episode(tmp_path)
    store = EpisodeReviewStore(tmp_path)

    image = store.artifact(
        "episode_0000",
        "model_calls/images/step_0000/front.png",
    )

    assert image.read_bytes() == b"png-0"
    with pytest.raises(ReviewDataError, match="not a model-visible"):
        store.artifact("episode_0000", "trajectory.jsonl")
    with pytest.raises(ReviewDataError, match="stay within"):
        store.artifact("episode_0000", "../summary.json")
    assert episode.is_dir()


def test_review_static_assets_are_packaged_and_do_not_use_remote_dependencies():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert (STATIC_DIR / "styles.css").is_file()
    assert 'src="/static/app.js"' in html
    assert 'id="group-select"' in html
    assert 'id="run-select"' in html
    assert "http://" not in html
    assert "https://" not in html
    assert "trajectory.jsonl" not in javascript
    assert "/api/runs/" in javascript
