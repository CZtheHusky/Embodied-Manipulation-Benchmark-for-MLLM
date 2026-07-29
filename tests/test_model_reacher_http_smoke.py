import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mllm_teleop_benchmark import BenchmarkRunner, load_config


pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")


class StubChatHandler(BaseHTTPRequestHandler):
    requests = []
    authorization_headers = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.__class__.requests.append(json.loads(self.rfile.read(length)))
        self.__class__.authorization_headers.append(self.headers.get("Authorization"))
        response = {
            "id": f"http-smoke-{len(self.__class__.requests)}",
            "model": "stub-vision-model",
            "choices": [
                {
                    "message": {
                        "content": '{"channels":{"joystick_0":0,"joystick_1":0}}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 8},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return None


class StubResponsesHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.__class__.requests.append(json.loads(self.rfile.read(length)))
        response = {
            "id": f"resp-{len(self.__class__.requests)}",
            "model": "stub-vision-model",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"channels":{"joystick_0":0,"joystick_1":0}}',
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 50, "output_tokens": 8},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return None


class RetryingResponsesHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.__class__.requests.append(json.loads(self.rfile.read(length)))
        if len(self.__class__.requests) == 1:
            payload = (
                'event: error\ndata: {"type":"error",'
                '"error":{"type":"server_error","message":"temporarily unavailable"}}\n\n'
            ).encode("utf-8")
            content_type = "text/event-stream"
        else:
            response = {
                "id": "resp-after-retry",
                "model": "stub-vision-model",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"channels":{"joystick_0":0,"joystick_1":0}}',
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 50, "output_tokens": 8},
            }
            payload = json.dumps(response).encode("utf-8")
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return None


class MissingPreviousResponsesHandler(BaseHTTPRequestHandler):
    requests = []
    success_count = 0

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.__class__.requests.append(request)
        if request.get("previous_response_id") == "resp-1":
            payload = json.dumps(
                {
                    "error": {
                        "message": (
                            "Previous response with id 'resp-1' not found."
                        )
                    }
                }
            ).encode("utf-8")
            self.send_response(400)
        else:
            self.__class__.success_count += 1
            response = {
                "id": f"resp-{self.__class__.success_count}",
                "model": "stub-vision-model",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"channels":{"joystick_0":0,"joystick_1":0}}',
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 50, "output_tokens": 8},
            }
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return None


def _request_image_count(request):
    return sum(
        1
        for message in request["messages"]
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "image_url"
    )


@pytest.mark.parametrize(
    ("image_window", "expected_image_counts"),
    [
        (0, [1, 1, 1, 1, 1]),
        (2, [1, 2, 3, 3, 3]),
        ("all", [1, 2, 3, 4, 5]),
    ],
)
def test_reacher_runs_five_steps_through_openai_compatible_http(monkeypatch, tmp_path, image_window, expected_image_counts):
    StubChatHandler.requests = []
    StubChatHandler.authorization_headers = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = load_config("configs/reacher_delta_eef_mllm.yaml")
        agent = replace(
            config.agent,
            model=replace(
                config.agent.model,
                name="stub-vision-model",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key_env="HTTP_SMOKE_API_KEY",
                require_api_key=True,
            ),
            history=replace(config.agent.history, image_window=image_window),
        )
        config = replace(
            config,
            task=replace(config.task, max_steps=5, success_threshold=-1.0),
            agent=agent,
            logging=replace(config.logging, output_dir=tmp_path),
        )
        monkeypatch.setenv("HTTP_SMOKE_API_KEY", "HTTP_SMOKE_SECRET_SENTINEL")

        episode = BenchmarkRunner(config).run().episodes[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert episode.steps == 5
    assert episode.termination_reason == "max_steps"
    assert len(StubChatHandler.requests) == 5
    assert [_request_image_count(request) for request in StubChatHandler.requests] == expected_image_counts
    assert StubChatHandler.authorization_headers == ["Bearer HTTP_SMOKE_SECRET_SENTINEL"] * 5
    assert episode.metrics["invalid_action_count"] == 0
    assert episode.metrics["model_call_count"] == 5
    assert episode.metrics["cache_metrics_status"] == "unavailable"

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert (episode_dir / "model_calls/step_0004/response.json").exists()
    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            assert b"HTTP_SMOKE_SECRET_SENTINEL" not in artifact.read_bytes()


def test_reacher_responses_api_retains_full_context_by_response_id(
    monkeypatch,
    tmp_path,
):
    StubResponsesHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubResponsesHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = load_config("configs/reacher_delta_eef_mllm.yaml")
        agent = replace(
            config.agent,
            model=replace(
                config.agent.model,
                api="responses",
                name="stub-vision-model",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key_env="HTTP_SMOKE_API_KEY",
                require_api_key=True,
            ),
            history=replace(
                config.agent.history,
                action_window="all",
                image_window="all",
            ),
        )
        config = replace(
            config,
            task=replace(config.task, max_steps=3, success_threshold=-1.0),
            agent=agent,
            logging=replace(config.logging, output_dir=tmp_path),
        )
        monkeypatch.setenv("HTTP_SMOKE_API_KEY", "HTTP_SMOKE_SECRET_SENTINEL")

        episode = BenchmarkRunner(config).run().episodes[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert episode.steps == 3
    assert episode.termination_reason == "max_steps"
    assert [
        request.get("previous_response_id")
        for request in StubResponsesHandler.requests
    ] == [None, "resp-1", "resp-2"]
    assert all(request["store"] is True for request in StubResponsesHandler.requests)
    assert all(
        sum(
            item.get("type") == "input_image"
            for message in request["input"]
            for item in message.get("content", ())
            if isinstance(item, dict)
        )
        == 1
        for request in StubResponsesHandler.requests
    )
    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [step["agent_trace"]["response_id"] for step in trajectory] == [
        "resp-1",
        "resp-2",
        "resp-3",
    ]
    assert all(step["agent_trace"]["api"] == "responses" for step in trajectory)


def test_reacher_retries_responses_error_before_stepping_simulator(
    monkeypatch,
    tmp_path,
):
    RetryingResponsesHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), RetryingResponsesHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = load_config("configs/reacher_delta_eef_mllm.yaml")
        agent = replace(
            config.agent,
            model=replace(
                config.agent.model,
                api="responses",
                name="stub-vision-model",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key_env="HTTP_SMOKE_API_KEY",
                require_api_key=True,
                max_retries=0,
                retry_backoff_seconds=0,
                retry_backoff_max_seconds=0,
                retry_until_success=True,
                stream=True,
            ),
            history=replace(
                config.agent.history,
                action_window="all",
                image_window="all",
            ),
        )
        config = replace(
            config,
            task=replace(config.task, max_steps=1, success_threshold=-1.0),
            agent=agent,
            logging=replace(config.logging, output_dir=tmp_path),
        )
        monkeypatch.setenv("HTTP_SMOKE_API_KEY", "HTTP_SMOKE_SECRET_SENTINEL")

        episode = BenchmarkRunner(config).run().episodes[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert len(RetryingResponsesHandler.requests) == 2
    assert RetryingResponsesHandler.requests[0] == RetryingResponsesHandler.requests[1]
    assert episode.steps == 1
    assert episode.termination_reason == "max_steps"
    assert episode.metrics["model_call_count"] == 1
    assert episode.metrics["model_retry_count"] == 1

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(trajectory) == 1
    assert trajectory[0]["validation"]["is_valid"]
    assert trajectory[0]["agent_trace"]["attempts"] == 2
    assert not (episode_dir / "model_calls/step_0000/error.json").exists()


def test_reacher_recovers_missing_previous_response_before_stepping_simulator(
    monkeypatch,
    tmp_path,
):
    MissingPreviousResponsesHandler.requests = []
    MissingPreviousResponsesHandler.success_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), MissingPreviousResponsesHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = load_config("configs/reacher_delta_eef_mllm.yaml")
        agent = replace(
            config.agent,
            model=replace(
                config.agent.model,
                api="responses",
                name="stub-vision-model",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key_env="HTTP_SMOKE_API_KEY",
                require_api_key=True,
                max_retries=0,
                retry_backoff_seconds=0,
                stream=False,
            ),
            history=replace(
                config.agent.history,
                action_window="all",
                image_window="all",
            ),
        )
        config = replace(
            config,
            task=replace(config.task, max_steps=2, success_threshold=-1.0),
            agent=agent,
            logging=replace(config.logging, output_dir=tmp_path),
        )
        monkeypatch.setenv("HTTP_SMOKE_API_KEY", "HTTP_SMOKE_SECRET_SENTINEL")

        episode = BenchmarkRunner(config).run().episodes[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    requests = MissingPreviousResponsesHandler.requests
    assert len(requests) == 3
    assert [request.get("previous_response_id") for request in requests] == [
        None,
        "resp-1",
        None,
    ]
    rebuilt = requests[2]
    assert sum(
        item.get("type") == "input_image"
        for message in rebuilt["input"]
        for item in message.get("content", ())
        if isinstance(item, dict)
    ) == 2
    rebuilt_text = json.dumps(rebuilt)
    assert "Observation step 0" in rebuilt_text
    assert "Execution feedback for step 0" in rebuilt_text
    assert "Observation step 1" in rebuilt_text
    assert episode.steps == 2
    assert episode.termination_reason == "max_steps"

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert trajectory[1]["agent_trace"]["context_recovery_count"] == 1
    assert trajectory[1]["agent_trace"]["attempts"] == 2
    assert (
        episode_dir
        / "model_calls/step_0001/recovery/previous_response_not_found.json"
    ).exists()
