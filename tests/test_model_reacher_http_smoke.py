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
