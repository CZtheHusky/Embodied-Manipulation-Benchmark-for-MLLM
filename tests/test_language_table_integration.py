from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner, load_config
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.control.language_table import LanguageTablePlanarController
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_controller,
    build_mapping,
    build_sim,
    build_task,
)
from mllm_teleop_benchmark.core.types import LanguageTableAction
from mllm_teleop_benchmark.sim.language_table.adapter import LanguageTableSimAdapter


CONFIG_PATH = "configs/language_table_block2block.yaml"
MLLM_CONFIG_PATH = "configs/language_table_block2block_mllm_chat_completions.yaml"
PRIVATE_SENTINEL = "PRIVATE_LANGUAGE_TABLE_TARGET_SENTINEL"


class FakeLanguageTableClient:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        text = json.dumps(
            {
                "channels": {"joystick_0": 1, "joystick_1": 0},
                "Reflection": "Use the public image and instruction to choose a planar move.",
            },
            separators=(",", ":"),
        )
        return ModelResponse(
            text=text,
            request_id=f"fake-language-table-{len(self.requests)}",
            model="fake-language-table-model",
            usage=ModelUsage(input_tokens=50, output_tokens=20),
            raw={
                "id": f"fake-language-table-{len(self.requests)}",
                "model": "fake-language-table-model",
                "choices": [{"message": {"content": text}}],
            },
            latency_ms=1.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


def test_language_table_configs_build_without_upstream_import() -> None:
    config = load_config(CONFIG_PATH)
    mllm_config = load_config(MLLM_CONFIG_PATH)

    sim = build_sim(config)
    task = build_task(config)
    schema = build_action_space(mllm_config).schema()

    assert isinstance(sim, LanguageTableSimAdapter)
    assert sim.env is None
    assert config.simulator.params["block_mode"] == "BLOCK_4"
    assert config.simulator.params["reward"] == "blocktoblock"
    assert config.camera.cameras[0].width == 320
    assert config.camera.cameras[0].height == 180
    assert task.instruction() == "Complete the current Language-Table instruction."
    assert schema == {
        "joystick_0": {"kind": "continuous", "min": -30.0, "max": 30.0},
        "joystick_1": {"kind": "continuous", "min": -30.0, "max": 30.0},
    }


def test_language_table_controller_maps_one_to_one_centimeter() -> None:
    config = load_config(CONFIG_PATH)
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )

    action = controller.translate(
        JoystickAction({"joystick_0": 1.0, "joystick_1": -1.0})
    )

    assert isinstance(controller, LanguageTablePlanarController)
    assert isinstance(action, LanguageTableAction)
    assert action.delta_xy == (0.01, -0.01)
    assert controller.metadata()["dimensions"] == ("dx", "dy")


def test_language_table_reset_render_and_step() -> None:
    pytest.importorskip("language_table")
    config = load_config(CONFIG_PATH)
    sim = build_sim(config)
    task = build_task(config)

    observation = task.reset(sim, seed=7)
    frame = sim.render(config.camera.cameras[0])
    result = sim.step(LanguageTableAction(delta_xy=(0.01, 0.0)))

    assert observation.state["instruction"] == task.instruction()
    assert observation.state["instruction"]
    assert frame.shape == (180, 320, 3)
    assert frame.dtype == np.uint8
    assert result.observation.step_id == 1
    assert result.truncated is False
    assert "distance_to_target" in result.info
    sim.close()


def test_language_table_two_step_benchmark(tmp_path: Path) -> None:
    pytest.importorskip("language_table")
    config = load_config(CONFIG_PATH)
    config = replace(
        config,
        benchmark=replace(config.benchmark, name="language_table_test"),
        task=replace(config.task, max_steps=2),
        logging=replace(config.logging, output_dir=tmp_path),
    )

    result = BenchmarkRunner(config).run()

    assert result.aggregate_metrics["num_episodes"] == 1
    assert 1 <= result.episodes[0].steps <= 2
    assert result.episodes[0].metrics["simulator_substeps"] == result.episodes[0].steps
    assert Path(result.episodes[0].log_dir, "trajectory.jsonl").is_file()


def test_language_table_fake_model_receives_public_rgb_and_dynamic_instruction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("language_table")
    client = FakeLanguageTableClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-language-table-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake-language-table.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_LANGUAGE_TABLE_API_KEY")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )
    original_get_state = LanguageTableSimAdapter.get_state

    def poisoned_get_state(self):
        state = original_get_state(self)
        state["private_target"] = PRIVATE_SENTINEL
        return state

    monkeypatch.setattr(LanguageTableSimAdapter, "get_state", poisoned_get_state)
    config = load_config(MLLM_CONFIG_PATH)
    config = replace(
        config,
        task=replace(config.task, max_steps=2),
        logging=replace(config.logging, output_dir=tmp_path, save_video=False),
    )

    episode = BenchmarkRunner(config).run((0,)).episodes[0]

    assert episode.steps == 2
    assert len(client.requests) == 2
    request_body = client.requests[-1].encoded_body().decode("utf-8")
    assert "move the green block to the yellow pentagon" in request_body
    assert PRIVATE_SENTINEL not in request_body
    assert "PRIVATE_LANGUAGE_TABLE_API_KEY" not in request_body
    assert '"reward"' not in request_body
    assert "distance_to_target" not in request_body
    images = sorted(Path(episode.log_dir).glob("model_calls/images/step_*/*.png"))
    assert len(images) == 2
    assert {path.name for path in images} == {"rgb.png"}
    for path in images:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape == (180, 320, 3)
        assert float(pixels.std()) > 1.0
