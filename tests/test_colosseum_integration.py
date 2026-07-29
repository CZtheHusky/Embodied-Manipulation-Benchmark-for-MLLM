from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.agents.base import AgentContext
from mllm_teleop_benchmark.agents.model_client import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from mllm_teleop_benchmark.agents.model_policy import MLLMPolicy
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import (
    AgentConfig,
    AgentHistoryConfig,
    AgentObservationConfig,
    ModelConfig,
)
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_camera_manager,
    build_controller,
    build_mapping,
    build_observation_provider,
    build_sim,
    build_task,
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import Observation, RLBenchAction
from mllm_teleop_benchmark.sim.colosseum.adapter import ColosseumSimAdapter
from mllm_teleop_benchmark.sim.rlbench.adapter import (
    RLBenchSimAdapter,
    _load_invalid_action_error_type,
)
from mllm_teleop_benchmark.sim.rlbench.observation import (
    RLBenchObservationProvider,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/colosseum_close_box_manip_obj_color_smoke.yaml"
MLLM_CONFIGS = (
    (
        "configs/colosseum_close_box_mllm_reflection.yaml",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/colosseum_close_box_mllm_reflection_perturbed.yaml",
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
)
COLOSSEUM_COMMIT = "098a910ae1d490524adb49b8d50735b00a1a62de"
RLBENCH_COMMIT = "7c3f425f4a0b6b5ce001ba7246354eb3c70555be"
PYREP_COMMIT = "076ca15c57f2495a4194da03565891ab1aaa317e"
EXISTING_RLBENCH_COMMIT = "f214b6ecc177babce12d3ad13dd92049c824c1d9"


class RecordingClient(ModelClient):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        text = (
            '{"channels":{"joystick_0":0,"joystick_1":0,"joystick_2":0}}'
        )
        return ModelResponse(
            text=text,
            request_id="colosseum-privacy",
            model="fake-colosseum-model",
            usage=ModelUsage(input_tokens=32, output_tokens=8),
            raw={"id": "colosseum-privacy"},
            latency_ms=1.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


class ColosseumFakeClient(ModelClient):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        text = json.dumps(
            {
                "channels": {
                    "joystick_0": 0,
                    "joystick_1": 0,
                    "joystick_2": 0,
                },
                "Reflection": (
                    "The translation channels remain unidentified because the "
                    "last command requested no motion."
                ),
            },
            separators=(",", ":"),
        )
        return ModelResponse(
            text=text,
            request_id=f"fake-colosseum-{len(self.requests)}",
            model="fake-colosseum-model",
            usage=ModelUsage(input_tokens=96, output_tokens=32),
            raw={"id": f"fake-colosseum-{len(self.requests)}"},
            latency_ms=1.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


def _load_manage_conda_env():
    script_path = ROOT / "scripts" / "manage_conda_env.py"
    spec = importlib.util.spec_from_file_location("manage_conda_env", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_runtime_module(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        if os.environ.get("MLLM_TELEOP_REQUIRE_COLOSSEUM") == "1":
            pytest.fail(
                f"Required Colosseum runtime module failed to import: "
                f"{module_name}: {exc}"
            )
        pytest.skip(f"Colosseum runtime module unavailable: {module_name}: {exc}")


def test_colosseum_conda_backend_has_independent_official_source_pins():
    module = _load_manage_conda_env()
    backend = module.BACKENDS["colosseum"]
    sources = {source["repo"]: source for source in backend["sources"]}

    assert backend["env_name"] == "mllm-teleop-colosseum"
    assert backend["env_file"] == "envs/environment-colosseum.yml"
    assert backend["require_flag"] == "MLLM_TELEOP_REQUIRE_COLOSSEUM"
    assert sources["https://github.com/robot-colosseum/robot-colosseum.git"][
        "commit"
    ] == COLOSSEUM_COMMIT
    assert sources["https://github.com/stepjam/RLBench.git"]["commit"] == RLBENCH_COMMIT
    assert sources["https://github.com/stepjam/PyRep.git"]["commit"] == PYREP_COMMIT
    assert {source["name"] for source in sources.values()} == {
        "colosseum",
        "colosseum-rlbench",
        "colosseum-pyrep",
    }
    assert module.BACKENDS["rlbench"]["sources"][1]["commit"] == (
        EXISTING_RLBENCH_COMMIT
    )
    assert "V4_1_0" in module.COPPELIASIM_URL


def test_colosseum_environment_file_is_separate():
    environment = (ROOT / "envs" / "environment-colosseum.yml").read_text(
        encoding="utf-8"
    )

    assert environment.startswith("name: mllm-teleop-colosseum\n")
    assert "hydra-core>=1.3.2" in environment
    assert "mllm-teleop-rlbench" not in environment


def test_invalid_action_error_uses_new_rlbench_location(monkeypatch):
    rlbench = ModuleType("rlbench")
    rlbench.__path__ = []
    backend = ModuleType("rlbench.backend")
    backend.__path__ = []
    exceptions = ModuleType("rlbench.backend.exceptions")

    class NewInvalidActionError(Exception):
        pass

    exceptions.InvalidActionError = NewInvalidActionError
    monkeypatch.setitem(sys.modules, "rlbench", rlbench)
    monkeypatch.setitem(sys.modules, "rlbench.backend", backend)
    monkeypatch.setitem(sys.modules, "rlbench.backend.exceptions", exceptions)

    assert _load_invalid_action_error_type() is NewInvalidActionError


def test_invalid_action_error_falls_back_to_legacy_rlbench_location(monkeypatch):
    rlbench = ModuleType("rlbench")
    rlbench.__path__ = []
    backend = ModuleType("rlbench.backend")
    backend.__path__ = []
    task_environment = ModuleType("rlbench.task_environment")

    class LegacyInvalidActionError(Exception):
        pass

    task_environment.InvalidActionError = LegacyInvalidActionError
    monkeypatch.setitem(sys.modules, "rlbench", rlbench)
    monkeypatch.setitem(sys.modules, "rlbench.backend", backend)
    monkeypatch.delitem(
        sys.modules,
        "rlbench.backend.exceptions",
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "rlbench.task_environment",
        task_environment,
    )

    assert _load_invalid_action_error_type() is LegacyInvalidActionError


def test_colosseum_config_builds_shared_rlbench_contract_without_runtime():
    config = load_config(CONFIG_PATH)
    action_space = build_action_space(config)
    mapping = build_mapping(config, seed=config.benchmark.seed)
    controller = build_controller(config, mapping, seed=config.benchmark.seed)
    sim = build_sim(config)
    task = build_task(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    provider = build_observation_provider(config, camera_manager)

    assert config.simulator.type == "colosseum"
    assert config.simulator.env_id == "close_box"
    assert config.simulator.params["spreadsheet_idx"] == 2
    assert config.simulator.params["perturbation"] == "manip_obj_color"
    assert config.task.type == "rlbench_generic"
    assert config.controller.type == "rlbench_delta_eef"
    assert tuple(action_space.channels) == (
        "joystick_0",
        "joystick_1",
        "joystick_2",
    )
    assert controller.metadata()["type"] == "rlbench_delta_eef"
    assert task.instruction() == "Close the box lid."
    assert isinstance(sim, ColosseumSimAdapter)
    assert isinstance(sim, RLBenchSimAdapter)
    assert isinstance(provider, RLBenchObservationProvider)


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_colosseum_mllm_configs_are_explicit_and_switchable(
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)
    action_space = build_action_space(config)

    assert config.simulator.type == "colosseum"
    assert config.simulator.env_id == "close_box"
    assert config.simulator.params["spreadsheet_idx"] == 2
    assert config.simulator.params["perturbation"] == "manip_obj_color"
    assert config.task.params["instruction"] == "Close the visible box lid."
    assert config.control_interface.active_dimensions == ("dx", "dy", "dz")
    assert config.control_interface.gripper == "none"
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.mapping_visible_to_agent is visible
    assert config.control_interface.frame_visible_to_agent is visible
    assert config.controller.scales == {"dx": 0.02, "dy": 0.02, "dz": 0.02}
    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 3
    assert config.agent.observation.cameras == ("front",)
    assert tuple(action_space.channels) == (
        "joystick_0",
        "joystick_1",
        "joystick_2",
    )


def test_colosseum_evaluator_metadata_records_official_condition_and_seed():
    config = load_config(CONFIG_PATH)
    adapter = ColosseumSimAdapter(config.simulator)
    adapter.episode_seed = 123
    adapter.variation_index = 0

    state = adapter.get_state()
    observation = adapter._observation()

    assert state["evaluator_metadata"] == {
        "task": "close_box",
        "perturbation": "manip_obj_color",
        "spreadsheet_idx": 2,
        "variation": 0,
        "seed": 123,
    }
    assert observation.metadata == {
        "simulator": "colosseum",
        "env_id": "close_box",
    }


def test_colosseum_evaluator_metadata_is_not_projected_into_model_input():
    config = load_config(CONFIG_PATH)
    action_space = build_action_space(config)
    private_values = {
        "task": "PRIVATE_COLOSSEUM_TASK",
        "perturbation": "PRIVATE_COLOSSEUM_PERTURBATION",
        "spreadsheet_idx": "PRIVATE_COLOSSEUM_SPREADSHEET_IDX",
        "variation": "PRIVATE_COLOSSEUM_VARIATION",
        "seed": "PRIVATE_COLOSSEUM_SEED",
    }
    observation = Observation(
        step_id=0,
        images={"front": np.zeros((12, 16, 3), dtype=np.uint8)},
        state={"evaluator_metadata": private_values},
        metadata={"simulator": "colosseum"},
    )
    client = RecordingClient()
    policy = MLLMPolicy(
        AgentConfig(
            type="mllm",
            model=ModelConfig(require_api_key=False),
            observation=AgentObservationConfig(
                cameras=("front",),
                image_max_side=32,
            ),
            history=AgentHistoryConfig(),
        ),
        client,
        "fake-colosseum-model",
    )
    context = AgentContext(
        observation=observation,
        task_instruction="Close the box lid.",
        action_space=action_space,
        action_schema=action_space.schema(),
        metadata={
            "control_interface": control_interface_metadata(
                config,
                config.benchmark.seed,
            )
        },
    )

    policy.act(context)

    encoded = client.requests[0].encoded_body().decode("utf-8")
    assert "Close the box lid." in encoded
    for private_value in private_values.values():
        assert private_value not in encoded


def test_colosseum_runtime_reset_step_render_smoke_when_installed():
    for module_name in ("colosseum", "rlbench", "pyrep"):
        _require_runtime_module(module_name)

    config = load_config(CONFIG_PATH)
    sim = build_sim(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    provider = build_observation_provider(config, camera_manager)
    provider.configure_sim(sim)
    try:
        try:
            observation = sim.reset(config.task, seed=42)
            assert observation.step_id == 0
            assert observation.metadata["simulator"] == "colosseum"
            assert sim.get_state()["evaluator_metadata"]["seed"] == 42

            frame = sim.render(config.camera.cameras[0])
            frame_array = np.asarray(frame)
            assert frame_array.shape == (240, 320, 3)
            assert frame_array.dtype == np.uint8
            assert float(frame_array.mean()) > 2.0
            assert float(frame_array.std()) > 2.0

            enabled_factors = [
                (
                    str(factor.variation),
                    str(factor.get("name", "any")),
                )
                for factor in sim._official_config.env.scene.factors
                if bool(factor.enabled)
            ]
            assert enabled_factors == [("object_color", "manip_obj_color")]

            result = sim.step(
                RLBenchAction(
                    delta_position=(0.0, 0.0, 0.0),
                    gripper=1.0,
                )
            )
            assert result.observation.step_id == 1
            assert result.info["evaluator_metadata"]["spreadsheet_idx"] == 2
        except (ImportError, RuntimeError) as exc:
            if os.environ.get("MLLM_TELEOP_REQUIRE_COLOSSEUM") == "1":
                raise
            pytest.skip(f"Colosseum/CoppeliaSim runtime unavailable: {exc}")
    finally:
        sim.close()


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_colosseum_fake_client_closes_public_rgb_loop(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    for module_name in ("colosseum", "rlbench", "pyrep"):
        _require_runtime_module(module_name)

    client = ColosseumFakeClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-colosseum-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake-colosseum.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_COLOSSEUM_API_KEY")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    original_get_state = ColosseumSimAdapter.get_state

    def poisoned_get_state(self):
        state = original_get_state(self)
        state["private_target_pose"] = "PRIVATE_COLOSSEUM_TARGET"
        state["private_native_action"] = "PRIVATE_COLOSSEUM_ACTION"
        return state

    monkeypatch.setattr(ColosseumSimAdapter, "get_state", poisoned_get_state)
    config = load_config(
        config_path,
        (
            "task.max_steps=4",
            f"logging.output_dir={tmp_path}",
            "logging.save_video=false",
        ),
    )

    episode = BenchmarkRunner(config).run((0,)).episodes[0]

    assert episode.steps == 4
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 4
    assert [_image_count(request) for request in client.requests] == [1, 2, 3, 4]
    assert [
        sum(message["role"] == "assistant" for message in request.messages)
        for request in client.requests
    ] == [0, 1, 2, 3]

    request_text = _request_text(client.requests[-1])
    assert "Prompt version: mllm_joystick_reflection_v2" in request_text
    assert "Task: Close the visible box lid." in request_text
    assert "active controllable dimensions are: dx, dy, dz" in request_text
    assert "Colosseum/RLBench world Cartesian x, y, and z axes" in request_text
    assert "gripper held open" in request_text
    if visible:
        assert "joystick_0 -> dx" in request_text
        assert "No additional control-frame rotation is applied" in request_text
    else:
        assert "exact joystick-to-dimension mapping is not disclosed" in request_text
        assert "exact physical-frame transform is not disclosed" in request_text

    encoded_requests = "\n".join(
        request.encoded_body().decode("utf-8") for request in client.requests
    )
    for forbidden in (
        "PRIVATE_COLOSSEUM_TARGET",
        "PRIVATE_COLOSSEUM_ACTION",
        "PRIVATE_COLOSSEUM_API_KEY",
        "manip_obj_color",
        "spreadsheet_idx",
        "variation_index",
        "rotation_matrix",
        "quaternion_xyzw",
        "permuted_sign_flipped",
        "random_rotation_3d",
    ):
        assert forbidden not in encoded_requests

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(trajectory) == 4
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert all(step["reflection"] for step in trajectory)

    pngs = sorted(episode_dir.glob("model_calls/images/step_*/*.png"))
    assert len(pngs) == 4
    for path in pngs:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape == (240, 320, 3)
        assert float(pixels.mean()) > 2.0
        assert float(pixels.std()) > 2.0

    manifest = json.loads(
        (episode_dir.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["model"]["reasoning_effort"] == "low"
    assert manifest["model"]["action_output_mode"] == "action_with_reflection"
    assert manifest["history"]["image_window"] == 3
    assert manifest["control_interface"]["mapping_strategy"] == mapping_strategy
    assert manifest["control_interface"]["frame_transform"] == frame_transform
    assert manifest["control_interface"]["mapping_visible_to_agent"] is visible
    assert manifest["control_interface"]["frame_visible_to_agent"] is visible


def _image_count(request: ModelRequest) -> int:
    return sum(
        1
        for message in request.messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "image_url"
    )


def _request_text(request: ModelRequest) -> str:
    values: list[str] = []
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, str):
            values.append(content)
        elif isinstance(content, list):
            values.extend(
                str(item["text"])
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
    return "\n".join(values)
