from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

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
