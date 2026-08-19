from __future__ import annotations

import pytest

from mllm_teleop_benchmark import load_config
from mllm_teleop_benchmark.core.execution_budget import execution_budget
from mllm_teleop_benchmark.core.registry import build_sim
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group


REACHER_MATRIX = (
    (
        "configs/reacher_delta_eef_mllm_chat_completions.yaml",
        "chat_completions",
        "mujoco_randomized",
        "random_view",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/reacher_delta_eef_mllm_chat_completions_perturbed.yaml",
        "chat_completions",
        "mujoco_randomized",
        "random_view",
        "permuted_sign_flipped",
        "random_rotation_2d",
        False,
    ),
    (
        "configs/reacher_delta_eef_mllm_chat_completions_default_view.yaml",
        "chat_completions",
        "fixed",
        "default_view",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/reacher_delta_eef_mllm_chat_completions_default_view_perturbed.yaml",
        "chat_completions",
        "fixed",
        "default_view",
        "permuted_sign_flipped",
        "random_rotation_2d",
        False,
    ),
    (
        "configs/reacher_delta_eef_mllm_responses.yaml",
        "responses",
        "mujoco_randomized",
        "random_view",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/reacher_delta_eef_mllm_responses_perturbed.yaml",
        "responses",
        "mujoco_randomized",
        "random_view",
        "permuted_sign_flipped",
        "random_rotation_2d",
        False,
    ),
    (
        "configs/reacher_delta_eef_mllm_responses_default_view.yaml",
        "responses",
        "fixed",
        "default_view",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/reacher_delta_eef_mllm_responses_default_view_perturbed.yaml",
        "responses",
        "fixed",
        "default_view",
        "permuted_sign_flipped",
        "random_rotation_2d",
        False,
    ),
)


@pytest.mark.parametrize(
    (
        "config_path",
        "api",
        "camera_manager",
        "camera_condition",
        "mapping_strategy",
        "frame_transform",
        "visible",
    ),
    REACHER_MATRIX,
)
def test_reacher_matrix_configs_are_explicit_and_distinct(
    config_path,
    api,
    camera_manager,
    camera_condition,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)

    assert config.benchmark.num_episodes == 10
    assert config.agent.model is not None
    assert config.agent.model.api == api
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == (
        "all" if api == "responses" else 3
    )
    assert config.agent.history.max_images_per_request >= (
        config.task.max_steps * (len(config.agent.observation.cameras) + 1)
        if api == "responses"
        else 128
    )
    assert config.logging.save_video is True
    assert config.camera.manager == camera_manager
    assert config.camera.params["experiment_condition"] == camera_condition
    assert config.camera.cameras[0].width == 512
    assert config.camera.cameras[0].height == 512
    assert config.control_interface.active_dimensions == ("dx", "dy")
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.mapping_visible_to_agent is visible
    assert config.control_interface.frame_visible_to_agent is visible

    budget = execution_budget(config)
    assert budget.max_agent_steps == 80
    assert budget.max_internal_substeps_per_agent_step == 30
    assert budget.required_simulator_steps == 2400
    assert budget.configured_simulator_step_horizon == 2400

    description = config.control_interface.control_frame_description
    assert description is not None
    if camera_condition == "random_view":
        assert "randomly sampled at episode reset" in description
        assert "fixed for that episode" in description
        assert "do not have a fixed world-axis projection across episodes" in description
    else:
        assert "environment's default pose" in description
        assert "remains fixed across episodes and steps" in description
        assert "consistent world-axis projection" in description

    group = make_experiment_group(config)
    assert group is not None
    assert f"api-{api}" in group
    assert f"camera-{camera_condition}" in group
    assert "episodes-10" in group
    if mapping_strategy != "fixed":
        assert "mapping-permuted_sign_flipped" in group
        assert "frame-random_rotation_2d" in group


def test_reacher_matrix_uses_unique_benchmark_names():
    names = [load_config(row[0]).benchmark.name for row in REACHER_MATRIX]

    assert len(names) == 8
    assert len(set(names)) == 8


def test_reacher_native_horizon_covers_worst_case_agent_command_budget():
    pytest.importorskip("gymnasium")
    config = load_config("configs/reacher_delta_eef_mllm_chat_completions.yaml")

    sim = build_sim(config)
    try:
        assert sim.config.params["max_episode_steps"] == 2400
        assert sim.env._max_episode_steps == 2400
    finally:
        sim.close()
