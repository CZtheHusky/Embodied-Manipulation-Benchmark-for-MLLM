from __future__ import annotations

import math
from pathlib import Path

import pytest

from mllm_teleop_benchmark.config.loader import load_config


TRANSLATION_STEP_M = 0.01
ROTATION_STEP_RAD = math.radians(1.8)
TRANSLATION_DIMS = frozenset({"dx", "dy", "dz"})
ROTATION_DIMS = frozenset({"droll", "dpitch", "dyaw"})


def _effective_step(config, dimension: str) -> float:
    simulator = config.simulator.type
    if simulator == "gymnasium":
        return float(config.simulator.env_kwargs["eef_step_scale"])

    native_gain = {
        "metaworld": 0.01,
        "maniskill": 0.1,
        "robocasa": 0.05 if dimension in TRANSLATION_DIMS else 0.5,
        "robosuite": 0.05 if dimension in TRANSLATION_DIMS else 0.5,
        "libero": 0.05 if dimension in TRANSLATION_DIMS else 0.5,
        "language_table": 1.0,
        "simpler_env": 1.0,
        "calvin": 0.02 if dimension in TRANSLATION_DIMS else 0.05,
        "dmcontrol": 1.0,
        "rlbench": 1.0,
        "colosseum": 1.0,
        "dummy": 1.0,
    }[simulator]
    return float(config.controller.scales[dimension]) * native_gain


@pytest.mark.parametrize(
    "config_path",
    sorted(Path("configs").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_eef_configs_share_the_same_physical_single_axis_step(config_path):
    """A simulator-specific scale must not silently change benchmark difficulty."""
    config = load_config(config_path)
    active_dimensions = set(config.control_interface.active_dimensions)

    for dimension in sorted(active_dimensions & TRANSLATION_DIMS):
        assert _effective_step(config, dimension) == pytest.approx(
            TRANSLATION_STEP_M
        )

    for dimension in sorted(active_dimensions & ROTATION_DIMS):
        assert _effective_step(config, dimension) == pytest.approx(
            ROTATION_STEP_RAD
        )


@pytest.mark.parametrize(
    "config_path",
    sorted(Path("configs").glob("*mllm*.yaml")),
    ids=lambda path: path.name,
)
def test_mllm_control_description_discloses_the_effective_step(config_path):
    config = load_config(config_path)
    active_dimensions = set(config.control_interface.active_dimensions)
    description = config.control_interface.control_frame_description
    schema = config.action_space.channels

    assert config.control_interface.channel_kind == "continuous"
    assert config.control_interface.channel_minimum == -30.0
    assert config.control_interface.channel_maximum == 30.0
    for channel_name in config.control_interface.channel_names()[
        : len(config.control_interface.active_dimensions)
    ]:
        channel = schema[channel_name]
        assert channel.kind == "continuous"
        assert channel.minimum == -30.0
        assert channel.maximum == 30.0

    if active_dimensions & TRANSLATION_DIMS:
        assert "0.01" in description
    if active_dimensions & ROTATION_DIMS:
        assert "1.8 degrees" in description
