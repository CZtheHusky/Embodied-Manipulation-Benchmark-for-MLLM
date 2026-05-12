from __future__ import annotations

from pathlib import Path

import yaml

from mllm_teleop_benchmark.config.schema import BenchmarkConfig


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return BenchmarkConfig.from_dict(data)
