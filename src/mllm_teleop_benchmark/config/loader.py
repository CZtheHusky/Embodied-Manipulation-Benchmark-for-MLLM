from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mllm_teleop_benchmark.config.schema import BenchmarkConfig


def load_config(path: str | Path, overrides: tuple[str, ...] = ()) -> BenchmarkConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    for override in overrides:
        _apply_override(data, override)
    return BenchmarkConfig.from_dict(data, source_path=str(config_path.resolve()))


def _apply_override(data: dict[str, Any], override: str) -> None:
    key_path, raw_value = override.split("=", 1)
    keys = [part for part in key_path.split(".") if part]
    value = yaml.safe_load(raw_value)
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[keys[-1]] = value
