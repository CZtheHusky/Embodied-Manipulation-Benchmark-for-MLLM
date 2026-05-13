from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mllm_teleop_benchmark.config.schema import BenchmarkConfig


def load_config(path: str | Path, overrides: tuple[str, ...] = ()) -> BenchmarkConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    for override in overrides:
        _apply_override(data, override)
    return BenchmarkConfig.from_dict(data)


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must use key.path=value syntax: {override}")
    key_path, raw_value = override.split("=", 1)
    keys = [part for part in key_path.split(".") if part]
    if not keys:
        raise ValueError(f"Override key path is empty: {override}")
    value = yaml.safe_load(raw_value)
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            existing = {}
            cursor[key] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot override through non-mapping config key: {key}")
        cursor = existing
    cursor[keys[-1]] = value
