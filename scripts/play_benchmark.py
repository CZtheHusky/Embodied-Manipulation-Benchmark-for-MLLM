#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a benchmark environment with keyboard controls.")
    parser.add_argument("--config", default="configs/reacher_delta_eef.yaml")
    parser.add_argument("--backend", default=None, help="Human player backend, e.g. pygame.")
    parser.add_argument("--runtime", default=None, help="Simulator runtime, e.g. sim_adapter_worker.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set control_interface.mapping_strategy=permuted")
    args = parser.parse_args()

    from mllm_teleop_benchmark.config.loader import load_config
    from mllm_teleop_benchmark.ui.registry import build_player

    overrides = list(args.overrides)
    backend = args.backend or os.environ.get("MLLM_TELEOP_PLAYER_BACKEND")
    runtime = args.runtime or os.environ.get("MLLM_TELEOP_PLAYER_RUNTIME")
    if backend:
        overrides.append(f"ui.backend={backend}")
    if runtime:
        overrides.append(f"ui.runtime={runtime}")

    config = load_config(args.config, tuple(overrides))
    result = build_player(config).run()
    print(result)


if __name__ == "__main__":
    main()
