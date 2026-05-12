#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from mllm_teleop_benchmark import BenchmarkRunner, load_config
from mllm_teleop_benchmark.utils.serialization import to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an MLLM teleoperation benchmark config.")
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    result = BenchmarkRunner(config).run()
    print(json.dumps(to_jsonable(result.aggregate_metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
