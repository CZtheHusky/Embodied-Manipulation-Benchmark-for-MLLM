from __future__ import annotations

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.ui.human_player import PygameHumanPlayer
from mllm_teleop_benchmark.ui.runtime import PlayerRuntime, SimAdapterWorkerRuntime


def build_runtime(config: BenchmarkConfig) -> PlayerRuntime:
    if config.ui.runtime == "sim_adapter_worker":
        return SimAdapterWorkerRuntime(config)
    raise ValueError(f"Unsupported player runtime: {config.ui.runtime}")


def build_player(config: BenchmarkConfig) -> PygameHumanPlayer:
    if config.ui.backend == "pygame":
        return PygameHumanPlayer(config, build_runtime(config))
    raise ValueError(f"Unsupported player backend: {config.ui.backend}")


__all__ = ["build_player", "build_runtime"]
