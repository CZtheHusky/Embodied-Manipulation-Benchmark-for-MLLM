from __future__ import annotations

from mllm_teleop_benchmark.agents.base import Agent
from mllm_teleop_benchmark.agents.mllm_policy import MLLMPolicy
from mllm_teleop_benchmark.agents.random_policy import RandomPolicy
from mllm_teleop_benchmark.agents.scripted_policy import ScriptedPolicy
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.dummy import DummySimAdapter
from mllm_teleop_benchmark.tasks.base import Task
from mllm_teleop_benchmark.tasks.reach_target import ReachTargetTask


def build_sim(config: BenchmarkConfig) -> SimAdapter:
    if config.simulator.type == "dummy":
        return DummySimAdapter(config.simulator)
    raise ValueError(f"Unsupported simulator type: {config.simulator.type}")


def build_task(config: BenchmarkConfig) -> Task:
    if config.task.type == "reach_target":
        return ReachTargetTask(config.task)
    raise ValueError(f"Unsupported task type: {config.task.type}")


def build_agent(config: BenchmarkConfig, seed: int) -> Agent:
    if config.agent.type == "random":
        return RandomPolicy(seed)
    if config.agent.type == "scripted":
        return ScriptedPolicy()
    if config.agent.type == "mllm":
        return MLLMPolicy()
    raise ValueError(f"Unsupported agent type: {config.agent.type}")
