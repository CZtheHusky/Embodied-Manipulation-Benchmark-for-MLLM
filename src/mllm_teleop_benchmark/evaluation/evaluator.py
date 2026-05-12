from __future__ import annotations

from mllm_teleop_benchmark.core.types import EpisodeResult


def aggregate_episode_results(results: list[EpisodeResult]) -> dict[str, object]:
    if not results:
        return {"num_episodes": 0, "success_rate": 0.0}
    successes = sum(1 for result in results if result.success)
    avg_steps = sum(result.steps for result in results) / len(results)
    return {
        "num_episodes": len(results),
        "success_rate": successes / len(results),
        "average_steps": avg_steps,
    }
