from __future__ import annotations

from mllm_teleop_benchmark.core.types import StepTransition


class MetricsRecorder:
    def __init__(self) -> None:
        self.invalid_action_count = 0
        self.steps = 0

    def update(self, transition: StepTransition) -> None:
        self.steps += 1
        if not transition.validation.is_valid:
            self.invalid_action_count += 1

    def finalize(self, success: bool, task_metrics: dict[str, object]) -> dict[str, object]:
        metrics = {
            "success": success,
            "steps": self.steps,
            "invalid_action_count": self.invalid_action_count,
        }
        metrics.update(task_metrics)
        return metrics
