from __future__ import annotations

from mllm_teleop_benchmark.core.types import (
    AgentTrace,
    CompoundSimulatorAction,
    StepTransition,
)


class MetricsRecorder:
    def __init__(self) -> None:
        self.invalid_action_count = 0
        self.steps = 0
        self.simulator_steps = 0
        self.model_decision_count = 0
        self.model_call_count = 0
        self.model_api_error_count = 0
        self.model_input_error_count = 0
        self.model_retry_count = 0
        self.model_latencies_ms: list[float] = []
        self.model_input_tokens: list[int] = []
        self.model_output_tokens: list[int] = []
        self.cached_input_tokens: list[int] = []
        self.cache_reported_input_tokens: list[int] = []
        self.model_request_bytes_total = 0
        self.model_image_count_total = 0
        self.common_prefix_bytes: list[int] = []

    def update(self, transition: StepTransition) -> None:
        self.steps += 1
        if isinstance(transition.sim_action, CompoundSimulatorAction):
            self.simulator_steps += len(transition.sim_action.actions)
        else:
            self.simulator_steps += 1
        if not transition.validation.is_valid:
            self.invalid_action_count += 1

    def update_agent_trace(self, trace: AgentTrace) -> None:
        if trace.actor_type != "mllm":
            return
        self.model_decision_count += 1
        if trace.attempts > 0:
            self.model_call_count += 1
            self.model_retry_count += max(0, trace.attempts - 1)
        if trace.status == "error":
            self.model_api_error_count += 1
        elif trace.status == "fatal":
            self.model_input_error_count += 1
        if trace.latency_ms is not None:
            self.model_latencies_ms.append(trace.latency_ms)
        if trace.input_tokens is not None:
            self.model_input_tokens.append(trace.input_tokens)
        if trace.output_tokens is not None:
            self.model_output_tokens.append(trace.output_tokens)
        if trace.cached_input_tokens is not None:
            self.cached_input_tokens.append(trace.cached_input_tokens)
            if trace.input_tokens is not None:
                self.cache_reported_input_tokens.append(trace.input_tokens)
        self.model_request_bytes_total += trace.request_bytes
        self.model_image_count_total += trace.image_count
        self.common_prefix_bytes.append(trace.common_prefix_bytes)

    def finalize(self, success: bool, task_metrics: dict[str, object]) -> dict[str, object]:
        metrics: dict[str, object] = {
            "success": success,
            "steps": self.steps,
            "simulator_steps": self.simulator_steps,
            "invalid_action_count": self.invalid_action_count,
        }
        if self.model_decision_count:
            cached_total = sum(self.cached_input_tokens)
            cache_input_total = sum(self.cache_reported_input_tokens)
            metrics.update(
                {
                    "model_decision_count": self.model_decision_count,
                    "model_call_count": self.model_call_count,
                    "model_api_error_count": self.model_api_error_count,
                    "model_input_error_count": self.model_input_error_count,
                    "model_retry_count": self.model_retry_count,
                    "model_latency_ms_mean": _mean(self.model_latencies_ms),
                    "model_latency_ms_p50": _percentile(self.model_latencies_ms, 0.50),
                    "model_latency_ms_p95": _percentile(self.model_latencies_ms, 0.95),
                    "input_tokens": sum(self.model_input_tokens) if self.model_input_tokens else None,
                    "output_tokens": sum(self.model_output_tokens) if self.model_output_tokens else None,
                    "cached_input_tokens": cached_total if self.cached_input_tokens else None,
                    "cache_hit_ratio": cached_total / cache_input_total if cache_input_total > 0 else None,
                    "cache_metrics_status": "reported" if self.cached_input_tokens else "unavailable",
                    "model_request_bytes_total": self.model_request_bytes_total,
                    "model_image_count_total": self.model_image_count_total,
                    "common_prefix_bytes_total": sum(self.common_prefix_bytes),
                    "common_prefix_bytes_last": self.common_prefix_bytes[-1] if self.common_prefix_bytes else 0,
                }
            )
        metrics.update(task_metrics)
        return metrics


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
