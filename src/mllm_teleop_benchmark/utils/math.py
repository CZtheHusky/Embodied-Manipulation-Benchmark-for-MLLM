from __future__ import annotations

from collections.abc import Iterable


def as_vec3(values: Iterable[float], field_name: str) -> tuple[float, float, float]:
    items = tuple(float(v) for v in values)
    if len(items) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values")
    return items


def add_vec3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def fit_vector_to_component_limit(
    values: Iterable[float],
    limit: float = 1.0,
) -> tuple[float, ...]:
    items = tuple(float(value) for value in values)
    component_limit = float(limit)
    if component_limit <= 0.0:
        raise ValueError("component limit must be positive")
    max_abs = max((abs(value) for value in items), default=0.0)
    if max_abs <= component_limit:
        return items
    scale = component_limit / max_abs
    return tuple(value * scale for value in items)


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
