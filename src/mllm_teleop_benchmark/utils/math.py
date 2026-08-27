from __future__ import annotations

from collections.abc import Iterable


def add_vec3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def fit_vector_to_component_limit(
    values: Iterable[float],
) -> tuple[float, ...]:
    items = tuple(float(value) for value in values)
    max_abs = max((abs(value) for value in items), default=0.0)
    if max_abs <= 1.0:
        return items
    scale = 1.0 / max_abs
    return tuple(value * scale for value in items)


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
