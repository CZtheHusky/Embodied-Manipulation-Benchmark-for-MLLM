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


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
