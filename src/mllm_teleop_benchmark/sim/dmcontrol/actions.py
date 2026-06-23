from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.core.types import DMControlAction


def action_to_native(action: DMControlAction, action_spec: Any, *, clip: bool = True) -> Any:
    """Convert and validate benchmark DMControl actions against env.action_spec()."""
    import numpy as np

    native = np.asarray(action.values, dtype=np.float64)
    expected_shape = tuple(getattr(action_spec, "shape", ()))
    if native.shape != expected_shape:
        raise ValueError(f"DMControl action shape {native.shape} does not match env.action_spec shape {expected_shape}")
    if not np.all(np.isfinite(native)):
        raise ValueError("DMControl action contains non-finite values")

    minimum = getattr(action_spec, "minimum", None)
    maximum = getattr(action_spec, "maximum", None)
    if minimum is None or maximum is None:
        return native

    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    if clip:
        return np.clip(native, low, high)
    if np.any(native < low) or np.any(native > high):
        raise ValueError("DMControl action is outside env.action_spec bounds")
    return native
