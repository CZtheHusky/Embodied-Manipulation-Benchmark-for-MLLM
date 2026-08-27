from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.core.types import DMControlAction


def action_to_native(action: DMControlAction, action_spec: Any) -> Any:
    """Convert benchmark DMControl actions against env.action_spec()."""
    import numpy as np

    native = np.asarray(action.values, dtype=np.float64)
    minimum = getattr(action_spec, "minimum", None)
    maximum = getattr(action_spec, "maximum", None)
    if minimum is None or maximum is None:
        return native

    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    return np.clip(native, low, high)
