from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.core.types import GymBoxAction


def action_to_native(action: GymBoxAction) -> Any:
    import numpy as np

    return np.asarray(action.values, dtype=np.float32)
