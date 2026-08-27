from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return value.tolist()
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
