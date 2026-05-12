from __future__ import annotations

import json
from pathlib import Path

from mllm_teleop_benchmark.core.types import StepTransition
from mllm_teleop_benchmark.utils.serialization import to_jsonable


class TrajectoryLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._transitions: list[StepTransition] = []

    def append(self, transition: StepTransition) -> None:
        self._transitions.append(transition)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for transition in self._transitions:
                handle.write(json.dumps(to_jsonable(transition), sort_keys=True) + "\n")
