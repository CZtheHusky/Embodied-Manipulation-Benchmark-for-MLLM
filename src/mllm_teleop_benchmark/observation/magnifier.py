from __future__ import annotations

from dataclasses import replace
import math

from PIL import Image

from mllm_teleop_benchmark.core.types import (
    MagnifierResult,
    MagnifierUpdate,
    MagnifierView,
    NormalizedBBox,
    Observation,
)
from mllm_teleop_benchmark.utils.images import rgb_array


class Magnifier:
    def __init__(self, allowed_rgb_keys: tuple[str, ...] = ()) -> None:
        self.allowed_rgb_keys = allowed_rgb_keys
        self._source_rgb_key: str | None = None
        self._bbox_xyxy: NormalizedBBox | None = None

    def apply_update(
        self,
        update: MagnifierUpdate,
        observation: Observation,
    ) -> MagnifierResult | None:
        if not update.requested:
            return None
        if update.error is not None:
            return self._result("rejected", f"Magnifier command rejected: {update.error}")
        if update.clear:
            self._source_rgb_key = None
            self._bbox_xyxy = None
            return self._result("cleared", "Magnifier cleared.")

        source_rgb_key = update.source_rgb_key
        bbox_xyxy = update.bbox_xyxy
        if source_rgb_key is None or bbox_xyxy is None:
            return self._result(
                "rejected",
                "Magnifier command rejected: source_rgb_key and bbox_xyxy are required.",
            )
        available = self._available_rgb_keys(observation)
        if source_rgb_key not in available:
            listed = ", ".join(available) if available else "none"
            return self._result(
                "rejected",
                "Magnifier command rejected: unknown source_rgb_key "
                f"{source_rgb_key!r}. Available source RGB keys: {listed}.",
            )
        try:
            rgb_array(observation.images[source_rgb_key])
        except Exception as exc:
            return self._result(
                "rejected",
                f"Magnifier command rejected: source_rgb_key {source_rgb_key!r} "
                f"is not a valid RGB image ({type(exc).__name__}).",
            )

        status = "activated" if self._source_rgb_key is None else "updated"
        self._source_rgb_key = source_rgb_key
        self._bbox_xyxy = bbox_xyxy
        return self._result(
            status,
            f"Magnifier {status} for source_rgb_key {source_rgb_key!r} at "
            f"bbox_xyxy=[{','.join(f'{value:g}' for value in bbox_xyxy)}].",
        )

    def observe(
        self,
        observation: Observation,
        result: MagnifierResult | None = None,
    ) -> Observation:
        feedback = [result.message] if result is not None else []
        view: MagnifierView | None = None
        if self._source_rgb_key is not None and self._bbox_xyxy is not None:
            if self._source_rgb_key not in observation.images:
                feedback.append(
                    "Magnifier view unavailable: active source_rgb_key "
                    f"{self._source_rgb_key!r} is missing from the current observation."
                )
            else:
                try:
                    image = crop_and_resize(
                        observation.images[self._source_rgb_key],
                        self._bbox_xyxy,
                    )
                except Exception as exc:
                    feedback.append(
                        "Magnifier view unavailable: failed to crop active source_rgb_key "
                        f"{self._source_rgb_key!r} ({type(exc).__name__})."
                    )
                else:
                    view = MagnifierView(
                        source_rgb_key=self._source_rgb_key,
                        bbox_xyxy=self._bbox_xyxy,
                        image=image,
                    )
        return replace(
            observation,
            magnifier=view,
            tool_feedback=tuple(feedback),
        )

    def _available_rgb_keys(self, observation: Observation) -> tuple[str, ...]:
        if self.allowed_rgb_keys:
            return tuple(
                key for key in self.allowed_rgb_keys if key in observation.images
            )
        return tuple(sorted(str(key) for key in observation.images))

    def _result(self, status: str, message: str) -> MagnifierResult:
        return MagnifierResult(
            status=status,
            message=message,
            active_source_rgb_key=self._source_rgb_key,
            active_bbox_xyxy=self._bbox_xyxy,
        )


def crop_and_resize(value: object, bbox_xyxy: NormalizedBBox):
    array = rgb_array(value)
    height, width = array.shape[:2]
    x_min, y_min, x_max, y_max = bbox_xyxy
    # Invalid model coordinates otherwise crop a different or empty source-image region.
    if not _valid_bbox(bbox_xyxy):
        raise ValueError("bbox_xyxy must satisfy 0 <= min < max <= 1")

    left = min(width - 1, math.floor(x_min * width))
    top = min(height - 1, math.floor(y_min * height))
    right = max(left + 1, min(width, math.ceil(x_max * width)))
    bottom = max(top + 1, min(height, math.ceil(y_max * height)))
    crop = array[top:bottom, left:right]
    crop_height, crop_width = crop.shape[:2]
    scale = min(width / crop_width, height / crop_height)
    target_width = max(1, min(width, round(crop_width * scale)))
    target_height = max(1, min(height, round(crop_height * scale)))
    image = Image.fromarray(crop)
    return rgb_array(
        image.resize(
            (target_width, target_height),
            resample=Image.Resampling.LANCZOS,
        )
    )


def _valid_bbox(bbox_xyxy: NormalizedBBox) -> bool:
    x_min, y_min, x_max, y_max = bbox_xyxy
    return (
        all(math.isfinite(value) for value in bbox_xyxy)
        and 0.0 <= x_min < x_max <= 1.0
        and 0.0 <= y_min < y_max <= 1.0
    )
