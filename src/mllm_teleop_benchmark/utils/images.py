from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

import imageio.v3 as iio
import numpy as np


@dataclass(frozen=True)
class EncodedPNG:
    width: int
    height: int
    content: bytes
    sha256: str


def encode_rgb_png(value: Any, max_side: int) -> EncodedPNG:
    array = resize_max_side(rgb_array(value), max_side)
    png = iio.imwrite("<bytes>", array, extension=".png")
    if not isinstance(png, bytes):
        raise TypeError("PNG encoder did not return bytes")
    return EncodedPNG(
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        content=png,
        sha256=hashlib.sha256(png).hexdigest(),
    )


def rgb_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Image must have RGB/RGBA shape, got {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.size == 0:
        raise ValueError("Image must not be empty")
    if array.dtype != np.uint8:
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.max(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def resize_max_side(array: np.ndarray, max_side: int) -> np.ndarray:
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return array
    scale = max_side / float(longest)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    rows = np.linspace(0, height - 1, target_height).round().astype(int)
    columns = np.linspace(0, width - 1, target_width).round().astype(int)
    return np.ascontiguousarray(array[rows][:, columns])


def safe_artifact_name(value: str, fallback: str = "camera") -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or fallback
