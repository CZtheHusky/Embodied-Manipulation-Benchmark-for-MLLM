from __future__ import annotations

import random


def make_rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def episode_seed(base_seed: int, episode_id: int) -> int:
    return int(base_seed) + int(episode_id)
