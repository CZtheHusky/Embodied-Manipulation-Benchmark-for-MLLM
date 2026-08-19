import numpy as np

from mllm_teleop_benchmark.core.types import MagnifierUpdate, Observation
from mllm_teleop_benchmark.observation.magnifier import Magnifier, crop_and_resize


def _observation(step_id: int, **images) -> Observation:
    return Observation(step_id=step_id, images=images, state={})


def test_crop_uses_original_rgb_and_fits_original_frame_with_aspect_preserved():
    source = np.zeros((100, 200, 3), dtype=np.uint8)
    source[:, :100, 0] = 255
    source[:, 100:, 2] = 255

    landscape = crop_and_resize(source, (0.25, 0.25, 0.75, 0.75))
    portrait = crop_and_resize(source, (0.4, 0.0, 0.6, 1.0))

    assert landscape.shape == (100, 200, 3)
    assert portrait.shape == (100, 40, 3)
    assert landscape.shape[1] / landscape.shape[0] == 2.0
    assert portrait.shape[1] / portrait.shape[0] == 0.4

    right_half = crop_and_resize(source, (0.5, 0.0, 1.0, 1.0))
    assert right_half.shape == (100, 100, 3)
    assert np.all(right_half[..., 0] == 0)
    assert np.all(right_half[..., 2] == 255)


def test_valid_update_persists_across_steps_and_tracks_current_raw_rgb():
    magnifier = Magnifier(("front", "wrist"))
    first = _observation(
        0,
        front=np.full((20, 40, 3), 10, dtype=np.uint8),
        wrist=np.full((20, 40, 3), 20, dtype=np.uint8),
    )
    result = magnifier.apply_update(
        MagnifierUpdate(
            requested=True,
            source_rgb_key="front",
            bbox_xyxy=(0.25, 0.25, 0.75, 0.75),
        ),
        first,
    )

    second = magnifier.observe(
        _observation(
            1,
            front=np.full((20, 40, 3), 90, dtype=np.uint8),
            wrist=np.full((20, 40, 3), 30, dtype=np.uint8),
        ),
        result,
    )
    third = magnifier.observe(
        _observation(
            2,
            front=np.full((32, 64, 3), 150, dtype=np.uint8),
            wrist=np.full((32, 64, 3), 40, dtype=np.uint8),
        )
    )

    assert second.magnifier is not None
    assert third.magnifier is not None
    assert second.magnifier.source_rgb_key == "front"
    assert second.magnifier.bbox_xyxy == (0.25, 0.25, 0.75, 0.75)
    assert int(second.magnifier.image.mean()) == 90
    assert int(third.magnifier.image.mean()) == 150
    assert third.magnifier.image.shape == (32, 64, 3)
    assert second.tool_feedback == (
        "Magnifier activated for source_rgb_key 'front' at "
        "bbox_xyxy=[0.25,0.25,0.75,0.75].",
    )


def test_bad_rgb_key_returns_public_feedback_and_keeps_previous_lens():
    magnifier = Magnifier(("front", "wrist"))
    observation = _observation(
        0,
        front=np.zeros((10, 20, 3), dtype=np.uint8),
        wrist=np.zeros((10, 20, 3), dtype=np.uint8),
    )
    magnifier.apply_update(
        MagnifierUpdate(
            requested=True,
            source_rgb_key="front",
            bbox_xyxy=(0.0, 0.0, 0.5, 0.5),
        ),
        observation,
    )

    rejected = magnifier.apply_update(
        MagnifierUpdate(
            requested=True,
            source_rgb_key="overhead_typo",
            bbox_xyxy=(0.1, 0.1, 0.9, 0.9),
        ),
        observation,
    )
    next_observation = magnifier.observe(observation, rejected)

    assert rejected is not None
    assert rejected.status == "rejected"
    assert "Available source RGB keys: front, wrist" in rejected.message
    assert next_observation.magnifier is not None
    assert next_observation.magnifier.source_rgb_key == "front"
    assert next_observation.magnifier.bbox_xyxy == (0.0, 0.0, 0.5, 0.5)
    assert next_observation.tool_feedback == (rejected.message,)


def test_new_valid_command_replaces_source_key_and_relative_bbox():
    magnifier = Magnifier(("front", "wrist"))
    observation = _observation(
        0,
        front=np.full((20, 40, 3), 25, dtype=np.uint8),
        wrist=np.full((20, 40, 3), 175, dtype=np.uint8),
    )
    magnifier.apply_update(
        MagnifierUpdate(
            requested=True,
            source_rgb_key="front",
            bbox_xyxy=(0.0, 0.0, 0.5, 0.5),
        ),
        observation,
    )

    result = magnifier.apply_update(
        MagnifierUpdate(
            requested=True,
            source_rgb_key="wrist",
            bbox_xyxy=(0.5, 0.5, 1.0, 1.0),
        ),
        observation,
    )
    updated = magnifier.observe(observation, result)

    assert result is not None
    assert result.status == "updated"
    assert updated.magnifier is not None
    assert updated.magnifier.source_rgb_key == "wrist"
    assert updated.magnifier.bbox_xyxy == (0.5, 0.5, 1.0, 1.0)
    assert int(updated.magnifier.image.mean()) == 175


def test_null_update_clears_persistent_lens():
    magnifier = Magnifier(("front",))
    observation = _observation(
        0,
        front=np.zeros((10, 20, 3), dtype=np.uint8),
    )
    magnifier.apply_update(
        MagnifierUpdate(
            requested=True,
            source_rgb_key="front",
            bbox_xyxy=(0.0, 0.0, 0.5, 0.5),
        ),
        observation,
    )

    result = magnifier.apply_update(
        MagnifierUpdate(requested=True, clear=True),
        observation,
    )
    cleared = magnifier.observe(observation, result)

    assert result is not None
    assert result.status == "cleared"
    assert cleared.magnifier is None
    assert cleared.tool_feedback == ("Magnifier cleared.",)
