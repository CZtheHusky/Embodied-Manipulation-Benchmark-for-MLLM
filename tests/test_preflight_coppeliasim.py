import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.preflight_coppeliasim import (
    PreflightFailure,
    inspect_rgb_frame,
    probe_camera_projection,
    project_world_point,
    run_close_box_scripted_success_canary,
    run_reset_canary,
    summarize_projection_probes,
)


def _native_observation(seed: int = 0):
    rows, columns = np.indices((80, 100))
    frame = np.stack(
        (
            (rows + seed) % 255,
            (columns + 2 * seed) % 255,
            (rows + columns + 3 * seed) % 255,
        ),
        axis=-1,
    ).astype(np.uint8)
    return SimpleNamespace(
        front_rgb=frame,
        gripper_pose=np.asarray(
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
        misc={
            "front_camera_intrinsics": np.asarray(
                [
                    [100.0, 0.0, 50.0],
                    [0.0, 100.0, 40.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "front_camera_extrinsics": np.eye(4),
        },
    )


class FakeAdapter:
    def __init__(self, *, successful_seed: int | None = None):
        self.successful_seed = successful_seed
        self.last_native_obs = None
        self.last_action_rejected = False
        self.current_seed = 0
        self.step_seeds = []

    def reset(self, _task_config, seed):
        self.current_seed = int(seed)
        self.last_native_obs = _native_observation(seed)
        return SimpleNamespace()

    def render(self, camera):
        assert camera.name == "front"
        return self.last_native_obs.front_rgb

    def get_state(self):
        return {"success": self.current_seed == self.successful_seed}

    def step(self, _action):
        self.step_seeds.append(self.current_seed)
        self.last_action_rejected = self.current_seed % 2 == 1
        return SimpleNamespace(
            info={"action_rejected": self.last_action_rejected}
        )


def test_projection_math_reports_pixel_displacements_and_visibility():
    native = _native_observation()

    origin = project_world_point(
        native.gripper_pose[:3],
        native.misc["front_camera_intrinsics"],
        native.misc["front_camera_extrinsics"],
        image_width=100,
        image_height=80,
    )
    probe = probe_camera_projection(
        native,
        camera_name="front",
        axis_offset_meters=0.05,
        min_pixel_displacement=2.0,
    )

    assert origin["visible"] is True
    assert origin["pixel"] == pytest.approx([50.0, 40.0])
    assert probe["axes"]["+x"]["pixel_displacement"] == pytest.approx(
        [2.5, 0.0]
    )
    assert probe["axes"]["+x"]["primary_image_direction"] == "image_right"
    assert probe["axes"]["+x"]["can_publicly_declare"] is True
    assert probe["axes"]["+y"]["pixel_displacement"] == pytest.approx(
        [0.0, 2.5]
    )
    assert probe["axes"]["+y"]["primary_image_direction"] == "image_down"
    assert probe["axes"]["+y"]["can_publicly_declare"] is True
    assert probe["axes"]["+z"]["pixel_displacement"] == pytest.approx(
        [0.0, 0.0]
    )
    assert probe["axes"]["+z"]["can_publicly_declare"] is False

    summary = summarize_projection_probes([probe])
    assert summary["public_direction_declaration"] == "partial"
    assert summary["allowed_axes"] == ["+x", "+y"]
    assert summary["withheld_axes"] == ["+z"]


@pytest.mark.parametrize(
    ("frame", "non_black", "non_constant"),
    [
        (np.zeros((8, 8, 3), dtype=np.uint8), False, False),
        (np.full((8, 8, 3), 127, dtype=np.uint8), True, False),
        (
            np.broadcast_to(
                np.asarray([20, 127, 240], dtype=np.uint8),
                (8, 8, 3),
            ),
            True,
            False,
        ),
        (_native_observation().front_rgb, True, True),
    ],
)
def test_frame_inspection_rejects_black_and_constant_frames(
    frame, non_black, non_constant
):
    report = inspect_rgb_frame(frame)

    assert report["non_black"] is non_black
    assert report["non_constant"] is non_constant
    assert report["valid"] is (non_black and non_constant)


def test_ten_seed_canary_rejects_initial_success_and_records_planning_rate():
    sim = FakeAdapter(successful_seed=5)
    camera = SimpleNamespace(name="front")

    report = run_reset_canary(
        sim,
        SimpleNamespace(),
        (camera,),
        tuple(range(10)),
    )

    assert report["status"] == "fail"
    assert report["seed_count"] == 10
    assert report["seeds"][5]["initial_native_success"] is True
    assert 5 not in sim.step_seeds
    assert report["planning_probe"] == {
        "delta_position": [0.0, 0.0, 0.0],
        "attempts": 9,
        "rejections": 4,
        "errors": 0,
        "rejection_rate": pytest.approx(4 / 9),
        "recordable": True,
    }
    assert any("already successful" in failure for failure in report["failures"])
    assert report["camera_summary"]["front"] == {
        "valid_frame_count": 10,
        "all_reset_frames_identical": False,
        "cross_seed_variation_assessed": True,
    }
    assert report["model_api_called"] is False
    assert report["artifact_scope"] == "preflight_report_only"
    serialized = json.dumps(report)
    assert "camera_intrinsics" not in serialized
    assert "camera_extrinsics" not in serialized
    assert "gripper_pose" not in serialized


def test_ten_seed_canary_fails_when_camera_becomes_black():
    class BlackFrameAdapter(FakeAdapter):
        def render(self, camera):
            if self.current_seed == 7:
                return np.zeros((80, 100, 3), dtype=np.uint8)
            return super().render(camera)

    report = run_reset_canary(
        BlackFrameAdapter(),
        SimpleNamespace(),
        (SimpleNamespace(name="front"),),
        tuple(range(10)),
    )

    assert report["status"] == "fail"
    assert report["seeds"][7]["cameras"]["front"]["non_black"] is False
    assert any("camera 'front'" in failure for failure in report["failures"])


def test_closebox_scripted_success_fails_loudly_without_official_capabilities():
    native_task = SimpleNamespace()
    sim = SimpleNamespace(
        config=SimpleNamespace(type="colosseum"),
        task_name="close_box",
        task=SimpleNamespace(_task=native_task),
        reset=lambda _task_config, _seed: None,
        get_state=lambda: {"success": False},
    )

    with pytest.raises(
        PreflightFailure,
        match=r"lacks official capabilities.*No guessed pose script was executed",
    ):
        run_close_box_scripted_success_canary(
            sim,
            SimpleNamespace(),
            seed=11,
        )


def test_closebox_scripted_success_uses_only_official_live_demo_waypoints():
    class NativeTask:
        def __init__(self):
            self.completed = False

        def get_waypoints(self):
            return [object(), object()]

        def success(self):
            return self.completed, self.completed

    class TaskEnvironment:
        def __init__(self, native_task):
            self._task = native_task
            self.calls = []

        def get_demos(self, amount, live_demos=False, max_attempts=10):
            self.calls.append((amount, live_demos, max_attempts))
            self._task.completed = True
            return [[object(), object(), object()]]

    native_task = NativeTask()
    task_environment = TaskEnvironment(native_task)
    sim = SimpleNamespace(
        config=SimpleNamespace(type="colosseum"),
        task_name="close_box",
        task=task_environment,
        reset=lambda _task_config, _seed: None,
        get_state=lambda: {"success": False},
    )

    report = run_close_box_scripted_success_canary(
        sim,
        SimpleNamespace(),
        seed=11,
    )

    assert task_environment.calls == [(1, True, 1)]
    assert report["status"] == "pass"
    assert report["method"] == "official_task_environment_live_demo_waypoints"
    assert report["waypoint_count"] == 2
    assert report["demo_observation_count"] == 3
    assert report["final_native_success"] is True
    assert report["guessed_pose_script_executed"] is False
