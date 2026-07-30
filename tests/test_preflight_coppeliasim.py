import json
from types import SimpleNamespace

import numpy as np
import pytest

from mllm_teleop_benchmark.core.types import RLBenchAction
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


def _restricted_closebox_config(
    *,
    max_steps=100,
    scales=None,
    continuous=False,
):
    return SimpleNamespace(
        simulator=SimpleNamespace(type="colosseum"),
        task=SimpleNamespace(max_steps=max_steps),
        control_interface=SimpleNamespace(
            active_dimensions=("dx", "dy", "dz"),
            gripper="none",
            channel_kind="continuous" if continuous else "discrete",
            channel_values=(-1.0, 0.0, 1.0),
            channel_minimum=-30.0 if continuous else None,
            channel_maximum=30.0 if continuous else None,
            mapping_strategy="fixed",
            frame_transform="identity",
        ),
        controller=SimpleNamespace(
            type="rlbench_delta_eef",
            scales=scales or {"dx": 0.01, "dy": 0.01, "dz": 0.01},
            params={"default_gripper": 1.0},
        ),
    )


class FakeWaypointObject:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float64)

    def get_position(self):
        return self.position.copy()


class FakeWaypoint:
    def __init__(self, position):
        self.object = FakeWaypointObject(position)

    def get_waypoint_object(self):
        return self.object

    def get_ext(self):
        raise AssertionError("waypoint extensions must not be inspected")

    def start_of_path(self):
        raise AssertionError("official waypoint executor must not run")

    def end_of_path(self):
        raise AssertionError("official waypoint executor must not run")


class FakeCloseBoxNativeTask:
    def __init__(self):
        self.completed = False
        self.waypoints = [
            FakeWaypoint((0.04, 0.0, 1.0)),
            FakeWaypoint((0.04, 0.04, 1.0)),
        ]

    def get_waypoints(self):
        return self.waypoints

    def success(self):
        return self.completed, self.completed


class FakeCloseBoxTaskEnvironment:
    def __init__(self, native_task):
        self._task = native_task
        self.get_demos_called = False

    def get_demos(self, *_args, **_kwargs):
        self.get_demos_called = True
        raise AssertionError("legacy full-control demo path must not be called")


class FakeRestrictedCloseBoxSim:
    def __init__(self, *, reject_first=False, reject_all=False):
        self.config = SimpleNamespace(type="colosseum")
        self.task_name = "close_box"
        self.native_task = FakeCloseBoxNativeTask()
        self.task = FakeCloseBoxTaskEnvironment(self.native_task)
        self.reject_first = reject_first
        self.reject_all = reject_all
        self.rejected_once = False
        self.last_action_rejected = False
        self.actions = []
        self.last_native_obs = None

    def reset(self, _task_config, _seed):
        self.native_task.completed = False
        self.rejected_once = False
        self.actions = []
        self.last_native_obs = SimpleNamespace(
            gripper_pose=np.asarray(
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float64,
            )
        )

    def get_state(self):
        return {"success": self.native_task.completed}

    def step(self, action):
        self.actions.append(action)
        rejected = self.reject_all or (
            self.reject_first and not self.rejected_once
        )
        self.rejected_once = self.rejected_once or rejected
        self.last_action_rejected = rejected
        if not rejected:
            pose = self.last_native_obs.gripper_pose.copy()
            pose[:3] += np.asarray(action.delta_position)
            self.last_native_obs.gripper_pose = pose
            self.native_task.completed = bool(
                np.allclose(pose[:3], (0.04, 0.04, 1.0), atol=1e-12)
            )
        return SimpleNamespace(info={"action_rejected": rejected})


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
        "acceptances": 5,
        "errors": 0,
        "acceptance_required": False,
        "minimum_eef_displacement_meters": 1e-6,
        "displacement_evidence_count": 0,
        "missing_displacement_evidence_count": 0,
        "moving_acceptances": 0,
        "displacement_magnitudes_meters": [],
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


def test_nonzero_planning_probe_fails_when_every_action_is_rejected():
    class RejectAllAdapter(FakeAdapter):
        def step(self, _action):
            self.step_seeds.append(self.current_seed)
            self.last_action_rejected = True
            return SimpleNamespace(info={"action_rejected": True})

    report = run_reset_canary(
        RejectAllAdapter(),
        SimpleNamespace(),
        (SimpleNamespace(name="front"),),
        tuple(range(3)),
        planning_delta=(0.01, 0.0, 0.0),
    )

    assert report["status"] == "fail"
    assert report["planning_probe"]["acceptance_required"] is True
    assert report["planning_probe"]["acceptances"] == 0
    assert any(
        "not accepted for any reset seed" in failure
        for failure in report["failures"]
    )


def test_nonzero_planning_probe_passes_with_measured_eef_displacement():
    class MovingAdapter(FakeAdapter):
        def step(self, action):
            result = super().step(action)
            if not self.last_action_rejected:
                pose = self.last_native_obs.gripper_pose.copy()
                pose[:3] += np.asarray(action.delta_position)
                self.last_native_obs.gripper_pose = pose
            return result

    report = run_reset_canary(
        MovingAdapter(),
        SimpleNamespace(),
        (SimpleNamespace(name="front"),),
        tuple(range(3)),
        planning_delta=(0.01, 0.0, 0.0),
    )

    assert report["status"] == "pass"
    assert report["planning_probe"]["acceptances"] == 2
    assert report["planning_probe"]["displacement_evidence_count"] == 2
    assert report["planning_probe"]["missing_displacement_evidence_count"] == 0
    assert report["planning_probe"]["moving_acceptances"] == 2
    assert report["planning_probe"][
        "displacement_magnitudes_meters"
    ] == pytest.approx([0.01, 0.01])
    assert report["seeds"][0]["planning"][
        "eef_displacement_magnitude_meters"
    ] == pytest.approx(0.01)


def test_nonzero_planning_probe_requires_motion_beyond_float_noise():
    class NoisyAcceptedAdapter(FakeAdapter):
        def step(self, action):
            result = super().step(action)
            if not self.last_action_rejected:
                pose = self.last_native_obs.gripper_pose.copy()
                pose[0] += 5e-7
                self.last_native_obs.gripper_pose = pose
            return result

    report = run_reset_canary(
        NoisyAcceptedAdapter(),
        SimpleNamespace(),
        (SimpleNamespace(name="front"),),
        tuple(range(3)),
        planning_delta=(0.01, 0.0, 0.0),
    )

    # Planner acceptance and float jitter do not prove the arm executed the target.
    assert report["planning_probe"]["acceptances"] == 2
    assert report["planning_probe"]["displacement_evidence_count"] == 2
    assert report["planning_probe"]["moving_acceptances"] == 0
    assert report["planning_probe"][
        "displacement_magnitudes_meters"
    ] == pytest.approx([5e-7, 5e-7])
    assert report["status"] == "fail"
    assert any(
        "no accepted nonzero planning probe moved the end effector" in failure
        for failure in report["failures"]
    )


def test_nonzero_accepted_probe_fails_without_post_step_pose_evidence():
    class MissingPostStepPoseAdapter(FakeAdapter):
        def step(self, action):
            result = super().step(action)
            if not self.last_action_rejected:
                self.last_native_obs.gripper_pose = None
            return result

    report = run_reset_canary(
        MissingPostStepPoseAdapter(),
        SimpleNamespace(),
        (SimpleNamespace(name="front"),),
        (0,),
        planning_delta=(0.01, 0.0, 0.0),
    )

    assert report["planning_probe"]["acceptances"] == 1
    assert report["planning_probe"]["displacement_evidence_count"] == 0
    assert report["planning_probe"]["missing_displacement_evidence_count"] == 1
    assert report["status"] == "fail"
    assert any(
        "accepted nonzero planning probe lacks finite gripper pose evidence"
        in failure
        for failure in report["failures"]
    )
    planning_report = report["seeds"][0]["planning"]
    assert planning_report == {
        "attempted": True,
        "action_rejected": False,
        "displacement_evidence_available": False,
    }
    serialized = json.dumps(report)
    assert "gripper_pose" not in serialized
    assert "[0.0, 0.0, 2.0]" not in serialized


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
    sim = FakeRestrictedCloseBoxSim()
    sim.task._task.get_waypoints = None

    with pytest.raises(
        PreflightFailure,
        match=r"lacks official Task.get_waypoints.*No guessed pose script",
    ):
        run_close_box_scripted_success_canary(
            sim,
            _restricted_closebox_config(),
            seed=11,
        )
    assert sim.task.get_demos_called is False


def test_closebox_scripted_success_uses_only_restricted_benchmark_actions():
    sim = FakeRestrictedCloseBoxSim(reject_first=True)
    report = run_close_box_scripted_success_canary(
        sim,
        _restricted_closebox_config(),
        seed=11,
    )

    allowed_components = {-0.01, 0.0, 0.01}
    assert sim.actions
    assert all(
        isinstance(action, RLBenchAction)
        and set(action.delta_position).issubset(allowed_components)
        and action.delta_rotation_rpy == (0.0, 0.0, 0.0)
        and action.gripper == 1.0
        for action in sim.actions
    )
    assert sim.task.get_demos_called is False
    assert report["status"] == "pass"
    assert report["method"] == "official_waypoint_positions_via_rlbench_action"
    assert report["waypoint_count"] == 2
    assert report["actions_attempted"] == 9
    assert report["planning_rejections"] == 1
    assert report["planning_rejection_rate"] == pytest.approx(1.0 / 9.0)
    assert report["adapter_native_success_checks"] >= len(sim.actions) + 1
    assert report["final_native_success"] is True
    assert report["official_live_demo_called"] is False
    assert report["waypoint_extensions_executed"] is False
    assert report["waypoint_coordinates_recorded"] is False
    assert report["guessed_pose_script_executed"] is False
    assert report["action_contract"]["action_budget"] == 100
    assert "waypoint_positions" not in report


def test_closebox_scripted_success_accepts_model_continuous_action_schema():
    sim = FakeRestrictedCloseBoxSim(reject_first=True)

    report = run_close_box_scripted_success_canary(
        sim,
        _restricted_closebox_config(continuous=True),
        seed=11,
    )

    assert report["status"] == "pass"
    assert all(
        set(action.delta_position).issubset({-0.01, 0.0, 0.01})
        for action in sim.actions
    )


def test_closebox_scripted_success_exhaustion_reports_planning_rejections():
    sim = FakeRestrictedCloseBoxSim(reject_all=True)

    with pytest.raises(
        PreflightFailure,
        match="action budget exhausted after 3 actions",
    ) as captured:
        run_close_box_scripted_success_canary(
            sim,
            _restricted_closebox_config(),
            seed=11,
            max_actions=3,
        )

    report = captured.value.report
    assert report is not None
    assert report["status"] == "fail"
    assert report["actions_attempted"] == 3
    assert report["planning_rejections"] == 3
    assert report["planning_rejection_rate"] == 1.0
    assert sim.task.get_demos_called is False


@pytest.mark.parametrize(
    ("scales", "message"),
    [
        (
            {"dx": 0.02, "dy": 0.01, "dz": 0.02},
            "dx/dy/dz scales must be equal",
        ),
        (
            {"dx": 0.02, "dy": 0.02, "dz": 0.02},
            "must equal the 0.01-meter benchmark control step",
        ),
    ],
)
def test_closebox_scripted_success_rejects_controller_scale_mismatch(
    scales, message
):
    sim = FakeRestrictedCloseBoxSim()

    with pytest.raises(
        PreflightFailure,
        match=message,
    ):
        run_close_box_scripted_success_canary(
            sim,
            _restricted_closebox_config(scales=scales),
            seed=11,
        )

    assert sim.actions == []
    assert sim.task.get_demos_called is False
