#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mllm_teleop_benchmark import load_config
from mllm_teleop_benchmark.core.registry import (
    build_camera_manager,
    build_observation_provider,
    build_sim,
)
from mllm_teleop_benchmark.core.types import RLBenchAction
from mllm_teleop_benchmark.utils.seeding import episode_seed

REPORT_VERSION = "coppeliasim_preflight_v1"
DEFAULT_AXIS_OFFSET_METERS = 0.05
DEFAULT_MIN_PIXEL_DISPLACEMENT = 2.0


class PreflightFailure(RuntimeError):
    """A preflight capability or validation failure that must be reported."""


def project_world_point(
    world_point: Sequence[float],
    intrinsics: Any,
    camera_to_world: Any,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, object]:
    """Project one world point using RLBench/PyRep camera matrices."""
    point = np.asarray(world_point, dtype=np.float64)
    intrinsic_matrix = np.asarray(intrinsics, dtype=np.float64)
    extrinsic_matrix = np.asarray(camera_to_world, dtype=np.float64)
    if point.shape != (3,):
        raise PreflightFailure(
            f"World point must have shape (3,), got {point.shape}"
        )
    if intrinsic_matrix.shape != (3, 3):
        raise PreflightFailure(
            "Camera intrinsics must have shape (3, 3), got "
            f"{intrinsic_matrix.shape}"
        )
    if extrinsic_matrix.shape != (4, 4):
        raise PreflightFailure(
            "Camera extrinsics must have shape (4, 4), got "
            f"{extrinsic_matrix.shape}"
        )
    if (
        not np.all(np.isfinite(point))
        or not np.all(np.isfinite(intrinsic_matrix))
        or not np.all(np.isfinite(extrinsic_matrix))
    ):
        raise PreflightFailure("Projection inputs must contain only finite values")
    if image_width <= 0 or image_height <= 0:
        raise PreflightFailure("Projection image dimensions must be positive")

    try:
        world_to_camera = np.linalg.inv(extrinsic_matrix)
    except np.linalg.LinAlgError as exc:
        raise PreflightFailure("Camera extrinsics are singular") from exc

    camera_point = world_to_camera @ np.asarray([*point, 1.0])
    projected = intrinsic_matrix @ camera_point[:3]
    denominator = float(projected[2])
    depth = float(camera_point[2])
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return {
            "pixel": None,
            "camera_depth": depth,
            "in_front": False,
            "in_frame": False,
            "visible": False,
        }

    pixel_x = float(projected[0] / denominator)
    pixel_y = float(projected[1] / denominator)
    in_front = depth > 1e-9
    in_frame = (
        0.0 <= pixel_x < float(image_width)
        and 0.0 <= pixel_y < float(image_height)
    )
    return {
        "pixel": [pixel_x, pixel_y],
        "camera_depth": depth,
        "in_front": in_front,
        "in_frame": in_frame,
        "visible": bool(in_front and in_frame),
    }


def inspect_rgb_frame(frame: Any) -> dict[str, object]:
    """Return deterministic black/constant checks without writing an image."""
    array = np.asarray(frame)
    errors: list[str] = []
    if array.ndim != 3:
        errors.append(f"expected HWC RGB array, got shape {array.shape}")
    elif array.shape[-1] not in (3, 4):
        errors.append(f"expected 3 or 4 channels, got shape {array.shape}")
    if errors:
        return {
            "valid": False,
            "non_black": False,
            "non_constant": False,
            "errors": errors,
        }

    rgb = array[..., :3]
    if not np.issubdtype(rgb.dtype, np.number):
        errors.append(f"expected numeric RGB values, got dtype {rgb.dtype}")
    elif not np.all(np.isfinite(rgb)):
        errors.append("RGB frame contains non-finite values")
    if errors:
        return {
            "valid": False,
            "non_black": False,
            "non_constant": False,
            "errors": errors,
        }

    values = rgb.astype(np.float64, copy=False)
    if values.size and float(np.min(values)) >= 0.0 and float(np.max(values)) <= 1.0:
        values = values * 255.0
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    channel_standard_deviation = np.std(values, axis=(0, 1))
    standard_deviation = float(np.max(channel_standard_deviation))
    non_black = maximum > 1.0
    non_constant = standard_deviation > 1.0
    return {
        "valid": bool(non_black and non_constant),
        "shape": [int(value) for value in rgb.shape],
        "dtype": str(rgb.dtype),
        "minimum": minimum,
        "maximum": maximum,
        "standard_deviation": standard_deviation,
        "non_black": non_black,
        "non_constant": non_constant,
        "sha256": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
        "errors": [],
    }


def probe_camera_projection(
    native_observation: Any,
    *,
    camera_name: str = "front",
    frame: Any | None = None,
    axis_offset_meters: float = DEFAULT_AXIS_OFFSET_METERS,
    min_pixel_displacement: float = DEFAULT_MIN_PIXEL_DISPLACEMENT,
) -> dict[str, object]:
    """Probe projected world-axis directions from a current native observation."""
    if axis_offset_meters <= 0.0:
        raise PreflightFailure("Axis offset must be positive")
    if min_pixel_displacement < 0.0:
        raise PreflightFailure("Minimum pixel displacement cannot be negative")

    misc = _native_value(native_observation, "misc")
    if not isinstance(misc, Mapping):
        raise PreflightFailure("Native observation is missing misc camera data")
    intrinsics_key = f"{camera_name}_camera_intrinsics"
    extrinsics_key = f"{camera_name}_camera_extrinsics"
    if intrinsics_key not in misc or extrinsics_key not in misc:
        raise PreflightFailure(
            "Native observation misc is missing "
            f"{intrinsics_key!r} or {extrinsics_key!r}"
        )

    gripper_pose = np.asarray(
        _native_value(native_observation, "gripper_pose"),
        dtype=np.float64,
    )
    if gripper_pose.ndim != 1 or gripper_pose.size < 3:
        raise PreflightFailure(
            "Native observation must expose gripper_pose with at least 3 values"
        )
    eef_position = gripper_pose[:3]
    if not np.all(np.isfinite(eef_position)):
        raise PreflightFailure("Native gripper_pose contains non-finite values")

    if frame is None:
        frame = _native_value(native_observation, f"{camera_name}_rgb")
    frame_array = np.asarray(frame)
    if frame_array.ndim != 3:
        raise PreflightFailure(
            f"Camera {camera_name!r} frame must be HWC, got {frame_array.shape}"
        )
    image_height, image_width = frame_array.shape[:2]
    base = project_world_point(
        eef_position,
        misc[intrinsics_key],
        misc[extrinsics_key],
        image_width=int(image_width),
        image_height=int(image_height),
    )

    axes: dict[str, object] = {}
    for axis_index, axis_name in enumerate(("+x", "+y", "+z")):
        endpoint = eef_position.copy()
        endpoint[axis_index] += axis_offset_meters
        projected = project_world_point(
            endpoint,
            misc[intrinsics_key],
            misc[extrinsics_key],
            image_width=int(image_width),
            image_height=int(image_height),
        )
        displacement: list[float] | None = None
        magnitude = 0.0
        direction: str | None = None
        if base["pixel"] is not None and projected["pixel"] is not None:
            displacement_array = np.asarray(projected["pixel"]) - np.asarray(
                base["pixel"]
            )
            displacement = [
                float(displacement_array[0]),
                float(displacement_array[1]),
            ]
            magnitude = float(np.linalg.norm(displacement_array))
            if magnitude >= min_pixel_displacement:
                direction = _primary_image_direction(displacement_array)
        claimable = bool(
            base["visible"]
            and projected["visible"]
            and direction is not None
        )
        axes[axis_name] = {
            "offset_meters": float(axis_offset_meters),
            "endpoint": projected,
            "pixel_displacement": displacement,
            "pixel_displacement_magnitude": magnitude,
            "primary_image_direction": direction,
            "can_publicly_declare": claimable,
            "reason": (
                "visible displacement exceeds threshold"
                if claimable
                else "EEF/offset is not jointly visible or displacement is too small"
            ),
        }

    return {
        "camera": camera_name,
        "image_size": [int(image_width), int(image_height)],
        "eef_projection": base,
        "minimum_pixel_displacement": float(min_pixel_displacement),
        "axes": axes,
    }


def run_reset_canary(
    sim: Any,
    task_config: Any,
    camera_specs: Sequence[Any],
    seeds: Sequence[int],
    *,
    planning_delta: Sequence[float] = (0.0, 0.0, 0.0),
    axis_offset_meters: float = DEFAULT_AXIS_OFFSET_METERS,
    min_pixel_displacement: float = DEFAULT_MIN_PIXEL_DISPLACEMENT,
) -> dict[str, object]:
    """Run reset, image, projection, and planning probes without an agent."""
    if len(seeds) == 0:
        raise PreflightFailure("Reset canary requires at least one seed")
    delta = tuple(float(value) for value in planning_delta)
    if len(delta) != 3 or not all(np.isfinite(delta)):
        raise PreflightFailure("Planning delta must contain three finite values")
    camera_names = tuple(str(spec.name) for spec in camera_specs)
    if not camera_names:
        raise PreflightFailure("Reset canary requires at least one camera")

    seed_reports: list[dict[str, object]] = []
    failures: list[str] = []
    camera_hashes: dict[str, list[str]] = {name: [] for name in camera_names}
    projections: list[dict[str, object]] = []
    planning_attempts = 0
    planning_rejections = 0
    planning_errors = 0

    for seed in seeds:
        seed_report: dict[str, object] = {
            "seed": int(seed),
            "reset_succeeded": False,
            "initial_native_success": None,
            "cameras": {},
            "projection": None,
            "planning": {"attempted": False},
        }
        try:
            sim.reset(task_config, int(seed))
            seed_report["reset_succeeded"] = True
            state = sim.get_state()
            if not isinstance(state, Mapping) or "success" not in state:
                raise PreflightFailure(
                    "Adapter state does not expose native reset success"
                )
            initial_success = bool(state["success"])
            seed_report["initial_native_success"] = initial_success
            if initial_success:
                failures.append(
                    f"seed {seed}: native task is already successful after reset"
                )

            rendered_frames: dict[str, Any] = {}
            for spec in camera_specs:
                name = str(spec.name)
                try:
                    frame = sim.render(spec)
                    rendered_frames[name] = frame
                    frame_report = inspect_rgb_frame(frame)
                except Exception as exc:
                    frame_report = {
                        "valid": False,
                        "non_black": False,
                        "non_constant": False,
                        "errors": [f"{type(exc).__name__}: {exc}"],
                    }
                seed_report["cameras"][name] = frame_report
                frame_hash = frame_report.get("sha256")
                if isinstance(frame_hash, str):
                    camera_hashes[name].append(frame_hash)
                if not frame_report["valid"]:
                    failures.append(
                        f"seed {seed}: camera {name!r} is black, constant, or invalid"
                    )

            if "front" in rendered_frames:
                try:
                    projection = probe_camera_projection(
                        sim.last_native_obs,
                        camera_name="front",
                        frame=rendered_frames["front"],
                        axis_offset_meters=axis_offset_meters,
                        min_pixel_displacement=min_pixel_displacement,
                    )
                    seed_report["projection"] = projection
                    projections.append(projection)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    seed_report["projection"] = {"error": message}
                    failures.append(f"seed {seed}: front projection failed: {message}")

            if not initial_success:
                seed_report["planning"] = {"attempted": True}
                planning_attempts += 1
                try:
                    step_result = sim.step(
                        RLBenchAction(
                            delta_position=delta,
                            delta_rotation_rpy=(0.0, 0.0, 0.0),
                            gripper=1.0,
                        )
                    )
                    rejected_value = step_result.info.get("action_rejected")
                    if rejected_value is None:
                        rejected_value = getattr(
                            sim, "last_action_rejected", None
                        )
                    if rejected_value is None:
                        raise PreflightFailure(
                            "Adapter did not report action_rejected"
                        )
                    rejected = bool(rejected_value)
                    planning_rejections += int(rejected)
                    seed_report["planning"] = {
                        "attempted": True,
                        "action_rejected": rejected,
                    }
                except Exception as exc:
                    planning_errors += 1
                    message = f"{type(exc).__name__}: {exc}"
                    seed_report["planning"] = {
                        "attempted": True,
                        "error": message,
                    }
                    failures.append(f"seed {seed}: planning probe failed: {message}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            seed_report["reset_error"] = message
            failures.append(f"seed {seed}: reset failed: {message}")
        seed_reports.append(seed_report)

    camera_summary: dict[str, object] = {}
    for camera_name, hashes in camera_hashes.items():
        all_identical = len(hashes) >= 2 and len(set(hashes)) == 1
        if all_identical:
            failures.append(
                f"camera {camera_name!r}: all successful reset frames are identical"
            )
        camera_summary[camera_name] = {
            "valid_frame_count": len(hashes),
            "all_reset_frames_identical": all_identical,
            "cross_seed_variation_assessed": len(hashes) >= 2,
        }

    return {
        "report_version": REPORT_VERSION,
        "kind": "reset_canary",
        "status": "pass" if not failures else "fail",
        "model_api_called": False,
        "artifact_scope": "preflight_report_only",
        "seed_count": len(seeds),
        "seeds": seed_reports,
        "camera_summary": camera_summary,
        "projection_summary": summarize_projection_probes(projections),
        "planning_probe": {
            "delta_position": list(delta),
            "attempts": planning_attempts,
            "rejections": planning_rejections,
            "errors": planning_errors,
            "rejection_rate": (
                float(planning_rejections / planning_attempts)
                if planning_attempts
                else None
            ),
            "recordable": planning_attempts > 0 and planning_errors == 0,
        },
        "failures": failures,
    }


def summarize_projection_probes(
    probes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    axis_summary: dict[str, object] = {}
    allowed_axes: list[str] = []
    withheld_axes: list[str] = []
    for axis_name in ("+x", "+y", "+z"):
        samples = [
            probe["axes"][axis_name]
            for probe in probes
            if isinstance(probe.get("axes"), Mapping)
            and axis_name in probe["axes"]
        ]
        directions = {
            str(sample["primary_image_direction"])
            for sample in samples
            if sample.get("can_publicly_declare")
            and sample.get("primary_image_direction") is not None
        }
        all_claimable = bool(samples) and all(
            bool(sample.get("can_publicly_declare")) for sample in samples
        )
        stable = all_claimable and len(directions) == 1
        if stable:
            allowed_axes.append(axis_name)
        else:
            withheld_axes.append(axis_name)
        axis_summary[axis_name] = {
            "sample_count": len(samples),
            "all_samples_claimable": all_claimable,
            "stable_direction": stable,
            "observed_primary_directions": sorted(directions),
            "can_publicly_declare": stable,
        }
    if len(allowed_axes) == 3:
        declaration = "all_axes"
    elif allowed_axes:
        declaration = "partial"
    else:
        declaration = "none"
    return {
        "sample_count": len(probes),
        "axis_summary": axis_summary,
        "public_direction_declaration": declaration,
        "allowed_axes": allowed_axes,
        "withheld_axes": withheld_axes,
    }


def run_close_box_scripted_success_canary(
    sim: Any,
    task_config: Any,
    *,
    seed: int,
) -> dict[str, object]:
    """Execute only the official CloseBox live-demo waypoint mechanism."""
    config = getattr(sim, "config", None)
    if getattr(config, "type", None) != "colosseum":
        raise PreflightFailure(
            "CloseBox scripted-success canary requires simulator.type=colosseum"
        )
    if getattr(sim, "task_name", None) != "close_box":
        raise PreflightFailure(
            "CloseBox scripted-success canary supports only official close_box"
        )

    sim.reset(task_config, int(seed))
    state = sim.get_state()
    if not isinstance(state, Mapping) or "success" not in state:
        raise PreflightFailure(
            "CloseBox adapter state does not expose native reset success"
        )
    initial_success = bool(state["success"])
    if initial_success:
        raise PreflightFailure(
            f"CloseBox seed {seed} is already successful immediately after reset"
        )
    task_environment = getattr(sim, "task", None)
    native_task = getattr(task_environment, "_task", None)
    get_waypoints = getattr(native_task, "get_waypoints", None)
    get_demos = getattr(task_environment, "get_demos", None)
    native_success = getattr(native_task, "success", None)
    missing = [
        name
        for name, capability in (
            ("Task.get_waypoints", get_waypoints),
            ("TaskEnvironment.get_demos", get_demos),
            ("Task.success", native_success),
        )
        if not callable(capability)
    ]
    if missing:
        raise PreflightFailure(
            "Cannot safely execute CloseBox scripted success: the pinned "
            "runtime lacks official capabilities "
            + ", ".join(missing)
            + ". No guessed pose script was executed."
        )

    waypoints = get_waypoints()
    if not isinstance(waypoints, Sequence) or len(waypoints) == 0:
        raise PreflightFailure(
            "Cannot safely execute CloseBox scripted success: official task "
            "exposes no waypoints. No guessed pose script was executed."
        )
    signature = inspect.signature(get_demos)
    if "live_demos" not in signature.parameters:
        raise PreflightFailure(
            "Cannot safely execute CloseBox scripted success: official "
            "TaskEnvironment.get_demos lacks live_demos support. No guessed "
            "pose script was executed."
        )

    kwargs: dict[str, object] = {"live_demos": True}
    if "max_attempts" in signature.parameters:
        kwargs["max_attempts"] = 1
    numpy_random_state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        demos = get_demos(1, **kwargs)
    except Exception as exc:
        raise PreflightFailure(
            "Official CloseBox live-demo waypoint execution failed; no "
            f"fallback pose script was attempted: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        np.random.set_state(numpy_random_state)

    if not isinstance(demos, Sequence) or len(demos) != 1:
        raise PreflightFailure(
            "Official CloseBox live-demo did not return exactly one demo"
        )
    result = native_success()
    if not isinstance(result, tuple) or len(result) != 2:
        raise PreflightFailure(
            "Official CloseBox Task.success() did not return (success, terminate)"
        )
    if not bool(result[0]):
        raise PreflightFailure(
            "Official CloseBox live-demo completed without native success"
        )
    return {
        "report_version": REPORT_VERSION,
        "kind": "closebox_scripted_success",
        "status": "pass",
        "model_api_called": False,
        "artifact_scope": "preflight_report_only",
        "seed": int(seed),
        "method": "official_task_environment_live_demo_waypoints",
        "waypoint_count": len(waypoints),
        "demo_observation_count": len(demos[0]),
        "initial_native_success": initial_success,
        "final_native_success": True,
        "final_native_terminate": bool(result[1]),
        "guessed_pose_script_executed": False,
        "failures": [],
    }


def _primary_image_direction(displacement: np.ndarray) -> str:
    horizontal = float(displacement[0])
    vertical = float(displacement[1])
    if abs(horizontal) >= abs(vertical):
        return "image_right" if horizontal > 0.0 else "image_left"
    return "image_down" if vertical > 0.0 else "image_up"


def _native_value(native_observation: Any, key: str) -> Any:
    if isinstance(native_observation, Mapping):
        return native_observation.get(key)
    return getattr(native_observation, key, None)


def _write_report(report: Mapping[str, object], output: str | None) -> None:
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run RLBench/Colosseum preflight probes without a model API."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reset_parser = subparsers.add_parser(
        "reset-canary",
        help="Run camera projection, reset, image, and planning probes.",
    )
    reset_parser.add_argument("--config", required=True)
    reset_parser.add_argument("--seed-count", type=int, default=10)
    reset_parser.add_argument("--base-seed", type=int)
    reset_parser.add_argument(
        "--planning-delta",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
    )
    reset_parser.add_argument(
        "--axis-offset-meters",
        type=float,
        default=DEFAULT_AXIS_OFFSET_METERS,
    )
    reset_parser.add_argument(
        "--min-pixel-displacement",
        type=float,
        default=DEFAULT_MIN_PIXEL_DISPLACEMENT,
    )
    reset_parser.add_argument("--output")

    closebox_parser = subparsers.add_parser(
        "closebox-scripted-success",
        help="Run the official Colosseum CloseBox live-demo waypoints.",
    )
    closebox_parser.add_argument("--config", required=True)
    closebox_parser.add_argument("--seed", type=int)
    closebox_parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    sim: Any | None = None
    report: dict[str, object]
    try:
        config = load_config(args.config)
        if config.simulator.type not in {"rlbench", "colosseum"}:
            raise PreflightFailure(
                "CoppeliaSim preflight supports only rlbench or colosseum configs"
            )
        sim = build_sim(config)
        camera_manager = build_camera_manager(
            config.camera, config.benchmark.seed
        )
        observation_provider = build_observation_provider(config, camera_manager)
        observation_provider.configure_sim(sim)

        if args.command == "reset-canary":
            if args.seed_count <= 0:
                raise PreflightFailure("--seed-count must be positive")
            base_seed = (
                config.benchmark.seed
                if args.base_seed is None
                else int(args.base_seed)
            )
            seeds = tuple(
                episode_seed(base_seed, episode_id)
                for episode_id in range(args.seed_count)
            )
            report = run_reset_canary(
                sim,
                config.task,
                camera_manager.get_active_cameras(),
                seeds,
                planning_delta=args.planning_delta,
                axis_offset_meters=args.axis_offset_meters,
                min_pixel_displacement=args.min_pixel_displacement,
            )
        else:
            seed = (
                config.benchmark.seed if args.seed is None else int(args.seed)
            )
            report = run_close_box_scripted_success_canary(
                sim,
                config.task,
                seed=seed,
            )
        report["config"] = str(Path(args.config).resolve())
        report["simulator"] = config.simulator.type
    except Exception as exc:
        report = {
            "report_version": REPORT_VERSION,
            "kind": args.command.replace("-", "_"),
            "status": "fail",
            "model_api_called": False,
            "artifact_scope": "preflight_report_only",
            "config": str(Path(args.config).resolve()),
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    finally:
        if sim is not None:
            try:
                sim.close()
            except Exception as exc:
                report["status"] = "fail"
                failures = report.setdefault("failures", [])
                if isinstance(failures, list):
                    failures.append(
                        f"simulator close failed: {type(exc).__name__}: {exc}"
                    )

    _write_report(report, args.output)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
