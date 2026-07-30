#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import numpy as np
from PIL import Image


EPISODE_RE = re.compile(r"episode_(\d+)$")
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".yaml", ".yml"}
HIDDEN_CONTROL_NAMES = {
    "permuted_sign_flipped",
    "random_rotation_2d",
    "random_rotation_3d",
    "rotation_matrix",
    "quaternion_xyzw",
}
PRIVILEGED_KEYS = {
    "reward",
    "success",
    "seed",
    "sim_action",
    "simulator_action",
    "distance_to_target",
    "eef_pos",
    "target_position",
    "robot_proprio",
    "native_action",
    "benchmark_applied_native_action",
    "benchmark_absolute_gripper",
    "target_qpos",
}
SECRET_PATTERNS = {
    "authorization": re.compile(
        r"""(?ix)\bauthorization\b["']?\s*[:=]\s*["']?(?:bearer\s+)?
        (?!none\b|null\b|\[redacted\]|<redacted>)[^\s"',}\]]{8,}"""
    ),
    "api_key": re.compile(
        r"""(?ix)
        \b(?:openai[_-]?api[_-]?key|x[_-]?api[_-]?key|api[_-]?key)\b
        ["']?\s*[:=]\s*["']?
        (?!none\b|null\b|\[redacted\]|<redacted>)[^\s"',}\]]{8,}
        |\bsk-[A-Za-z0-9_-]{8,}
        """
    ),
    "base64_data_url": re.compile(
        r"(?i)data:[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+(?:;[^,\s]*)?;base64,"
    ),
}


class RunAuditor:
    def __init__(self, root: Path, ffprobe_binary: str) -> None:
        self.root = root
        self.ffprobe_binary = ffprobe_binary
        self.ffprobe_path = shutil.which(ffprobe_binary)
        self.anomalies: list[dict[str, str]] = []
        self.image_references = 0
        self.image_files: dict[str, dict[str, Any]] = {}
        self.image_missing: set[str] = set()
        self.image_invalid: set[str] = set()
        self.image_mismatches: set[str] = set()
        self.black_screens: dict[str, dict[str, float]] = {}
        self.image_evidence: dict[tuple[int, str, int], str] = {}
        self.image_reference_hash_conflicts: set[tuple[int, str, int]] = set()
        self.frozen_cameras: set[tuple[int, str]] = set()
        self.native_action_required = False
        self.robocasa_action_required = False
        self.rlbench_action_required = False
        self.response_model_required = False
        self.requested_model: str | None = None
        self.response_models: dict[str, str] = {}
        self.response_model_missing: set[str] = set()
        self.response_model_invalid: set[str] = set()

    def run(self) -> dict[str, Any]:
        manifest = self.read_json(self.root / "run_manifest.json", "incomplete")
        summary = self.read_json(self.root / "run_summary.json", "incomplete")
        schema_version = _nested_int(manifest, "schema_version")
        self.response_model_required = (
            schema_version is not None and schema_version >= 2
        )
        self.requested_model = _nested_str(manifest, "model", "requested_model")
        if self.response_model_required and not self.requested_model:
            self.add(
                "error",
                "requested_model_missing",
                "run_manifest.json",
                "model.requested_model must be a non-empty string",
            )
        simulator_type = _nested_str(manifest, "simulator", "type")
        benchmark_name = _nested_str(manifest, "benchmark", "name")
        self.native_action_required = simulator_type == "simpler_env" or (
            simulator_type is None
            and isinstance(benchmark_name, str)
            and benchmark_name.startswith("simpler_env_")
        )
        self.robocasa_action_required = simulator_type == "robocasa" or (
            simulator_type is None
            and isinstance(benchmark_name, str)
            and benchmark_name.startswith("robocasa_")
        )
        self.rlbench_action_required = simulator_type in {"rlbench", "colosseum"}
        source_revision = self.audit_source_revision(manifest)
        requested = _nested_int(manifest, "benchmark", "num_episodes")
        if manifest is not None and requested is None:
            self.add("error", "requested_episode_count_invalid", "run_manifest.json")

        summarized = self.summary_episodes(summary)
        if requested is not None and summary is not None and requested != len(summarized):
            self.add(
                "incomplete",
                "requested_episode_count_mismatch",
                "run_summary.json",
                f"requested={requested}, summarized={len(summarized)}",
            )
        episode_dirs = self.episode_dirs()
        if not episode_dirs and not summarized:
            self.add("incomplete", "episodes_missing", ".")
        if summary is not None:
            for episode_id in sorted(set(episode_dirs) - set(summarized)):
                self.add(
                    "incomplete",
                    "episode_not_in_run_summary",
                    self.rel(episode_dirs[episode_id]),
                )
        for episode_id in sorted(set(summarized) - set(episode_dirs)):
            self.add(
                "incomplete",
                "episode_directory_missing",
                f"episode_{episode_id:04d}",
            )

        episodes = [
            self.audit_episode(episode_id, episode_dirs.get(episode_id))
            for episode_id in sorted(set(episode_dirs) | set(summarized))
        ]
        response_model = self.response_model_summary()
        privacy = self.scan_prompt_privacy()
        secrets = self.scan_secrets()
        self.anomalies = _dedupe(self.anomalies)

        totals = Counter()
        trace_statuses: Counter[str] = Counter()
        for episode in episodes:
            trace_statuses.update(episode["agent_trace"]["statuses"])
            totals.update(
                {
                    key: episode["counts"][key]
                    for key in (
                        "steps",
                        "valid_steps",
                        "invalid_steps",
                        "attempts",
                        "retries",
                        "requests_present",
                        "responses_present",
                    )
                }
            )
        status = _status(self.anomalies)
        return {
            "schema_version": 2,
            "run_dir": str(self.root),
            "status": status,
            "complete": not any(
                item["severity"] == "incomplete" for item in self.anomalies
            ),
            "counts": {
                "requested_episodes": requested,
                "summarized_episodes": len(summarized) if summary is not None else None,
                "episode_directories": len(episode_dirs),
                "completed_episodes": sum(
                    episode["completion_status"] == "complete" for episode in episodes
                ),
                "successful_episodes": sum(
                    episode["success"] is True for episode in episodes
                ),
                **dict(totals),
                "agent_trace_statuses": dict(sorted(trace_statuses.items())),
            },
            "episodes": episodes,
            "checks": {
                "images": self.image_summary(),
                "secrets": secrets,
                "prompt_privacy": privacy,
                "model_view": self.video_summary(episodes),
                "source_revision": source_revision,
                "native_actions": self.native_action_summary(episodes),
                "response_model": response_model,
            },
            "anomalies": self.anomalies,
        }

    def audit_source_revision(
        self,
        manifest: dict[str, Any] | None,
    ) -> dict[str, Any]:
        schema_version = _nested_int(manifest, "schema_version")
        if schema_version is None or schema_version < 2:
            return {"status": "legacy_unreported"}

        source = manifest.get("source") if isinstance(manifest, dict) else None
        if not isinstance(source, dict):
            self.add("error", "source_revision_missing", "run_manifest.json")
            return {"status": "fail"}

        commit = source.get("git_commit")
        dirty = source.get("git_dirty")
        config_path = source.get("config_path")
        capture_status = source.get("status")
        if capture_status != "reported":
            self.add(
                "error",
                "source_revision_unavailable",
                "run_manifest.json",
            )
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
            self.add("error", "git_commit_invalid", "run_manifest.json")
        if dirty is not False:
            self.add("error", "source_tree_dirty", "run_manifest.json")
        if not isinstance(config_path, str) or not config_path:
            self.add("error", "source_config_path_missing", "run_manifest.json")

        failed = any(
            item["code"]
            in {
                "source_revision_unavailable",
                "git_commit_invalid",
                "source_tree_dirty",
                "source_config_path_missing",
            }
            for item in self.anomalies
        )
        return {
            "status": "fail" if failed else "pass",
            "capture_status": capture_status,
            "git_commit": commit,
            "git_dirty": dirty,
            "config_path": config_path,
        }

    def audit_episode(self, episode_id: int, episode_dir: Path | None) -> dict[str, Any]:
        path = episode_dir or self.root / f"episode_{episode_id:04d}"
        action_audit_required = (
            self.native_action_required
            or self.robocasa_action_required
            or self.rlbench_action_required
        )
        report = {
            "episode_id": episode_id,
            "path": self.rel(path),
            "completion_status": "incomplete" if episode_dir is None else "complete",
            "success": None,
            "termination_reason": None,
            "counts": {
                "steps": 0,
                "summary_steps": None,
                "valid_steps": 0,
                "invalid_steps": 0,
                "attempts": 0,
                "retries": 0,
                "requests_present": 0,
                "responses_present": 0,
            },
            "agent_trace": {"statuses": {}, "missing": 0},
            "native_action": {
                "required": action_audit_required,
                "status": (
                    "incomplete" if action_audit_required else "not_required"
                ),
                "checked_steps": 0,
                "rejected_steps": 0,
                "invalid_steps": 0,
            },
            "images": {
                "manifest_references": 0,
                "unique_pngs": 0,
                "cameras": {},
            },
            "review": {
                "manifest_status": None,
                "model_view_status": "fail",
                "observed": None,
            },
        }
        if episode_dir is None:
            return report

        mark = len(self.anomalies)
        summary = self.read_json(episode_dir / "summary.json", "incomplete")
        trajectory = self.read_trajectory(episode_dir / "trajectory.jsonl")
        report["counts"]["steps"] = len(trajectory)
        if summary is not None:
            report["success"] = summary.get("success")
            report["termination_reason"] = summary.get("termination_reason")
            if not trajectory:
                self.add(
                    "error",
                    "episode_has_no_model_steps",
                    self.rel(episode_dir / "trajectory.jsonl"),
                )
            summary_steps = summary.get("steps")
            report["counts"]["summary_steps"] = summary_steps
            if not _is_int(summary_steps):
                self.add("error", "episode_steps_invalid", self.rel(episode_dir / "summary.json"))
            elif summary_steps != len(trajectory):
                self.add(
                    "error",
                    "episode_step_count_mismatch",
                    self.rel(episode_dir / "trajectory.jsonl"),
                )

        statuses: Counter[str] = Counter()
        episode_images: set[str] = set()
        image_references = 0
        for row_index, transition in enumerate(trajectory):
            step_id = transition.get("step_id")
            if not _is_int(step_id):
                step_id = row_index
                self.add(
                    "error",
                    "trajectory_step_id_invalid",
                    self.rel(episode_dir / "trajectory.jsonl"),
                )
            validation = transition.get("validation")
            valid = validation.get("is_valid") if isinstance(validation, dict) else None
            if valid is True:
                report["counts"]["valid_steps"] += 1
            else:
                report["counts"]["invalid_steps"] += 1
                self.add(
                    "error",
                    "validation_not_valid",
                    self.rel(episode_dir / "trajectory.jsonl"),
                    f"step={step_id}, valid={valid!r}",
                )

            if action_audit_required:
                report["native_action"]["checked_steps"] += 1
                if self.native_action_required:
                    action_valid = self.audit_simpler_native_action(
                        transition,
                        episode_dir / "trajectory.jsonl",
                        step_id,
                    )
                elif self.robocasa_action_required:
                    action_valid = self.audit_robocasa_sim_action(
                        transition,
                        episode_dir / "trajectory.jsonl",
                        step_id,
                    )
                else:
                    action_valid, rejected = self.audit_rlbench_sim_action(
                        transition,
                        episode_dir / "trajectory.jsonl",
                        step_id,
                    )
                    if rejected:
                        report["native_action"]["rejected_steps"] += 1
                if not action_valid:
                    report["native_action"]["invalid_steps"] += 1

            trace = transition.get("agent_trace")
            if not isinstance(trace, dict):
                report["agent_trace"]["missing"] += 1
                trace = {}
            status = trace.get("status")
            status_name = status if isinstance(status, str) else "missing"
            statuses[status_name] += 1
            if status_name != "ok":
                self.add(
                    "error",
                    "agent_trace_not_ok",
                    self.rel(episode_dir / "trajectory.jsonl"),
                    f"step={step_id}, status={status_name}",
                )
            if trace.get("actor_type") != "mllm":
                self.add(
                    "error",
                    "agent_trace_not_mllm",
                    self.rel(episode_dir / "trajectory.jsonl"),
                    f"step={step_id}",
                )
            attempts = trace.get("attempts")
            if not _is_int(attempts) or attempts < 0:
                self.add(
                    "error",
                    "agent_trace_attempts_invalid",
                    self.rel(episode_dir / "trajectory.jsonl"),
                    f"step={step_id}",
                )
            else:
                report["counts"]["attempts"] += attempts
                report["counts"]["retries"] += max(attempts - 1, 0)

            call_dir = self.safe_path(episode_dir, trace.get("artifact_dir"))
            if call_dir is None:
                self.add(
                    "error",
                    "model_call_artifact_dir_invalid",
                    self.rel(episode_dir / "trajectory.jsonl"),
                    f"step={step_id}",
                )
                continue
            request_path = call_dir / "request.json"
            response_path = call_dir / "response.json"
            request = self.read_model_request(request_path, step_id)
            if request is not None:
                report["counts"]["requests_present"] += 1
                image_references += self.audit_images(
                    request,
                    request_path,
                    episode_id,
                    episode_dir,
                    episode_images,
                )
            if response_path.is_file():
                report["counts"]["responses_present"] += 1
                if self.response_model_required:
                    self.audit_response_model(response_path, step_id)
            else:
                self.add(
                    "error",
                    "model_response_missing",
                    self.rel(response_path),
                    f"step={step_id}",
                )

        report["agent_trace"]["statuses"] = dict(sorted(statuses.items()))
        if action_audit_required and trajectory:
            all_actions_rejected = (
                self.rlbench_action_required
                and report["native_action"]["rejected_steps"]
                == report["native_action"]["checked_steps"]
            )
            if all_actions_rejected:
                self.add(
                    "error",
                    "episode_all_simulator_actions_rejected",
                    self.rel(episode_dir / "trajectory.jsonl"),
                    (
                        f"episode={episode_id}, "
                        f"steps={report['native_action']['checked_steps']}"
                    ),
                )
            report["native_action"]["status"] = (
                "fail"
                if report["native_action"]["invalid_steps"] or all_actions_rejected
                else "pass"
            )
        report["review"] = self.audit_video(episode_dir)
        if any(item["severity"] == "incomplete" for item in self.anomalies[mark:]):
            report["completion_status"] = "incomplete"
        camera_summary = self.image_camera_summary(episode_id)
        if report["completion_status"] == "complete":
            self.audit_frozen_cameras(
                episode_id,
                episode_dir,
                camera_summary,
                visual_change_expected=any(
                    _has_nonzero_parsed_action(transition)
                    for transition in trajectory
                ),
            )
        report["images"] = {
            "manifest_references": image_references,
            "unique_pngs": len(episode_images),
            "cameras": camera_summary,
        }
        return report

    def audit_simpler_native_action(
        self,
        transition: dict[str, Any],
        trajectory_path: Path,
        step_id: int,
    ) -> bool:
        valid = True

        def fail(code: str, detail: str | None = None) -> None:
            nonlocal valid
            valid = False
            self.add(
                "error",
                code,
                self.rel(trajectory_path),
                f"step={step_id}" + (f", {detail}" if detail else ""),
            )

        info = transition.get("info")
        if not isinstance(info, dict):
            fail("native_action_info_missing")
            return False
        applied = info.get("benchmark_applied_native_action")
        if not isinstance(applied, dict):
            fail("native_action_audit_missing")
            return False

        values = _finite_vector(applied.get("values"), length=7)
        if applied.get("shape") != [7] or values is None:
            fail("native_action_audit_invalid")
            return False

        sim_action = transition.get("sim_action")
        sim_values = (
            _finite_vector(sim_action.get("values"), length=7)
            if isinstance(sim_action, dict)
            else None
        )
        if sim_values is None:
            fail("sim_action_invalid")
        elif not np.allclose(values[:6], sim_values[:6], atol=1e-7, rtol=0.0):
            fail("native_motion_mismatch")

        if info.get("absolute_gripper_target") is not True:
            fail("absolute_gripper_mode_missing")
            return valid
        gripper = info.get("benchmark_absolute_gripper")
        if not isinstance(gripper, dict):
            fail("absolute_gripper_audit_missing")
            return False

        controller_command = gripper.get("controller_command")
        applied_command = gripper.get("applied_env_command")
        if (
            not _is_finite_number(controller_command)
            or float(controller_command) not in (-1.0, 0.0, 1.0)
        ):
            fail("absolute_gripper_controller_command_invalid")
        if not _is_finite_number(applied_command) or float(applied_command) != 0.0:
            fail("absolute_gripper_env_command_invalid")
        if values[6] != 0.0:
            fail("native_gripper_command_not_zero")
        if (
            sim_values is not None
            and _is_finite_number(controller_command)
            and not np.isclose(
                float(controller_command),
                float(sim_values[6]),
                atol=1e-7,
                rtol=0.0,
            )
        ):
            fail("absolute_gripper_controller_mismatch")

        expected_update = (
            {
                -1.0: "open",
                0.0: "preserved",
                1.0: "closed",
            }.get(float(controller_command))
            if _is_finite_number(controller_command)
            else None
        )
        if gripper.get("target_update") != expected_update:
            fail("absolute_gripper_target_update_invalid")
        target = gripper.get("target_qpos")
        try:
            target_values = np.asarray(target, dtype=np.float64)
        except (TypeError, ValueError):
            target_values = np.asarray([], dtype=np.float64)
        if target_values.size == 0 or not np.all(np.isfinite(target_values)):
            fail("absolute_gripper_target_invalid")
        return valid

    def audit_robocasa_sim_action(
        self,
        transition: dict[str, Any],
        trajectory_path: Path,
        step_id: int,
    ) -> bool:
        valid = True

        def fail(code: str) -> None:
            nonlocal valid
            valid = False
            self.add(
                "error",
                code,
                self.rel(trajectory_path),
                f"step={step_id}",
            )

        sim_action = transition.get("sim_action")
        if not isinstance(sim_action, dict):
            fail("robocasa_sim_action_missing")
            return False
        values = sim_action.get("values")
        if not isinstance(values, dict):
            fail("robocasa_sim_action_values_invalid")
            return False

        expected = {
            "action.end_effector_position",
            "action.end_effector_rotation",
            "action.gripper_close",
            "action.base_motion",
            "action.control_mode",
        }
        if set(values) != expected:
            fail("robocasa_sim_action_keys_invalid")

        position = _finite_vector(
            values.get("action.end_effector_position"),
            length=3,
        )
        rotation = _finite_vector(
            values.get("action.end_effector_rotation"),
            length=3,
        )
        if position is None or np.any(np.abs(position) > 1.0):
            fail("robocasa_sim_action_position_invalid")
        if rotation is None or np.any(np.abs(rotation) > 1.0):
            fail("robocasa_sim_action_rotation_invalid")

        gripper = _finite_vector(
            values.get("action.gripper_close"),
            length=1,
        )
        if gripper is None or float(gripper[0]) not in {0.0, 1.0}:
            fail("robocasa_sim_action_gripper_invalid")

        base = _finite_vector(
            values.get("action.base_motion"),
            length=4,
        )
        if base is None or not np.allclose(base, 0.0):
            fail("robocasa_sim_action_base_invalid")

        control_mode = _finite_vector(
            values.get("action.control_mode"),
            length=1,
        )
        if control_mode is None or float(control_mode[0]) != 1.0:
            fail("robocasa_sim_action_control_mode_invalid")

        metadata = sim_action.get("metadata")
        if not isinstance(metadata, dict):
            fail("robocasa_sim_action_metadata_invalid")
        elif metadata.get("controller") != "robocasa_osc":
            fail("robocasa_sim_action_controller_invalid")
        return valid

    def audit_rlbench_sim_action(
        self,
        transition: dict[str, Any],
        trajectory_path: Path,
        step_id: int,
    ) -> tuple[bool, bool]:
        valid = True

        def fail(code: str, detail: str | None = None) -> None:
            nonlocal valid
            valid = False
            self.add(
                "error",
                code,
                self.rel(trajectory_path),
                f"step={step_id}" + (f", {detail}" if detail else ""),
            )

        sim_action = transition.get("sim_action")
        if not isinstance(sim_action, dict):
            fail("sim_action_missing")
        else:
            if not _is_finite_number_vector(
                sim_action.get("delta_position"), length=3
            ):
                fail("sim_action_delta_position_invalid")
            if not _is_finite_number_vector(
                sim_action.get("delta_rotation_rpy"), length=3
            ):
                fail("sim_action_delta_rotation_rpy_invalid")
            gripper = sim_action.get("gripper")
            if not _is_finite_number(gripper) or not 0.0 <= float(gripper) <= 1.0:
                fail("sim_action_gripper_invalid")
            metadata = sim_action.get("metadata")
            if not isinstance(metadata, dict):
                fail("sim_action_metadata_invalid")
            elif metadata.get("controller") != "rlbench_delta_eef":
                fail("sim_action_controller_invalid")

        info = transition.get("info")
        if not isinstance(info, dict):
            fail("sim_action_info_missing")
            return valid, False
        rejected = info.get("action_rejected")
        if not isinstance(rejected, bool):
            fail("sim_action_rejection_invalid")
            return valid, False
        return valid, rejected

    def audit_images(
        self,
        request: dict[str, Any],
        request_path: Path,
        episode_id: int,
        episode_dir: Path,
        episode_images: set[str],
    ) -> int:
        records = request.get("images")
        if not isinstance(records, list):
            self.add("error", "request_images_invalid", self.rel(request_path))
            return 0
        for record in records:
            self.image_references += 1
            artifact = record.get("artifact") if isinstance(record, dict) else None
            expected = record.get("sha256") if isinstance(record, dict) else None
            camera = record.get("camera") if isinstance(record, dict) else None
            step_id = record.get("step_id") if isinstance(record, dict) else None
            path = self.safe_path(episode_dir, artifact)
            if (
                path is None
                or not isinstance(expected, str)
                or not isinstance(camera, str)
                or not camera
                or not _is_int(step_id)
            ):
                self.add("error", "request_image_record_invalid", self.rel(request_path))
                continue
            evidence_key = (episode_id, camera, step_id)
            previous_hash = self.image_evidence.get(evidence_key)
            if previous_hash is None:
                self.image_evidence[evidence_key] = expected
            elif previous_hash != expected:
                self.image_reference_hash_conflicts.add(evidence_key)
                self.add(
                    "error",
                    "model_image_reference_hash_conflict",
                    self.rel(request_path),
                    f"episode={episode_id}, camera={camera!r}, step={step_id}",
                )
            display = self.rel(path)
            episode_images.add(display)
            if not path.is_file():
                self.image_missing.add(display)
                self.add("error", "model_image_missing", display)
                continue
            info = self.image_files.get(display)
            if info is None:
                info = _inspect_png(path)
                self.image_files[display] = info
            if not info["valid"]:
                self.image_invalid.add(display)
                self.add("error", "model_image_invalid_png", display)
                continue
            if info["sha256"] != expected:
                self.image_mismatches.add(display)
                self.add("error", "model_image_hash_mismatch", display)
            if info["black"]:
                self.black_screens[display] = {
                    "pixel_mean": info["mean"],
                    "pixel_stddev": info["stddev"],
                }
                self.add("error", "model_image_black_screen", display)
        return len(records)

    def image_camera_summary(
        self,
        episode_id: int | None = None,
    ) -> dict[str, dict[str, int]]:
        steps_by_camera: dict[str, set[tuple[int, int]]] = {}
        hashes_by_camera: dict[str, set[str]] = {}
        for (observed_episode, camera, step_id), digest in self.image_evidence.items():
            if episode_id is not None and observed_episode != episode_id:
                continue
            steps_by_camera.setdefault(camera, set()).add(
                (observed_episode, step_id)
            )
            hashes_by_camera.setdefault(camera, set()).add(digest)
        return {
            camera: {
                "observed_steps": len(steps_by_camera[camera]),
                "unique_hashes": len(hashes_by_camera[camera]),
            }
            for camera in sorted(steps_by_camera)
        }

    def audit_frozen_cameras(
        self,
        episode_id: int,
        episode_dir: Path,
        camera_summary: dict[str, dict[str, int]],
        *,
        visual_change_expected: bool,
    ) -> None:
        if not visual_change_expected:
            return
        for camera, counts in camera_summary.items():
            if counts["observed_steps"] < 2 or counts["unique_hashes"] != 1:
                continue
            self.frozen_cameras.add((episode_id, camera))
            self.add(
                "error",
                "model_image_camera_frozen",
                self.rel(episode_dir),
                (
                    f"episode={episode_id}, camera={camera!r}, "
                    f"observed_steps={counts['observed_steps']}"
                ),
            )

    def scan_prompt_privacy(self) -> dict[str, Any]:
        findings: list[dict[str, str]] = []
        scanned = 0
        pattern = "episode_*/model_calls/step_*/request.json"
        for path in sorted(self.root.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                self.add("error", "prompt_request_invalid", self.rel(path))
                continue
            scanned += 1
            _walk_prompt(payload, "$", self.rel(path), findings)
        for finding in findings:
            self.add(
                "error",
                f"prompt_privacy_{finding['kind']}",
                finding["path"],
                f"{finding['name']} at {finding['json_path']}",
            )
        return {
            "status": "fail" if findings else "pass",
            "scanned_requests": scanned,
            "findings": findings,
        }

    def scan_secrets(self) -> dict[str, Any]:
        findings = []
        scanned = 0
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            scanned += 1
            for line_number, line in enumerate(text.splitlines(), start=1):
                for kind, pattern in SECRET_PATTERNS.items():
                    if pattern.search(line):
                        findings.append(
                            {"kind": kind, "path": self.rel(path), "line": line_number}
                        )
        findings = _dedupe(findings)
        for finding in findings:
            self.add(
                "error",
                "secret_leak",
                f"{finding['path']}:{finding['line']}",
                finding["kind"],
            )
        counts = Counter(finding["kind"] for finding in findings)
        return {
            "status": "fail" if findings else "pass",
            "scanned_files": scanned,
            "counts": {
                name: counts[name]
                for name in ("authorization", "api_key", "base64_data_url")
            },
            "findings": findings,
        }

    def audit_video(self, episode_dir: Path) -> dict[str, Any]:
        manifest = self.read_json(episode_dir / "review/manifest.json", "error")
        path = episode_dir / "review/model_view.mp4"
        result = {
            "manifest_status": manifest.get("status") if manifest else None,
            "model_view_status": "fail",
            "observed": None,
        }
        if not path.is_file() or path.stat().st_size == 0:
            self.add("error", "model_view_missing", self.rel(path))
            return result
        if self.ffprobe_path is None:
            result["model_view_status"] = "unavailable"
            self.add("unavailable", "ffprobe_unavailable", self.rel(path))
            return result
        probe = _probe_video(path, self.ffprobe_path)
        result["model_view_status"] = probe["status"]
        result["observed"] = probe["observed"]
        if probe["status"] != "pass":
            self.add(
                "error",
                "model_view_invalid",
                self.rel(path),
                ",".join(probe["problems"]),
            )
        return result

    def image_summary(self) -> dict[str, Any]:
        failed = (
            self.image_missing
            or self.image_invalid
            or self.image_mismatches
            or self.black_screens
            or self.image_reference_hash_conflicts
            or self.frozen_cameras
        )
        return {
            "status": "fail" if failed else "pass",
            "manifest_references": self.image_references,
            "unique_pngs": len(self.image_files),
            "cameras": self.image_camera_summary(),
            "missing_paths": sorted(self.image_missing),
            "invalid_png_paths": sorted(self.image_invalid),
            "hash_mismatch_paths": sorted(self.image_mismatches),
            "black_screens": [
                {"path": path, **values}
                for path, values in sorted(self.black_screens.items())
            ],
        }

    def video_summary(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        statuses = Counter(
            episode["review"]["model_view_status"] for episode in episodes
        )
        status = (
            "fail"
            if statuses["fail"]
            else "unavailable"
            if statuses["unavailable"]
            else "pass"
        )
        return {
            "status": status,
            "ffprobe": {
                "binary": self.ffprobe_binary,
                "status": "available" if self.ffprobe_path else "unavailable",
            },
            "expected": {
                "codec": "h264",
                "width": 1280,
                "height": 720,
                "fps": 2.0,
                "audio_streams": 0,
            },
            "episode_statuses": dict(sorted(statuses.items())),
        }

    def native_action_summary(
        self,
        episodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        checked = sum(
            episode["native_action"]["checked_steps"] for episode in episodes
        )
        invalid = sum(
            episode["native_action"]["invalid_steps"] for episode in episodes
        )
        rejected = sum(
            episode["native_action"]["rejected_steps"] for episode in episodes
        )
        required = (
            self.native_action_required
            or self.robocasa_action_required
            or self.rlbench_action_required
        )
        if not required:
            status = "not_required"
        elif invalid or any(
            episode["native_action"]["status"] == "fail" for episode in episodes
        ):
            status = "fail"
        elif checked:
            status = "pass"
        else:
            status = "incomplete"
        return {
            "required": required,
            "status": status,
            "checked_steps": checked,
            "rejected_steps": rejected,
            "invalid_steps": invalid,
        }

    def audit_response_model(self, path: Path, step_id: int) -> None:
        display = self.rel(path)
        response = self.read_json(path)
        if response is None:
            self.response_model_invalid.add(display)
            return
        model = response.get("model")
        if not isinstance(model, str) or not model:
            self.response_model_missing.add(display)
            self.add(
                "error",
                "response_model_missing",
                display,
                f"step={step_id}, top-level model must be a non-empty string",
            )
            return
        self.response_models[display] = model

    def response_model_summary(self) -> dict[str, Any]:
        if not self.response_model_required:
            return {
                "status": "legacy_unreported",
                "requested": self.requested_model,
                "observed": [],
                "missing_count": 0,
                "mismatch_paths": [],
            }

        observed = sorted(set(self.response_models.values()))
        mismatch_paths = (
            sorted(
                path
                for path, model in self.response_models.items()
                if model != self.requested_model
            )
            if self.requested_model
            else []
        )
        for path in mismatch_paths:
            self.add(
                "error",
                "response_model_mismatch",
                path,
                (
                    f"requested={self.requested_model!r}, "
                    f"observed={self.response_models[path]!r}"
                ),
            )
        if len(observed) > 1:
            self.add(
                "error",
                "response_model_mixed",
                ".",
                f"observed={json.dumps(observed, sort_keys=True)}",
            )

        failed = (
            not self.requested_model
            or self.response_model_missing
            or self.response_model_invalid
            or mismatch_paths
            or len(observed) > 1
        )
        return {
            "status": "fail" if failed else "pass",
            "requested": self.requested_model,
            "observed": observed,
            "missing_count": len(self.response_model_missing),
            "mismatch_paths": mismatch_paths,
        }

    def read_trajectory(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            self.add("incomplete", "trajectory_missing", self.rel(path))
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            self.add("error", "trajectory_unreadable", self.rel(path))
            return []
        rows = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                self.add(
                    "error",
                    "trajectory_json_invalid",
                    f"{self.rel(path)}:{line_number}",
                )
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                self.add(
                    "error",
                    "trajectory_step_invalid",
                    f"{self.rel(path)}:{line_number}",
                )
        return rows

    def read_model_request(self, path: Path, step_id: int) -> dict[str, Any] | None:
        if not path.is_file():
            self.add("error", "model_request_missing", self.rel(path), f"step={step_id}")
            return None
        return self.read_json(path)

    def read_json(
        self, path: Path, missing_severity: str | None = None
    ) -> dict[str, Any] | None:
        if not path.is_file():
            if missing_severity:
                self.add(
                    missing_severity, f"{path.stem}_missing", self.rel(path)
                )
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            self.add("error", f"{path.stem}_invalid", self.rel(path))
            return None
        if not isinstance(value, dict):
            self.add("error", f"{path.stem}_invalid", self.rel(path))
            return None
        return value

    def summary_episodes(
        self, summary: dict[str, Any] | None
    ) -> dict[int, dict[str, Any]]:
        if summary is None:
            return {}
        records = summary.get("episodes")
        if not isinstance(records, list):
            self.add("error", "run_summary_episodes_invalid", "run_summary.json")
            return {}
        result = {}
        for record in records:
            episode_id = record.get("episode_id") if isinstance(record, dict) else None
            if not _is_int(episode_id) or episode_id < 0 or episode_id in result:
                self.add("error", "run_summary_episode_invalid", "run_summary.json")
            else:
                result[episode_id] = record
        return result

    def episode_dirs(self) -> dict[int, Path]:
        result = {}
        for path in sorted(self.root.iterdir()):
            match = EPISODE_RE.fullmatch(path.name)
            if not match or not path.is_dir():
                continue
            episode_id = int(match.group(1))
            if episode_id in result:
                self.add("error", "episode_directory_duplicate", self.rel(path))
            else:
                result[episode_id] = path
        return result

    def safe_path(self, base: Path, relative: object) -> Path | None:
        if not isinstance(relative, str) or not relative:
            return None
        value = Path(relative)
        if value.is_absolute() or ".." in value.parts:
            return None
        root = base.resolve()
        target = (base / value).resolve()
        return target if target.is_relative_to(root) else None

    def rel(self, path: Path | str) -> str:
        value = Path(path)
        try:
            return str(value.relative_to(self.root))
        except ValueError:
            return str(value)

    def add(
        self,
        severity: str,
        code: str,
        path: Path | str,
        detail: str | None = None,
    ) -> None:
        anomaly = {"severity": severity, "code": code, "path": str(path)}
        if detail:
            anomaly["detail"] = detail
        self.anomalies.append(anomaly)


def audit_run(run_dir: Path, *, ffprobe_binary: str = "ffprobe") -> dict[str, Any]:
    root = run_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"Run directory is not a directory: {root}")
    return RunAuditor(root, ffprobe_binary).run()


def _inspect_png(path: Path) -> dict[str, Any]:
    try:
        content = path.read_bytes()
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
        if not pixels.size:
            raise ValueError
    except (OSError, ValueError):
        return {"valid": False, "sha256": None, "mean": None, "stddev": None, "black": None}
    mean = float(pixels.mean())
    stddev = float(pixels.std())
    return {
        "valid": True,
        "sha256": hashlib.sha256(content).hexdigest(),
        "mean": round(mean, 6),
        "stddev": round(stddev, 6),
        "black": mean <= 1.0 or stddev <= 1.0,
    }


def _walk_prompt(
    value: Any,
    json_path: str,
    artifact_path: str,
    findings: list[dict[str, str]],
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{json_path}.{key}"
            name = str(key).lower()
            if name in PRIVILEGED_KEYS:
                findings.append(
                    {
                        "kind": "privileged_key",
                        "name": name,
                        "path": artifact_path,
                        "json_path": child,
                    }
                )
            _walk_prompt(nested, child, artifact_path, findings)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _walk_prompt(nested, f"{json_path}[{index}]", artifact_path, findings)
    elif isinstance(value, str):
        lowered = value.lower()
        for name in sorted(HIDDEN_CONTROL_NAMES):
            if name in lowered:
                findings.append(
                    {
                        "kind": "hidden_control_name",
                        "name": name,
                        "path": artifact_path,
                        "json_path": json_path,
                    }
                )


def _probe_video(path: Path, ffprobe_path: str) -> dict[str, Any]:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, check=False, text=True, timeout=30
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        payload = {}
    streams = payload.get("streams") if isinstance(payload, dict) else None
    streams = streams if isinstance(streams, list) else []
    videos = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    audio_count = sum(
        isinstance(stream, dict) and stream.get("codec_type") == "audio"
        for stream in streams
    )
    video = videos[0] if videos else {}
    fps = _fps(video)
    observed = {
        "video_streams": len(videos),
        "codec": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": fps,
        "audio_streams": audio_count,
    }
    problems = []
    if len(videos) != 1:
        problems.append("video_stream_count")
    if video.get("codec_name") != "h264":
        problems.append("codec")
    if video.get("width") != 1280 or video.get("height") != 720:
        problems.append("dimensions")
    if fps is None or abs(fps - 2.0) > 1e-6:
        problems.append("fps")
    if audio_count:
        problems.append("audio")
    return {
        "status": "fail" if problems else "pass",
        "problems": problems,
        "observed": observed,
    }


def _fps(stream: dict[str, Any]) -> float | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if not isinstance(value, str) or value in {"0/0", "N/A"}:
            continue
        try:
            return float(Fraction(value))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def _nested_int(value: dict[str, Any] | None, *keys: str) -> int | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if _is_int(current) else None


def _nested_str(value: dict[str, Any] | None, *keys: str) -> str | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, str) else None


def _finite_vector(value: object, *, length: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _is_finite_number_vector(value: object, *, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_is_finite_number(item) for item in value)
    )


def _has_nonzero_parsed_action(transition: Mapping[str, object]) -> bool:
    action = transition.get("parsed_action")
    return isinstance(action, Mapping) and any(
        _is_finite_number(value) and float(value) != 0.0
        for value in action.values()
    )


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def _status(anomalies: list[dict[str, str]]) -> str:
    severities = {item["severity"] for item in anomalies}
    if "incomplete" in severities:
        return "incomplete"
    if "error" in severities:
        return "fail"
    if "unavailable" in severities:
        return "unavailable"
    return "pass"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _dedupe(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit an offline MLLM benchmark run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    try:
        report = audit_run(args.run_dir, ffprobe_binary=args.ffprobe)
    except ValueError as exc:
        report = {
            "schema_version": 2,
            "run_dir": str(args.run_dir),
            "status": "error",
            "complete": False,
            "anomalies": [
                {
                    "severity": "error",
                    "code": "run_directory_invalid",
                    "path": str(args.run_dir),
                    "detail": str(exc),
                }
            ],
        }
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
