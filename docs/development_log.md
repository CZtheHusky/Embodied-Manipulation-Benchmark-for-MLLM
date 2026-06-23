# Development Log

This file records implementation decisions that matter for future maintenance.

## 2026-05-12: Initial benchmark framework

- Added a config-driven Python package under `src/mllm_teleop_benchmark`.
- Added core OOP boundaries: `SimAdapter`, `Task`, `Controller`, `ObservationProvider`, `CameraManager`, `Agent`, `MetricsRecorder`, and episode logging.
- Added `DummySimAdapter` and `ReachTargetTask` to validate the full runner without a real simulator.
- Added strict joystick parsing/validation and fixed joystick mapping.
- Added conda environment management with `environment.yml`.
- Validation: `pytest -q` and `scripts/run_benchmark.py --config configs/dummy_reach.yaml`.

## 2026-05-12: Reacher Gymnasium/MuJoCo integration

- Added local Gymnasium/MuJoCo Reacher delta-EEF environment instead of forking Gymnasium.
- Added repo-owned `reacher_pos.xml` with position actuators.
- Added `GymnasiumSimAdapter`, `GymnasiumObservationProvider`, `GymnasiumReacherTask`, and `ReacherDeltaEEFJoystickController`.
- Added `configs/reacher_delta_eef.yaml`.
- Kept Gymnasium as an external dependency; benchmark-specific env code lives under `sim/gymnasium_envs`.

## 2026-05-12: Manual Reacher player GL fix

- Single-process Pygame + MuJoCo rendering caused OpenGL context conflicts on the local X11 desktop.
- Symptoms included black Pygame windows, black post-step frames, `EGL_BAD_ACCESS`, and `X_GLXMakeCurrent BadAccess`.
- The stable solution is a two-process player:
  - Main process: Pygame window and keyboard events.
  - Worker process: Gymnasium/MuJoCo environment and `rgb_array` rendering with EGL.
- Implemented this in the direct Gymnasium player with `env_worker`, `WorkerCommand`, and `WorkerResult`.
- Initial manual player was for human debugging/evaluation and did not yet fully reuse the benchmark parser/controller/logger path.

## 2026-05-12: Manual player rejoined benchmark action path

- Kept the two-process manual player because it is the stable rendering path on the local X11 desktop.
- Moved keyboard action interpretation back to the main process benchmark pipeline:
  - keyboard input -> joystick action dict
  - `ActionParser` / `ActionValidator`
  - configured joystick mapping and controller
  - `GymBoxAction` sent to the worker process
- The worker now handles environment lifecycle, stepping, rendering, and state return only.
- Manual play records standard `StepTransition` entries through `EpisodeLogger`, so human play logs are comparable to policy evaluation logs.

## 2026-05-12: Agent-visible control interface

- Added `control_interface` as the preferred config section for the virtual joystick interface shown to humans and MLLMs.
- Added `control_interface` as the preferred config section while temporarily keeping legacy `action_space` and `mapping` support.
- Reacher now exposes only two virtual joysticks, `dx` and `dy`, and no gripper channel.
- Prompt payloads now include a control instruction derived from the control interface.
- Mapping remains separate from frame transforms: joystick-to-dimension mapping is applied first, then coordinate-frame transforms. This preserves room for future joystick permutation/sign confusion and rotated control frames, independently or together.

## 2026-05-12: Human play information panel and motion-only confusion

- Human play windows now reserve a top information panel above the environment RGB frame.
- Panel content is generated from `ControlInstructionBuilder`, the same source used by MLLM prompt payloads.
- Control-interface mapping strategies can permute and sign-flip motion dimensions.
- Gripper channels are fixed as the final virtual joystick and are never permuted or sign-flipped.
- Removed diagonal human-only macro keys so human play uses the same joystick action schema exposed to MLLM policies.
- Added CLI config overrides through `--set key.path=value` for both benchmark and human play scripts.

## 2026-05-12: Cross-environment control-frame rotation

- Added shared 2D and 3D rotation frame transforms at the benchmark control layer.
- `rotation_2d` rotates `dx/dy` around z; `random_rotation_2d` samples uniformly on SO(2).
- `rotation_3d` rotates `dx/dy/dz` with fixed roll/pitch/yaw; `random_rotation_3d` samples uniformly on SO(3) by default.
- Euler-range random 3D rotation is still available through explicit `sampling=euler_ranges`.
- Gymnasium box controllers, including Reacher, now apply the same frame transform path as EE delta-pose controllers.

## 2026-05-13: Pusher delta-EEF integration

- Confirmed that Gymnasium `Reacher-v5` is a 2D two-joint torque environment, while Gymnasium `Pusher-v5` is the 7-DoF MuJoCo arm matching the requested 3D use case.
- Added local `MLLMTeleop/PusherDeltaEEF-v0` instead of forking Gymnasium.
- Added repo-owned `pusher_pos.xml`, derived from Gymnasium's Pusher XML, with joint position actuators replacing native torque motors.
- Added `PusherDeltaEEFEnv`, which accepts `dx/dy/dz` delta end-effector commands and translates them to 7 arm joint position targets using damped least-squares MuJoCo Jacobian IK.
- Added `PusherDeltaEEFJoystickController`, `GymnasiumPusherTask`, and `configs/pusher_delta_eef.yaml`.
- Generalized the two-process manual player as `DirectGymnasiumPlayer`.
- Added reusable `ControlDebugRecorder` for optional control-history logging. Human play now records source/control id, raw joystick action, expected EEF delta, actual EEF delta, target delta, clamp status, and before/after state snapshots.
- Added reusable `FrameDebugRecorder` for optional raw RGB frame logging. Human play uses it through `ui.debug_frames`.

## 2026-05-13: Config cleanup and debug defaults

- Removed legacy benchmark YAML support for top-level `action_space`, `mapping`, and standalone `frame_transform`; configs must now declare `control_interface`.
- Human play defaults `debug_frames` and `debug_control_history` to enabled.
- Benchmark/MLLM evaluation still does not create debug frame or control-history outputs by default.
- Removed old UI names and compatibility config keys: `direct_reacher`, `debug_screenshots`, and `debug_key_events`.

## 2026-05-13: Debug session directory layout

- Added `DebugSession` as the shared source of debug run ids, directory paths, event tags, and manifest metadata.
- Debug output is grouped as `debug/<env_slug>/<control_slug>/<actor_slug>/<run_id>/episode_XXXX`.
- Human play now writes frame and control-history debug artifacts into the same per-run session directory, avoiding overwrites across repeated runs.
- Each debug run writes `manifest.json` with environment, benchmark, task, control interface, controller, actor, seed, episode, and run timestamp metadata.

## 2026-05-13: Hidden control-complexity details

- Human panel metadata and MLLM prompt metadata now expose only whether mapping/frame complexity is present.
- Exact joystick permutation, sign flips, and frame rotation values are not shown to humans or MLLM-facing prompt metadata.
- Full internal mapping/frame details remain available in logs/debug artifacts for offline analysis.

## 2026-05-13: ManiSkill optional backend

- Added ManiSkill as an optional simulator backend without forking or modifying ManiSkill source code.
- Added `ManiSkillSimAdapter`, `ManiSkillObservationProvider`, `ManiSkillDeltaEEFController`, and `ManiSkillTask`.
- Added `maniskill_randomized` camera management for episode-level camera sampling only; camera views are fixed within each episode.
- Added `configs/maniskill_pickcube_delta_eef.yaml` as the first ManiSkill config.
- ManiSkill remains an optional dependency through the `maniskill` extra; core tests must pass when it is not installed.

## 2026-05-14: Conda-only backend environments

- Standardized on conda as the only supported environment manager.
- Changed root `environment.yml` to the lightweight `mllm-teleop-core` environment.
- Added backend-specific env files under `envs/` for core, Gymnasium/MuJoCo, ManiSkill, and robosuite.
- Moved simulator/UI packages out of core `pyproject.toml` dependencies and into optional metadata groups plus conda env files.
- Documented the policy in `docs/environment_management.md`.

## 2026-05-14: ManiSkill direct human play

- Added a shared direct-player base for human keyboard play so Gymnasium and ManiSkill reuse the same Pygame loop, action parsing, controller translation, logging, and debug recorders.
- Added `direct_maniskill` as a two-process human-play backend for ManiSkill PickCube.
- Added gripper keyboard toggling for the final virtual joystick through `G` or `Space`.
- Updated ManiSkill play docs and the ManiSkill conda env to include Pygame.

## 2026-05-14: CALVIN benchmark integration

- Added CALVIN as an optional simulator backend using `calvin-env`.
- Implemented `CalvinSimAdapter` wrapping `PlayTableSimEnv`.
- Implemented `CalvinDeltaEEFController` mapping virtual joysticks to CALVIN's 7-DoF delta EEF action space.
- Implemented `CalvinObservationProvider` for `static` and `gripper` camera extraction.
- Implemented `CalvinTask` for language-conditioned task success tracking.
- Added `configs/calvin_delta_eef.yaml`.
- Added `direct_calvin` multi-process human play UI worker.
- Added `envs/environment-calvin.yml` for dependency management.
- Verified with `tests/test_calvin_integration.py` (mocked sim).

## 2026-05-14: CALVIN camera randomization support

- Added `RandomizedCalvinCameraManager` in `camera.py` for episode-level `position` and `look_at` sampling.
- Updated `CalvinSimAdapter` to apply sampled PyBullet view matrices during `reset()`.
- Updated `CalvinObservationProvider` to pass sampled camera parameters from the manager to the adapter.
- Updated `configs/calvin_delta_eef.yaml` to enable `calvin_randomized` by default for the `static` camera.
- Verified deterministic sampling with new unit tests in `test_calvin_integration.py`.

## 2026-06-23: CALVIN real-environment smoke validation

- Confirmed `calvin-env` is not a PyPI package. The installable source is `mees/calvin_env`, verified at commit `797142c588c21e76717268b7b430958dbd13bf48`, package version `0.0.1`.
- Updated `envs/environment-calvin.yml` to install `calvin_env` as editable VCS source and this repository as `-e ..`.
- Created/updated the local `mllm-teleop-calvin` conda environment and verified `calvin_env` imports from `envs/src/calvin-env`.
- Replaced the unconditional CALVIN pytest skip with dependency detection. When `calvin_env` is installed, the test now constructs real `PlayTableSimEnv`, runs reset, extracts `static` and `gripper` RGB images, and steps a 7-value action.
- Added old Gym API compatibility for CALVIN reset/step/render and normalized the `env_id: null` config-loader string case inside the CALVIN adapter.
- Changed CALVIN gripper mapping from continuous interpolation to discrete open/close, because the real robot implementation asserts gripper action is exactly `-1` or `1`.
- Verification: `compileall` passed; `PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-calvin/bin/python -m pytest -q tests/test_calvin_integration.py -rs` passed with 4 tests; `PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-calvin/bin/python scripts/run_benchmark.py --config configs/calvin_delta_eef.yaml` completed one 100-step episode with `success_rate: 0.0`.
- Remaining risks: CALVIN emits Hydra/gym deprecation warnings, camera randomization is still a best-effort internal patch, and success metrics need a task oracle for real language-task evaluation beyond the smoke run.


- public commands or setup: update `README.md`
- architecture or component boundaries: update `docs/architecture.md`
- environment-specific behavior: update a focused doc such as `docs/reacher_integration.md`
- important implementation decisions or incidents: append here
