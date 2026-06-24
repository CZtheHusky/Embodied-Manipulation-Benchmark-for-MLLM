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
- Generalized the two-process manual player for Gymnasium.
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
- Added a two-process human-play backend for ManiSkill PickCube.
- Added gripper keyboard toggling for the final virtual joystick through `G` or `Space`.
- Updated ManiSkill play docs and the ManiSkill conda env to include Pygame.

## 2026-05-13: Robosuite integration

- Selected `robosuite` as the next simulation environment due to its standardized 3D tasks and MuJoCo backend.
- Updated `docs/architecture.md` to formally define integration constraints:
    - Prefer native APIs but allow forking if necessary.
    - Strict version pinning for all simulation dependencies.
    - Decoupled control pipeline (confusions/transforms calculated before simulator step).
    - API-driven visual randomization (e.g., `CameraModder`).
- Pinned `robosuite==1.4.1` in the robosuite backend conda env and optional dependency metadata, while keeping the root core environment simulator-free.
- Implemented `RobosuiteSimAdapter` wrapping `suite.make()` with support for off-screen rendering and runtime camera randomization.
- Implemented `RobosuiteOSCController` mapping benchmark actions to robosuite's 7-DoF `OSC_POSE` controller.
- Implemented `RobosuiteLiftTask` as the first standardized 3D manipulation task.
- Added `configs/robosuite_lift.yaml` for evaluated random policy on the Lift task.
- Verified with `tests/test_robosuite_integration.py` and successfully ran a sample benchmark episode.

## 2026-06-23: Benchmark-risk hardening

- Fixed visible mappings are now disclosed to both human panels and MLLM-facing prompts; hidden/complex mappings still hide exact permutations and signs.
- Benchmark episode logs are grouped under per-run directories so repeated runs do not overwrite prior artifacts.
- `agent.type: mllm` now fails during runner construction until an API-backed policy exists.
- Robosuite import defaults `NUMBA_DISABLE_JIT=1`, camera randomization is seed-sampled, and optional-backend tests can be forced with `MLLM_TELEOP_REQUIRE_ROBOSUITE=1`.
- Control-frame rotation now also transforms small delta-rotation vectors for 6D EEF controllers.

## 2026-06-23: DMControl and CALVIN backend port

- Recovered the active goals from the sibling Gemini worktrees and delegated separate real-runtime validation for DMControl and CALVIN before merging.
- Added DMControl support through `DMControlSimAdapter`, `DMControlObservationProvider`, `DMControlController`, `DMControlTask`, human play, `configs/dmcontrol_example.yaml`, and `envs/environment-dmcontrol.yml`.
- Added CALVIN support through `CalvinSimAdapter`, `CalvinObservationProvider`, `CalvinDeltaEEFController`, `CalvinTask`, `calvin_randomized` camera sampling, human play, `configs/calvin_delta_eef.yaml`, and `envs/environment-calvin.yml`.
- Preserved the benchmark-layer control pipeline: joystick mapping and frame transforms happen before simulator-native action arrays for both new backends.
- Verified DMControl in `mllm-teleop-dmcontrol` with unit tests, a direct-worker step smoke, and a full random-policy benchmark smoke. Verified CALVIN in `mllm-teleop-calvin` with real reset/image/step tests and a 100-step random-policy benchmark smoke.
- Remaining known gaps: DMControl free-camera orbit randomization is unsupported and fails loudly; CALVIN camera randomization is best-effort without a visual oracle; CALVIN task success still needs a stricter language-task oracle.

## 2026-06-23: uv-first environment management

- Added uv input and lock files for core, Gymnasium, DMControl, robosuite, ManiSkill CPU/CUDA, and CALVIN.
- Added `scripts/manage_uv_env.py` to lock, create, run, and smoke-test per-backend uv environments under `.venvs/`.
- Kept conda env files as rollback/fallback while uv backend smoke coverage is validated.
- Defaulted ManiSkill uv resolution to CPU PyTorch and added `maniskill-cu126` for NVIDIA CUDA 12.6 hosts.
- CALVIN remains special: uv resolves the pinned Git dependency, but environment creation clones the upstream repo and installs it from a local editable checkout so Hydra runtime configs are available.
- Pinned robosuite's MuJoCo dependency to `mujoco>=2.3,<3.0`; `robosuite==1.4.1` fails against MuJoCo 3.x controller APIs.
- Made uv environment creation idempotent by clearing the target `.venvs/<backend>` before syncing from the lock.
- Passed ManiSkill's `--torch-backend cpu` to both lock and sync so `torch==2.12.1+cpu` installs reproducibly.
- Added CUDA availability as the first `maniskill-cu126` smoke step so hosts without an NVIDIA driver fail loudly.
- Verified `maniskill-cu126` on station with `torch==2.12.1+cu126`, CUDA 12.6, and NVIDIA GeForce RTX 2080; GPU smoke passed after running outside the restricted command sandbox.

## Documentation rule

- public commands or setup: update `README.md`
- architecture or component boundaries: update `docs/architecture.md`
- environment-specific behavior: update a focused doc such as `docs/reacher_integration.md`
- important implementation decisions or incidents: append here
