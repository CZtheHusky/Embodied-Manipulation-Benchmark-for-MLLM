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
- Implemented this in `DirectReacherPlayer` with `env_worker`, `WorkerCommand`, and `WorkerResult`.
- Current manual player is for human debugging/evaluation. It does not yet fully reuse the benchmark parser/controller/logger path.

## Documentation rule

Every development pass should update one or more docs when it changes:

- public commands or setup: update `README.md`
- architecture or component boundaries: update `docs/architecture.md`
- environment-specific behavior: update a focused doc such as `docs/reacher_integration.md`
- important implementation decisions or incidents: append here
