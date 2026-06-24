# Framework Assessment

Current state as of 2026-05-13.

## What Is Solid

- The benchmark now has clear boundaries between config loading, action parsing, mapping, controller translation, simulator adapters, tasks, observation providers, agents, metrics, and logging.
- `control_interface` is the single source of truth for virtual joystick channels. It derives the action schema and mapping config instead of duplicating them in YAML.
- Mapping confusion and control-frame rotation are independent components. They can be enabled independently or together.
- Reacher and Pusher use local Gymnasium registrations instead of forking Gymnasium.
- Human play uses the same action parser, validator, mapping, controller path, simulator adapters, task reset, and observation providers intended for MLLM policies.
- Human play separates Pygame/control in the main process from simulator runtime in a worker process through `ui.runtime: sim_adapter_worker`.
- `ControlDebugRecorder` and `FrameDebugRecorder` are reusable logging components, not environment-specific utilities.
- Debug output is grouped by environment, control interface, actor, run id, and episode, with a manifest for each debug run.

## Current Weak Spots

- `core/registry.py` uses hand-written `if`/`elif` construction. This is simple now, but a plugin-style registry will scale better as more simulators, tasks, and controllers are added.
- `FrameDebugRecorder` and `ControlDebugRecorder` are available to MLLM/evaluation code but are not yet integrated into `EpisodeManager`. That is intentional because debug should default off for MLLM runs, but an explicit opt-in hook should be added later.
- Environment state keys for EEF logging are convention-based (`fingertip_position`, `ee_position`, `end_effector_position`). Future adapters should document these fields explicitly.

## Recommended Next Refactors

1. Add typed `DebugConfig` dataclass.
2. Add explicit opt-in debug hooks to `EpisodeManager` for MLLM/policy evaluation.
3. Replace the manual `core/registry.py` conditionals with small component registries.
4. Standardize simulator state metadata keys for EEF pose, target pose, object pose, and clamp status.
5. Add new human-play backends through `PlayerRuntimeHooks` first, and only create backend-specific runtimes when the generic `SimAdapterWorkerRuntime` cannot support a simulator.
