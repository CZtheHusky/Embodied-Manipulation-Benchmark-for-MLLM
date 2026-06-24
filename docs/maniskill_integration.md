# ManiSkill Integration

ManiSkill is integrated as an optional backend. The benchmark does not fork or modify ManiSkill source code.

## Components

- Adapter: `sim/maniskill/adapter.py`
- Observation provider: `sim/maniskill/observation.py`
- Controller: `control/maniskill.py`
- Task wrapper: `tasks/maniskill.py`
- Human play UI: `ui/human_player.py` with `ui.runtime: sim_adapter_worker`
- Example config: `configs/maniskill_pickcube_delta_eef.yaml`

## Dependency

ManiSkill is optional because it is a heavier simulator dependency. Use the CPU uv environment by default:

```bash
python3 scripts/manage_uv_env.py create maniskill-cpu
```

On NVIDIA CUDA 12.6 hosts, use the GPU uv environment:

```bash
python3 scripts/manage_uv_env.py create maniskill-cu126
python3 scripts/manage_uv_env.py smoke maniskill-cu126
```

Core tests and non-ManiSkill environments continue to work when `mani_skill` is not installed.

## Control

The initial config targets `PickCube-v1` with ManiSkill `control_mode: pd_ee_delta_pose`.

The benchmark control path remains simulator-independent:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> ManiSkillDeltaEEFController -> env.step()
```

Joystick permutation, sign changes, and coordinate-frame rotation are handled by this repository before native ManiSkill actions are produced. ManiSkill receives a clipped native Box action.

The gripper, when present, remains the final virtual joystick. The default mapping is `0.0 -> -1.0` and `1.0 -> 1.0`, configurable through controller params.

## Human Play

The ManiSkill play UI uses the shared Pygame human player and SimAdapter worker:

```bash
python3 scripts/manage_uv_env.py run maniskill-cpu -- python scripts/play_benchmark.py --config configs/maniskill_pickcube_delta_eef.yaml
```

The config selects `ui.backend: pygame` and `ui.runtime: sim_adapter_worker`. The main process owns Pygame, keyboard input, action parsing, validation, mapping, frame transform, controller translation, logging, and debug files. The worker process owns `ManiSkillSimAdapter`, task reset, step, observation extraction, and RGB rendering fallback.

Keyboard controls are virtual joystick controls, not privileged task shortcuts. `W/S`, `A/D`, `Q/E`, `J/L`, `K/I`, and `U/O` drive motion joysticks in order. `G` or `Space` toggles the final gripper joystick when the configured control interface has a gripper.

## Observation And Cameras

ManiSkill observations are read from `obs_mode: rgb` sensor data. The observation provider extracts per-camera RGB frames and falls back to `env.render()` only if no image is available in the native observation.

Camera randomization is episode-level only. `maniskill_randomized` samples camera position during `CameraManager.reset(seed)` and passes the sampled sensor config before ManiSkill env creation/reset. The project does not support changing camera views inside an episode.

## Current Limitations

- Initial support is single-env (`num_envs: 1`) and Box-action focused.
- ManiSkill is not installed in the core uv environment.
- Runtime camera mutation is intentionally out of scope.
- Task success defaults to the `success` key in ManiSkill info and can be changed with `task.params.success_key`.
