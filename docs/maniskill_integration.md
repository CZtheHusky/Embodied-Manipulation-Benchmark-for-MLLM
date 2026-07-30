# ManiSkill Integration

ManiSkill is integrated as an optional backend. The benchmark does not fork or modify ManiSkill source code.

## Components

- Adapter: `sim/maniskill/adapter.py`
- Observation provider: `sim/maniskill/observation.py`
- Controller: `control/maniskill.py`
- Task wrapper: `tasks/maniskill.py`
- Human play UI: `ui/human_player.py` with `ui.runtime: sim_adapter_worker`
- Example config: `configs/maniskill_pickcube_delta_eef.yaml`
- Fixed MLLM config: `configs/maniskill_pickcube_delta_eef_mllm_chat_completions_fixed.yaml`
- Perturbed MLLM config: `configs/maniskill_pickcube_delta_eef_mllm_chat_completions_perturbed.yaml`

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

Local CPU verification uses a fake in-process model client and does not make a
network request:

```bash
PYTHONPATH=src .venvs/maniskill-cpu/bin/python -m pytest tests/test_maniskill_backend.py -q
```

The smoke runs both MLLM configurations for five real PickCube steps, checks
the `1, 2, 3, 4, 4` image-window progression, retains all prior actions,
decodes a non-black `512x512` RGB request image, and checks the request privacy
boundary.

## Control

The initial config targets `PickCube-v1` with ManiSkill `control_mode: pd_ee_delta_pose`.

The benchmark control path remains simulator-independent:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> ManiSkillDeltaEEFController -> env.step()
```

Joystick permutation, sign changes, and coordinate-frame rotation are handled by this repository before native ManiSkill actions are produced. ManiSkill receives a clipped native Box action.

The six motion joysticks accept only `-1`, `0`, or `1`. The final virtual
joystick is the direct gripper and accepts any value in `[0, 1]`: `0` means
fully open and `1` means fully closed. The controller converts that public
range to ManiSkill's default native gripper range, `0.0 -> -1.0` and
`1.0 -> 1.0`. The gripper is never permuted, sign-flipped, or frame-rotated.

Mapping is applied before the frame transform. The transform rotates both the
3D translation vector (`dx`, `dy`, `dz`) and the small delta-rotation vector
(`droll`, `dpitch`, `dyaw`).

For Panda `pd_ee_delta_pose`, the pinned upstream controller uses
`root_translation:root_aligned_body_rotation`. The Panda root has identity
orientation in PickCube, so this is the fixed ManiSkill world frame: `+x`
points from the robot base toward the table center, `+y` runs laterally across
the table toward the render-camera side, and `+z` points upward. The public
prompt states that both translation and rotation-vector axes are root-aligned,
not tool-local.

The benchmark scales a magnitude-1 translation command to `0.01 m` and a
magnitude-1 rotation command to `1.8 degrees` before the pinned
`PDEEPoseController` applies its native range. The pinned controller scales a
positive normalized rotation input by
its negative `rot_lower` bound. `ManiSkillDeltaEEFController` therefore applies
the explicit configured `rotation_input_sign: -1` after the benchmark frame
transform. This makes a public positive `droll`, `dpitch`, or `dyaw` command
produce a positive root-frame rotation, matching the shared joystick contract.

## PickCube MLLM Configurations

PickCube has two standalone MLLM configurations:

| Configuration | Mapping | Control frame | Model-visible rule |
| --- | --- | --- | --- |
| `configs/maniskill_pickcube_delta_eef_mllm_chat_completions_fixed.yaml` | `fixed` | `identity` | The exact ordered mapping and the identity-frame rule are public. |
| `configs/maniskill_pickcube_delta_eef_mllm_chat_completions_perturbed.yaml` | `permuted_sign_flipped` | `random_rotation_3d` | The permutation/sign and 3D-frame rules are public, but their episode realization is hidden. |

Both use the OpenAI-compatible Chat Completions API,
`chat_completions` with optional Reflection, `image_window: 3`, and `action_window: all`. They
send the official `render_camera` RGB stream. ManiSkill sensor observations
intentionally hide the green PickCube goal site, so the benchmark uses the
human render camera where both the red cube and green goal marker are public.
Both MLLM configs use base seed `9`; the ten-episode run therefore covers
seeds `9` through `18`, which are verified to be non-successful immediately
after reset. `task.params.reject_initial_success: true` fails before the first
model request if a future seed is already successful instead of counting a
zero-step episode as model success. The discarded seed-7 batch is preserved
under `outputs/maniskill_pickcube_delta_eef_mllm_chat_completions_fixed/invalid/initial-success/`
because its seed-8 episode started with the cube already placed.
Select either YAML directly:

```bash
python3 scripts/manage_uv_env.py run maniskill-cpu -- python scripts/run_benchmark.py --config configs/maniskill_pickcube_delta_eef_mllm_chat_completions_fixed.yaml
python3 scripts/manage_uv_env.py run maniskill-cpu -- python scripts/run_benchmark.py --config configs/maniskill_pickcube_delta_eef_mllm_chat_completions_perturbed.yaml
```

The model request is a strict projection containing only the task
instruction, public control text and action schema, camera names, RGB images,
step ids, and the permitted normalized action/reflection history. Adapter
state, reward, cube/goal/EEF poses, distance, native simulator actions, seeds,
camera metadata, and sampled mapping/frame values remain outside the request.
They may still exist in simulator state and benchmark logs for evaluation.

The shared `ControlInstructionBuilder` states the motion joystick values, 3D
translation/rotation semantics, physical controller frame, fixed mapping or
hidden-complexity rule, identity-frame behavior, and configured public gripper
commands. Hidden mapping and frame samples remain fixed within an episode. The
task instruction therefore contains only the task objective instead of
duplicating backend control prose.

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
- MLLM evaluation rejects initial-success states; it does not silently resample
  them, so changing the episode range requires an explicit seed-range check.
