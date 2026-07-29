# robosuite Integration

robosuite is integrated as an optional backend. It is not part of the root core environment.

## Components

- Adapter: `sim/robosuite_adapter.py`
- Controller: `control/robosuite_controller.py`
- Task wrapper: `tasks/robosuite_task.py`
- Human/random-agent config: `configs/robosuite_lift.yaml`
- Fixed MLLM config: `configs/robosuite_lift_mllm_reflection.yaml`
- Perturbed MLLM config: `configs/robosuite_lift_mllm_reflection_perturbed.yaml`
- Environment: `envs/uv/robosuite.in` and `envs/uv/locks/robosuite.lock.txt`

## Dependency

Use the backend-specific uv environment:

```bash
python3 scripts/manage_uv_env.py create robosuite
```

The backend pins `robosuite==1.4.1` and `mujoco>=2.3,<3.0`. robosuite 1.4.1 calls the MuJoCo 2.x Python API and fails with MuJoCo 3.x at controller reset. Core tests and non-robosuite environments must continue to work without robosuite installed. The adapter sets `NUMBA_DISABLE_JIT=1` by default before importing robosuite because this pinned install can otherwise fail during numba cache setup.

To require the optional backend smoke test instead of skipping it:

```bash
MLLM_TELEOP_REQUIRE_ROBOSUITE=1 pytest tests/test_robosuite_integration.py -rs
```

Manual play uses the same uv environment:

```bash
python3 scripts/manage_uv_env.py run robosuite -- python scripts/play_benchmark.py --config configs/robosuite_lift.yaml
```

## MLLM configs

Both Lift MLLM configs use OpenAI-compatible Chat Completions with the
`action_with_reflection` output contract. They send the current pair of RGB
frames, up to three prior pairs of RGB frames, and all prior normalized model
actions and reflections:

```text
agent.model.api: chat_completions
agent.action_output_mode: action_with_reflection
agent.history.image_window: 3
agent.history.action_window: all
```

Choose the public fixed baseline:

```bash
python3 scripts/manage_uv_env.py run robosuite -- \
  python scripts/run_benchmark.py --config configs/robosuite_lift_mllm_reflection.yaml
```

Or choose the control-perturbed condition:

```bash
python3 scripts/manage_uv_env.py run robosuite -- \
  python scripts/run_benchmark.py --config configs/robosuite_lift_mllm_reflection_perturbed.yaml
```

The configs resolve model name, base URL, and API key from the standard
environment-variable fields. Tests replace the client with an in-process fake
and do not load `.env.model-api.local` or contact any endpoint.

## Control

The initial config targets robosuite `Lift` with Panda and OSC pose control.

The benchmark control path remains:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> RobosuiteOSCController -> env.step()
```

The robosuite controller emits a `RobosuiteAction` wrapper. The adapter converts it to the native numpy action only at the simulator boundary.

Lift intentionally exposes only end-effector translation and the gripper. The
public joystick schema has four channels:

| Channel role | Public values | Native OSC action |
| --- | --- | --- |
| Three motion channels | Discrete `-1`, `0`, or `1` | `dx`, `dy`, `dz` in indices 0-2 |
| Final gripper channel | Continuous `[0, 1]` | Gripper command in index 6 |

Native OSC orientation indices 3-5 are always zero. This is why
`random_rotation_3d`, rather than a 2D transform or a 6D pose transform, is the
matching frame perturbation for Lift.

robosuite 1.4.1's `OSC_POSE` controller adds its scaled translation delta to
the current end-effector position, so its Cartesian action is expressed along
world x/y/z axes. The benchmark controller first applies its configured
translation scales and then the benchmark frame transform before calling
robosuite. The original `0.05` benchmark scales are preserved in both MLLM
conditions.

The benchmark gripper value is converted with `2 * value - 1`. Panda therefore
receives `-1` for public value `0` (open), `0` for public value `0.5` (hold),
and `1` for public value `1` (close). The final gripper channel is never
permuted, sign-flipped, or rotated.

The two conditions differ only in the motion-control complexity:

- Fixed: `joystick_0/1/2 -> +dx/+dy/+dz`; the identity frame transform passes
  those world-axis deltas through unchanged.
- Perturbed: an episode-seeded signed permutation maps the first three
  joysticks to `dx/dy/dz`, then an episode-seeded uniform SO(3) rotation maps
  the resulting delta into the OSC world axes. The sampled mapping, signs,
  quaternion, and rotation matrix remain private.

## Cameras

Camera settings are sampled from the episode seed and applied during reset through robosuite runtime APIs. The project does not mutate static assets to randomize camera views. `camera_randomization` supports fixed `pos`, `quat`, and `fovy` values, plus `pos_range` and `fovy_range` for seeded episode-level sampling.

The MLLM configs request `agentview` and `robot0_eye_in_hand` only. Adapter
state and observation metadata are not part of a Chat Completions request.
Tests poison every private field with unique sentinels and verify that request
text contains none of them. The forbidden set covers simulator state, reward,
object pose, end-effector pose, distance, simulator action, seed, camera
randomization, sampled joystick mapping/signs, and sampled frame rotation.

## Public prompt status

The shared prompt currently exposes the task, RGB camera names, active
`dx/dy/dz` end-effector translation semantics, action schema, visible fixed
mapping or complexity presence, the absence of an additional frame rotation
in the fixed condition, and the fact that the gripper is the final
direct-control joystick. It does not serialize simulator state or exact
perturbation values.

Three backend-independent wording requirements remain in the shared
`ControlInstructionBuilder` and are intentionally not patched by this
robosuite-only integration:

1. Bind the named control frame to robosuite OSC world x/y/z axes; the shared
   prompt now says that no additional rotation is applied but does not identify
   the backend frame.
2. State explicitly that gripper `0` means open, `0.5` means hold, and `1`
   means close; the JSON schema currently supplies only the `[0, 1]` range.
3. State that hidden mapping and frame transforms are constant within an
   episode while continuing to withhold their sampled values.

Until those shared sentences are added through an approved core change, the
YAML schema and this backend document are complete, but the model-facing prose
does not fully state all robosuite control semantics.

## Local verification

Run the robosuite-only suite in the pinned environment:

```bash
PYTHONPATH=src NUMBA_DISABLE_JIT=1 \
  .venvs/robosuite/bin/python -m pytest tests/test_robosuite_integration.py -q -rs
```

This includes config validation, controller checks, real simulator reset/step
smoke, non-black RGB assertions, and a two-step Fake Client episode for each
MLLM condition.

## Current Limitations

- Initial support is focused on Lift.
- The adapter is intentionally lightweight and should be hardened as more robosuite tasks are added.
- The remaining public-prompt wording requirements above require a shared-core change.
