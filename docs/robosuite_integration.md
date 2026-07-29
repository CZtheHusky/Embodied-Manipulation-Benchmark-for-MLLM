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

Both MLLM configs use a 100-decision API budget. The native robosuite Lift
environment retains its 1000-step horizon, so these results are a local
API-budgeted protocol and must not be reported as the upstream 1000-step Lift
success rate.

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
MuJoCo world x/y/z axes. Its input is already normalized: upstream maps native
`-1` and `1` to approximately `-0.05 m` and `0.05 m`. Both translation-only
MLLM configs therefore use benchmark `dx/dy/dz` scales of `1.0`. A benchmark
scale of `0.05` would apply a second scale and reduce a full command to an
upstream input of `0.05`, or approximately `0.0025 m` after native OSC scaling.

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

After the frame transform, the benchmark checks the normalized OSC component
bounds. If any translation component exceeds magnitude `1`, it uniformly
divides all three components by their largest absolute component. This
preserves the rotated direction while possibly shortening a diagonal step and
prevents robosuite's native per-component clipping from distorting the intended
direction. The same public bound policy is configured in both conditions.

Lift success comes from robosuite's native `_check_success()` oracle, which
requires the cube to be at least `0.04 m` above the table. The config records
that native threshold for audit clarity but does not reimplement the oracle.
All Lift configs reject an initially successful reset before any model request.
The adapter also closes the previous native environment and clears cached
reward/info state before each episode reset.

## Cameras

The Lift configs preserve robosuite's upstream `agentview` task camera. A
position-only override was removed because retaining the old orientation while
moving the camera pointed it away from the robot workspace; the frames were
non-black but unsuitable for visual evaluation. Custom camera settings, when
needed for a new task, must specify and visually validate position and
orientation together.

For the upright upstream `agentview`, the approximate image-plane basis in the
MuJoCo world frame is:

- image right: `+world y`, represented by `[0, 1, 0]`
- image up: `-0.706 world x + 0.708 world z`, represented by
  `[-0.706, 0, 0.708]`

These vectors are stored as config metadata and covered by numerical norm,
orthogonality, and component canaries. They describe this fixed Lift camera,
not a general robosuite camera convention.

The adapter still supports seeded episode-level camera settings through
robosuite runtime APIs and does not mutate static assets.
`camera_randomization` supports fixed `pos`, `quat`, and `fovy` values, plus
`pos_range` and `fovy_range` for seeded sampling.

The MLLM configs request `agentview` and `robot0_eye_in_hand` only. Adapter
state and observation metadata are not part of a Chat Completions request.
Tests poison every private field with unique sentinels and verify that request
text contains none of them. The forbidden set covers simulator state, reward,
object pose, end-effector pose, distance, simulator action, seed, camera
randomization, sampled joystick mapping/signs, and sampled frame rotation.

MuJoCo's direct offscreen render is vertically inverted. The adapter applies
one vertical flip both when formatting robosuite observation images and when
serving `render(camera)`, so UI frames and model-visible PNGs share the same
upright public orientation.

## Public prompt status

The shared prompt currently exposes the task, RGB camera names, active
`dx/dy/dz` end-effector translation semantics, action schema, visible fixed
mapping or complexity presence, the absence of an additional frame rotation
in the fixed condition, and the fact that the gripper is the final
direct-control joystick. It does not serialize simulator state or exact
perturbation values.

The shared `ControlInstructionBuilder` also identifies the robosuite OSC world
frame, states that the control is normalized translation-only with native
`+/-1 -> +/-0.05 m`, gives the approximate `agentview` image-plane basis, states
that gripper `0` opens, `0.5` holds, and `1` closes, and tells the model that
hidden mapping/frame samples remain constant within an episode. The sampled
permutation, signs, quaternion, and matrix remain undisclosed.

## Local verification

Run the robosuite-only suite in the pinned environment:

```bash
PYTHONPATH=src NUMBA_DISABLE_JIT=1 \
  .venvs/robosuite/bin/python -m pytest tests/test_robosuite_integration.py -q -rs
```

This includes config validation, controller checks, an exactly-one-flip canary,
real simulator reset/step smoke, and a four-step Fake Client episode for each
MLLM condition. The Fake Client test decodes the model request PNGs and checks
that they are upright, non-black, and retain the visible Lift cube.

## Current Limitations

- Initial support is focused on Lift.
- The adapter is intentionally lightweight and should be hardened as more robosuite tasks are added.
