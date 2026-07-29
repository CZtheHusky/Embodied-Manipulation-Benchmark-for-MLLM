# DMControl Integration

This backend integrates DeepMind Control Suite environments through
`suite.load`.

## Benchmark Validity

The native `manipulator/bring_ball` action has five actuator-space entries:

```text
root, shoulder, elbow, wrist, grasp
```

They are motor controls, not `dx/dy/dz/droll/gripper`. The legacy
`dmcontrol` controller remains available for native-action examples, but its
configured dimension labels are not a verified Cartesian contract.

The MLLM configurations use the separate `dmcontrol_planar_eef` controller.
Its public action is:

```text
dx, dy, gripper
```

- `dx` is end-effector translation along MuJoCo world x; world `+x` projects
  image-right in the fixed camera.
- `dy` is end-effector translation along MuJoCo world z; world `+z` projects
  image-up in the fixed camera.
- gripper `0` applies maximum opening effort, `0.5` applies no gripper effort,
  and `1` applies maximum closing effort.

A mapped motion magnitude of `1` requests a `0.03 m` Cartesian target
displacement before the Cartesian-impedance conversion. The upstream
`manipulator.xml` defines the fixed camera with its positive image-x axis
aligned to world `+x` and positive image-y/up axis aligned to world `+z`. The
`hand` camera uses MuJoCo `mode="track"` and moves with the hand, so the fixed
camera's image-axis relation is never claimed for that view.

The controller emits `DMControlPlanarEEFAction`, not a five-element native
motor vector. At the simulator boundary, `DMControlSimAdapter` reads the
current MuJoCo physics state, computes the x/z translational Jacobian for the
`grasp` site, and applies a Cartesian impedance command through the four arm
motors. Bias forces are compensated and native controls are clipped to the
upstream action spec. The gripper command is converted from public `[0, 1]` to
the native `[-1, 1]` grasp motor range.

Physics state, Jacobians, joint state, actuator controls, rewards, object and
target positions, seeds, and sampled control perturbations remain inside the
simulator/evaluator boundary. They are not projected into model requests.

## MLLM Configurations

| Configuration | Mapping | Frame |
| --- | --- | --- |
| `configs/dmcontrol_manipulator_bring_ball_mllm.yaml` | `fixed` | `identity` |
| `configs/dmcontrol_manipulator_bring_ball_mllm_perturbed.yaml` | `permuted_sign_flipped` | `random_rotation_2d` |

Both use:

- Chat Completions
- `action_with_reflection`
- reasoning effort `low`
- three historical image turns plus the current turn
- complete normalized action/reflection history
- fixed and hand RGB cameras

For the perturbed condition, the existence of mapping and frame complexity is
public, but the sampled permutation, signs, and rotation angle are hidden and
fixed for the episode.

## Architecture

- `DMControlSimAdapter` owns reset, stepping, rendering, native bounds, and the
  planar EEF-to-motor conversion.
- `DMControlController` supports the legacy native-action route.
- `DMControlPlanarEEFController` maps public joystick channels and applies the
  benchmark mapping/frame transforms before creating a typed planar command.
- `DMControlObservationProvider` extracts camera images and evaluator state.
- `DMControlTask` converts reward/state into benchmark metrics.

## Verification

Run the dedicated suite without any model endpoint:

```bash
MUJOCO_GL=egl PYTHONPATH=src \
  .venvs/dmcontrol/bin/python -m pytest \
  tests/test_dmcontrol_integration.py -q
```

The tests verify:

- upstream actuator names and the native action contract;
- baseline and perturbed configuration contracts;
- fixed mapping and seeded permutation/sign/2D rotation behavior;
- pinned camera XML semantics and the distinction between the fixed and
  hand-tracking views;
- positive and negative single-axis commands move the real `grasp` site in
  the expected world x/z direction;
- gripper `0` opens and gripper `1` closes in real physics;
- real reset, step, named-camera render, and worker-process shutdown;
- fixed and hand RGB frames are non-black;
- a four-step Fake Client episode preserves image/action history;
- model request text excludes simulator state, native motors, Jacobians,
  rewards, seeds, and sampled perturbation values.

The physical-direction checks use benchmark seed 7. Each signed x-axis command
moves the grasp site by more than `0.5 mm` in the requested x direction in one
control step; each signed visual-y command moves it by more than `0.5 mm` in
the requested MuJoCo world-z direction. Gripper verification compares the
finger aperture after 20 opening steps with the aperture after 20 closing
steps.

## Environment Management

```bash
python3 scripts/manage_uv_env.py create dmcontrol
python3 scripts/manage_uv_env.py smoke dmcontrol
```

Run either MLLM condition through the dedicated environment:

```bash
python3 scripts/manage_uv_env.py run dmcontrol -- \
  python scripts/run_benchmark.py \
  --config configs/dmcontrol_manipulator_bring_ball_mllm.yaml

python3 scripts/manage_uv_env.py run dmcontrol -- \
  python scripts/run_benchmark.py \
  --config configs/dmcontrol_manipulator_bring_ball_mllm_perturbed.yaml
```

These commands require the model endpoint variables named in the YAML. The
dedicated test suite uses an in-process Fake Client and makes no network
request.

## Limitations

- The planar controller is specific to the DMControl `manipulator` model and
  its named arm actuators and `grasp` site. Other domains fail loudly.
- This is a two-dimensional x/z task. It must not be compared as a 3D
  Cartesian-control condition.
- Free-camera orbit randomization is unsupported and fails loudly.
- Composer tasks may require adapter changes.
