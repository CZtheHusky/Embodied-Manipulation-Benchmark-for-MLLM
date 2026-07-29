# DMControl Integration

This backend integrates the DeepMind Control Suite through `suite.load`.

## Verification Status

The adapter/runtime route is verified in the dedicated `dmcontrol` uv
environment:

- `manipulator/bring_ball` can reset, step, render RGB, and close.
- Native actions are shape-checked and clipped to `env.action_spec()`.
- Named MuJoCo cameras can be configured at runtime.
- The simulator can run in the shared worker-process human-play runtime.

DMControl is not currently ready for the MLLM Cartesian perturbation matrix.
The native five-element action for `manipulator/bring_ball` controls actuators
named `root`, `shoulder`, `elbow`, `wrist`, and `grasp`. It is not an
end-effector `dx/dy/dz/droll/gripper` command. A shape-compatible step therefore
does not prove the semantic validity of Cartesian joystick mapping or a 3D
control-frame rotation.

The repository intentionally has no
`dmcontrol_manipulator_bring_ball_mllm*.yaml` entry until that mismatch is
resolved.

## Architecture

- `DMControlSimAdapter` owns the environment and validates native actions.
- `DMControlObservationProvider` extracts camera images and simulator state.
- `DMControlTask` converts reward/state into benchmark metrics.
- `DMControlController` maps configured joystick channels into a native action
  array.

The last component is generic and shape-compatible, but the legacy
`configs/dmcontrol_example.yaml` dimension labels must not be interpreted as a
verified Cartesian control contract for `manipulator/bring_ball`.

## Environment Management

```bash
python3 scripts/manage_uv_env.py create dmcontrol
python3 scripts/manage_uv_env.py smoke dmcontrol
```

Run the adapter-level example:

```bash
python3 scripts/manage_uv_env.py run dmcontrol -- \
  python scripts/run_benchmark.py --config configs/dmcontrol_example.yaml
```

Run dedicated tests:

```bash
PYTHONPATH=src MUJOCO_GL=egl \
  .venvs/dmcontrol/bin/python -m pytest tests/test_dmcontrol_integration.py -q
```

## Requirements Before MLLM Evaluation

One of these routes must be implemented and verified:

1. Add a real Cartesian controller that converts public end-effector deltas
   into the task's native actuator commands using current simulator state.
2. Select a DMControl task with a native Cartesian action contract.
3. Define a separate actuator-space benchmark condition and omit physical
   control-frame rotation from that condition.

The first two routes can support the same fixed and
`permuted_sign_flipped + random_rotation_3d` comparison as the other 3D
backends. The third is a different experiment and must not be labeled as the
same Cartesian perturbation.

## Limitations

- `suite.load` environments are supported; Composer tasks may need adapter
  changes.
- Free-camera orbit randomization is unsupported and fails loudly.
- Human play keeps DMControl in a worker process for OpenGL isolation.
