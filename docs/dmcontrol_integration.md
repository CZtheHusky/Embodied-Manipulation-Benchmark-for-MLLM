# DMControl Integration

This backend integrates the [DeepMind Control Suite (DMControl)](https://github.com/google-deepmind/dm_control) into the benchmark.

## Verification Status

Verified on 2026-07-29 with the `dmcontrol` uv environment:

- `pytest` runs a short real `manipulator/bring_ball` episode and validates action shape/bounds handling.
- A `sim_adapter_worker` smoke produced an RGB frame and stepped once with clipped out-of-bounds action input.
- The MLLM integration tests run both control configurations for four real environment steps with an in-process fake Chat Completions client. They verify two non-black RGB camera streams, five-dimensional simulator actions, three-frame image history, complete action history, prompt privacy, and run-manifest control labels without making a network request.
- `scripts/run_benchmark.py --config configs/dmcontrol_example.yaml` completed one 250-step random-policy episode. The random policy did not solve the task, so the observed `success_rate` was `0.0`.

## Architecture

The integration follows the benchmark's decoupled design:

- **SimAdapter (`DMControlSimAdapter`)**: Manages the life cycle of the DMControl environment loaded via `suite.load` and validates/clips native actions against `env.action_spec()`.
- **Controller (`DMControlController`)**: Translates `JoystickAction` into a native numpy array action for DMControl. It handles joystick mapping and coordinate frame transforms in the benchmark layer.
- **ObservationProvider (`DMControlObservationProvider`)**: Extracts images from DMControl cameras and retrieves environment state. It supports explicit named fixed-camera pose/FOV edits through MuJoCo model fields.
- **Task (`DMControlTask`)**: Maps DMControl rewards and state to benchmark success criteria and metrics.

## Control Complexity

One of the core requirements is supporting control confusion (mapping permutations, sign flips) and camera randomization without modifying DMControl's source code or MJCF XML assets.

- **Joystick Mapping & Frame Rotation**: These are computed in `DMControlController` before the final action is passed to the simulator.
- **Action Bounds**: Final `DMControlAction` values are checked against `env.action_spec()` shape and clipped to bounded action ranges in `DMControlSimAdapter`, including manual play because the worker uses the same adapter.
- **Camera Configuration**: Explicit named fixed-camera parameters are applied at runtime by modifying `physics.model.cam_pos`, `physics.model.cam_quat`, and `physics.model.cam_fovy`. The shared `mujoco_randomized` free-camera orbit manager is not supported for DMControl yet.

## Configuration

Example configuration (`configs/dmcontrol_example.yaml`):

```yaml
simulator:
  type: dmcontrol
  params:
    domain_name: manipulator
    task_name: bring_ball
    mujoco_gl: egl
  env_kwargs: {}

task:
  type: dmcontrol_generic
  max_steps: 250
  success_threshold: 0.95
  params:
    instruction: "Bring the ball to the target area."
    success_key: "reward"

controller:
  type: dmcontrol
  scales:
    dx: 0.1
    dy: 0.1
    dz: 0.1
    droll: 0.1
    gripper: 1.0
  params:
    dimensions: [dx, dy, dz, droll, gripper]
```

### MLLM configurations

The `manipulator/bring_ball` MLLM setup has two standalone configurations:

| Configuration | Mapping | Control frame | Model-visible rule |
| --- | --- | --- | --- |
| `configs/dmcontrol_manipulator_bring_ball_mllm.yaml` | `fixed` | `identity` | Fixed joystick mapping is public. |
| `configs/dmcontrol_manipulator_bring_ball_mllm_perturbed.yaml` | `permuted_sign_flipped` | `random_rotation_3d` | Complexity is disclosed, but the sampled permutation, signs, and rotation are hidden. |

Both configurations use Chat Completions with `action_with_reflection`, `image_window=3`, `action_window=all`, and the `fixed` and `hand` RGB cameras. The benchmark controller emits the environment's five-element native action vector in the configured order `dx`, `dy`, `dz`, `droll`, and final direct gripper channel. Because the configured translational control subspace is three-dimensional, the perturbed configuration uses `random_rotation_3d`.

The model request contains only the task instruction, public control-interface text, action schema, camera names, RGB images, step ids, and permitted action/reflection history. DMControl observation state, reward, target/object positions, distance, simulator actions, seeds, and sampled mapping/frame values remain available for evaluation or logging but are never added to the model request. The perturbed config sets both `mapping_visible_to_agent` and `frame_visible_to_agent` to `false`; changing those flags alone does not make shared public metadata reveal hidden randomized values.

At commit `981ff05`, the shared `ControlInstructionBuilder` explicitly states the joystick values, active dimensions, fixed mapping, and the nondisclosure rule for complex mapping/frame transforms. It represents the baseline identity frame as the absence of frame complexity rather than literally saying `identity` in the model-facing text. If an explicit identity-frame sentence is required, that belongs in the shared prompt builder and its shared tests; the DMControl integration intentionally does not override or duplicate the common prompt.

## Environment Management

Use the dedicated uv environment for DMControl:

```bash
python3 scripts/manage_uv_env.py create dmcontrol
```

## Running

### Benchmark

```bash
python3 scripts/manage_uv_env.py run dmcontrol -- python scripts/run_benchmark.py --config configs/dmcontrol_example.yaml
```

### MLLM benchmark

Select the baseline or perturbed YAML directly:

```bash
python3 scripts/manage_uv_env.py run dmcontrol -- python scripts/run_benchmark.py --config configs/dmcontrol_manipulator_bring_ball_mllm.yaml
python3 scripts/manage_uv_env.py run dmcontrol -- python scripts/run_benchmark.py --config configs/dmcontrol_manipulator_bring_ball_mllm_perturbed.yaml
```

These commands require the model endpoint variables named in the YAML. The local fake-client verification does not read model API env files or make an API request:

```bash
PYTHONPATH=src .venvs/dmcontrol/bin/python -m pytest tests/test_dmcontrol_integration.py -q
```

### Manual Play

```bash
python3 scripts/manage_uv_env.py run dmcontrol -- python scripts/play_benchmark.py --config configs/dmcontrol_example.yaml
```

## Limitations

- Currently supports `suite.load` environments. `composer` environments may require minor updates to the `SimAdapter`.
- Camera pose/FOV edits require explicit named fixed-camera config. The existing `RandomizedMujocoCameraManager` emits free-camera orbit parameters (`distance`, `azimuth`, `elevation`, `lookat`); DMControl `physics.render` does not expose a stable backend hook for applying those through this provider, so the backend fails loudly instead of silently pretending support.
- Manual play intentionally keeps `dm_control` in a separate worker process for GL isolation. The worker uses `DMControlSimAdapter` and `DMControlObservationProvider`, so manual play and benchmark runs share reset, action clipping, camera, observation, and state semantics.
