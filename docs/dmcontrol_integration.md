# DMControl Integration

This backend integrates the [DeepMind Control Suite (DMControl)](https://github.com/google-deepmind/dm_control) into the benchmark.

## Verification Status

Verified on 2026-06-23 with the `dmcontrol` uv environment:

- `pytest` runs a short real `manipulator/bring_ball` episode and validates action shape/bounds handling.
- A direct-worker smoke produced an RGB frame and stepped once with clipped out-of-bounds action input.
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
- **Action Bounds**: Final `DMControlAction` values are checked against `env.action_spec()` shape and clipped to bounded action ranges in both the adapter path and the direct manual-play worker.
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

### Manual Play

```bash
python3 scripts/manage_uv_env.py run dmcontrol -- python scripts/play_benchmark.py --config configs/dmcontrol_example.yaml --backend direct_dmcontrol
```

## Limitations

- Currently supports `suite.load` environments. `composer` environments may require minor updates to the `SimAdapter`.
- Camera pose/FOV edits require explicit named fixed-camera config. The existing `RandomizedMujocoCameraManager` emits free-camera orbit parameters (`distance`, `azimuth`, `elevation`, `lookat`); DMControl `physics.render` does not expose a stable backend hook for applying those through this provider, so the backend fails loudly instead of silently pretending support.
- `DirectDMControlPlayer` intentionally keeps a separate worker process for GL isolation. The worker still bypasses `DMControlSimAdapter`, but it now shares the same native action validation/clipping helper used by the adapter. Rendering uses the configured camera spec size and lets the Pygame side scale/display frames.
