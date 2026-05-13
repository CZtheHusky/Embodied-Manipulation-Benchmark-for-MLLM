# Reacher Integration

The Reacher environment is implemented locally in this repository and registered with Gymnasium. Gymnasium is not forked or vendored.

## Components

- Env registration: `mllm_teleop_benchmark.sim.gymnasium_envs.registry`
- Env implementation: `sim/gymnasium_envs/envs/reacher_delta_eef.py`
- XML asset: `sim/gymnasium_envs/envs/assets/reacher_pos.xml`
- Benchmark adapter: `sim/gymnasium_envs/adapter.py`
- Observation provider: `sim/gymnasium_envs/observation.py`
- Controller: `control/gymnasium_box.py`
- Task: `tasks/gymnasium_reacher.py`
- Config: `configs/reacher_delta_eef.yaml`

## Control

`ReacherDeltaEEFEnv` exposes a 2D Gymnasium box action in `[-1, 1]`.

- action[0] controls world-frame end-effector delta x
- action[1] controls world-frame end-effector delta y
- `eef_step_scale` converts normalized commands to meter-scale deltas

The benchmark-facing `control_interface` exposes exactly two virtual joysticks for planar motion.

The human panel and MLLM prompt describe the same public interface: control mode, joystick count, scale summary, active dimensions when configured as visible, and whether mapping/frame complexity is present. They do not reveal the exact joystick-to-axis assignment, sign assignment, or frame rotation when those control-complexity mechanisms are enabled.

No gripper channel is exposed for Reacher because the environment has no gripper. This keeps human play and MLLM play aligned with the actual controllable dimensions.

The environment uses a small `PlanarReacherKinematics` helper:

- maintain an internal end-effector target in world XY
- clamp target to the reachable annulus
- solve joint targets with MuJoCo Jacobian pseudo-inverse
- send joint position targets to position actuators from `reacher_pos.xml`

## Camera And Rendering

Benchmark episodes use Gymnasium `rgb_array` rendering through `GymnasiumSimAdapter.render()`.

Manual play uses a separate two-process path because MuJoCo/OpenGL and Pygame/SDL can conflict in one process. See `docs/manual_play.md`.

Avoid reintroducing `mujoco.Renderer` direct rendering into the default path unless it is isolated and tested on both headless and desktop setups.

## Current Limitations

- Reacher only supports delta-EEF control in the current local env.
- Pusher is not implemented yet.
- Camera randomization metadata is supported in the benchmark path; the direct manual player uses the worker-rendered view.
