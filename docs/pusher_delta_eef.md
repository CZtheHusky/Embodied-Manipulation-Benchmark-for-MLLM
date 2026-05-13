# Pusher Delta-EEF Integration

The Pusher environment is implemented as a local Gymnasium/MuJoCo environment and registered as `MLLMTeleop/PusherDeltaEEF-v0`.

Gymnasium's native `Pusher-v5` is a 7-DoF MuJoCo arm. Its native action is a 7D torque command over the arm joints:

- `r_shoulder_pan_joint`
- `r_shoulder_lift_joint`
- `r_upper_arm_roll_joint`
- `r_elbow_flex_joint`
- `r_forearm_roll_joint`
- `r_wrist_flex_joint`
- `r_wrist_roll_joint`

This is not a delta end-effector control interface. The benchmark therefore owns a local XML asset derived from Gymnasium's Pusher model and replaces the torque motors with joint position actuators.

## Control Interface

The benchmark-facing action is 3D delta EEF position:

- `dx`
- `dy`
- `dz`

There is no gripper and no `rpy` channel in the initial Pusher integration. The task is object pushing, and the native environment has no gripper. Keeping the exposed channels to `dx/dy/dz` keeps human play and MLLM play aligned with the controllable task semantics.

The controller path is:

```text
virtual joystick action
-> JoystickMapping
-> ControlFrameTransform
-> PusherDeltaEEFJoystickController
-> 3D Gymnasium box action
-> PusherDeltaEEFEnv
-> Jacobian IK
-> 7 joint position targets
```

This means joystick permutation, sign flips, and SO(3) frame rotation can be enabled from config without changing task code.

## IK Design

`PusherDeltaEEFEnv` tracks a target end-effector position for the `tips_arm` body. Each 3D action is clipped to `[-1, 1]`, scaled by `eef_step_scale`, accumulated into the target, and clamped to a configured workspace.

The environment computes a damped least-squares update using the translational MuJoCo body Jacobian:

```text
dq = J.T @ inv(J @ J.T + damping^2 I) @ position_error
```

The solved 7D joint vector is sent to the position actuators as the MuJoCo control.

## Success and Metrics

The task succeeds when `object_goal_distance <= success_threshold`. The environment also reports:

- `fingertip_object_distance`
- `object_position`
- `goal_position`
- `fingertip_position`
- `ee_target_xyz`
- `ee_target_reachability_clamped`
- `goal_hold_counter`

## Manual Play

Run:

```bash
python scripts/play_benchmark.py --config configs/pusher_delta_eef.yaml
```

Default keys:

- `W/S`: `joystick_0`
- `A/D`: `joystick_1`
- `Q/E`: `joystick_2`
- `R`: reset
- `Esc`: quit

The visible control panel is generated from the same control-interface metadata intended for MLLM prompting.

## Current Limitations

- The initial implementation controls only EEF position, not orientation.
- The IK solver is designed for benchmark interaction, not high-performance robotics control.
- The scripted/random policies are not expected to solve Pusher reliably.
- The local XML is derived from Gymnasium's Pusher model to avoid forking Gymnasium.
