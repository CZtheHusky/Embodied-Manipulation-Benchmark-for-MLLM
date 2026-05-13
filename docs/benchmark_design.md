# Benchmark Design

This benchmark evaluates whether an MLLM can act as a closed-loop embodied manipulation agent through visual feedback and a structured virtual joystick interface.

The core motivation is that human teleoperators often complete manipulation tasks by observing a robot through one or more cameras and issuing incremental controls through a physical device. MLLMs cannot directly use those physical devices, so the benchmark exposes virtual joystick channels that can be described in text and emitted as structured JSON actions.

The benchmark exposes an agent-visible control interface rather than a fixed global joystick count. Simple planar tasks can expose only `dx` and `dy`; full manipulation tasks can expose more end-effector pose dimensions and a gripper when the embodiment supports one.

A full end-effector delta pose interface can contain:

- `joystick_0 -> dx`
- `joystick_1 -> dy`
- `joystick_2 -> dz`
- `joystick_3 -> droll`
- `joystick_4 -> dpitch`
- `joystick_5 -> dyaw`
- `joystick_6 -> gripper`

Motion channels are usually discrete delta commands in `{-1, 0, 1}` and are scaled by configuration. A gripper channel is exposed only when the task and embodiment support it; direct gripper control uses `[0, 1]`.

For Reacher-style planar tasks, the current interface is only:

- `joystick_0 -> dx`
- `joystick_1 -> dy`

The v0 implementation intentionally uses a dummy simulator and a reach-target task so the benchmark pipeline can be tested before binding to ManiSkill, LIBERO, RLBench, robosuite, OmniGibson, BEHAVIOR-1K, Isaac Sim, or other environments.

Future benchmark variants should extend the same interfaces to support randomized mappings, sign flips, hidden mapping metadata, camera randomization, multi-view observations, wrist cameras, control-frame transforms, online calibration, and MLLM-backed policies. Joystick mapping and control-frame transforms are intentionally separate so permutation/sign confusion and rotated control frames can be enabled independently or together.

Mapping confusion only applies to motion dimensions. If a gripper channel is present, it is always the final virtual joystick and is not permuted or sign-flipped.

Control-frame rotation is defined at the benchmark control layer rather than inside a specific simulator. Planar tasks use 2D rotation over `dx/dy`, sampled uniformly on SO(2) for random rotation. Spatial tasks use 3D rotation over `dx/dy/dz`, sampled uniformly on SO(3) for random rotation.

This benchmark does not try to solve high-frequency low-level servo control training. It evaluates visual spatial understanding, action grounding, control-interface understanding, and closed-loop decision making at the virtual teleoperation layer.
