# Benchmark Design

This benchmark evaluates whether an MLLM can act as a closed-loop embodied manipulation agent through visual feedback and a structured virtual joystick interface.

The core motivation is that human teleoperators often complete manipulation tasks by observing a robot through one or more cameras and issuing incremental controls through a physical device. MLLMs cannot directly use those physical devices, so the benchmark exposes virtual joystick channels that can be described in text and emitted as structured JSON actions.

The benchmark exposes an agent-visible control interface rather than a fixed global joystick count. Simple planar tasks can expose only `dx` and `dy`; full manipulation tasks can expose more end-effector pose dimensions and a gripper when the embodiment supports one.

A full end-effector delta pose interface can contain virtual joystick channels for `dx`, `dy`, `dz`, `droll`, `dpitch`, `dyaw`, and optionally a final gripper channel.

Motion channels are usually discrete delta commands in `{-1, 0, 1}` and are scaled by configuration. A gripper channel is exposed only when the task and embodiment support it; direct gripper control uses `[0, 1]`.

For Reacher-style planar tasks, the current interface uses two virtual joysticks for planar motion. The benchmark can tell the agent that planar motion channels exist without revealing the exact joystick-to-physical-axis relationship when mapping or frame complexity is enabled.

The v0 implementation intentionally uses a dummy simulator and a reach-target task so the benchmark pipeline can be tested before binding to ManiSkill, LIBERO, RLBench, robosuite, OmniGibson, BEHAVIOR-1K, Isaac Sim, or other environments.

Future benchmark variants should extend the same interfaces to support randomized mappings, sign flips, hidden mapping metadata, camera randomization, multi-view observations, wrist cameras, control-frame transforms, online calibration, and MLLM-backed policies. Joystick mapping and control-frame transforms are intentionally separate so permutation/sign confusion and rotated control frames can be enabled independently or together.

Mapping confusion only applies to motion dimensions. If a gripper channel is present, it is always the final virtual joystick and is not permuted or sign-flipped.

Human play panels and MLLM prompts must expose the same public control-interface metadata. For mapping permutations, sign flips, frame rotations, or similar control-complexity mechanisms, the visible interface only states whether such complexity is present. It must not reveal the exact permutation, sign assignment, rotation type, rotation matrix, or sampled parameters. Full internal values may still be written to debug/log artifacts for offline diagnosis.

Control-frame rotation is defined at the benchmark control layer rather than inside a specific simulator. Planar tasks use 2D rotation over `dx/dy`, sampled uniformly on SO(2) for random rotation. Spatial tasks use 3D rotation over `dx/dy/dz`, sampled uniformly on SO(3) for random rotation.

This benchmark does not try to solve high-frequency low-level servo control training. It evaluates visual spatial understanding, action grounding, control-interface understanding, and closed-loop decision making at the virtual teleoperation layer.
