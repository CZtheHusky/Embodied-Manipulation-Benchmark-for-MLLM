# Benchmark Design

This benchmark evaluates whether an MLLM can act as a closed-loop embodied manipulation agent through visual feedback and a structured virtual joystick interface.

The core motivation is that human teleoperators often complete manipulation tasks by observing a robot through one or more cameras and issuing incremental controls through a physical device. MLLMs cannot directly use those physical devices, so the benchmark exposes virtual joystick channels that can be described in text and emitted as structured JSON actions.

The initial version focuses on a simple end-effector delta pose action space:

- `joystick_0 -> dx`
- `joystick_1 -> dy`
- `joystick_2 -> dz`
- `joystick_3 -> droll`
- `joystick_4 -> dpitch`
- `joystick_5 -> dyaw`
- `joystick_6 -> gripper`

The first six channels are discrete delta commands in `{-1, 0, 1}` and are scaled by configuration. The gripper channel is a direct continuous command in `[0, 1]`.

The v0 implementation intentionally uses a dummy simulator and a reach-target task so the benchmark pipeline can be tested before binding to ManiSkill, LIBERO, RLBench, robosuite, OmniGibson, BEHAVIOR-1K, Isaac Sim, or other environments.

Future benchmark variants should extend the same interfaces to support randomized mappings, sign flips, hidden mapping metadata, camera randomization, multi-view observations, wrist cameras, control-frame transforms, online calibration, and MLLM-backed policies.

This benchmark does not try to solve high-frequency low-level servo control training. It evaluates visual spatial understanding, action grounding, control-interface understanding, and closed-loop decision making at the virtual teleoperation layer.
