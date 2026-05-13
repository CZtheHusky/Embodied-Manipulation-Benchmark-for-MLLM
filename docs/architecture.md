# Architecture

The code is organized around replaceable interfaces rather than a single simulator or action mode.

`BenchmarkRunner` loads a typed config and runs episodes. `EpisodeManager` owns the step loop and connects a `Task`, `SimAdapter`, `ObservationProvider`, `Agent`, `ActionParser`, `ActionValidator`, `Controller`, `MetricsRecorder`, and `EpisodeLogger`.

The simulator boundary is `SimAdapter`. Real environments should implement `reset`, `step`, `render`, and `get_state` without changing agents, controllers, or logging.

The action boundary is split into three layers:

- `ControlInterfaceConfig` describes the agent-visible virtual joystick interface.
- `ActionParser` converts raw policy output into `JoystickAction`.
- `ActionValidator` enforces the configured action schema.
- `Controller` translates valid joystick actions into simulator actions.

`action_space` and `mapping` remain as compatibility config sections, but new configs should prefer `control_interface`. When present, `control_interface` generates the joystick schema and fixed mapping used by both human play and MLLM play.

`JoystickMapping` and `ControlFrameTransform` are separate dependencies of the controller. Mapping is applied first, from virtual joystick channels to control-frame dimensions. Frame transform is applied second, from control-frame deltas to simulator/controller-frame deltas. This supports future joystick permutation/sign confusion, coordinate-frame rotation, and combinations of both without changing task logic.

Frame transforms are cross-environment control-layer components. `rotation_2d` rotates `dx/dy` around z; `rotation_3d` rotates `dx/dy/dz` with roll/pitch/yaw. `random_rotation_2d` samples uniformly on SO(2), and `random_rotation_3d` samples uniformly on SO(3) by default. Random variants use the episode seed so benchmark and human-play behavior is reproducible.

Agents consume `AgentContext`, which contains observation, task instruction, action schema, visible metadata, and recent history. `MLLMPolicy` is currently a placeholder; future API-backed policies should use `PromptBuilder` and keep prompt/action schema versions in logs.

`ControlInstructionBuilder` is the shared source for MLLM control instructions and human-play UI panel lines. Human evaluators should see the same control mode, joystick count, visible mapping, scale summary, and frame-transform status that an MLLM would receive at episode start.

Control-interface mapping strategies only confuse motion dimensions (`dx`, `dy`, `dz`, `droll`, `dpitch`, `dyaw`). If a gripper exists, it is always the final virtual joystick and is never permuted or sign-flipped.

Episode logging writes:

- `config.yaml`
- `summary.json`
- `trajectory.jsonl`

The dummy simulator stores a small kinematic state and renders metadata dictionaries instead of image arrays. Real image and video persistence can be added later through `ObservationProvider` and `EpisodeLogger`.

## Gymnasium/MuJoCo Environments

Gymnasium is used as a dependency, not as a vendored or forked repository. Benchmark-specific environments live inside this package under `sim/gymnasium_envs`.

The Reacher delta-EEF environment owns its XML asset, kinematics helper, goal termination helper, and MuJoCo camera configuration helper. The benchmark interacts with it only through `GymnasiumSimAdapter`, `GymnasiumObservationProvider`, and a controller that translates joystick actions into a native Gymnasium box action.

The manual Reacher interface currently uses a separate two-process path: Pygame runs in the main process, while Gymnasium/MuJoCo runs in a worker process. This is an intentional stability choice after single-process Pygame/MuJoCo rendering caused GL context conflicts on X11.

The stable worker path still uses benchmark-level joystick parsing, validation, mapping, controller translation, metrics, and trajectory logging in the main process. The worker is a rendering and environment-isolation layer, not a separate simulator abstraction.
