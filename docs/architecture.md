# Architecture

The code is organized around replaceable interfaces rather than a single simulator or action mode.

`BenchmarkRunner` loads a typed config and runs episodes. `EpisodeManager` owns the step loop and connects a `Task`, `SimAdapter`, `ObservationProvider`, `Agent`, `ActionParser`, `ActionValidator`, `Controller`, `MetricsRecorder`, and `EpisodeLogger`.

The simulator boundary is `SimAdapter`. Real environments should implement `reset`, `step`, `render`, and `get_state` without changing agents, controllers, or logging.

The action boundary is split into three layers:

- `ActionParser` converts raw policy output into `JoystickAction`.
- `ActionValidator` enforces the configured action schema.
- `Controller` translates valid joystick actions into simulator actions.

`JoystickMapping` and `ControlFrameTransform` are separate dependencies of the controller. This keeps future randomized mappings and hidden control frames out of task logic.

Agents consume `AgentContext`, which contains observation, task instruction, action schema, visible metadata, and recent history. `MLLMPolicy` is currently a placeholder; future API-backed policies should use `PromptBuilder` and keep prompt/action schema versions in logs.

Episode logging writes:

- `config.yaml`
- `summary.json`
- `trajectory.jsonl`

The dummy simulator stores a small kinematic state and renders metadata dictionaries instead of image arrays. Real image and video persistence can be added later through `ObservationProvider` and `EpisodeLogger`.
