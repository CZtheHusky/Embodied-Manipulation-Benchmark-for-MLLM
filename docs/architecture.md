# Architecture

The code is organized around replaceable interfaces rather than a single simulator or action mode.

`BenchmarkRunner` loads a typed config and runs episodes. `EpisodeManager` owns the step loop and connects a `Task`, `SimAdapter`, `ObservationProvider`, `Agent`, `ActionParser`, `ActionValidator`, `Controller`, `MetricsRecorder`, and `EpisodeLogger`.

The simulator boundary is `SimAdapter`. Real environments should implement `reset`, `step`, `render`, and `get_state` without changing agents, controllers, or logging.

`ObservationProvider.configure_sim` is an optional pre-reset hook for simulators that need episode-level camera configuration before environment creation or reset. The project does not change camera views inside an episode.

The action boundary is split into three layers:

- `ControlInterfaceConfig` describes the agent-visible virtual joystick interface.
- `ActionParser` converts raw policy output into `JoystickAction`.
- `ActionValidator` enforces the configured action schema.
- `Controller` translates valid joystick actions into simulator actions.

`action_space` and `mapping` remain as compatibility config sections, but new configs should prefer `control_interface`. When present, `control_interface` generates the joystick schema and fixed mapping used by both human play and MLLM play.

`JoystickMapping` and `ControlFrameTransform` are separate dependencies of the controller. Mapping is applied first, from virtual joystick channels to control-frame dimensions. Frame transform is applied second, from control-frame deltas to simulator/controller-frame deltas. This supports future joystick permutation/sign confusion, coordinate-frame rotation, and combinations of both without changing task logic.

Frame transforms are cross-environment control-layer components. `rotation_2d` rotates `dx/dy` around z; `rotation_3d` rotates `dx/dy/dz` and small delta-rotation vectors with roll/pitch/yaw. `random_rotation_2d` samples uniformly on SO(2), and `random_rotation_3d` samples uniformly on SO(3) by default. Random variants use the episode seed so benchmark and human-play behavior is reproducible.

Agents consume `AgentContext` and return `AgentDecision`, which carries a raw action, `AgentTrace`, and optional artifacts. `MLLMPolicy` projects that broad internal context into a strict public request containing only task text, public RGB, public controls/schema, and the model's normalized action history. `ModelClient` receives the projected request and never receives simulator state. The legacy text prompt helper exposes camera names rather than raw observation metadata.

`ControlInstructionBuilder` is the shared source for MLLM control instructions and human-play UI panel lines. Human evaluators should see the same control mode, joystick count, scale summary, visible fixed mapping, and complexity-presence flags that an MLLM would receive at episode start. Exact permutation, sign flips, and frame rotation parameters are not disclosed for hidden or complex control settings.

Control-interface mapping strategies only confuse motion dimensions (`dx`, `dy`, `dz`, `droll`, `dpitch`, `dyaw`). If a gripper exists, it is always the final virtual joystick and is never permuted or sign-flipped.

Episode logging writes:

- `config.yaml`
- `summary.json`
- `trajectory.jsonl`
- `model_calls/step_NNNN/{request,response,error}.json` for API agents
- `model_calls/images/step_NNNN/<camera>.png` for exact model inputs
- `review/model_view.mp4` and `review/manifest.json` when automated MLLM
  review video is enabled
- `review/evaluator_debug.mp4` only when evaluator debug video is explicitly
  enabled

Benchmark runs are grouped under `logging.output_dir/<run_id>/episode_XXXX/` so repeated runs do not overwrite earlier artifacts.

Review videos are derived after the episode finishes. `model_view.mp4` reads the
exact model-visible PNG artifacts and their hashes rather than re-rendering the
simulator. Its panel receives a public-only projection of task, control, action,
and API trace fields. The optional evaluator video uses a separate whitelist
projection for reward, distance, success, target/EEF position, and translated
simulator action and is permanently labeled `EVALUATOR ONLY`. Video generation
failure is recorded without changing the episode result.

The final post-action public observation is encoded with the same deterministic
RGB-to-PNG utility as model inputs and stored under `review/terminal/`. Raw PNG,
request/response, trajectory, and hashes remain authoritative; MP4 files are
review conveniences and do not participate in scoring.

The dummy simulator stores a small kinematic state and renders metadata dictionaries rather than public RGB arrays. It remains useful for non-visual pipeline tests; API-controlled visual episodes require an RGB-producing observation provider. Model input PNGs are persisted independently of `logging.save_images`; review videos are controlled by `logging.save_video`.

## Environment Integration Constraints

To ensure the benchmark remains stable and reproducible while allowing for necessary adaptations, environment integrations should follow these principles:

1. **Prefer Native APIs but Allow Forks**: Prioritize interacting with environments through their public APIs (e.g., `make`, `reset`, `step`, `render`). While zero-intrusion is the ideal for maintainability, controlled forks or internal modifications are permitted when required to expose necessary state, metrics, or randomization hooks that are not accessible via standard APIs.
2. **Version Pinning**: Simulation dependencies should be pinned in backend-specific uv lock files and matching optional dependency metadata when the backend can be installed as a normal Python dependency. They must not be added to the root core environment unless they are genuinely simulator-independent. CALVIN is an exception: it uses a pinned local editable checkout because upstream `calvin_env` needs runtime Hydra configs from its source tree.
3. **Decoupled Control Pipeline**: All control complexities—including virtual joystick mapping, sign flips, and coordinate-frame transforms—MUST be computed within the benchmark's controller layer BEFORE calling the simulator's step function. The simulator receives only the final, normalized action arrays.
4. **API-Driven Visual Randomization**: Modifications to visual parameters such as camera FOV, position, and rotation must be performed programmatically via the environment's runtime API (e.g., `CameraModder` in robosuite or direct MuJoCo model access) during the `reset` phase. Static asset files (like XML or URDF) should not be mutated to achieve randomization.

## Gymnasium/MuJoCo Environments

Gymnasium is used as a dependency, not as a vendored or forked repository. Benchmark-specific environments live inside this package under `sim/gymnasium_envs`.

The Reacher delta-EEF environment owns its XML asset, kinematics helper, goal termination helper, and MuJoCo camera configuration helper. The benchmark interacts with it only through `GymnasiumSimAdapter`, `GymnasiumObservationProvider`, and a controller that translates joystick actions into a native Gymnasium box action.

The manual Reacher interface currently uses a separate two-process path: Pygame runs in the main process, while Gymnasium/MuJoCo runs in a worker process. This is an intentional stability choice after single-process Pygame/MuJoCo rendering caused GL context conflicts on X11.

The stable worker path still uses benchmark-level joystick parsing, validation, mapping, controller translation, metrics, and trajectory logging in the main process. The worker is a rendering and environment-isolation layer, not a separate simulator abstraction.

## ManiSkill Environments

ManiSkill is an optional dependency, not a vendored or forked repository. The backend lives under `sim/maniskill` and is selected with `simulator.type: maniskill`.

The ManiSkill path uses `ManiSkillSimAdapter`, `ManiSkillObservationProvider`, `ManiSkillDeltaEEFController`, and `ManiSkillTask`. Control complexity remains in the benchmark layer before native ManiSkill Box actions are emitted. Camera randomization is episode-level through the camera manager and is applied before ManiSkill env creation/reset.

## robosuite Environments

robosuite is an optional backend selected with `simulator.type: robosuite` and installed through the `robosuite` uv environment.

The robosuite path uses `RobosuiteSimAdapter`, `RobosuiteOSCController`, and `RobosuiteLiftTask`. Control complexity remains in the benchmark layer before a native robosuite OSC action is emitted. Camera changes are sampled from the episode seed and applied during reset through robosuite's runtime camera utilities.

## DMControl Environments

DeepMind Control Suite is an optional backend selected with `simulator.type: dmcontrol` and installed through the `dmcontrol` uv environment.

The DMControl path uses `DMControlSimAdapter`, `DMControlObservationProvider`, `DMControlController`, and `DMControlTask`. Actions are validated and clipped against `env.action_spec()` before stepping. Explicit named fixed-camera pose/FOV edits are applied through MuJoCo model fields; the shared free-camera orbit randomization is not claimed as supported for DMControl.

## CALVIN Environments

CALVIN is an optional backend selected with `simulator.type: calvin` and installed through the `calvin` uv environment.

The CALVIN path uses `CalvinSimAdapter`, `CalvinObservationProvider`, `CalvinDeltaEEFController`, and `CalvinTask`. The controller maps benchmark joystick commands into CALVIN's 7-value delta end-effector action. `calvin_randomized` camera sampling is episode-level and patches CALVIN's PyBullet camera internals on a best-effort basis.
