# Meta-World Integration

Meta-World is integrated as an optional backend through its Gymnasium API. The initial scope is deliberately small: one MT1 environment, `reach-v3`, with Box actions and RGB rendering.

Reference API shape:

```python
import gymnasium as gym
import metaworld

env = gym.make("Meta-World/MT1", env_name="reach-v3", seed=seed)
```

## Components

- Adapter: `sim/metaworld/adapter.py`
- Observation provider: `sim/metaworld/observation.py`
- Controller: `control/metaworld.py`
- Task wrapper: `tasks/metaworld_task.py`
- Example config: `configs/metaworld_reach.yaml`
- MLLM baseline: `configs/metaworld_reach_mllm_reflection.yaml`
- MLLM perturbed: `configs/metaworld_reach_mllm_reflection_perturbed.yaml`

## Dependency

Meta-World is optional and is not installed in the core environment. Use the dedicated uv backend:

```bash
python3 scripts/manage_uv_env.py create metaworld
python3 scripts/manage_uv_env.py smoke metaworld
```

Core tests and non-Meta-World configs continue to build without `metaworld` installed because simulator imports stay lazy.

## Control

The first config uses Meta-World's normalized Box action:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> MetaworldBoxActionController -> env.step()
```

`metaworld_box` maps `dx`, `dy`, and `dz` to the first three native action slots. The fourth Meta-World action slot is gripper control. For `reach-v3`, the config leaves benchmark gripper control disabled and sends `default_gripper_value: 1.0`.

The MLLM configs expose the fourth native slot as the final direct gripper
joystick. Virtual command `0` means open and `1` means closed; the controller
maps those public endpoints to native values `-1` and `1`. The three
translation joysticks remain discrete `-1`, `0`, or `1` commands. The public
frame description identifies the Meta-World end-effector controller axes
without exposing episode state or a sampled transform.

## MLLM Evaluation

Both MLLM configurations use the shared `MLLMPolicy` and `SimAdapter`; there is
no Meta-World-specific agent. They use OpenAI-compatible Chat Completions,
`action_with_reflection`, `reasoning_effort: low`, three historical RGB frames,
and all prior normalized action text. Review video is disabled by default and
can be enabled by the real evaluation command.

The baseline uses `fixed + identity`. The perturbed configuration uses
`permuted_sign_flipped + random_rotation_3d`. The perturbation only changes the
three translation controls; the gripper remains the final unflipped channel.
The model is told that mapping and frame complexity are present and must be
inferred from visual observations and its own action history, but it is not
given an example, sampled value, permutation, signs, rotation, seed, simulator
action, reward, distance, target coordinates, state, or end-effector pose.

Local four-step integration checks replace the network client with a fake
client while still running the real Meta-World environment:

```bash
python3 scripts/manage_uv_env.py run metaworld -- \
  python -m pytest -q tests/test_metaworld_integration.py -rs
```

For a real API run, provide API settings through the ignored local environment
file or process environment and explicitly enable video:

```bash
python3 scripts/manage_uv_env.py run metaworld -- \
  python scripts/run_benchmark.py \
  --config configs/metaworld_reach_mllm_reflection.yaml \
  --set logging.save_video=true
```

Use `configs/metaworld_reach_mllm_reflection_perturbed.yaml` for the hidden
mapping and random 3D control-frame condition.

## Running

```bash
python3 scripts/manage_uv_env.py run metaworld -- python scripts/run_benchmark.py --config configs/metaworld_reach.yaml
```

Manual play uses the shared Pygame worker runtime:

```bash
python3 scripts/manage_uv_env.py run metaworld -- python scripts/play_benchmark.py --config configs/metaworld_reach.yaml
```

The runtime smoke in `tests/test_metaworld_integration.py` skips when `metaworld` is absent. When installed, it creates the MT1 `reach-v3` environment, calls reset, step, render, and close.

## Current Limitations

- Only MT1 `reach-v3` is configured.
- MT10, MT50, ML benchmarks, vector envs, and task-specific expert policies are out of scope.
- Camera selection is passed at environment creation through `camera_name`; per-camera runtime switching is not implemented.
