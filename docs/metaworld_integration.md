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

## Running

```bash
python3 scripts/manage_uv_env.py run metaworld -- python scripts/run_benchmark.py --config configs/metaworld_reach.yaml
```

The runtime smoke in `tests/test_metaworld_integration.py` skips when `metaworld` is absent. When installed, it creates the MT1 `reach-v3` environment, calls reset, step, render, and close.

## Current Limitations

- Only MT1 `reach-v3` is configured.
- MT10, MT50, ML benchmarks, vector envs, and task-specific expert policies are out of scope.
- Camera selection is passed at environment creation through `camera_name`; per-camera runtime switching is not implemented.
- There is no Meta-World direct human-play worker yet. The supported closed loop is the benchmark runner path.
