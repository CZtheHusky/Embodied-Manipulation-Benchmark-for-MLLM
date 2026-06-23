# robosuite Integration

robosuite is integrated as an optional backend. It is not part of the root core environment.

## Components

- Adapter: `sim/robosuite_adapter.py`
- Controller: `control/robosuite_controller.py`
- Task wrapper: `tasks/robosuite_task.py`
- Example config: `configs/robosuite_lift.yaml`
- Environment: `envs/environment-robosuite.yml`

## Dependency

Use the backend-specific conda environment:

```bash
conda env create -f envs/environment-robosuite.yml
conda activate mllm-teleop-robosuite
```

The backend pins `robosuite==1.4.1`. Core tests and non-robosuite environments must continue to work without robosuite installed. The adapter sets `NUMBA_DISABLE_JIT=1` by default before importing robosuite because this pinned install can otherwise fail during numba cache setup.

To require the optional backend smoke test instead of skipping it:

```bash
MLLM_TELEOP_REQUIRE_ROBOSUITE=1 pytest tests/test_robosuite_integration.py -rs
```

## Control

The initial config targets robosuite `Lift` with Panda and OSC pose control.

The benchmark control path remains:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> RobosuiteOSCController -> env.step()
```

The robosuite controller emits a `RobosuiteAction` wrapper. The adapter converts it to the native numpy action only at the simulator boundary.

## Cameras

Camera settings are sampled from the episode seed and applied during reset through robosuite runtime APIs. The project does not mutate static assets to randomize camera views. `camera_randomization` supports fixed `pos`, `quat`, and `fovy` values, plus `pos_range` and `fovy_range` for seeded episode-level sampling.

## Current Limitations

- Initial support is focused on Lift.
- The adapter is intentionally lightweight and should be hardened as more robosuite tasks are added.
- Human play for robosuite is not yet implemented.
