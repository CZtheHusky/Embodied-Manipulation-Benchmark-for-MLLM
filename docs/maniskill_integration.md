# ManiSkill Integration

ManiSkill is integrated as an optional backend. The benchmark does not fork or modify ManiSkill source code.

## Components

- Adapter: `sim/maniskill/adapter.py`
- Observation provider: `sim/maniskill/observation.py`
- Controller: `control/maniskill.py`
- Task wrapper: `tasks/maniskill.py`
- Example config: `configs/maniskill_pickcube_delta_eef.yaml`

## Dependency

ManiSkill is optional because it is a heavier simulator dependency:

```bash
conda env create -f envs/environment-maniskill.yml
conda activate mllm-teleop-maniskill
```

Core tests and non-ManiSkill environments continue to work when `mani_skill` is not installed.

## Control

The initial config targets `PickCube-v1` with ManiSkill `control_mode: pd_ee_delta_pose`.

The benchmark control path remains simulator-independent:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> ManiSkillDeltaEEFController -> env.step()
```

Joystick permutation, sign changes, and coordinate-frame rotation are handled by this repository before native ManiSkill actions are produced. ManiSkill receives a clipped native Box action.

The gripper, when present, remains the final virtual joystick. The default mapping is `0.0 -> -1.0` and `1.0 -> 1.0`, configurable through controller params.

## Observation And Cameras

ManiSkill observations are read from `obs_mode: rgb` sensor data. The observation provider extracts per-camera RGB frames and falls back to `env.render()` only if no image is available in the native observation.

Camera randomization is episode-level only. `maniskill_randomized` samples camera position during `CameraManager.reset(seed)` and passes the sampled sensor config before ManiSkill env creation/reset. The project does not support changing camera views inside an episode.

## Current Limitations

- Initial support is single-env (`num_envs: 1`) and Box-action focused.
- ManiSkill is not installed in the core conda environment.
- Runtime camera mutation is intentionally out of scope.
- Task success defaults to the `success` key in ManiSkill info and can be changed with `task.params.success_key`.
