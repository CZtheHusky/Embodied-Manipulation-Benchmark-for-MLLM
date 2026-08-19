# Camera Resolution Policy

The benchmark configures simulator-native RGB resolution before an observation
is exposed to a human or model. This is separate from `image_max_side`, which
only downscales an already-rendered public image while preparing an API request
and never enlarges it.

## Policy

- A public external camera uses a longest edge of 512 pixels.
- A source aspect ratio is preserved. For example, `320x240` becomes
  `512x384`, and `640x512` becomes `512x410` after integer rounding.
- A camera without an upstream size or aspect ratio uses `512x512`.
- Auxiliary hand, wrist, gripper, or eye-in-hand cameras retain their upstream
  default size.
- If the only RGB camera is hand-mounted, it is treated as the main public
  camera and its longest edge is set to 512.
- Resizing is applied through each simulator's native camera or renderer
  configuration. Post-render image resizing is not used to satisfy this policy.

## Audited Defaults And Benchmark Targets

| Backend | Audited upstream RGB default | Benchmark public RGB |
| --- | --- | --- |
| Gymnasium Reacher/Pusher | no RGB observation; MuJoCo render defaults to `480x480` | `front 512x512` |
| Meta-World | no RGB observation; render defaults to `480x480` | `topview 512x512`; `behindGripper 480x480` |
| ManiSkill PickCube | `base_camera 128x128`; `render_camera 512x512` | `render_camera 512x512` |
| robosuite Lift | `agentview 256x256` by default | `agentview 512x512`; `robot0_eye_in_hand 256x256` |
| RoboCasa / RoboCasa365 Gym wrapper | left, right, and eye-in-hand cameras are `256x256` | left/right `512x512`; eye-in-hand `256x256` |
| SimplerEnv Google Robot | `overhead_camera 640x512` | `overhead_camera 512x410` |
| dm_control manipulator | `physics.render()` defaults to `320x240` | `fixed 512x384`; `hand 320x240` |
| CALVIN | `static 200x200`; `gripper 84x84` | `static 512x512`; `gripper 84x84` |
| RLBench | five RGB cameras default to `128x128` | `front 512x512` |
| Colosseum | inherited RLBench camera default is `128x128` | `front 512x512` |
| LIBERO | `agentview` and `robot0_eye_in_hand` default to `128x128` | `agentview 512x512`; eye-in-hand `128x128` |
| Dummy | no upstream camera default | `front 512x512` |

The audit targets the pinned or locked sources used by this repository:
Gymnasium 1.3.0, Meta-World 3.0.0, ManiSkill 3.0.1, robosuite 1.4.1,
dm_control 1.0.43, RoboCasa `8f3c96e`, SimplerEnv `06accac`,
ManiSkill2_real2sim `ef7a4d4`, CALVIN `797142c`, RLBench `f214b6e`,
Colosseum `098a910` with RLBench `7c3f425`, and LIBERO `8f1084e`.

## Backend Application Paths

- Gymnasium passes `width` and `height` into `MujocoEnv` construction.
- Meta-World uses cached native `mujoco.Renderer` instances per requested size.
- robosuite and LIBERO pass per-camera width and height lists to their native
  environment constructors.
- RoboCasa keeps the Gym wrapper sensors at `256x256` and uses the underlying
  MuJoCo renderer for external `512x512` views; the eye-in-hand image remains
  the wrapper's native `256x256` observation.
- SimplerEnv merges `width=512` and `height=410` into its native flat
  `camera_cfgs` before environment construction.
- CALVIN updates the native camera dimensions and recomputes the projection
  matrix before reset.
- RLBench and Colosseum set `CameraConfig.image_size` before CoppeliaSim launch.
- dm_control renders each configured camera directly at its requested size.

`tests/test_camera_resolution_policy.py` guards the complete config matrix.
Backend integration tests additionally assert the actual RGB array shapes.
