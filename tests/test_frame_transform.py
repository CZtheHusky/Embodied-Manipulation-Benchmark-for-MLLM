from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import FrameTransformConfig
from mllm_teleop_benchmark.mapping.frame_transform import IdentityControlFrameTransform, RotationControlFrameTransform


def test_identity_transform_keeps_delta():
    config = load_config("configs/dummy_reach.yaml")
    transform = IdentityControlFrameTransform.from_config(config.frame_transform)
    assert transform.transform_delta_position((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    assert transform.transform_delta_rotation((0.1, 0.2, 0.3)) == (0.1, 0.2, 0.3)


def test_rotation_2d_rotates_xy_about_z():
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_2d", visible_to_agent=True, params={"angle_degrees": 90.0})
    )
    x, y, z = transform.transform_delta_position((1.0, 0.0, 2.0))
    assert abs(x) < 1e-9
    assert abs(y - 1.0) < 1e-9
    assert z == 2.0


def test_rotation_3d_rotates_xyz():
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0.0, 0.0, 90.0]})
    )
    x, y, z = transform.transform_delta_position((1.0, 0.0, 0.0))
    assert abs(x) < 1e-9
    assert abs(y - 1.0) < 1e-9
    assert abs(z) < 1e-9


def test_random_rotation_2d_is_seed_deterministic():
    config = FrameTransformConfig(type="random_rotation_2d", visible_to_agent=True, params={"angle_range_degrees": [10.0, 20.0]})
    first = RotationControlFrameTransform.from_config(config, seed=5).metadata()
    second = RotationControlFrameTransform.from_config(config, seed=5).metadata()
    assert first == second
    assert first["params"]["sampling"] == "uniform_so2"


def test_random_rotation_3d_defaults_to_uniform_so3():
    config = FrameTransformConfig(type="random_rotation_3d", visible_to_agent=True, params={})
    transform = RotationControlFrameTransform.from_config(config, seed=5)
    metadata = transform.metadata()
    assert metadata["params"]["sampling"] == "uniform_so3"
    quat = metadata["params"]["quaternion_xyzw"]
    assert abs(sum(value * value for value in quat) - 1.0) < 1e-9
    x, y, z = transform.transform_delta_position((1.0, 0.0, 0.0))
    assert abs((x * x + y * y + z * z) - 1.0) < 1e-9


def test_random_rotation_3d_can_use_euler_ranges_explicitly():
    config = FrameTransformConfig(type="random_rotation_3d", visible_to_agent=True, params={"sampling": "euler_ranges", "yaw_range_degrees": [10.0, 20.0]})
    metadata = RotationControlFrameTransform.from_config(config, seed=5).metadata()
    assert metadata["params"]["sampling"] == "euler_ranges"
    assert "rpy_radians" in metadata["params"]
