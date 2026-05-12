from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.mapping.frame_transform import IdentityControlFrameTransform


def test_identity_transform_keeps_delta():
    config = load_config("configs/dummy_reach.yaml")
    transform = IdentityControlFrameTransform.from_config(config.frame_transform)
    assert transform.transform_delta_position((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    assert transform.transform_delta_rotation((0.1, 0.2, 0.3)) == (0.1, 0.2, 0.3)
