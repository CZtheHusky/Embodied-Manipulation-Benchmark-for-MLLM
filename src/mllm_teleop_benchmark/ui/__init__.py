from importlib import import_module

_LAZY_EXPORTS = {
    "DirectCalvinPlayer": "mllm_teleop_benchmark.ui.direct_calvin_player",
    "DirectDMControlPlayer": "mllm_teleop_benchmark.ui.direct_dmcontrol_player",
    "DirectGymnasiumPlayer": "mllm_teleop_benchmark.ui.direct_gymnasium_player",
    "DirectManiSkillPlayer": "mllm_teleop_benchmark.ui.direct_maniskill_player",
    "PygameBenchmarkApp": "mllm_teleop_benchmark.ui.pygame_app",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
