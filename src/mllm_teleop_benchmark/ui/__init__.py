from importlib import import_module

_LAZY_EXPORTS = {
    "PygameHumanPlayer": "mllm_teleop_benchmark.ui.human_player",
    "PlayerCommand": "mllm_teleop_benchmark.ui.runtime",
    "PlayerRuntimeResult": "mllm_teleop_benchmark.ui.runtime",
    "SimAdapterWorkerRuntime": "mllm_teleop_benchmark.ui.runtime",
    "build_player": "mllm_teleop_benchmark.ui.registry",
    "build_runtime": "mllm_teleop_benchmark.ui.registry",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
