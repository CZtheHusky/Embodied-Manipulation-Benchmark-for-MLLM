from importlib import import_module

_LAZY_EXPORTS = {
    "PygameHumanPlayer": "mllm_teleop_benchmark.ui.human_player",
    "PlayerCommand": "mllm_teleop_benchmark.ui.runtime",
    "PlayerRuntimeResult": "mllm_teleop_benchmark.ui.runtime",
    "SimAdapterWorkerRuntime": "mllm_teleop_benchmark.ui.runtime",
    "build_player": "mllm_teleop_benchmark.ui.registry",
    "build_runtime": "mllm_teleop_benchmark.ui.registry",
}


def __getattr__(name: str):
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError:
        # Python's module attribute protocol requires AttributeError for unknown lazy exports.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
