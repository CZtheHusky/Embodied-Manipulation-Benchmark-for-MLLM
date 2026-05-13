#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os


def _configure_render_backend(backend: str | None) -> None:
    # Must run before importing packages that may import MuJoCo or PyOpenGL.
    if backend == "benchmark":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a benchmark environment with keyboard controls.")
    parser.add_argument("--config", default="configs/reacher_delta_eef.yaml")
    parser.add_argument("--backend", choices=["benchmark", "direct_gymnasium"], default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set control_interface.mapping_strategy=permuted")
    args = parser.parse_args()

    backend = args.backend or os.environ.get("MLLM_TELEOP_PLAYER_BACKEND")
    _configure_render_backend(backend)

    from mllm_teleop_benchmark.config.loader import load_config
    from mllm_teleop_benchmark.ui import DirectGymnasiumPlayer, PygameBenchmarkApp

    config = load_config(args.config, tuple(args.overrides))
    ui_config = config.raw.get("ui", {}) if isinstance(config.raw.get("ui", {}), dict) else {}
    backend = backend or str(ui_config.get("backend", "direct_gymnasium"))
    if backend == "direct_gymnasium":
        result = DirectGymnasiumPlayer(config).run()
    else:
        result = PygameBenchmarkApp(config).run()
    print(result)


if __name__ == "__main__":
    main()
