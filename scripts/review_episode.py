#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mllm_teleop_benchmark.review.server import serve_review


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a read-only web UI for MLLM episode artifacts."
    )
    parser.add_argument(
        "path",
        help=(
            "A log root, experiment group, run, or episode directory. "
            "Nested reviewable runs are discovered recursively."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve_review(args.path, args.host, args.port)


if __name__ == "__main__":
    main()
