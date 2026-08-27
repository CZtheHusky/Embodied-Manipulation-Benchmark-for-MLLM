#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CONFIGS = (
    "configs/pusher_delta_eef_mllm_chat_completions.yaml",
    "configs/pusher_delta_eef_mllm_chat_completions_perturbed.yaml",
    "configs/pusher_delta_eef_mllm_chat_completions_default_view.yaml",
    "configs/pusher_delta_eef_mllm_chat_completions_default_view_perturbed.yaml",
    "configs/pusher_delta_eef_mllm_responses.yaml",
    "configs/pusher_delta_eef_mllm_responses_perturbed.yaml",
    "configs/pusher_delta_eef_mllm_responses_default_view.yaml",
    "configs/pusher_delta_eef_mllm_responses_default_view_perturbed.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the eight-condition Pusher MLLM evaluation matrix."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of condition runs to execute concurrently (maximum: 4).",
    )
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 4:
        parser.error("--concurrency must be between 1 and 4")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (
        repo_root
        / "outputs"
        / "pusher_delta_eef_mllm"
        / "launcher"
        / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now().astimezone().isoformat()
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(_run_condition, repo_root, output_dir, config): config
            for config in CONFIGS
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "passed" if result["return_code"] == 0 else "failed"
            print(f"[{status}] {result['config']} -> {result['log']}", flush=True)

    results.sort(key=lambda item: CONFIGS.index(str(item["config"])))
    summary = {
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "concurrency": args.concurrency,
        "model": os.environ.get("MLLM_TELEOP_MODEL"),
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # A zero process status would otherwise advertise a failed matrix as completed.
    if any(result["return_code"] != 0 for result in results):
        raise SystemExit(f"One or more condition runs failed; see {summary_path}")
    print(f"All condition runs completed: {summary_path}", flush=True)


def _run_condition(
    repo_root: Path,
    output_dir: Path,
    config: str,
) -> dict[str, object]:
    config_path = repo_root / config
    log_path = output_dir / f"{config_path.stem}.log"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_benchmark.py"),
        "--config",
        str(config_path),
    ]
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "config": config,
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "log": str(log_path.relative_to(repo_root)),
    }


if __name__ == "__main__":
    main()
