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
    "configs/reacher_delta_eef_mllm_chat_completions.yaml",
    "configs/reacher_delta_eef_mllm_chat_completions_perturbed.yaml",
    "configs/reacher_delta_eef_mllm_chat_completions_default_view.yaml",
    "configs/reacher_delta_eef_mllm_chat_completions_default_view_perturbed.yaml",
    "configs/reacher_delta_eef_mllm_responses.yaml",
    "configs/reacher_delta_eef_mllm_responses_perturbed.yaml",
    "configs/reacher_delta_eef_mllm_responses_default_view.yaml",
    "configs/reacher_delta_eef_mllm_responses_default_view_perturbed.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 16-condition Pusher and Reacher MLLM matrix."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of condition runs to execute concurrently (maximum: 4).",
    )
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        metavar="CONFIG[@EPISODE_ID]",
        help=(
            "Run only this matrix config, optionally selecting one episode while "
            "preserving its derived seed. Repeat to define a recovery batch."
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 4:
        parser.error("--concurrency must be between 1 and 4")

    missing_env = [
        name
        for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "MLLM_TELEOP_MODEL")
        if not os.environ.get(name)
    ]
    if missing_env:
        parser.error(
            "missing required model API environment variables: "
            + ", ".join(missing_env)
        )

    jobs = _parse_jobs(args.job, parser)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (
        repo_root
        / "outputs"
        / "gymnasium_mllm_matrix"
        / "launcher"
        / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now().astimezone().isoformat()
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _run_condition,
                repo_root,
                output_dir,
                job_index,
                config,
                episode_id,
            ): job_index
            for job_index, (config, episode_id) in enumerate(jobs)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _write_summary(
                output_dir,
                started_at,
                args.concurrency,
                jobs,
                results,
            )
            status = "passed" if result["return_code"] == 0 else "failed"
            print(f"[{status}] {result['config']} -> {result['log']}", flush=True)

    summary_path = _write_summary(
        output_dir,
        started_at,
        args.concurrency,
        jobs,
        results,
        finished=True,
    )
    if any(result["return_code"] != 0 for result in results):
        raise SystemExit(f"One or more condition runs failed; see {summary_path}")
    print(f"All condition runs completed: {summary_path}", flush=True)


def _run_condition(
    repo_root: Path,
    output_dir: Path,
    job_index: int,
    config: str,
    episode_id: int | None,
) -> dict[str, object]:
    config_path = repo_root / config
    suffix = f"__episode-{episode_id:04d}" if episode_id is not None else ""
    log_path = output_dir / f"{config_path.stem}{suffix}.log"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_benchmark.py"),
        "--config",
        str(config_path),
    ]
    if episode_id is not None:
        command.extend(("--episode-id", str(episode_id)))
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
        "job_index": job_index,
        "config": config,
        "episode_ids": [episode_id] if episode_id is not None else None,
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "log": str(log_path.relative_to(repo_root)),
    }


def _write_summary(
    output_dir: Path,
    started_at: str,
    concurrency: int,
    jobs: tuple[tuple[str, int | None], ...],
    results: list[dict[str, object]],
    *,
    finished: bool = False,
) -> Path:
    ordered = sorted(results, key=lambda item: int(item["job_index"]))
    summary = {
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat() if finished else None,
        "concurrency": concurrency,
        "model": os.environ.get("MLLM_TELEOP_MODEL"),
        "matrix_condition_count": len(CONFIGS),
        "expected_job_count": len(jobs),
        "completed_condition_count": len(ordered),
        "results": ordered,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _parse_jobs(
    specs: list[str],
    parser: argparse.ArgumentParser,
) -> tuple[tuple[str, int | None], ...]:
    if not specs:
        return tuple((config, None) for config in CONFIGS)

    jobs: list[tuple[str, int | None]] = []
    for spec in specs:
        config, separator, raw_episode_id = spec.partition("@")
        if config not in CONFIGS:
            parser.error(f"--job config is not part of the matrix: {config}")
        episode_id = None
        if separator:
            try:
                episode_id = int(raw_episode_id)
            except ValueError:
                parser.error(f"--job episode id must be an integer: {spec}")
            if episode_id < 0:
                parser.error(f"--job episode id must be non-negative: {spec}")
        job = (config, episode_id)
        if job in jobs:
            parser.error(f"duplicate --job: {spec}")
        jobs.append(job)
    return tuple(jobs)


if __name__ == "__main__":
    main()
