from __future__ import annotations

import json

from scripts.sanitize_model_artifacts import sanitize_artifacts


def test_sanitize_artifacts_redacts_reasoning_and_records_hashes(tmp_path):
    response = tmp_path / "run" / "episode_0000" / "model_calls" / "step_0000"
    response.mkdir(parents=True)
    artifact = response / "response.json"
    artifact.write_text(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "{\"channels\":{}}",
                            "reasoning_content": "PRIVATE_REASONING",
                        }
                    }
                ],
                "usage": {"reasoning_tokens": 12},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    dry_run = sanitize_artifacts(tmp_path, write=False)
    assert dry_run["changed_response_artifacts"] == 1
    assert "PRIVATE_REASONING" in artifact.read_text(encoding="utf-8")

    report = sanitize_artifacts(tmp_path, write=True)
    sanitized = artifact.read_text(encoding="utf-8")

    assert report["scanned_response_artifacts"] == 1
    assert report["changed_response_artifacts"] == 1
    assert report["files"][0]["sha256_before"] != report["files"][0]["sha256_after"]
    assert "PRIVATE_REASONING" not in sanitized
    assert "[REDACTED: hidden reasoning]" in sanitized
    assert '"reasoning_tokens": 12' in sanitized

    repeated = sanitize_artifacts(tmp_path, write=True)
    assert repeated["changed_response_artifacts"] == 0
