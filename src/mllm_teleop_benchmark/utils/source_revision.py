from __future__ import annotations

from pathlib import Path
import subprocess


def capture_source_revision(config_path: str | None) -> dict[str, object]:
    package_root = Path(__file__).resolve().parents[3]
    git_root = _git_output(("rev-parse", "--show-toplevel"), package_root)
    if git_root is None:
        return {
            "status": "unavailable",
            "git_commit": None,
            "git_dirty": None,
            "config_path": config_path,
        }

    root = Path(git_root).resolve()
    commit = _git_output(("rev-parse", "HEAD"), root)
    status = _git_output(
        ("status", "--porcelain", "--untracked-files=no"),
        root,
        strip=False,
    )
    return {
        "status": "reported" if commit is not None and status is not None else "unavailable",
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "config_path": _display_path(config_path, root),
    }


def _git_output(
    args: tuple[str, ...],
    cwd: Path,
    *,
    strip: bool = True,
) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() if strip else result.stdout


def _display_path(config_path: str | None, root: Path) -> str | None:
    if config_path is None:
        return None
    path = Path(config_path).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
