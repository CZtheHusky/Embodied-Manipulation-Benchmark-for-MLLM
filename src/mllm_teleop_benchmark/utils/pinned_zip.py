from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo


@dataclass(frozen=True)
class InstalledZip:
    sha256: str
    roots: tuple[str, ...]
    file_count: int


def install_zip_exact(archive: Path, target: Path) -> InstalledZip:
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        members = zip_file.infolist()
        roots = _validated_roots(members)
        file_count = sum(not member.is_dir() for member in members)
        with TemporaryDirectory(
            prefix=".pinned-asset-",
            dir=target.parent,
        ) as temporary:
            temporary_root = Path(temporary)
            payload = temporary_root / "payload"
            backup = temporary_root / "backup"
            payload.mkdir()
            backup.mkdir()
            zip_file.extractall(payload)
            _replace_roots(payload, target, backup, roots)
    return InstalledZip(
        sha256=_sha256(archive),
        roots=roots,
        file_count=file_count,
    )


def _validated_roots(members: list[ZipInfo]) -> tuple[str, ...]:
    roots: set[str] = set()
    if not members:
        raise ValueError("Pinned asset archive is empty")
    for member in members:
        path = PurePosixPath(member.filename)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(
                f"Archive member escapes extraction directory: {member.filename}"
            )
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(
                f"Archive member is a symbolic link: {member.filename}"
            )
        roots.add(path.parts[0])
    return tuple(sorted(roots))


def _replace_roots(
    payload: Path,
    target: Path,
    backup: Path,
    roots: tuple[str, ...],
) -> None:
    moved: list[tuple[Path, Path, bool]] = []
    try:
        for root in roots:
            source = payload / root
            destination = target / root
            previous = backup / root
            if not source.exists():
                raise ValueError(
                    f"Archive root was not extracted: {root}"
                )
            had_previous = destination.exists() or destination.is_symlink()
            if had_previous:
                previous.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(previous)
            moved.append((destination, previous, had_previous))
            source.replace(destination)
    except Exception:
        for destination, previous, had_previous in reversed(moved):
            _remove_path(destination)
            if had_previous and (
                previous.exists() or previous.is_symlink()
            ):
                previous.replace(destination)
        raise


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
