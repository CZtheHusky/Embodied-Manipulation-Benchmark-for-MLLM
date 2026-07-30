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
    files: tuple[str, ...]


def install_zip_exact(
    archive: Path,
    target: Path,
    *,
    previous_files: tuple[str, ...] = (),
) -> InstalledZip:
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        members = zip_file.infolist()
        roots, files = _validated_members(members)
        previous = _validated_relative_paths(previous_files)
        with TemporaryDirectory(
            prefix=".pinned-asset-",
            dir=target.parent,
        ) as temporary:
            temporary_root = Path(temporary)
            payload = temporary_root / "payload"
            backup = temporary_root / "backup"
            payload.mkdir()
            backup.mkdir()
            _extract_files(zip_file, members, payload)
            _replace_files(
                payload,
                target,
                backup,
                current_files=files,
                previous_files=previous,
            )
    return InstalledZip(
        sha256=_sha256(archive),
        roots=roots,
        file_count=len(files),
        files=files,
    )


def _validated_members(
    members: list[ZipInfo],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    roots: set[str] = set()
    directories: set[str] = set()
    files: set[str] = set()
    if not members:
        raise ValueError("Pinned asset archive is empty")
    for member in members:
        path = _validated_relative_path(member.filename)
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(
                f"Archive member is a symbolic link: {member.filename}"
            )
        roots.add(path.parts[0])
        normalized = path.as_posix()
        collection = directories if member.is_dir() else files
        if normalized in collection:
            raise ValueError(
                f"Archive contains a duplicate member: {normalized}"
            )
        collection.add(normalized)
    if directories & files:
        conflict = min(directories & files)
        raise ValueError(
            f"Archive member is both a file and directory: {conflict}"
        )
    for filename in files:
        path = PurePosixPath(filename)
        if any(
            PurePosixPath(*path.parts[:index]).as_posix() in files
            for index in range(1, len(path.parts))
        ):
            raise ValueError(
                f"Archive file conflicts with a parent path: {filename}"
            )
    return tuple(sorted(roots)), tuple(sorted(files))


def _extract_files(
    zip_file: ZipFile,
    members: list[ZipInfo],
    payload: Path,
) -> None:
    for member in members:
        if member.is_dir():
            continue
        relative = _validated_relative_path(member.filename)
        destination = payload.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            zip_file.open(member) as source,
            destination.open("xb") as output,
        ):
            shutil.copyfileobj(source, output)


def _validated_relative_paths(
    values: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted({_validated_relative_path(value).as_posix() for value in values})
    )


def _validated_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(
            f"Archive member escapes extraction directory: {value}"
        )
    return path


def _replace_files(
    payload: Path,
    target: Path,
    backup: Path,
    *,
    current_files: tuple[str, ...],
    previous_files: tuple[str, ...],
) -> None:
    moved: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for relative in sorted(set(previous_files) | set(current_files)):
            destination = _safe_destination(target, relative)
            previous = backup / relative
            if destination.is_symlink() or destination.is_file():
                previous.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(previous)
                moved.append((destination, previous))

        _remove_empty_parents(target, previous_files)
        for relative in current_files:
            source = payload / relative
            destination = _safe_destination(target, relative)
            if not source.is_file():
                raise ValueError(
                    f"Archive file was not extracted: {relative}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_dir():
                try:
                    destination.rmdir()
                except OSError as error:
                    raise ValueError(
                        "Asset file conflicts with a non-empty directory: "
                        f"{relative}"
                    ) from error
            source.replace(destination)
            installed.append(destination)
    except Exception:
        for destination in reversed(installed):
            _remove_path(destination)
        _remove_empty_parents(target, current_files)
        for destination, previous in reversed(moved):
            if previous.exists() or previous.is_symlink():
                if destination.is_dir():
                    destination.rmdir()
                destination.parent.mkdir(parents=True, exist_ok=True)
                previous.replace(destination)
        raise


def _safe_destination(target: Path, relative: str) -> Path:
    if target.is_symlink():
        raise ValueError(
            f"Asset installation root is a symbolic link: {target}"
        )
    path = _validated_relative_path(relative)
    destination = target.joinpath(*path.parts)
    current = target
    for part in path.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ValueError(
                "Asset destination contains a symbolic-link parent: "
                f"{relative}"
            )
    resolved_parent = destination.parent.resolve()
    if not resolved_parent.is_relative_to(target.resolve()):
        raise ValueError(
            f"Asset destination escapes installation root: {relative}"
        )
    return destination


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _remove_empty_parents(
    target: Path,
    previous_files: tuple[str, ...],
) -> None:
    target_resolved = target.resolve()
    parents = {
        (target / relative).parent
        for relative in previous_files
    }
    for parent in sorted(
        parents,
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        current = parent
        while current != target:
            if not current.resolve().is_relative_to(target_resolved):
                break
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
