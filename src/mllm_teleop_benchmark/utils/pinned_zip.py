from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from tempfile import mkstemp, TemporaryDirectory
from zipfile import ZipFile, ZipInfo


@dataclass(frozen=True)
class InstalledZip:
    sha256: str
    roots: tuple[str, ...]
    file_count: int
    files: tuple[str, ...]


@dataclass(frozen=True)
class ZipInstallRequest:
    archive: Path
    target: Path
    previous_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PreparedZip:
    request: ZipInstallRequest
    payload: Path
    result: InstalledZip


def install_zip_exact(
    archive: Path,
    target: Path,
    *,
    previous_files: tuple[str, ...] = (),
) -> InstalledZip:
    return install_zip_batch_exact(
        (
            ZipInstallRequest(
                archive=archive,
                target=target,
                previous_files=previous_files,
            ),
        )
    )[0]


def install_zip_batch_exact(
    requests: tuple[ZipInstallRequest, ...],
) -> tuple[InstalledZip, ...]:
    normalized: list[ZipInstallRequest] = []
    device: int | None = None
    for request in requests:
        target = request.target.absolute()
        # Replacing through a symlink could install assets outside the requested root.
        if target.is_symlink():
            raise ValueError(
                f"Asset installation root is a symbolic link: {target}"
            )
        target.mkdir(parents=True, exist_ok=True)
        current_device = target.stat().st_dev
        if device is None:
            device = current_device
        # Batch replace and rollback rely on same-filesystem atomic renames.
        elif current_device != device:
            raise ValueError(
                "Pinned asset batch targets must share one filesystem"
            )
        normalized.append(
            ZipInstallRequest(
                archive=request.archive.absolute(),
                target=target,
                previous_files=tuple(
                    sorted(
                        {
                            _validated_relative_path(value).as_posix()
                            for value in request.previous_files
                        }
                    )
                ),
            )
        )

    temporary_parent = normalized[0].target.parent
    with TemporaryDirectory(
        prefix=".pinned-asset-batch-",
        dir=temporary_parent,
    ) as temporary:
        temporary_root = Path(temporary)
        prepared = tuple(
            _prepare_zip(request, temporary_root / f"payload-{index}")
            for index, request in enumerate(normalized)
        )
        _replace_batch(prepared, temporary_root / "backup")
        return tuple(item.result for item in prepared)


def write_json_atomic(
    path: Path,
    value: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Marker replacement must not follow a caller-controlled symlink.
    if path.is_symlink():
        raise ValueError(f"Marker path is a symbolic link: {path}")
    descriptor, temporary_name = mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Repeat after writing the temporary file to close the replacement race window.
        if path.is_symlink():
            raise ValueError(f"Marker path is a symbolic link: {path}")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _prepare_zip(
    request: ZipInstallRequest,
    payload: Path,
) -> _PreparedZip:
    payload.mkdir()
    with ZipFile(request.archive) as zip_file:
        members = zip_file.infolist()
        roots, files = _validated_members(members)
        _extract_files(zip_file, members, payload)
    return _PreparedZip(
        request=request,
        payload=payload,
        result=InstalledZip(
            sha256=_sha256(request.archive),
            roots=roots,
            file_count=len(files),
            files=files,
        ),
    )


def _validated_members(
    members: list[ZipInfo],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    roots: set[str] = set()
    directories: set[str] = set()
    files: set[str] = set()
    # Installing an empty archive would otherwise remove previously owned files.
    if not members:
        raise ValueError("Pinned asset archive is empty")
    for member in members:
        path = _validated_relative_path(member.filename)
        # Zip symlinks can redirect later extraction outside the payload root.
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(
                f"Archive member is a symbolic link: {member.filename}"
            )
        roots.add(path.parts[0])
        normalized = path.as_posix()
        collection = directories if member.is_dir() else files
        collection.add(normalized)
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

def _validated_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    # These components would make payload extraction escape its temporary root.
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(
            f"Archive member escapes extraction directory: {value}"
        )
    return path


def _replace_batch(
    prepared: tuple[_PreparedZip, ...],
    backup: Path,
) -> None:
    backup.mkdir()
    current: dict[Path, tuple[Path, Path]] = {}
    previous: dict[Path, Path] = {}
    for item in prepared:
        target = item.request.target
        for relative in item.result.files:
            destination = _safe_destination(target, relative)
            # Silent last-writer-wins would corrupt archive ownership and rollback state.
            if destination in current:
                raise ValueError(
                    "Pinned archives own the same destination: "
                    f"{destination}"
                )
            current[destination] = (
                item.payload.joinpath(*PurePosixPath(relative).parts),
                target,
            )
        for relative in item.request.previous_files:
            destination = _safe_destination(target, relative)
            previous.setdefault(destination, target)

    current_paths = set(current)
    affected = set(previous) | current_paths
    for destination, (_source, target) in current.items():
        parent = destination.parent
        while parent != target:
            # Following an unowned symlink/file parent could write outside asset ownership.
            if (
                (parent.is_symlink() or parent.is_file())
                and parent not in affected
            ):
                raise ValueError(
                    "Asset file has an unowned non-directory parent: "
                    f"{destination}"
                )
            parent = parent.parent

    moved: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    created_directories: list[Path] = []
    removed_directories: list[Path] = []
    try:
        for index, destination in enumerate(
            sorted(affected, key=lambda path: (len(path.parts), str(path)))
        ):
            if destination.is_symlink() or destination.is_file():
                previous_path = backup / str(index)
                destination.replace(previous_path)
                moved.append((destination, previous_path))

        for destination, (source, target) in sorted(
            current.items(),
            key=lambda item: (len(item[0].parts), str(item[0])),
        ):
            # _prepare_zip extracts every path listed in InstalledZip before replacement.
            assert source.is_file()
            _create_parent_directories(
                target,
                destination.parent,
                created_directories,
            )
            if destination.is_dir():
                try:
                    destination.rmdir()
                except OSError as error:
                    raise ValueError(
                        "Asset file conflicts with a non-empty directory: "
                        f"{destination}"
                    ) from error
                removed_directories.append(destination)
            source.replace(destination)
            installed.append(destination)

        current_set = set(current)
        for destination, target in previous.items():
            if destination not in current_set and any(
                moved_destination == destination
                for moved_destination, _backup in moved
            ):
                _prune_empty_parents(
                    destination.parent,
                    target,
                    removed_directories,
                )
    except Exception:
        _rollback_batch(
            installed=installed,
            moved=moved,
            created_directories=created_directories,
            removed_directories=removed_directories,
        )
        raise


def _safe_destination(target: Path, relative: str) -> Path:
    # Recheck at each destination to catch root replacement after batch preparation.
    if target.is_symlink():
        raise ValueError(
            f"Asset installation root is a symbolic link: {target}"
        )
    path = _validated_relative_path(relative)
    destination = target.joinpath(*path.parts)
    current = target
    for part in path.parts[:-1]:
        current /= part
        # Parent symlinks can redirect a lexically safe relative path.
        if current.is_symlink():
            raise ValueError(
                "Asset destination contains a symbolic-link parent: "
                f"{relative}"
            )
    resolved_parent = destination.parent.resolve()
    # Resolution catches links introduced between the component checks above.
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


def _create_parent_directories(
    target: Path,
    parent: Path,
    created: list[Path],
) -> None:
    missing: list[Path] = []
    current = parent
    while current != target and not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
        created.append(directory)


def _prune_empty_parents(
    parent: Path,
    target: Path,
    removed: list[Path],
) -> None:
    current = parent
    while current != target:
        try:
            current.rmdir()
        except OSError:
            break
        removed.append(current)
        current = current.parent


def _rollback_batch(
    *,
    installed: list[Path],
    moved: list[tuple[Path, Path]],
    created_directories: list[Path],
    removed_directories: list[Path],
) -> None:
    for destination in reversed(installed):
        _remove_path(destination)
    for directory in reversed(created_directories):
        try:
            directory.rmdir()
        except OSError:
            pass
    for directory in sorted(
        set(removed_directories),
        key=lambda path: len(path.parts),
    ):
        directory.mkdir(exist_ok=True)
    for destination, previous in reversed(moved):
        destination.parent.mkdir(parents=True, exist_ok=True)
        previous.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
