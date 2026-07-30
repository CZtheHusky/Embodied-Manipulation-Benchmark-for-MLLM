from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from mllm_teleop_benchmark.utils.pinned_zip import (
    ZipInstallRequest,
    install_zip_batch_exact,
    install_zip_exact,
    write_json_atomic,
)


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_exact_install_replaces_only_archive_roots(tmp_path):
    target = tmp_path / "assets"
    (target / "owned").mkdir(parents=True)
    (target / "owned/stale.txt").write_text("stale", encoding="utf-8")
    (target / "owned/tracked.xml").write_text("tracked", encoding="utf-8")
    (target / "unrelated").mkdir()
    (target / "unrelated/keep.txt").write_text("keep", encoding="utf-8")
    archive = _zip(
        tmp_path / "assets.zip",
        {"owned/current.txt": b"current"},
    )

    result = install_zip_exact(
        archive,
        target,
        previous_files=("owned/stale.txt",),
    )

    assert result.roots == ("owned",)
    assert result.file_count == 1
    assert result.files == ("owned/current.txt",)
    assert (target / "owned/current.txt").read_bytes() == b"current"
    assert not (target / "owned/stale.txt").exists()
    assert (
        target / "owned/tracked.xml"
    ).read_text(encoding="utf-8") == "tracked"
    assert (target / "unrelated/keep.txt").read_text(encoding="utf-8") == "keep"


def test_exact_install_rejects_symlink_members(tmp_path):
    archive_path = tmp_path / "symlink.zip"
    with ZipFile(archive_path, "w") as archive:
        member = ZipInfo("owned/link")
        member.create_system = 3
        member.external_attr = 0o120777 << 16
        archive.writestr(member, "../outside")

    with pytest.raises(ValueError, match="symbolic link"):
        install_zip_exact(archive_path, tmp_path / "assets")


def test_exact_install_handles_file_to_directory_change(tmp_path):
    target = tmp_path / "assets"
    target.mkdir()
    (target / "owned").write_text("old", encoding="utf-8")
    archive = _zip(
        tmp_path / "assets.zip",
        {"owned/current.txt": b"current"},
    )

    install_zip_exact(
        archive,
        target,
        previous_files=("owned",),
    )

    assert (target / "owned/current.txt").read_bytes() == b"current"


def test_exact_install_handles_directory_to_file_change(tmp_path):
    target = tmp_path / "assets"
    (target / "owned").mkdir(parents=True)
    (target / "owned/stale.txt").write_text("old", encoding="utf-8")
    archive = _zip(
        tmp_path / "assets.zip",
        {"owned": b"current"},
    )

    install_zip_exact(
        archive,
        target,
        previous_files=("owned/stale.txt",),
    )

    assert (target / "owned").read_bytes() == b"current"


def test_exact_install_rolls_back_when_directory_has_unowned_file(tmp_path):
    target = tmp_path / "assets"
    (target / "owned").mkdir(parents=True)
    (target / "owned/stale.txt").write_text("old", encoding="utf-8")
    (target / "owned/keep.txt").write_text("keep", encoding="utf-8")
    archive = _zip(
        tmp_path / "assets.zip",
        {"owned": b"current"},
    )

    with pytest.raises(ValueError, match="non-empty directory"):
        install_zip_exact(
            archive,
            target,
            previous_files=("owned/stale.txt",),
        )

    assert (target / "owned/stale.txt").read_text(encoding="utf-8") == "old"
    assert (target / "owned/keep.txt").read_text(encoding="utf-8") == "keep"


def test_exact_install_rejects_traversal_member(tmp_path):
    archive = _zip(
        tmp_path / "traversal.zip",
        {"../outside.txt": b"outside"},
    )

    with pytest.raises(ValueError, match="escapes extraction directory"):
        install_zip_exact(archive, tmp_path / "assets")

    assert not (tmp_path / "outside.txt").exists()


def test_exact_install_rejects_symlinked_destination_parent(tmp_path):
    target = tmp_path / "assets"
    outside = tmp_path / "outside"
    target.mkdir()
    outside.mkdir()
    (target / "owned").symlink_to(outside, target_is_directory=True)
    archive = _zip(
        tmp_path / "assets.zip",
        {"owned/current.txt": b"current"},
    )

    with pytest.raises(ValueError, match="symbolic-link parent"):
        install_zip_exact(archive, target)

    assert not (outside / "current.txt").exists()


def test_batch_install_rolls_back_earlier_archive(tmp_path):
    first_target = tmp_path / "first"
    second_target = tmp_path / "second"
    first_target.mkdir()
    (first_target / "old.txt").write_text("old", encoding="utf-8")
    (second_target / "conflict").mkdir(parents=True)
    (second_target / "conflict/stale.txt").write_text(
        "stale",
        encoding="utf-8",
    )
    (second_target / "conflict/keep.txt").write_text(
        "keep",
        encoding="utf-8",
    )
    first_archive = _zip(
        tmp_path / "first.zip",
        {"new.txt": b"new"},
    )
    second_archive = _zip(
        tmp_path / "second.zip",
        {"conflict": b"replacement"},
    )

    with pytest.raises(ValueError, match="non-empty directory"):
        install_zip_batch_exact(
            (
                ZipInstallRequest(
                    first_archive,
                    first_target,
                    ("old.txt",),
                ),
                ZipInstallRequest(
                    second_archive,
                    second_target,
                    ("conflict/stale.txt",),
                ),
            )
        )

    assert (first_target / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (first_target / "new.txt").exists()
    assert (
        second_target / "conflict/stale.txt"
    ).read_text(encoding="utf-8") == "stale"
    assert (
        second_target / "conflict/keep.txt"
    ).read_text(encoding="utf-8") == "keep"


def test_batch_rollback_preserves_preexisting_empty_directory(tmp_path):
    first_target = tmp_path / "first"
    second_target = tmp_path / "second"
    (first_target / "empty").mkdir(parents=True)
    (second_target / "conflict").mkdir(parents=True)
    (second_target / "conflict/keep.txt").write_text(
        "keep",
        encoding="utf-8",
    )
    first_archive = _zip(
        tmp_path / "first.zip",
        {"empty/new.txt": b"new"},
    )
    second_archive = _zip(
        tmp_path / "second.zip",
        {"conflict": b"replacement"},
    )

    with pytest.raises(ValueError, match="non-empty directory"):
        install_zip_batch_exact(
            (
                ZipInstallRequest(first_archive, first_target),
                ZipInstallRequest(second_archive, second_target),
            )
        )

    assert (first_target / "empty").is_dir()
    assert list((first_target / "empty").iterdir()) == []


def test_atomic_json_rejects_marker_symlink(tmp_path):
    outside = tmp_path / "outside.json"
    marker = tmp_path / "marker.json"
    outside.write_text("original", encoding="utf-8")
    marker.symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic link"):
        write_json_atomic(marker, {"revision": "new"})

    assert outside.read_text(encoding="utf-8") == "original"
