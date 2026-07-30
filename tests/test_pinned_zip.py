from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from mllm_teleop_benchmark.utils.pinned_zip import install_zip_exact


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_exact_install_replaces_only_archive_roots(tmp_path):
    target = tmp_path / "assets"
    (target / "owned").mkdir(parents=True)
    (target / "owned/stale.txt").write_text("stale", encoding="utf-8")
    (target / "unrelated").mkdir()
    (target / "unrelated/keep.txt").write_text("keep", encoding="utf-8")
    archive = _zip(
        tmp_path / "assets.zip",
        {"owned/current.txt": b"current"},
    )

    result = install_zip_exact(archive, target)

    assert result.roots == ("owned",)
    assert result.file_count == 1
    assert (target / "owned/current.txt").read_bytes() == b"current"
    assert not (target / "owned/stale.txt").exists()
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
