"""Exercise release preparation against disposable Git repositories."""

from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = runpy.run_path(str(ROOT / "scripts/prepare_release.py"))
prepare = TOOLS["prepare"]
NOW = datetime(2026, 10, 27, 2, tzinfo=timezone.utc)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=Release Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "Test release state")


def tag(root, name):
    git(root, "-c", "tag.gpgsign=false", "tag", name)


def state_path(root):
    directory = Path(git(root, "rev-parse", "--git-common-dir"))
    return (root / directory).resolve() / "multicamera-release.json"


@pytest.fixture
def repo(tmp_path):
    package = tmp_path / "src/multicamera"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(
        '# 版本来源\r\n__version__ = "1.2.3"  # keep comment\r\n'.encode("utf-8"))
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndynamic = ["version"]\n'
        '[tool.setuptools.dynamic]\nversion = {attr = "multicamera.__version__"}\n')
    git(tmp_path, "init")
    commit(tmp_path)
    return tmp_path


@pytest.mark.parametrize(("bump", "expected"), [(None, "1.2.4"), ("minor", "1.3.0"),
                                                ("none", "1.2.3")])
def test_prepare_version_only_leaves_commit_and_tags_untouched(repo, bump, expected):
    head = git(repo, "rev-parse", "HEAD")
    original = (repo / "src/multicamera/__init__.py").read_bytes()
    plan = prepare(repo, bump=bump)
    assert plan["version"] == expected
    assert plan["tag"] == f"v{expected}"
    assert (repo / "src/multicamera/__init__.py").read_bytes() == original.replace(
        b"1.2.3", expected.encode())
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "tag", "--list") == ""
    assert git(repo, "diff", "--cached") == ""
    assert json.loads(state_path(repo).read_text())["reserved_tags"] == [f"v{expected}"]


def test_preview_uses_maximum_local_tag_and_reservation(repo):
    tag(repo, "v1.2.3-build.20261027.2")
    tag(repo, "v1.2.3-build.20261027.9")
    tag(repo, "v1.2.4-build.20261027.99")
    tag(repo, "v1.2.3-build.20261026.99")
    tag(repo, "v1.2.3-build.20261027.010")
    assert prepare(repo, preview=True, now=NOW)["tag"] == "v1.2.3-build.20261027.10"
    assert prepare(repo, preview=True, now=NOW)["tag"] == "v1.2.3-build.20261027.11"
    assert prepare(repo, preview=True, now=NOW)["version"] == "1.2.3"
    assert not git(repo, "status", "--porcelain")


def test_preview_combined_with_bump_and_finalization(repo):
    plan = prepare(repo, bump="patch", preview=True, now=NOW)
    assert plan["tag"] == "v1.2.4-build.20261027.1"
    commit(repo)
    assert prepare(repo, bump="none")["tag"] == "v1.2.4"


def test_linked_worktrees_share_preview_reservations(repo):
    linked = repo.parent / (repo.name + "-linked")
    git(repo, "worktree", "add", "--detach", str(linked), "HEAD")
    try:
        assert prepare(repo, preview=True, now=NOW)["tag"].endswith(".1")
        assert prepare(linked, preview=True, now=NOW)["tag"].endswith(".2")
        assert state_path(repo) == state_path(linked)
    finally:
        git(repo, "worktree", "remove", str(linked))


def test_preview_uses_shanghai_date_and_resets_each_day(repo):
    # 16:30 UTC is already the next calendar day in Shanghai.
    before_midnight = datetime(2026, 10, 26, 15, 59, tzinfo=timezone.utc)
    after_midnight = datetime(2026, 10, 26, 16, 30, tzinfo=timezone.utc)
    assert prepare(repo, preview=True, now=before_midnight)["tag"].endswith("20261026.1")
    assert prepare(repo, preview=True, now=after_midnight)["tag"].endswith("20261027.1")


def test_dirty_worktree_refused_without_modifying_source_or_reservations(repo):
    (repo / "unrelated.txt").write_text("in progress")
    original = (repo / "src/multicamera/__init__.py").read_bytes()
    with pytest.raises(ValueError, match="clean"):
        prepare(repo)
    assert (repo / "src/multicamera/__init__.py").read_bytes() == original
    assert not state_path(repo).exists()


def test_dry_run_on_dirty_worktree_does_not_reserve_or_write(repo):
    (repo / "unrelated.txt").write_text("in progress")
    first = prepare(repo, preview=True, dry_run=True, now=NOW)
    second = prepare(repo, preview=True, dry_run=True, now=NOW)
    assert first == second
    assert first["worktree"] == "dirty"
    assert first["tag"] == "v1.2.3-build.20261027.1"
    assert not state_path(repo).exists()
    assert not state_path(repo).with_suffix(".lock").exists()


@pytest.mark.parametrize("preview", [False, True])
def test_existing_stable_version_cannot_be_prepared_again(repo, preview):
    tag(repo, "v1.2.3")
    with pytest.raises(ValueError, match="already exists"):
        prepare(repo, bump="none", preview=preview)


def test_stable_reservation_cannot_be_overwritten(repo):
    prepare(repo, bump="none")
    original = state_path(repo).read_bytes()
    with pytest.raises(ValueError, match="already prepared"):
        prepare(repo, bump="none")
    assert state_path(repo).read_bytes() == original


@pytest.mark.parametrize("content", ['not json', '[]', '{"reserved_tags": [123]}'])
def test_corrupt_reservations_fail_without_version_change(repo, content):
    state_path(repo).write_text(content)
    with pytest.raises(ValueError):
        prepare(repo)
    assert TOOLS["CHECK"]["product_version"](repo) == "1.2.3"


def test_concurrent_prepare_refused_and_existing_lock_preserved(repo):
    lock = state_path(repo).with_suffix(".lock")
    lock.write_text("other process")
    with pytest.raises(ValueError, match="lock"):
        prepare(repo)
    assert lock.read_text() == "other process"


def test_failed_reservation_does_not_change_version(repo, monkeypatch):
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setitem(prepare.__globals__, "atomic_write", fail)
    with pytest.raises(OSError, match="disk full"):
        prepare(repo)
    assert TOOLS["CHECK"]["product_version"](repo) == "1.2.3"
    assert not state_path(repo).with_suffix(".lock").exists()


def test_failed_version_write_never_reuses_reserved_preview_number(repo, monkeypatch):
    original_write = prepare.__globals__["atomic_write"]
    def fail_version(path, content):
        if path.name == "__init__.py":
            raise OSError("cannot replace version")
        original_write(path, content)
    monkeypatch.setitem(prepare.__globals__, "atomic_write", fail_version)
    with pytest.raises(OSError):
        prepare(repo, bump="patch", preview=True, now=NOW)
    monkeypatch.setitem(prepare.__globals__, "atomic_write", original_write)
    plan = prepare(repo, bump="patch", preview=True, now=NOW)
    assert plan["tag"] == "v1.2.4-build.20261027.2"


def test_command_dry_run_is_parseable_and_does_not_change_real_checkout():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_release.py"), "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["dry_run"] is True
