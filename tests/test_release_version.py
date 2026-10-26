"""Release checks reject inconsistent metadata and tags before packaging."""

from pathlib import Path
import runpy
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECK = runpy.run_path(str(ROOT / "scripts/check_release.py"))


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src/multicamera").mkdir(parents=True)
    (tmp_path / "src/multicamera/__init__.py").write_text('__version__ = "1.2.3"\n')
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndynamic = ["version"]\n'
        '[tool.setuptools.dynamic]\nversion = {attr = "multicamera.__version__"}\n')
    return tmp_path


@pytest.mark.parametrize("tag", [None, "v1.2.3", "v1.2.3-build.20261027.1"])
def test_matching_versions(repo, tag):
    assert CHECK["validate"](repo, tag=tag) == "1.2.3"


@pytest.mark.parametrize("tag", ["v1.2.4", "1.2.3", "v1.2.3-rc.1",
                                 "v1.2.3-build.20260230.1", "v1.2.3-build.20261027.0",
                                 "v01.2.3", "v1.2.3-build.20261027.01"])
def test_invalid_or_mismatched_tag(repo, tag):
    with pytest.raises(ValueError):
        CHECK["validate"](repo, tag=tag)


@pytest.mark.parametrize("value", ['"1.2"', '"01.2.3"', '"1.2.3-rc.1"', '123'])
def test_invalid_product_version(repo, value):
    (repo / "src/multicamera/__init__.py").write_text(f"__version__ = {value}\n")
    with pytest.raises(ValueError):
        CHECK["validate"](repo)


def test_duplicate_metadata_source_rejected(repo):
    path = repo / "pyproject.toml"
    path.write_text(path.read_text().replace('[project]', '[project]\nversion = "1.2.3"'))
    with pytest.raises(ValueError, match="derive version"):
        CHECK["validate"](repo)


def test_stale_installed_metadata_rejected(repo, monkeypatch):
    monkeypatch.setitem(CHECK["validate"].__globals__, "installed_version", lambda _: "1.2.2")
    with pytest.raises(ValueError, match="stale"):
        CHECK["validate"](repo, installed=True)


def test_clean_checkout_and_tag_target(repo):
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init")
    git("add", ".")
    git("-c", "user.name=Release Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "Initial")
    git("-c", "tag.gpgsign=false", "tag", "v1.2.3")
    assert CHECK["validate"](repo, tag="v1.2.3", require_clean=True) == "1.2.3"
    (repo / "untracked.txt").write_text("new work")
    with pytest.raises(ValueError, match="clean"):
        CHECK["validate"](repo, require_clean=True)
    git("add", ".")
    git("-c", "user.name=Release Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "Next")
    with pytest.raises(ValueError, match="HEAD"):
        CHECK["validate"](repo, tag="v1.2.3", require_clean=True)


def test_repository_metadata_uses_runtime_version():
    from multicamera import __version__
    assert CHECK["validate"](ROOT) == __version__


def test_packaging_records_requested_tag_when_preview_and_stable_share_commit(repo, monkeypatch):
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init")
    git("add", ".")
    git("-c", "user.name=Release Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "Release")
    for tag in ("v1.2.3", "v1.2.3-build.20261027.1"):
        git("-c", "tag.gpgsign=false", "tag", tag)
    monkeypatch.setitem(CHECK["validate"].__globals__, "installed_version", lambda _: "1.2.3")
    for tag in ("v1.2.3", "v1.2.3-build.20261027.1"):
        identity = CHECK["packaging_identity"](root=repo, tag=tag)
        assert identity["tag"] == tag
        assert identity["version"] == "1.2.3"
        assert identity["dirty"] is False
        assert identity["commit"] != "未知"
    (repo / "untracked").write_text("unfinished")
    with pytest.raises(ValueError, match="clean"):
        CHECK["packaging_identity"](root=repo, tag="v1.2.3")
