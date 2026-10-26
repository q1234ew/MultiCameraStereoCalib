"""Validate release versions without importing Qt, OpenCV, or the application."""

import argparse
import ast
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 build/dev environments
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"(?:0|[1-9][0-9]*)"
VERSION = rf"{NUMBER}\.{NUMBER}\.{NUMBER}"


def product_version(root=ROOT):
    tree = ast.parse((root / "src/multicamera/__init__.py").read_text(encoding="utf-8"))
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)]
    if len(values) != 1 or not isinstance(values[0], str) or not re.fullmatch(VERSION, values[0]):
        raise ValueError("__version__ must be a single literal MAJOR.MINOR.PATCH string")
    return values[0]


def validate(root=ROOT, tag=None, require_clean=False, installed=False):
    version = product_version(root)
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    if ("version" in config["project"]
            or "version" not in config["project"].get("dynamic", [])
            or config["tool"]["setuptools"]["dynamic"]["version"]
            != {"attr": "multicamera.__version__"}):
        raise ValueError("pyproject.toml must derive version from multicamera.__version__")
    if tag is not None:
        match = re.fullmatch(rf"v({VERSION})(?:-build\.([0-9]{{8}})\.([1-9][0-9]*))?", tag)
        if not match or match[1] != version:
            raise ValueError(f"Tag {tag!r} must match v{version} or v{version}-build.YYYYMMDD.N")
        if match[2]:
            datetime.strptime(match[2], "%Y%m%d")
    if installed and installed_version("multicamera") != version:
        raise ValueError("Installed package version is stale; reinstall with pip install -e .")
    if require_clean:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                cwd=root, check=True, capture_output=True, text=True)
        if status.stdout.strip():
            raise ValueError("Release requires a clean, committed checkout")
        if tag:
            def revision(ref):
                return subprocess.check_output(
                    ["git", "rev-parse", "--verify", ref], cwd=root, text=True).strip()
            if revision(f"refs/tags/{tag}^{{commit}}") != revision("HEAD"):
                raise ValueError("Release tag does not point to HEAD")
    return version


def packaging_identity(root=ROOT, tag=None):
    """Use the requested release tag when several tags point at the same commit."""
    version = validate(root=root, tag=tag, require_clean=tag is not None, installed=True)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL, timeout=5).strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "未知", None
    if tag is None:
        try:
            tag = git("describe", "--tags", "--exact-match")
        except (OSError, subprocess.SubprocessError):
            pass
    return {"version": version, "commit": commit, "tag": tag, "dirty": dirty}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Expected stable or build preview tag")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--installed", action="store_true", help="Check installed package metadata")
    parser.add_argument("--print-version", action="store_true")
    args = parser.parse_args()
    try:
        version = validate(tag=args.tag, require_clean=args.require_clean,
                           installed=args.installed)
    except (ValueError, KeyError, OSError, PackageNotFoundError, subprocess.SubprocessError) as exc:
        print(f"Release validation failed: {exc}", file=sys.stderr)
        return 1
    print(version if args.print_version else f"Release version checks passed: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
