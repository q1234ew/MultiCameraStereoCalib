"""Prepare a local release version and tag proposal; never commit, tag, fetch, or push."""

import argparse
import ast
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile


CHECK = runpy.run_path(str(Path(__file__).with_name("check_release.py")))
ROOT = CHECK["ROOT"]
SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def atomic_write(path, content):
    """Replace a file without exposing a partially written version or reservation list."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        if path.exists():
            temporary.chmod(path.stat().st_mode)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def version_content(path, version):
    source = path.read_bytes().decode("utf-8")
    tree = ast.parse(source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "__version__"
                              for t in node.targets))
    value = assignment.value
    lines = source.encode("utf-8").splitlines(keepends=True)
    start = sum(map(len, lines[:value.lineno - 1])) + value.col_offset
    end = sum(map(len, lines[:value.end_lineno - 1])) + value.end_col_offset
    data = source.encode("utf-8")
    return data[:start] + json.dumps(version).encode("ascii") + data[end:]


def prepare(root=ROOT, bump=None, preview=False, dry_run=False, now=None):
    root = Path(root).resolve()
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Run against the repository root")
    head = git(root, "rev-parse", "HEAD")
    git_dir = Path(git(root, "rev-parse", "--git-common-dir"))
    if not git_dir.is_absolute():
        git_dir = (root / git_dir).resolve()
    state_path = git_dir / "multicamera-release.json"
    lock_path = git_dir / "multicamera-release.lock"
    lock = None
    try:
        if not dry_run:
            try:
                lock = lock_path.open("x")
            except FileExistsError as exc:
                raise ValueError("Another release preparation is running (release lock exists)") from exc
        current = CHECK["validate"](root=root, require_clean=not dry_run)
        dirty = bool(git(root, "status", "--porcelain", "--untracked-files=all"))
        bump = bump or ("none" if preview else "patch")
        major, minor, patch = map(int, current.split("."))
        if bump == "patch":
            patch += 1
        elif bump == "minor":
            minor, patch = minor + 1, 0
        elif bump != "none":
            raise ValueError("Bump must be patch, minor, or none")
        target = f"{major}.{minor}.{patch}"
        tags = set(git(root, "tag", "--list").splitlines())
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        if not isinstance(state, dict):
            raise ValueError("Invalid local release reservation object")
        reserved = state.get("reserved_tags", [])
        if not isinstance(reserved, list) or any(not isinstance(tag, str) for tag in reserved):
            raise ValueError("Invalid local release reservations; inspect multicamera-release.json")
        if f"v{target}" in tags:
            raise ValueError(f"v{target} already exists; prepare a newer product version")
        tag = f"v{target}"
        if preview:
            now = now or datetime.now(SHANGHAI)
            if now.tzinfo is None:
                raise ValueError("Release date must include a timezone")
            day = now.astimezone(SHANGHAI).strftime("%Y%m%d")
            pattern = re.compile(rf"v{re.escape(target)}-build\.{day}\.([1-9][0-9]*)")
            sequences = [int(match[1]) for existing in tags | set(reserved)
                         if (match := pattern.fullmatch(existing))]
            tag += f"-build.{day}.{max(sequences, default=0) + 1}"
        elif tag in reserved:
            raise ValueError(f"{tag} was already prepared locally; inspect the local release record")
        plan = {"dry_run": dry_run, "previous_version": current, "version": target,
                "tag": tag, "prerelease": preview, "commit_before_preparation": head,
                "worktree": "dirty" if dirty else "clean",
                "next_steps": ["Review changes and update the changelog",
                               'Reinstall metadata: python -m pip install -e ".[dev]"',
                               "Run regression checks and commit the release changes",
                               f"Create local tag: git tag {tag}",
                               "Validate: python scripts/check_release.py --installed "
                               f"--require-clean --tag {tag}",
                               "Push only when separately authorized"]}
        if not dry_run:
            if git(root, "rev-parse", "HEAD") != head:
                raise ValueError("HEAD changed during release preparation; retry")
            CHECK["validate"](root=root, require_clean=True)
            content = version_content(root / "src/multicamera/__init__.py", target)
            # Reserve first: a failed version write may leave a gap, never a reused build number.
            atomic_write(state_path, json.dumps(
                {"reserved_tags": reserved + [tag], "last_plan": plan}, indent=2).encode("utf-8"))
            if target != current:
                atomic_write(root / "src/multicamera/__init__.py", content)
        return plan
    finally:
        if lock is not None:
            lock.close()
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bump", choices=("patch", "minor", "none"),
                        help="Default: patch; preview defaults to none")
    parser.add_argument("--preview", action="store_true", help="Reserve a dated preview build tag")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing any files")
    args = parser.parse_args()
    try:
        plan = prepare(bump=args.bump, preview=args.preview, dry_run=args.dry_run)
    except (ValueError, KeyError, OSError, SyntaxError, subprocess.SubprocessError) as exc:
        print(f"Release preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
