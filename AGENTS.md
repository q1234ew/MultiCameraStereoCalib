# Repository Guidelines

## Project Structure & Module Organization

MultiCamera is a Python 3.10+ desktop application using PySide6 and OpenCV for stereo and auxiliary-camera calibration.

- `src/multicamera/`: application package; `calibration/` contains solvers and quality checks, `streaming/` handles cameras, `io/` manages sessions and exports, and `fusion/` provides optional point-cloud features.
- `src/multicamera/ui/`: widgets, themes, and background workers in `threads/`.
- `tests/`: application regression tests; `scripts/tests/`: tests for research and diagnostic tools.
- `assets/` and `src/multicamera/ui/icons/`: visual resources; `scripts/` and `packaging/windows/`: diagnostics and packaging.
- `docs/`: technical documentation and append-only changelog.

## Build, Test, and Development Commands

Run from the repository root with the virtual environment active:

```bash
python -m pip install -e ".[dev]"       # Install application and development tools
python -m multicamera                  # Launch the desktop application
python -m pytest -q tests              # Run application regressions
python -m ruff check src scripts       # Check Python style
python -m compileall -q src scripts    # Check syntax
python -m multicamera --self-test-calibration --report calibration_self_test.json
```

On Windows, install with `-c packaging/windows/constraints.txt`. Build using `scripts\build_windows.bat`; create installers with `scripts\build_windows_installer.bat` and Inno Setup 6. Open3D is optional (`.[pointcloud]`); preserve lazy loading.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and descriptive type annotations. Ruff targets Python 3.10 with a 100-character line length. Follow surrounding patterns; keep device access and expensive calibration work out of the UI thread.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions. Add focused regressions for changed calibration, persistence, export, or UI behavior. Reuse the shared `app` fixture for Qt tests; `tests/conftest.py` enables offscreen rendering and widget cleanup. Run `python -m pytest -q scripts/tests` when changing diagnostic tools. No numeric coverage threshold is configured. Windows releases require source regressions, source calibration self-tests, and final EXE calibration self-tests; local tests do not establish hardware accuracy.

## Commit & Pull Request Guidelines

Recent commits use imperative, descriptive subjects such as `Fix Windows configuration overflow`. Keep commits focused. PRs should explain the problem, behavior change, validation results, and relevant issues; include screenshots for UI changes. Update technical documentation and append changelog entries for meaningful behavior changes.

## Data & Calibration Contracts

Keep generated `sessions/`, `logs/`, `data/`, and build outputs untracked. Remove private camera addresses before sharing logs. Preserve millimeter-to-meter conversion and transform directions: `T_rgb_left_aux` maps auxiliary-camera coordinates into RGB-left coordinates. Keep experimental mrcal tooling separate from the application runtime.

## Versioning & Release Management

- Use `MAJOR.MINOR.PATCH` for product versions (currently `0.1.0`). Increment PATCH for compatible fixes, MINOR for new capabilities, and MAJOR for breaking changes. During `0.x`, use a MINOR increment for breaking changes and document migration steps. Documentation-only changes do not require a version bump.
- Change only `src/multicamera/__init__.py` (`__version__`) when bumping the product version. Setuptools derives package metadata through `pyproject.toml`; Windows build scripts pass this version into Inno Setup and artifact filenames. Reinstall the editable package after a bump (`python -m pip install -e ".[dev]"`) to refresh installed metadata. Update version-specific documentation examples.
- Downloadable Windows binaries must include the product version and architecture in their filenames: `MultiCamera_Setup_<version>_x64.exe` and `MultiCamera_portable_<version>_x64.zip`. Packaging scripts, Release uploads, download instructions, metrics, and checksum records must use the same names. Read historical unversioned archives only for compatibility; never rename or replace already published assets.
- Tag stable releases `vX.Y.Z`. Follow the existing preview convention `vX.Y.Z-build.YYYYMMDD.N`, using the Shanghai calendar date and a new sequence number for each build that day. Keep the base product version aligned with the tag; retain numeric versions in installer metadata. Never move a published tag or replace an already published binary with different contents; use a new tag for corrected builds.
- Build release candidates from a clean, committed checkout. Verify the embedded identity from `MultiCamera.spec` and `build_info.py` matches the intended commit/tag and reports no uncommitted changes. Include the product version, full commit, tag, Open3D variant, changes, known limitations, and Actions run link in release notes.
- Before tagging, run the checks above and `python scripts/check_release.py --installed --require-clean`; pass `--tag vX.Y.Z` after creating the local tag to verify its format, version and HEAD target. Preview dates must be valid and sequence numbers positive. CI repeats this check before packaging; direct PyInstaller builds also validate installed metadata. The Windows workflow `.github/workflows/build-windows.yml` runs source regressions, source calibration self-tests, packaging, and final EXE calibration self-tests. Require all to pass. Verify the Release contains the expected installer, portable ZIP, both calibration reports, and `SHA256SUMS.txt` with matching hashes.
- Pushing `v*` tags triggers publication. Tags containing `-` are prereleases; manual runs publish `windows-build-<run_id>` prereleases and include installers only when selected. Trigger publication only within the user's authorized release scope. Report local validation, build completion, and published download availability separately; claim a release is available only after checking its published assets.


### Local Release Preparation

Use `python scripts/prepare_release.py --dry-run` to inspect the plan without writes, even with unfinished work. Actual preparation requires a clean checkout and changes only the product version when needed; it never stages, commits, creates tags, fetches, pushes, or triggers Actions.

- `python scripts/prepare_release.py`: increment PATCH and propose `vX.Y.Z`.
- `python scripts/prepare_release.py --bump minor`: increment MINOR and reset PATCH.
- `python scripts/prepare_release.py --preview`: keep the current product version and reserve `vX.Y.Z-build.YYYYMMDD.N`.
- Add `--bump patch` or `--bump minor` to `--preview` to start a new target version. For subsequent previews, commit the version change and use `--preview` alone.
- `python scripts/prepare_release.py --bump none`: prepare the current tested preview version for stable release without another bump.

Preview dates use Shanghai time. Sequences advance from the maximum matching local Git tag or local reservation; other dates and product versions do not affect them. Reservations and the last plan live in Git's common metadata directory (shared by linked worktrees) (`multicamera-release.json`) and stay untracked. A lock prevents concurrent preparation; a failed write may consume a number rather than reuse it. Existing stable tags and repeated stable preparations are rejected. Inspect a stale lock or failed preparation record before manually repairing it; never reset records merely to reuse a number.

The command uses local information only. Before preparing a release, update local tags through a separately authorized fetch when other machines may have published builds. Review the plan and changelog, reinstall editable metadata, run checks, commit, create the proposed local tag, then run `check_release.py --installed --require-clean --tag <tag>`. Pushing remains a separate authorized action. A preparation plan does not reserve a tag on GitHub or establish a successful build.
