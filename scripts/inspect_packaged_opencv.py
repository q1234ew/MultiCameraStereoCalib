"""Inspect the native cv2 binding from an existing Windows portable ZIP.

Run in a separate Windows Python process, not in the application's interpreter.
This probes the archived binary, not a newly installed cv2 or the old GUI EXE.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

from multicamera.diagnostics import opencv_runtime_info


def inspect_archive(archive: Path) -> dict:
    if sys.platform != "win32":
        raise RuntimeError("The archived Windows cv2.pyd must be inspected on Windows")
    with archive.open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    # Windows keeps a loaded .pyd locked until this short-lived process exits.
    with tempfile.TemporaryDirectory(prefix="multicamera-opencv-audit-",
                                     ignore_cleanup_errors=True) as temp:
        root = Path(temp).resolve()
        with zipfile.ZipFile(archive) as package:
            for name in package.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise ValueError(f"Unsafe archive member: {name}")
            package.extractall(root)
        candidates = list(root.rglob("cv2*.pyd"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one cv2 native extension, found {candidates}")
        native_path = candidates[0]
        handles = [os.add_dll_directory(str(directory)) for directory in
                   dict.fromkeys((native_path.parent, native_path.parent.parent))]
        try:
            # numpy is provided by the probe interpreter; cv2 is loaded ONLY from the archive.
            import numpy  # noqa: F401

            spec = importlib.util.spec_from_file_location("cv2", native_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["cv2"] = module
            spec.loader.exec_module(module)
            info = opencv_runtime_info(module, hash_native=True)
            info["probe_environment_distributions"] = info.pop("distributions")
            info["archive"] = str(archive.resolve())
            info["archive_sha256"] = digest
            info["probe_type"] = "archived_native_binding_not_old_executable"
            return info
        finally:
            for handle in handles:
                handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "failed"}
    try:
        report["opencv"] = inspect_archive(args.archive)
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
