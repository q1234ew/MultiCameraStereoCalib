"""Headless, deterministic smoke test run by the final packaged executable."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path


def _check(condition, message):
    if not condition:
        raise RuntimeError(message)


def _exercise_calibration(report: dict) -> dict:
    import cv2
    import numpy as np

    from .board.charuco_board import CharucoBoard, CharucoBoardConfig
    from .board.detector import CharucoDetector, DetectionResult
    from .calibration.intrinsic import IntrinsicCalibrator
    from .calibration.models import MultiCameraRig, StereoPairCalibration
    from .calibration.stereo import StereoCalibrator
    from .io.export import export_rig_calibration_bundle

    report["stage"] = "charuco_detection"
    board = CharucoBoard(CharucoBoardConfig())
    detection = CharucoDetector(board).detect(board.generate_image(pixel_size=80))
    _check(detection.num_corners == board.num_corners, "Generated board detection incomplete")
    left = IntrinsicCalibrator(board, min_frames=20)
    right = IntrinsicCalibrator(board, min_frames=20)
    stereo = StereoCalibrator(board, min_frames=20)
    k = np.array([[900., 0., 640.], [0., 880., 360.], [0., 0., 1.]])
    expected_t = np.array([[-.12], [0.], [0.]])
    rng = np.random.default_rng(42)
    for i in range(24):
        rv = np.array([.3 * np.sin(i), .2 * np.cos(i), .03 * i])
        tv = np.array([[-.3 + .015 * i], [-.2 + .01 * np.sin(i)], [1.2 + .04 * i]])
        detections = []
        for offset in (np.zeros((3, 1)), expected_t):
            ids = rng.permutation(board.num_corners)[:board.num_corners - (i % 4) * 2]
            points, _ = cv2.projectPoints(board.object_points[ids], rv, tv + offset, k, None)
            detections.append(DetectionResult(
                points.astype(np.float32).reshape(-1, 1, 2),
                ids.astype(np.int32).reshape(-1, 1), [], None, (1280, 720),
            ))
        _check(left.add_detection(detections[0]), "Left observation rejected")
        _check(right.add_detection(detections[1]), "Right observation rejected")
        _check(stereo.add_detection_pair(*detections), "Stereo observation rejected")
    report["stage"] = "intrinsics_left"
    intr_l = left.calibrate()
    report["stage"] = "intrinsics_right"
    intr_r = right.calibrate()
    report["stage"] = "stereo_calibration"
    result = stereo.calibrate(intr_l, intr_r)
    for intr in (intr_l, intr_r):
        _check(np.isfinite(intr.rms_error) and intr.rms_error < .001, "Intrinsic RMS too high")
        np.testing.assert_allclose(intr.camera_matrix, k, atol=.03, rtol=0)
        _check(len(intr.per_frame_errors) == 24, "Per-frame errors missing")
    _check(np.isfinite(result.rms_error) and result.rms_error < .001, "Stereo RMS too high")
    np.testing.assert_allclose(result.R, np.eye(3), atol=1e-4, rtol=0)
    np.testing.assert_allclose(result.T.reshape(3, 1), expected_t, atol=1e-4, rtol=0)
    for matrix in (result.Q, result.R1, result.R2, result.P1, result.P2, result.map1_left):
        _check(matrix is not None and np.isfinite(matrix).all(), "Rectification output invalid")

    rig = MultiCameraRig(reference_camera="selftest_L")
    rig.pairs["selftest"] = StereoPairCalibration(
        "selftest", "selftest_L", "selftest_R", intr_l, intr_r, result,
    )
    # Exercise the real exporters, then reopen their output using consumer APIs.
    report["stage"] = "export_and_readback"
    with tempfile.TemporaryDirectory(prefix="multicamera-selftest-") as temp:
        paths = export_rig_calibration_bundle(rig, temp)
        restored = MultiCameraRig.from_dict(json.loads(paths[0].read_text(encoding="utf-8")))
        np.testing.assert_allclose(restored.pairs["selftest"].stereo.T, result.T)
        for path in paths[1:3]:
            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            try:
                _check(fs.isOpened(), f"Cannot reopen {path.name}")
                for key, expected in (("M1", intr_l.camera_matrix), ("M2", intr_r.camera_matrix),
                                      ("R", result.R), ("T", result.T), ("Q", result.Q)):
                    np.testing.assert_allclose(fs.getNode(key).mat(), expected)
                _check(fs.getNode("left_camera_id").string() == "selftest_L", "Left ID changed")
                _check(fs.getNode("right_camera_id").string() == "selftest_R", "Right ID changed")
            finally:
                fs.release()
        kalibr = paths[3].read_text(encoding="utf-8")
        _check(all(key in kalibr for key in ("cam0:", "cam1:", "T_cn_cnm1:")),
               "Kalibr chain incomplete")
        _check(all(path.stat().st_size > 0 for path in paths), "Empty export file")
        exports = [path.name for path in paths]
    return {
        "detected_corners": detection.num_corners, "frames": 24,
        "left_rms": float(intr_l.rms_error), "right_rms": float(intr_r.rms_error),
        "stereo_rms": float(result.rms_error), "baseline_m": float(np.linalg.norm(result.T)),
        "exports": exports,
        "coverage": "pinhole intrinsics, stereo, rectification, JSON/OpenCV readback, Kalibr export",
    }


def run_self_test(report_path: str | Path) -> int:
    report = {
        "schema_version": 1, "status": "failed", "stage": "runtime",
        "frozen": bool(getattr(sys, "frozen", False)), "executable": sys.executable,
        "python": sys.version, "platform": platform.platform(),
        "bundle_root": str(getattr(sys, "_MEIPASS", "")),
        "build_commit": os.environ.get("GITHUB_SHA", ""),
    }
    start = time.perf_counter()
    try:
        from .diagnostics import REQUIRED_APIS, opencv_runtime_info

        report["opencv"] = opencv_runtime_info(hash_native=True)
        _check(set(report["opencv"]["distributions"]) == {"opencv-contrib-python"},
               "Expected only opencv-contrib-python; conflicting or missing distribution metadata")
        missing = [name for name in REQUIRED_APIS if not report["opencv"]["apis"][name]]
        _check(not missing, f"Missing required OpenCV APIs: {missing}")
        if report["frozen"]:
            root = Path(report["bundle_root"]).resolve()
            _check(Path(report["opencv"]["native_path"]).is_relative_to(root),
                   "OpenCV native library loaded from outside the bundle")
        report["stage"] = "calibration_and_export"
        report["calibration"] = _exercise_calibration(report)
        report["status"] = "passed"
        report["stage"] = "complete"
    except Exception:
        report["error"] = traceback.format_exc()
    report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    try:
        path = Path(report_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                        encoding="utf-8")
    except Exception:
        if sys.stderr is not None:
            traceback.print_exc()
        return 2
    if sys.stdout is not None:
        print(f"Calibration self-test: {report['status']}; report: {path}")
    return 0 if report["status"] == "passed" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run calibration without GUI, cameras or Open3D")
    parser.add_argument("--self-test-calibration", action="store_true")
    parser.add_argument("--report", default="calibration_self_test.json")
    args = parser.parse_args(argv)
    return run_self_test(args.report)


if __name__ == "__main__":
    raise SystemExit(main())
