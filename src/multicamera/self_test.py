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


def _exercise_fisheye(report: dict) -> dict:
    import cv2
    import numpy as np

    from .board.charuco_board import CharucoBoard, CharucoBoardConfig
    from .board.detector import DetectionResult
    from .calibration.intrinsic import IntrinsicCalibrator
    from .calibration.models import CameraModel, MultiCameraRig, StereoPairCalibration
    from .calibration.stereo import StereoCalibrator
    from .io.export import export_rig_calibration_bundle

    report["stage"] = "fisheye_observations"
    board = CharucoBoard(CharucoBoardConfig())
    frames, size = 16, (1280, 720)
    left, right = [IntrinsicCalibrator(board, CameraModel.FISHEYE, min_frames=frames)
                   for _ in range(2)]
    stereo = StereoCalibrator(board, CameraModel.FISHEYE, min_frames=frames)
    k = np.array([[900., 0., 640.], [0., 880., 360.], [0., 0., 1.]])
    d = np.array([.03, -.01, .002, -.001])
    expected_r = cv2.Rodrigues(np.array([.01, -.02, .005]))[0]
    expected_t = np.array([[-.12], [0.], [0.]])
    obj = board.object_points.astype(np.float64).reshape(-1, 1, 3)
    ids = np.arange(board.num_corners, dtype=np.int32)
    observations = []
    for i in range(frames):
        rv = np.array([.3 * np.sin(i), .25 * np.cos(i), .03 * i])
        tv = np.array([[-.12 + .01 * i], [-.1 + .015 * np.sin(i)], [.7 + .025 * i]])
        right_rv = cv2.Rodrigues(expected_r @ cv2.Rodrigues(rv)[0])[0]
        keep = ids != i if i % 3 else np.ones(len(ids), dtype=bool)
        detections = []
        for rotation, translation in ((rv, tv), (right_rv, expected_r @ tv + expected_t)):
            points = cv2.fisheye.projectPoints(obj, rotation, translation, k, d)[0]
            detections.append(DetectionResult(
                points[keep].astype(np.float32), ids[keep].reshape(-1, 1), [], None, size,
            ))
        _check(left.add_detection(detections[0]), "Fisheye left observation rejected")
        _check(right.add_detection(detections[1]), "Fisheye right observation rejected")
        _check(stereo.add_detection_pair(*detections), "Fisheye stereo observation rejected")
        observations.append(detections)
    counts = stereo.frame_common_corner_counts
    _check(len(set(counts)) > 1, "Fisheye variable corner counts not exercised")
    report["stage"] = "fisheye_intrinsics_left"
    intr_l = left.calibrate()
    report["stage"] = "fisheye_intrinsics_right"
    intr_r = right.calibrate()
    for intr in (intr_l, intr_r):
        _check(intr.model == CameraModel.FISHEYE, "Fisheye intrinsic model changed")
        _check(np.isfinite(intr.rms_error) and intr.rms_error < .001,
               "Fisheye intrinsic RMS too high")
        np.testing.assert_allclose(intr.camera_matrix, k, atol=.03, rtol=0)
        np.testing.assert_allclose(intr.dist_coeffs.ravel(), d, atol=.003, rtol=0)
        _check(len(intr.per_frame_errors) == frames, "Fisheye per-frame errors missing")

    report["stage"] = "fisheye_variable_corner_guard"
    try:
        stereo.calibrate(intr_l, intr_r)
    except cv2.error as exc:
        raise RuntimeError("Fisheye variable corners reached an OpenCV assertion") from exc
    except ValueError as exc:
        rejection = str(exc)
        _check("跨帧共同角点" in rejection and "鱼眼兼容" in rejection,
               "Fisheye variable-corner rejection is not actionable")
    else:
        raise RuntimeError("Fisheye variable corners were not explicitly rejected")
    report["stage"] = "fisheye_stereo_calibration"
    compatible = stereo.copy_with_common_ids()
    _check(compatible.num_frames == frames, "Fisheye compatibility discarded frames")
    _check(compatible.frame_ids == stereo.frame_ids, "Fisheye compatibility changed frame IDs")
    _check(len(set(compatible.frame_common_corner_counts)) == 1,
           "Fisheye compatibility did not equalize corner counts")
    _check(stereo.frame_common_corner_counts == counts, "Fisheye source observations mutated")
    result = compatible.calibrate(intr_l, intr_r)
    _check(result.model == CameraModel.FISHEYE, "Fisheye stereo model changed")
    _check(np.isfinite(result.rms_error) and result.rms_error < .001, "Fisheye stereo RMS too high")
    np.testing.assert_allclose(result.R, expected_r, atol=1e-4, rtol=0)
    np.testing.assert_allclose(result.T.reshape(3, 1), expected_t, atol=1e-4, rtol=0)

    report["stage"] = "fisheye_rectification"
    for matrix in (result.Q, result.R1, result.R2, result.P1, result.P2,
                   result.map1_left, result.map2_left, result.map1_right, result.map2_right):
        _check(matrix is not None and np.isfinite(matrix).all(), "Fisheye rectification invalid")
    for mapping in (result.map1_left, result.map2_left, result.map1_right, result.map2_right):
        _check(mapping.shape == size[::-1], "Fisheye rectification map size changed")
    # Measure perpendicular to the rectified disparity axis using identical corner IDs.
    axis = 1 if abs(result.P2[0, 3]) > abs(result.P2[1, 3]) else 0
    residuals = []
    for det_l, det_r in observations:
        rectified = [cv2.fisheye.undistortPoints(
            det.charuco_corners.astype(np.float64), intr.camera_matrix, intr.dist_coeffs,
            R=rotation, P=projection,
        ).reshape(-1, 2) for det, intr, rotation, projection in (
            (det_l, intr_l, result.R1, result.P1), (det_r, intr_r, result.R2, result.P2),
        )]
        residuals.extend((rectified[0][:, axis] - rectified[1][:, axis]).tolist())
    rectified_rms = float(np.sqrt(np.mean(np.square(residuals))))
    _check(np.isfinite(rectified_rms) and rectified_rms < .001,
           "Fisheye rectified same-ID residual too high")

    rig = MultiCameraRig(reference_camera="fisheye_L", pairs={
        "fisheye": StereoPairCalibration(
            "fisheye", "fisheye_L", "fisheye_R", intr_l, intr_r, result,
        ),
    })
    report["stage"] = "fisheye_export_and_readback"
    with tempfile.TemporaryDirectory(prefix="multicamera-fisheye-selftest-") as temp:
        paths = export_rig_calibration_bundle(rig, temp)
        restored = MultiCameraRig.from_dict(json.loads(paths[0].read_text(encoding="utf-8")))
        pair = restored.pairs["fisheye"]
        for intr, expected in ((pair.left_intrinsics, intr_l), (pair.right_intrinsics, intr_r)):
            _check(intr.model == CameraModel.FISHEYE, "JSON lost fisheye intrinsic model")
            np.testing.assert_allclose(intr.camera_matrix, expected.camera_matrix)
            np.testing.assert_allclose(intr.dist_coeffs, expected.dist_coeffs)
        _check(pair.stereo.model == CameraModel.FISHEYE, "JSON lost fisheye stereo model")
        np.testing.assert_allclose(pair.stereo.R, result.R)
        np.testing.assert_allclose(pair.stereo.T, result.T)
        np.testing.assert_allclose(pair.stereo.Q, result.Q)
        for path in paths[1:3]:
            fs = cv2.FileStorage(path.read_text(encoding='utf-8'),
                                 cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            try:
                _check(fs.isOpened(), f"Cannot reopen fisheye {path.name}")
                for key in ("camera_model", "left_model", "right_model"):
                    _check(fs.getNode(key).string() == "fisheye", "OpenCV lost fisheye model")
                _check(fs.getNode("distortion_model").string() == "equidistant",
                       "OpenCV lost equidistant distortion model")
                for key, expected in (("M1", intr_l.camera_matrix), ("D1", intr_l.dist_coeffs),
                                      ("M2", intr_r.camera_matrix), ("D2", intr_r.dist_coeffs),
                                      ("R", result.R), ("T", result.T), ("Q", result.Q),
                                      ("R1", result.R1), ("R2", result.R2),
                                      ("P1", result.P1), ("P2", result.P2)):
                    np.testing.assert_allclose(fs.getNode(key).mat(), expected)
            finally:
                fs.release()
        _check(all(path.stat().st_size > 0 for path in paths), "Empty fisheye export file")
        exports = [path.name for path in paths]
    return {
        "model": "fisheye", "frames": frames, "compatible_frames": compatible.num_frames,
        "original_common_corner_counts": counts,
        "compatible_common_corners": compatible.frame_common_corner_counts[0],
        "variable_corner_rejection": rejection,
        "left_rms": float(intr_l.rms_error), "right_rms": float(intr_r.rms_error),
        "stereo_rms": float(result.rms_error), "baseline_m": float(np.linalg.norm(result.T)),
        "rectified_rms_px": rectified_rms, "rectified_error_axis": "xy"[axis],
        "rectified_corner_count": len(residuals), "exports": exports,
        "coverage": "fisheye intrinsics, explicit common-ID stereo, rectification, JSON/OpenCV readback",
    }


def _exercise_health(report: dict) -> dict:
    import copy
    import cv2
    import numpy as np
    from .board.charuco_board import CharucoBoard, CharucoBoardConfig
    from .board.detector import DetectionResult
    from .calibration.models import CameraIntrinsics
    from .calibration.stereo import StereoCalibrator
    from .calibration.health import check_intrinsics, check_stereo_health

    report['stage'] = 'rational_fidelity_and_health'
    board = CharucoBoard(CharucoBoardConfig())
    matrix = np.array([[800., 0., 640.], [0., 800., 360.], [0., 0., 1.]])
    distortion = np.array([.05, -.02, 0., 0., 0., .01, .002, 0.])
    left = CameraIntrinsics(matrix, distortion, (1280, 720))
    right = copy.deepcopy(left)
    stereo = StereoCalibrator(board, min_frames=6)
    ids = np.arange(board.num_corners, dtype=np.int32).reshape(-1, 1)
    for i in range(8):
        rv = np.array([.2*np.sin(i), .2*np.cos(i), .02*i])
        tv = np.array([-.2, -.15, 1.+.05*i])
        detections = []
        for shift in (0., -.05):
            points = cv2.projectPoints(board.object_points, rv, tv+[shift, 0., 0.],
                                       matrix, distortion)[0].astype(np.float32)
            detections.append(DetectionResult(points, ids, [], None, (1280, 720)))
        stereo.add_detection_pair(*detections)
    result = stereo.calibrate(left, right)
    _check(left.dist_coeffs.size >= 8 and left.dist_coeffs.ravel()[5] == distortion[5],
           'Rational denominator lost during stereo solve')
    _check(np.isfinite(left.rms_error) and result.rms_error < .001,
           'Rational fit metrics or fixed-intrinsic semantics lost')
    good = check_stereo_health(left, right, result)
    _check(good['status'] == 'ok', 'Healthy rational model rejected')
    bad = copy.deepcopy(left)
    bad.camera_matrix[:2, :2] = [[463., 0.], [0., 463.]]
    bad.dist_coeffs = np.array([.058, -.034, .0002, .0001, -.0004])
    _check(check_intrinsics(bad)['status'] == 'failed', 'Folded distortion not rejected')
    zoomed = copy.deepcopy(result)
    for projection in (zoomed.P1, zoomed.P2):
        projection[0, 0] = projection[1, 1] = 8000.
    _check(check_stereo_health(left, right, zoomed)['status'] == 'failed',
           'Overzoomed rectification not rejected')
    no_baseline = copy.deepcopy(result)
    no_baseline.P2[:, 3] = 0.
    _check(check_stereo_health(left, right, no_baseline)['status'] == 'failed',
           'Projection baseline inconsistent with R/T not rejected')
    no_q = copy.deepcopy(result)
    no_q.Q[:] = 0.
    _check(check_stereo_health(left, right, no_q)['status'] == 'failed',
           'Zero Q not rejected')
    wrong_scale = copy.deepcopy(result)
    wrong_scale.Q[3] *= 2.
    _check(check_stereo_health(left, right, wrong_scale)['status'] == 'failed',
           'Wrong metric reconstruction scale not rejected')
    return {'rational_stereo_rms': float(result.rms_error), 'health': good,
            'invalid_lens_rejected': True, 'overzoom_rejected': True,
            'rational_denominator_preserved': True, 'projection_baseline_rejected': True,
            'zero_q_rejected': True, 'wrong_depth_scale_rejected': True}


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
    # The synthetic pinhole lens is distortion-free; keep the Kalibr-incompatible k3 fixed.
    intr_l = left.calibrate(flags=cv2.CALIB_FIX_K3)
    report["stage"] = "intrinsics_right"
    intr_r = right.calibrate(flags=cv2.CALIB_FIX_K3)
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
        directory = Path(temp) / '标定 会话'
        paths = export_rig_calibration_bundle(rig, directory, stem='标定 文件')
        restored = MultiCameraRig.from_dict(json.loads(paths[0].read_text(encoding="utf-8")))
        np.testing.assert_allclose(restored.pairs["selftest"].stereo.T, result.T)
        for path in paths[1:3]:
            fs = cv2.FileStorage(path.read_text(encoding='utf-8'),
                                 cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
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
        report['stage'] = 'unicode_image_roundtrip'
        from .io.image_io import write_png

        images = {
            '彩色 原图.png': np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3),
            '灰度 原图.png': np.arange(16 * 24, dtype=np.uint16).reshape(16, 24) * 100,
        }
        for name, image in images.items():
            path = directory / name
            write_png(path, image)
            restored_image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8),
                                           cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(restored_image, image)
        unicode_io = dict(exports=exports, images=list(images), pixel_exact=True)
    return {
        "detected_corners": detection.num_corners, "frames": 24,
        "left_rms": float(intr_l.rms_error), "right_rms": float(intr_r.rms_error),
        "stereo_rms": float(result.rms_error), "baseline_m": float(np.linalg.norm(result.T)),
        "exports": exports,
        "unicode_file_io": unicode_io,
        "coverage": "pinhole intrinsics, stereo, rectification, JSON/OpenCV readback, Kalibr export",
        "fisheye": _exercise_fisheye(report),
        "health_checks": _exercise_health(report),
    }


def run_self_test(report_path: str | Path) -> int:
    from .build_info import build_identity

    report = {
        "schema_version": 1, "status": "failed", "stage": "runtime",
        "frozen": bool(getattr(sys, "frozen", False)), "executable": sys.executable,
        "python": sys.version, "platform": platform.platform(),
        "bundle_root": str(getattr(sys, "_MEIPASS", "")),
        "build_commit": os.environ.get("GITHUB_SHA", ""),
        "build_identity": build_identity(),
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
