"""Calibrate partial ChArUco observations without the deprecated ArUco binding."""

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import DetectionResult
from multicamera.calibration.intrinsic import IntrinsicCalibrator


@pytest.fixture
def observations():
    board = CharucoBoard(CharucoBoardConfig())
    cal = IntrinsicCalibrator(board, min_frames=20)
    k = np.array([[900., 0., 640.], [0., 880., 360.], [0., 0., 1.]])
    rng = np.random.default_rng(42)
    for i in range(24):
        ids = rng.permutation(board.num_corners)[:board.num_corners - (i % 4) * 2]
        rv = np.array([.3 * np.sin(i), .2 * np.cos(i), .03 * i])
        tv = np.array([[-.3 + .015 * i], [-.2 + .01 * np.sin(i)], [1.2 + .04 * i]])
        corners, _ = cv2.projectPoints(board.object_points[ids], rv, tv, k, np.zeros(5))
        assert cal.add_detection(DetectionResult(
            corners.astype(np.float32), ids.astype(np.int32).reshape(-1, 1),
            [], None, (1280, 720),
        ))
    return cal, k


@pytest.mark.parametrize("flags", [0, cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3])
def test_calibrates_without_legacy_binding(observations, monkeypatch, flags):
    cal, expected_k = observations
    monkeypatch.delattr(cv2.aruco, "calibrateCameraCharuco", raising=False)
    intr = cal.calibrate(flags=flags)
    assert intr.image_size == (1280, 720)
    assert intr.rms_error < .001
    assert len(intr.per_frame_errors) == 24
    assert max(intr.per_frame_errors) < .001
    assert len(intr.rvecs) == len(intr.tvecs) == 24
    np.testing.assert_allclose(intr.camera_matrix, expected_k, atol=.02)
    if flags:
        np.testing.assert_array_equal(intr.dist_coeffs.flatten()[2:5], np.zeros(3))


def test_matches_legacy_result_where_available(observations):
    legacy = getattr(cv2.aruco, "calibrateCameraCharuco", None)
    if legacy is None:
        pytest.skip("Legacy API is absent in this OpenCV build")
    cal, _ = observations
    rms, k, dist, _, _ = legacy(
        cal._all_corners, cal._all_ids, cal.board.board, (1280, 720), None, None,
    )
    intr = cal.calibrate()
    np.testing.assert_allclose(intr.camera_matrix, k, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(intr.dist_coeffs, dist, atol=1e-5)
    assert intr.rms_error == pytest.approx(rms, abs=1e-7)
