"""Noisy low-tilt views expose premature joint-stereo termination."""

from copy import deepcopy

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import DetectionResult
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.stereo import StereoCalibrator


def test_joint_refinement_matches_long_reference_on_noisy_low_tilt_views():
    board = CharucoBoard(CharucoBoardConfig(cols=15, rows=10))
    left_cal, right_cal = IntrinsicCalibrator(board), IntrinsicCalibrator(board)
    stereo_cal = StereoCalibrator(board)
    k = np.array([[463., 0., 647.], [0., 463., 348.], [0., 0., 1.]])
    distortion = np.array([.07, -.057, -.00025, -.0003, .009])
    rotation = cv2.Rodrigues(np.array([.0014, .0033, -.004]))[0]
    translation = np.array([-.05, -.0002, -.0004])
    rng = np.random.default_rng(24)
    ids = np.arange(board.num_corners, dtype=np.int32).reshape(-1, 1)
    for _ in range(16):
        rv = rng.uniform([-.15, -.15, -.05], [.15, .15, .05])
        tv = np.array([-.25, -.18, rng.uniform(.6, 1.)])
        r = cv2.Rodrigues(rv)[0]
        poses = [(rv, tv), (cv2.Rodrigues(rotation @ r)[0], rotation @ tv + translation)]
        detections = []
        for pose in poses:
            corners = cv2.projectPoints(board.object_points, *pose, k, distortion)[0]
            corners += rng.normal(0, .2, corners.shape)
            detections.append(DetectionResult(corners.astype(np.float32), ids, [], None, (1280, 720)))
        assert left_cal.add_detection(detections[0])
        assert right_cal.add_detection(detections[1])
        assert stereo_cal.add_detection_pair(*detections)
    initial = [left_cal.calibrate(), right_cal.calibrate()]

    def reference(criteria=None):
        kwargs = {} if criteria is None else {"criteria": criteria}
        return cv2.stereoCalibrate(
            stereo_cal._obj_points, stereo_cal._img_points_left, stereo_cal._img_points_right,
            initial[0].camera_matrix.copy(), initial[0].dist_coeffs.copy(),
            initial[1].camera_matrix.copy(), initial[1].dist_coeffs.copy(), (1280, 720),
            flags=cv2.CALIB_USE_INTRINSIC_GUESS, **kwargs,
        )

    default = reference()
    long = reference((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-10))
    left, right = deepcopy(initial)
    actual = stereo_cal.calibrate(left, right, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    # This fixture must expose early stopping, rather than pass with an easy exact fit.
    assert actual.rms_error < default[0] - 1e-4
    assert actual.rms_error == pytest.approx(long[0], abs=1e-8)
    for solved, expected in zip((left.camera_matrix, left.dist_coeffs,
                                 right.camera_matrix, right.dist_coeffs, actual.R, actual.T), long[1:7]):
        np.testing.assert_allclose(solved, expected, rtol=1e-6, atol=1e-7)
    # Joint refinement still invalidates old monocular metrics until explicitly recomputed.
    assert np.isnan(left.rms_error) and np.isnan(right.rms_error)
    left_cal.recompute_errors(left)
    right_cal.recompute_errors(right)
    assert np.isfinite([left.rms_error, right.rms_error]).all()
    assert len(left.per_frame_errors) == len(right.per_frame_errors) == 16
