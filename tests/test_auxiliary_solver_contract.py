"""Auxiliary solver contracts, using known AUX-to-RGB camera geometry."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from multicamera.calibration import auxiliary
from multicamera.calibration.auxiliary import AuxIntrinsicCalibrator, CrossModalExtrinsicCalibrator
from multicamera.calibration.models import CameraIntrinsics, CameraModel
from multicamera.calibration.planar import (
    PatternBoardConfig, PatternType, PlanarDetectionResult, PlanarPatternDetector,
)


BOARD = PatternBoardConfig(cols=6, rows=4, square_size=.08)
SIZE = (960, 720)
K = np.array([[430., 0., 480.], [0., 440., 360.], [0., 0., 1.]])
OBJECT = PlanarPatternDetector._make_object_points(BOARD).astype(np.float64)


def intrinsics(model=CameraModel.PINHOLE, size=SIZE):
    dist = (np.array([-.12, .025, -.004, .0005]) if model == CameraModel.FISHEYE
            else np.array([-.08, .015, .001, -.002, .002]))
    return CameraIntrinsics(K.copy(), dist, size, model)


def transform(rvec, tvec):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    result[:3, 3] = tvec
    return result


EXPECTED = transform([.04, -.13, .025], [.16, -.025, .035])


def detection(intr, pose):
    rvec = cv2.Rodrigues(pose[:3, :3])[0]
    tvec = pose[:3, 3].reshape(3, 1).copy()
    if intr.model == CameraModel.FISHEYE:
        image = cv2.fisheye.projectPoints(
            OBJECT.reshape(-1, 1, 3), rvec, tvec, intr.camera_matrix, intr.dist_coeffs,
        )[0]
    else:
        image = cv2.projectPoints(OBJECT, rvec, tvec, intr.camera_matrix, intr.dist_coeffs)[0]
    return PlanarDetectionResult(True, OBJECT.copy(), image.reshape(-1, 2),
                                 intr.image_size, BOARD.pattern_type)


def pair(rgb, aux, index=0):
    aux_pose = transform([.12 + index * .05, -.21 + index * .03, .05 - index * .02],
                         [-.32 + index * .12, -.16 + index * .04, .85 + index * .09])
    return detection(rgb, EXPECTED @ aux_pose), detection(aux, aux_pose)


@pytest.mark.parametrize('rgb_model', list(CameraModel))
@pytest.mark.parametrize('aux_model', list(CameraModel))
def test_synthetic_external_transform_maps_aux_to_rgb(rgb_model, aux_model):
    rgb, aux = intrinsics(rgb_model), intrinsics(aux_model, (1000, 760))
    cal = CrossModalExtrinsicCalibrator(BOARD)
    for i in range(5):
        assert cal.add_detection_pair(*pair(rgb, aux, i))
    result = cal.calibrate(rgb, aux)
    np.testing.assert_allclose(result.T_rgb_left_aux, EXPECTED, atol=2e-7)
    point_aux = np.array([.1, .2, 1.5, 1.])
    np.testing.assert_allclose(result.T_rgb_left_aux @ point_aux, EXPECTED @ point_aux,
                               atol=2e-7)
    assert result.frame_count == 5
    assert result.rms_error < 1e-5
    np.testing.assert_allclose(result.T_rgb_left_aux[3], [0, 0, 0, 1])


@pytest.mark.parametrize('model', list(CameraModel))
def test_detector_float32_column_points_with_pixel_noise(model):
    rgb, aux = intrinsics(), intrinsics(model)
    cal = CrossModalExtrinsicCalibrator(BOARD)
    rng = np.random.default_rng(42)
    for i in range(10):
        detections = pair(rgb, aux, i)
        for det in detections:
            det.object_points = det.object_points.astype(np.float32).reshape(-1, 1, 3)
            det.image_points = (det.image_points + rng.normal(0, .08, (len(OBJECT), 2)))
            det.image_points = det.image_points.astype(np.float32).reshape(-1, 1, 2)
        assert cal.add_detection_pair(*detections)
    result = cal.calibrate(rgb, aux)
    np.testing.assert_allclose(result.T_rgb_left_aux[:3, :3], EXPECTED[:3, :3], atol=.004)
    np.testing.assert_allclose(result.T_rgb_left_aux[:3, 3], EXPECTED[:3, 3], atol=.004)
    assert .04 < result.rms_error < .2


@pytest.mark.parametrize('model', list(CameraModel))
def test_errors_are_pixel_rms_per_point(model):
    intr = intrinsics(model)
    pose = transform([.1, -.2, .03], [-.2, -.1, 1.])
    det = detection(intr, pose)
    det.image_points += [3., 4.]
    rvec, tvec = cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3].copy()
    assert auxiliary._reprojection_error(det, rvec, tvec, intr) == pytest.approx(5.)
    error_fn = (auxiliary._per_frame_errors_fisheye if model == CameraModel.FISHEYE
                else auxiliary._per_frame_errors)
    assert error_fn([OBJECT.reshape(-1, 1, 3)], [det.image_points], [rvec], [tvec],
                    intr.camera_matrix, intr.dist_coeffs) == pytest.approx([5.])


def test_frame_rms_pools_both_cameras_not_their_arithmetic_mean(monkeypatch):
    rgb, aux = intrinsics(), intrinsics()
    cal = CrossModalExtrinsicCalibrator(BOARD, min_frames=1)
    assert cal.add_detection_pair(*pair(rgb, aux))
    errors = iter([5., 10.])
    monkeypatch.setattr(auxiliary, '_reprojection_error', lambda *args: next(errors))
    result = cal.calibrate(rgb, aux)
    assert result.per_frame_errors == pytest.approx([np.sqrt(62.5)])
    assert result.rms_error == pytest.approx(np.sqrt(62.5))


def corrupt(det, kind):
    if kind == 'nan_image':
        det.image_points[0, 0] = np.nan
    elif kind == 'inf_object':
        det.object_points[0, 0] = np.inf
    elif kind == 'image_count':
        det.image_points = det.image_points[:-1]
    elif kind == 'object_count':
        det.object_points = det.object_points[:-1]
    elif kind == 'image_shape':
        det.image_points = det.image_points.reshape(2, -1)
    elif kind == 'object_shape':
        det.object_points = det.object_points.reshape(3, -1)
    elif kind == 'collapsed_object':
        det.object_points[:] = 0
    elif kind == 'nonplanar':
        det.object_points[0, 2] = .01
    elif kind == 'wrong_scale':
        det.object_points *= 1000
    elif kind == 'object_order':
        det.object_points = det.object_points[::-1].copy()
    elif kind == 'collapsed_image':
        det.image_points[:] = 1
    elif kind == 'collinear_image':
        det.image_points[:, 1] = 10
    elif kind == 'pattern':
        det.pattern_type = PatternType.CIRCLES_GRID
    elif kind == 'invalid':
        det.valid = False
    elif kind == 'none_image':
        det.image_points = None
    elif kind == 'none_object':
        det.object_points = None
    else:
        det.image_size = {'zero_size': (0, 720), 'float_size': (960., 720),
                          'bool_size': (True, 720), 'short_size': (960,)}[kind]
    return det


@pytest.mark.parametrize('kind', [
    'nan_image', 'inf_object', 'image_count', 'object_count', 'image_shape', 'object_shape',
    'collapsed_object', 'nonplanar', 'wrong_scale', 'object_order', 'collapsed_image',
    'collinear_image', 'pattern', 'invalid', 'none_image', 'none_object', 'zero_size',
    'float_size', 'bool_size', 'short_size',
])
@pytest.mark.parametrize('side', [0, 1])
def test_bad_detections_rejected_without_changing_state(kind, side):
    rgb, aux = intrinsics(), intrinsics()
    detections = list(pair(rgb, aux))
    detections[side] = corrupt(detections[side], kind)
    mono = AuxIntrinsicCalibrator(BOARD)
    cal = CrossModalExtrinsicCalibrator(BOARD)
    assert not mono.add_detection(detections[side])
    assert not cal.add_detection_pair(*detections)
    assert mono.num_frames == cal.num_frames == 0
    assert mono._image_size is None
    assert mono.add_detection(pair(rgb, aux)[side])
    assert cal.add_detection_pair(*pair(rgb, aux))


@pytest.mark.parametrize('side', [0, 1])
def test_resolution_changes_rejected_per_camera(side):
    rgb, aux = intrinsics(), intrinsics(size=(1000, 760))
    detections = list(pair(rgb, aux))
    mono = AuxIntrinsicCalibrator(BOARD)
    cal = CrossModalExtrinsicCalibrator(BOARD)
    assert mono.add_detection(detections[side])
    assert cal.add_detection_pair(*detections)
    detections[side] = replace(detections[side], image_size=(320, 240))
    assert not mono.add_detection(detections[side])
    assert not cal.add_detection_pair(*detections)
    assert mono.num_frames == cal.num_frames == 1


@pytest.mark.parametrize('field,value', [
    ('image_size', (320, 240)), ('image_size', (960., 720)),
    ('camera_matrix', np.eye(2)), ('camera_matrix', np.full((3, 3), np.nan)),
    ('camera_matrix', np.diag([0., 440., 1.])),
    ('camera_matrix', np.diag([430., -440., 1.])),
    ('camera_matrix', np.diag([430., 440., 2.])),
    ('dist_coeffs', np.zeros(3)), ('dist_coeffs', np.zeros((2, 2))),
    ('dist_coeffs', np.full(5, np.inf)), ('model', 'unsupported'),
])
@pytest.mark.parametrize('side', [0, 1])
def test_bad_intrinsics_raise_before_pnp(monkeypatch, field, value, side):
    rgb, aux = intrinsics(), intrinsics()
    cal = CrossModalExtrinsicCalibrator(BOARD, min_frames=1)
    assert cal.add_detection_pair(*pair(rgb, aux))
    setattr((rgb, aux)[side], field, value)

    def unexpected(*args, **kwargs):
        pytest.fail('Invalid intrinsics reached solvePnP')

    monkeypatch.setattr(cv2, 'solvePnP', unexpected)
    with pytest.raises(ValueError):
        cal.calibrate(rgb, aux)


def test_fisheye_requires_exactly_four_distortion_coefficients():
    rgb, aux = intrinsics(), intrinsics(CameraModel.FISHEYE)
    cal = CrossModalExtrinsicCalibrator(BOARD, min_frames=1)
    assert cal.add_detection_pair(*pair(rgb, aux))
    aux.dist_coeffs = np.zeros(5)
    with pytest.raises(ValueError, match='distortion'):
        cal.calibrate(rgb, aux)


def test_observations_are_snapshotted_and_intrinsics_remain_unchanged():
    rgb, aux = intrinsics(), intrinsics(CameraModel.FISHEYE)
    detections = pair(rgb, aux)
    cal = CrossModalExtrinsicCalibrator(BOARD, min_frames=1)
    mono = AuxIntrinsicCalibrator(BOARD)
    assert mono.add_detection(detections[1])
    assert cal.add_detection_pair(*detections)
    originals = [(i.camera_matrix.copy(), i.dist_coeffs.copy()) for i in (rgb, aux)]
    saved_image = detections[1].image_points.copy()
    for det in detections:
        det.image_points[:] = np.nan
        det.object_points[:] = 0
    np.testing.assert_array_equal(mono._image_points[0], saved_image)
    result = cal.calibrate(rgb, aux)
    np.testing.assert_allclose(result.T_rgb_left_aux, EXPECTED, atol=2e-7)
    for intr, (k, dist) in zip((rgb, aux), originals):
        np.testing.assert_array_equal(intr.camera_matrix, k)
        np.testing.assert_array_equal(intr.dist_coeffs, dist)


@pytest.mark.parametrize('failure', ['false', 'nan', 'behind', 'opencv'])
def test_failed_pose_skipped_and_minimum_valid_frames_enforced(monkeypatch, failure):
    rgb, aux = intrinsics(), intrinsics()
    cal = CrossModalExtrinsicCalibrator(BOARD, min_frames=3)
    for i in range(4):
        assert cal.add_detection_pair(*pair(rgb, aux, i))
    original = cv2.solvePnP
    calls = 0

    def solve(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure == 'opencv':
                raise cv2.error('synthetic pose failure')
            return (failure != 'false', np.full((3, 1), np.nan) if failure == 'nan'
                    else np.zeros((3, 1)), np.array([[0.], [0.], [-1.]]))
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, 'solvePnP', solve)
    result = cal.calibrate(rgb, aux)
    assert result.frame_count == 3
    np.testing.assert_allclose(result.T_rgb_left_aux, EXPECTED, atol=2e-7)
    cal.min_frames = 4
    calls = 0
    with pytest.raises(ValueError, match='valid solvePnP'):
        cal.calibrate(rgb, aux)


def test_symmetric_board_order_is_not_silently_flipped():
    rgb, aux = intrinsics(), intrinsics()
    rgb_det, aux_det = pair(rgb, aux)
    aux_det.image_points = aux_det.image_points[::-1].copy()
    cal = CrossModalExtrinsicCalibrator(BOARD, min_frames=1)
    assert cal.add_detection_pair(rgb_det, aux_det)
    result = cal.calibrate(rgb, aux)
    # Reversing symmetric-board corners yields a different valid board pose.
    # A tiny reprojection error cannot certify cross-modal correspondence.
    assert result.rms_error < 1e-5
    assert not np.allclose(result.T_rgb_left_aux, EXPECTED, atol=.01)


@pytest.mark.parametrize('model', list(CameraModel))
def test_mono_synthetic_calibration_still_works(model):
    intr = intrinsics(model)
    cal = AuxIntrinsicCalibrator(BOARD, model, min_frames=10)
    for i in range(12):
        pose = transform([-.25 + .05 * i, -.2 + .09 * (i % 5), -.1 + .03 * i],
                         [-.25 + .04 * (i % 4), -.17 + .03 * (i % 3), .9 + .03 * i])
        assert cal.add_detection(detection(intr, pose))
    result = cal.calibrate()
    assert result.per_frame_error_metric == 'euclidean_rms_px'
    np.testing.assert_allclose(result.camera_matrix, K, atol=.02)
    assert result.rms_error < 1e-3
    assert max(result.per_frame_errors) < 1e-3
