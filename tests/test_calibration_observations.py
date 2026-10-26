"""Input ownership, image geometry and reprojection-error contracts."""

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import DetectionResult
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.models import CameraIntrinsics, CameraModel


@pytest.fixture
def sample():
    board = CharucoBoard(CharucoBoardConfig())
    k = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    rv, tv = np.array([.1, -.2, .03]), np.array([-.3, -.2, 1.3])
    corners = cv2.projectPoints(board.object_points, rv, tv, k, np.zeros(5))[0].astype(np.float32)
    det = DetectionResult(corners, np.arange(board.num_corners, dtype=np.int32)[:, None], [], None, (640, 480))
    return IntrinsicCalibrator(board, min_frames=1), det, k, rv, tv


def test_mixed_resolution_rejected_without_corrupting_dataset(sample):
    cal, det, *_ = sample
    assert cal.add_detection(det)
    revision = cal.data_revision
    det.image_size = (320, 240)
    assert not cal.add_detection(det)
    assert '分辨率' in cal.last_rejection_reason
    assert cal.num_frames == 1 and cal._image_size == (640, 480)
    assert cal.data_revision == revision


@pytest.mark.parametrize('defect', ['duplicate', 'negative', 'out_of_range', 'float_ids', 'nan', 'length', 'shape', 'size'])
def test_malformed_observations_are_not_accepted(sample, defect):
    cal, det, *_ = sample
    if defect == 'duplicate':
        det.charuco_ids[1] = det.charuco_ids[0]
    elif defect == 'negative':
        det.charuco_ids[0] = -1
    elif defect == 'out_of_range':
        det.charuco_ids[-1] = cal.board.num_corners
    elif defect == 'float_ids':
        det.charuco_ids = det.charuco_ids.astype(float)
    elif defect == 'nan':
        det.charuco_corners[0] = np.nan
    elif defect == 'length':
        det.charuco_corners = det.charuco_corners[:-1]
    elif defect == 'shape':
        det.charuco_corners = det.charuco_corners.reshape(-1)
    else:
        det.image_size = (0, 480)
    assert not cal.add_detection(det)
    assert cal.last_rejection_reason
    assert cal.num_frames == 0 and cal.data_revision == 0


def test_observation_arrays_owned_and_images_keep_index_alignment(sample):
    cal, det, *_ = sample
    original = det.charuco_corners.copy()
    assert cal.add_detection(det)
    image = np.full((480, 640, 3), 123, np.uint8)
    assert cal.add_detection(det, image)
    det.charuco_corners[:] = 0
    det.charuco_ids[:] = 0
    np.testing.assert_array_equal(cal._all_corners[0], original)
    assert cal.stored_images[0] is None
    assert np.all(cal.stored_images[1] == 123)
    clone = cal.copy_with_indices([0, 1])
    assert clone.stored_images[0] is None and np.all(clone.stored_images[1] == 123)
    clone.pop_last_frame()
    assert clone.num_frames == len(clone.stored_images) == 1


def test_image_and_detection_sizes_must_match(sample):
    cal, det, *_ = sample
    assert not cal.add_detection(det, np.zeros((240, 320, 3), np.uint8))
    assert cal.num_frames == 0 and '原图' in cal.last_rejection_reason


@pytest.mark.parametrize('model', list(CameraModel))
def test_frame_rms_uses_euclidean_pixel_distance(sample, model):
    cal, det, k, rv, tv = sample
    obj = cal.board.object_points.astype(np.float64).reshape(-1, 1, 3)
    dist = np.zeros(4 if model == CameraModel.FISHEYE else 5)
    if model == CameraModel.FISHEYE:
        projected = cv2.fisheye.projectPoints(obj, rv, tv, k, dist)[0]
    else:
        projected = cv2.projectPoints(obj, rv, tv, k, dist)[0]
    det.charuco_corners = (projected + np.array([3., 4.])).astype(np.float32)
    assert cal.add_detection(det)
    if model == CameraModel.FISHEYE:
        errors = cal._compute_per_frame_errors_fisheye(k, dist, [rv], [tv], [obj], [det.charuco_corners])
    else:
        errors = cal._compute_per_frame_errors_pinhole(k, dist, [rv], [tv])
    assert errors == pytest.approx([5.], abs=1e-4)


@pytest.mark.parametrize('model', list(CameraModel))
def test_recomputed_errors_belong_to_refined_intrinsics(sample, model):
    cal, det, k, rv, tv = sample
    dist = np.zeros(4 if model == CameraModel.FISHEYE else 5)
    if model == CameraModel.FISHEYE:
        det.charuco_corners = cv2.fisheye.projectPoints(
            cal.board.object_points.astype(np.float64).reshape(-1, 1, 3), rv, tv, k, dist,
        )[0].astype(np.float32)
    assert cal.add_detection(det)
    intr = CameraIntrinsics(k.copy(), dist, det.image_size, model, rms_error=99., per_frame_errors=[99.])
    cal.recompute_errors(intr)
    assert intr.rms_error < .001 and max(intr.per_frame_errors) < .001
    assert len(intr.rvecs) == len(intr.tvecs) == 1
