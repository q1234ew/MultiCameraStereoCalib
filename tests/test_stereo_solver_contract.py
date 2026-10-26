"""Stereo solver outputs, mutation boundaries, and input contracts."""

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import DetectionResult
from multicamera.calibration.models import CameraIntrinsics, CameraModel
from multicamera.calibration.stereo import StereoCalibrator


SIZE = (640, 480)
K = np.array([[500., 0., 320.], [0., 510., 240.], [0., 0., 1.]])


@pytest.fixture
def board():
    return CharucoBoard(CharucoBoardConfig(cols=8, rows=6))


def intrinsics(model=CameraModel.PINHOLE, count=None):
    count = count or (4 if model == CameraModel.FISHEYE else 5)
    return CameraIntrinsics(
        K.copy(), np.zeros(count), SIZE, model, rms_error=.2,
        per_frame_errors=[.2], rvecs=[np.zeros(3)], tvecs=[np.ones(3)],
    )


def detection(board, size=SIZE):
    points, _ = cv2.projectPoints(
        board.object_points, np.zeros(3), np.array([-.1, -.1, 1.]), K, None,
    )
    return DetectionResult(
        points.astype(np.float32),
        np.arange(board.num_corners, dtype=np.int32).reshape(-1, 1),
        [], None, size,
    )


def ready_calibrator(board, model=CameraModel.PINHOLE):
    cal = StereoCalibrator(board, model, min_frames=1)
    det = detection(board)
    assert cal.add_detection_pair(det, det)
    return cal


def install_solver(monkeypatch, model, callback):
    namespace = cv2.fisheye if model == CameraModel.FISHEYE else cv2
    monkeypatch.setattr(namespace, 'stereoCalibrate', callback)


def solver_output(model, k1, d1, k2, d2):
    r = np.eye(3)
    t = np.array([[-.12], [0.], [0.]])
    result = (.1, k1, d1, k2, d2, r, t)
    if model == CameraModel.PINHOLE:
        result += (StereoCalibrator._essential_from_rt(r, t), np.zeros((3, 3)))
    return result


def assert_rectification_matches(cal, result, left, right):
    if cal.model == CameraModel.FISHEYE:
        r1, r2, p1, p2, q = cv2.fisheye.stereoRectify(
            left.camera_matrix, left.dist_coeffs.reshape(4, 1),
            right.camera_matrix, right.dist_coeffs.reshape(4, 1),
            SIZE, result.R, result.T, flags=cv2.CALIB_ZERO_DISPARITY,
        )
        map_fn = cv2.fisheye.initUndistortRectifyMap
    else:
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            left.camera_matrix, left.dist_coeffs,
            right.camera_matrix, right.dist_coeffs,
            SIZE, result.R, result.T, alpha=0,
        )
        map_fn = cv2.initUndistortRectifyMap
    for name, expected in [('R1', r1), ('R2', r2), ('P1', p1), ('P2', p2), ('Q', q)]:
        np.testing.assert_allclose(getattr(result, name), expected)
    for side, intr, r, p in [('left', left, r1, p1), ('right', right, r2, p2)]:
        maps = map_fn(intr.camera_matrix, intr.dist_coeffs, r, p, SIZE, cv2.CV_32FC1)
        for index, expected in enumerate(maps, 1):
            np.testing.assert_array_equal(getattr(result, f'map{index}_{side}'), expected)


@pytest.mark.parametrize('model', list(CameraModel))
def test_joint_refine_commits_returned_intrinsics_and_uses_them_for_maps(board, monkeypatch, model):
    cal = ready_calibrator(board, model)
    left, right = intrinsics(model), intrinsics(model)
    old_k, old_d = left.camera_matrix, left.dist_coeffs
    k1, k2 = K.copy(), K.copy()
    k1[0, 0] += 12
    k2[1, 1] += 7
    count = 4 if model == CameraModel.FISHEYE else 8
    d1, d2 = np.zeros((count, 1)), np.zeros((count, 1))
    d1[0], d2[0] = .02, -.01
    flag = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS if model == CameraModel.FISHEYE
            else cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL)

    def solve(*args, **kwargs):
        assert kwargs['flags'] == flag
        assert not np.shares_memory(args[3], left.camera_matrix)
        assert not np.shares_memory(args[4], left.dist_coeffs)
        # Returned arrays deliberately differ from the mutable input arrays.
        args[3][:] = 0
        args[4][:] = 99
        return solver_output(model, k1, d1, k2, d2)

    install_solver(monkeypatch, model, solve)
    result = cal.calibrate(left, right, flags=flag)
    for actual, expected in [(left.camera_matrix, k1), (left.dist_coeffs, d1),
                             (right.camera_matrix, k2), (right.dist_coeffs, d2)]:
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)
    np.testing.assert_array_equal(old_k, K)
    np.testing.assert_array_equal(old_d, np.zeros_like(old_d))
    for intr in (left, right):
        assert np.isnan(intr.rms_error)
        assert intr.per_frame_errors is intr.rvecs is intr.tvecs is None
    assert result.rms_error == .1
    assert_rectification_matches(cal, result, left, right)


@pytest.mark.parametrize('model', list(CameraModel))
def test_fixed_intrinsics_keep_fit_metrics_and_input_buffers(board, monkeypatch, model):
    cal = ready_calibrator(board, model)
    left, right = intrinsics(model), intrinsics(model)
    old_k, old_d, old_poses = left.camera_matrix, left.dist_coeffs, left.rvecs
    flag = (cv2.fisheye.CALIB_FIX_INTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
            if model == CameraModel.FISHEYE else cv2.CALIB_FIX_INTRINSIC)

    def solve(*args, **kwargs):
        assert kwargs['flags'] == flag
        for received, original in zip(args[3:7],
                                      [left.camera_matrix, left.dist_coeffs,
                                       right.camera_matrix, right.dist_coeffs]):
            assert not np.shares_memory(received, original)
        return solver_output(model, *args[3:7])

    install_solver(monkeypatch, model, solve)
    cal.calibrate(left, right)
    np.testing.assert_array_equal(old_k, K)
    np.testing.assert_array_equal(old_d, np.zeros_like(old_d))
    assert left.rms_error == .2
    assert left.per_frame_errors == [.2]
    assert left.rvecs is old_poses


@pytest.mark.parametrize('model', list(CameraModel))
@pytest.mark.parametrize('failure', ['solver', 'rectification'])
def test_failure_never_partially_updates_intrinsics(board, monkeypatch, model, failure):
    cal = ready_calibrator(board, model)
    left, right = intrinsics(model), intrinsics(model)
    old = [(i.camera_matrix, i.dist_coeffs, i.per_frame_errors, i.rvecs) for i in (left, right)]

    def solve(*args, **kwargs):
        args[3][0, 0] = 555
        args[4].flat[0] = .05
        if failure == 'solver':
            raise RuntimeError('solver failed')
        return solver_output(model, *args[3:7])

    def fail_rectify(*args):
        raise RuntimeError('rectification failed')

    install_solver(monkeypatch, model, solve)
    if failure == 'rectification':
        method = '_rectify_fisheye' if model == CameraModel.FISHEYE else '_rectify_pinhole'
        monkeypatch.setattr(cal, method, fail_rectify)
    with pytest.raises(RuntimeError):
        cal.calibrate(left, right)
    for intr, (k, d, errors, poses) in zip((left, right), old):
        assert intr.camera_matrix is k and intr.dist_coeffs is d
        np.testing.assert_array_equal(k, K)
        np.testing.assert_array_equal(d, np.zeros_like(d))
        assert intr.per_frame_errors is errors and intr.rvecs is poses
        assert intr.rms_error == .2


@pytest.mark.parametrize('flag', [cv2.fisheye.CALIB_FIX_INTRINSIC,
                                 cv2.fisheye.CALIB_USE_INTRINSIC_GUESS,
                                 cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                                 | cv2.fisheye.CALIB_FIX_SKEW])
@pytest.mark.parametrize('with_poses', [False, True])
def test_explicit_fisheye_flags_are_forwarded_unchanged(board, monkeypatch, flag, with_poses):
    cal = ready_calibrator(board, CameraModel.FISHEYE)

    def solve(*args, **kwargs):
        assert kwargs['flags'] == flag
        result = solver_output(CameraModel.FISHEYE, *args[3:7])
        return result + ([], []) if with_poses else result

    install_solver(monkeypatch, CameraModel.FISHEYE, solve)
    cal.calibrate(intrinsics(CameraModel.FISHEYE), intrinsics(CameraModel.FISHEYE), flag)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('field,value', [
    ('model', CameraModel.FISHEYE), ('image_size', (320, 240)),
    ('image_size', (0, 480)), ('image_size', (640.0, 480)),
    ('camera_matrix', np.eye(2)),
    ('camera_matrix', np.array([[0., 0., 320.], [0., 500., 240.], [0., 0., 1.]])),
    ('camera_matrix', np.array([[500., 0., np.nan], [0., 500., 240.], [0., 0., 1.]])),
    ('camera_matrix', np.array([[500., 0., 320.], [0., 500., 240.], [0., 1., 1.]])),
    ('dist_coeffs', np.zeros(6)), ('dist_coeffs', np.zeros((2, 2))),
    ('dist_coeffs', np.array([0., 0., 0., 0., np.inf])),
])
def test_invalid_intrinsics_rejected_before_solver(board, monkeypatch, side, field, value):
    cal = ready_calibrator(board)
    left, right = intrinsics(), intrinsics()
    setattr(left if side == 'left' else right, field, value)

    def unexpected(*args, **kwargs):
        pytest.fail('Invalid input reached OpenCV')

    install_solver(monkeypatch, CameraModel.PINHOLE, unexpected)
    with pytest.raises(ValueError):
        cal.calibrate(left, right)


def test_fisheye_requires_four_distortion_coefficients(board):
    cal = ready_calibrator(board, CameraModel.FISHEYE)
    with pytest.raises(ValueError, match='distortion'):
        cal.calibrate(intrinsics(CameraModel.FISHEYE, 5), intrinsics(CameraModel.FISHEYE))


def test_observation_size_rejection_is_atomic(board):
    cal = ready_calibrator(board)
    for left_size, right_size in [(SIZE, (320, 240)), ((320, 240), (320, 240)),
                                  ((0, 480), (0, 480))]:
        with pytest.raises(ValueError):
            cal.add_detection_pair(detection(board, left_size), detection(board, right_size))
        assert cal.num_frames == 1 and cal._image_size == SIZE
    assert cal.pop_last_frame()
    assert cal.add_detection_pair(detection(board, (320, 240)), detection(board, (320, 240)))
    assert cal._image_size == (320, 240)


@pytest.mark.parametrize('count,extra', [(4, 0), (5, cv2.CALIB_RATIONAL_MODEL)])
def test_real_joint_solver_with_reallocated_distortion(board, monkeypatch, count, extra):
    cal = StereoCalibrator(board)
    obj = board.object_points
    ids = np.arange(board.num_corners, dtype=np.int32).reshape(-1, 1)
    distortion = np.array([.08, -.02, .001, -.001, .005])
    for i in range(16):
        rv = np.array([.3 * np.sin(i), .25 * np.cos(i), .03 * i])
        tv = np.array([-.12 + .01 * i, -.1 + .015 * np.sin(i), .7 + .025 * i])
        observations = []
        for offset in [0., -.12]:
            pts, _ = cv2.projectPoints(obj, rv, tv + [offset, 0., 0.], K, distortion)
            observations.append(DetectionResult(pts.astype(np.float32), ids, [], None, SIZE))
        assert cal.add_detection_pair(*observations)
    left, right = intrinsics(count=count), intrinsics(count=count)
    original = cv2.stereoCalibrate
    returned = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        returned.append(result)
        return result

    monkeypatch.setattr(cv2, 'stereoCalibrate', capture)
    result = cal.calibrate(left, right, cv2.CALIB_USE_INTRINSIC_GUESS | extra)
    output = returned[0]
    for actual, expected in [(left.camera_matrix, output[1]), (left.dist_coeffs, output[2]),
                             (right.camera_matrix, output[3]), (right.dist_coeffs, output[4])]:
        np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(result.rms_error)
    assert left.dist_coeffs.size != count
    assert np.isnan(left.rms_error) and left.rvecs is None
    assert_rectification_matches(cal, result, left, right)


@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('compatibility', [False, True])
def test_real_fisheye_solver_contract(board, monkeypatch, joint, compatibility):
    cal = StereoCalibrator(board, CameraModel.FISHEYE)
    obj = board.object_points.astype(np.float64).reshape(-1, 1, 3)
    ids = np.arange(board.num_corners, dtype=np.int32).reshape(-1, 1)
    distortion = np.array([.03, -.01, .002, -.001])
    for i in range(16):
        rv = np.array([.3 * np.sin(i), .25 * np.cos(i), .03 * i])
        tv = np.array([-.12 + .01 * i, -.1 + .015 * np.sin(i), .7 + .025 * i])
        observations = []
        for offset in [0., -.12]:
            pts, _ = cv2.fisheye.projectPoints(obj, rv, tv + [offset, 0., 0.], K, distortion)
            keep = (ids.ravel() != i if compatibility and i % 3 else
                    np.ones(len(ids), dtype=bool))
            observations.append(DetectionResult(
                pts[keep].astype(np.float32), ids[keep], [], None, SIZE,
            ))
        assert cal.add_detection_pair(*observations)
    if compatibility:
        assert len(set(cal.frame_common_corner_counts)) > 1
        source = cal
        cal = cal.copy_with_common_ids()
        assert cal.num_frames == source.num_frames
        assert len(set(cal.frame_common_corner_counts)) == 1
    left, right = intrinsics(CameraModel.FISHEYE), intrinsics(CameraModel.FISHEYE)
    left.dist_coeffs[:] = right.dist_coeffs[:] = distortion
    original = cv2.fisheye.stereoCalibrate
    returned = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        returned.append(result)
        return result

    monkeypatch.setattr(cv2.fisheye, 'stereoCalibrate', capture)
    flag = cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_FIX_SKEW if joint else 0
    result = cal.calibrate(left, right, flag)
    output = returned[0]
    assert result.rms_error < .001
    for actual, expected in [(left.camera_matrix, output[1]), (left.dist_coeffs, output[2]),
                             (right.camera_matrix, output[3]), (right.dist_coeffs, output[4])]:
        np.testing.assert_array_equal(actual, expected)
    if joint:
        assert np.isnan(left.rms_error) and left.rvecs is None
    else:
        assert left.rms_error == .2 and left.rvecs is not None
    assert_rectification_matches(cal, result, left, right)


def partial_calibrator(board, id_sets, model=CameraModel.FISHEYE):
    cal = StereoCalibrator(board, model, min_frames=len(id_sets))
    for i, ids in enumerate(id_sets):
        observations = []
        for extra, offset in [(30, 0.), (31, 5.)]:
            order = np.array([*ids, extra], dtype=np.int32)
            if offset:
                order = order[::-1]
            det = detection(board)
            observations.append(DetectionResult(
                det.charuco_corners[order] + np.array([offset, 0.], dtype=np.float32),
                order.reshape(-1, 1), [], None, SIZE,
            ))
        assert cal.add_detection_pair(*observations, frame_id=f'source-{i + 1}')
    return cal


@pytest.mark.parametrize('flag', [0, cv2.fisheye.CALIB_FIX_INTRINSIC,
                                 cv2.fisheye.CALIB_USE_INTRINSIC_GUESS])
def test_ragged_fisheye_reports_source_counts_without_calling_solver(board, monkeypatch, flag):
    cal = partial_calibrator(board, [list(range(12)), list(range(10))])
    before = [a.copy() for a in cal._obj_points]

    def unexpected(*args, **kwargs):
        pytest.fail('Ragged fisheye observations reached OpenCV')

    install_solver(monkeypatch, CameraModel.FISHEYE, unexpected)
    with pytest.raises(ValueError) as exc:
        cal.calibrate(intrinsics(CameraModel.FISHEYE), intrinsics(CameraModel.FISHEYE), flag)
    message = str(exc.value)
    assert '鱼眼兼容：跨帧共同角点' in message
    assert 'source-1: 12 点' in message and 'source-2: 10 点' in message
    assert cal.frame_common_corner_counts == [12, 10]
    for original, actual in zip(before, cal._obj_points):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize('model', list(CameraModel))
def test_common_id_copy_aligns_all_arrays_and_does_not_change_source(board, model):
    cal = partial_calibrator(board, [
        [12, 2, 8, 0, 9, 3, 1, 7, 4, 6, 5],
        [11, 1, 8, 2, 7, 3, 6, 4, 5, 9, 10],
        [9, 8, 7, 6, 5, 4, 3, 2, 1],
    ], model)
    before = [[a.copy() for a in arrays] for arrays in
              (cal._obj_points, cal._img_points_left, cal._img_points_right, cal._corner_ids)]
    clone = cal.copy_with_common_ids()
    common = np.arange(1, 10)
    assert clone.num_frames == cal.num_frames == 3
    assert clone.min_frames == cal.min_frames and clone.model == cal.model
    assert clone.board is cal.board and clone._image_size == cal._image_size == SIZE
    assert clone.frame_ids == cal.frame_ids and clone.frame_ids is not cal.frame_ids
    for i, ids in enumerate(clone.frame_corner_ids):
        np.testing.assert_array_equal(ids, common)
        source_ids = cal.frame_corner_ids[i]
        lookup = {int(cid): j for j, cid in enumerate(source_ids)}
        take = [lookup[int(cid)] for cid in common]
        for source, selected in [(cal._obj_points, clone._obj_points),
                                 (cal._img_points_left, clone._img_points_left),
                                 (cal._img_points_right, clone._img_points_right)]:
            np.testing.assert_array_equal(selected[i], source[i][take])
            assert not np.shares_memory(source[i], selected[i])
        np.testing.assert_array_equal(clone._obj_points[i], board.object_points[common])
    for snapshots, source in zip(before, (cal._obj_points, cal._img_points_left,
                                          cal._img_points_right, cal._corner_ids)):
        for original, actual in zip(snapshots, source):
            np.testing.assert_array_equal(actual, original)
    clone._corner_ids[0][0] = -1
    clone._obj_points[0][:] = -1
    np.testing.assert_array_equal(cal._corner_ids[0], before[3][0])
    np.testing.assert_array_equal(cal._obj_points[0], before[0][0])


@pytest.mark.parametrize('id_sets,remaining', [
    ([list(range(8)), list(range(8, 16))], 0),
    ([list(range(8)), list(range(3, 11))], 5),
])
def test_common_id_copy_rejects_insufficient_intersection(board, id_sets, remaining):
    cal = partial_calibrator(board, id_sets)
    with pytest.raises(ValueError, match=f'跨帧共同角点仅 {remaining} 个'):
        cal.copy_with_common_ids()
    assert cal.num_frames == len(id_sets)
    assert cal.frame_common_corner_counts == [len(ids) for ids in id_sets]


@pytest.mark.parametrize('missing', ['all', 'frame', 'length'])
def test_common_id_copy_rejects_missing_ids(board, missing):
    cal = partial_calibrator(board, [list(range(8)), list(range(9))])
    if missing == 'all':
        cal._corner_ids.clear()
    elif missing == 'frame':
        cal._corner_ids[1] = None
    else:
        cal._corner_ids[1] = cal._corner_ids[1][1:]
    with pytest.raises(ValueError, match='角点 ID'):
        cal.copy_with_common_ids()
    assert cal.frame_common_corner_counts == [8, 9]


def test_corner_id_lifecycle_and_public_copies(board):
    cal = partial_calibrator(board, [list(range(8)), list(range(10)), list(range(12))])
    exposed = cal.frame_corner_ids
    exposed[0][:] = -1
    exposed.pop()
    np.testing.assert_array_equal(cal.frame_corner_ids[0], np.arange(8))
    assert len(cal.frame_corner_ids) == 3
    clone = cal.copy_with_indices([2, 0])
    assert clone.frame_ids == ['source-3', 'source-1']
    assert clone.frame_common_corner_counts == [12, 8]
    for i, source_idx in enumerate([2, 0]):
        np.testing.assert_array_equal(clone._corner_ids[i], cal._corner_ids[source_idx])
        assert not np.shares_memory(clone._corner_ids[i], cal._corner_ids[source_idx])
    assert cal.pop_last_frame()
    assert cal.frame_ids == ['source-1', 'source-2']
    assert len(cal.frame_corner_ids) == 2
    cal.reset()
    assert cal.frame_ids == [] and cal.frame_corner_ids == [] and cal._image_size is None
    with pytest.raises(ValueError, match='没有双目观测'):
        cal.copy_with_common_ids()


def test_pinhole_ragged_views_are_not_rejected_by_fisheye_guard(board, monkeypatch):
    cal = partial_calibrator(board, [list(range(12)), list(range(10))], CameraModel.PINHOLE)

    def solve(*args, **kwargs):
        assert [len(a) for a in args[0]] == [12, 10]
        return solver_output(CameraModel.PINHOLE, *args[3:7])

    install_solver(monkeypatch, CameraModel.PINHOLE, solve)
    assert cal.calibrate(intrinsics(), intrinsics()).model == CameraModel.PINHOLE
