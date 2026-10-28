"""Synthetic numerical contracts for the UI-independent capture recommender."""

from copy import deepcopy

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration import adaptive_guidance as ag
from multicamera.calibration.models import CameraIntrinsics, CameraModel


@pytest.fixture
def board():
    return CharucoBoard(CharucoBoardConfig(cols=9, rows=7, square_length=.035,
                                          marker_length=.025))


def intrinsics(model=CameraModel.PINHOLE):
    return CameraIntrinsics(np.array([[650., 0, 640], [0, 660., 480], [0, 0, 1.]]),
                            np.zeros(4 if model == CameraModel.FISHEYE else 5),
                            (1280, 960), model)


def observation(board, frame='0', model=CameraModel.PINHOLE, angles=(0., 0., 0.),
                center=(0., 0.), depth=.8, ids=None, camera_shift=0.):
    cal = intrinsics(model)
    rv = np.radians(angles).reshape(3, 1)
    R = cv2.Rodrigues(rv)[0]
    midpoint = board.object_points.mean(axis=0)
    tv = np.array([center[0] + camera_shift, center[1], depth]) - R @ midpoint
    ids = np.arange(board.num_corners) if ids is None else np.asarray(ids)
    project = cv2.fisheye.projectPoints if model == CameraModel.FISHEYE else cv2.projectPoints
    points = project(board.object_points[ids].astype(float).reshape(-1, 1, 3), rv, tv,
                     cal.camera_matrix, cal.dist_coeffs)[0]
    return ag.GuidanceObservation(points, ids.astype(np.int32), cal.image_size, str(frame))


def paired(board, count=1, **kwargs):
    return {camera: [observation(board, str(i), camera_shift=shift, **kwargs)
                     for i in range(count)] for camera, shift in [('L', 0.), ('R', -.04)]}


def analyze(board, observations, provided=None, model=CameraModel.PINHOLE):
    return ag.analyze_guidance(board, model, observations, provided or {})


def diverse(board, model=CameraModel.PINHOLE, count=12):
    angles = [(-22, -24, 0), (24, -18, 4), (-20, 25, -5), (25, 24, 0),
              (0, -28, 8), (28, 0, -8), (-28, 0, 3), (0, 28, -3),
              (18, -25, 0), (-24, 16, 4), (12, 30, -3), (-30, -12, 6)]
    return {cam: [observation(board, str(i), model, angles[i % len(angles)],
                              center=((i % 3 - 1) * .12, (i % 4 - 1.5) * .065),
                              depth=.65 + (i % 3) * .12, camera_shift=shift)
                  for i in range(count)] for cam, shift in [('L', 0.), ('R', -.04)]}


@pytest.mark.parametrize('model', list(CameraModel))
@pytest.mark.parametrize('angles,yaw_sign,pitch_sign', [((0, 25, 0), 1, 0),
                                                        ((0, -25, 0), -1, 0),
                                                        ((-25, 0, 0), 0, 1),
                                                        ((25, 0, 0), 0, -1)])
def test_pose_near_edge_convention(board, model, angles, yaw_sign, pitch_sign):
    obs = observation(board, model=model, angles=angles)
    pose = ag.measure_pose(board, obs, intrinsics(model))
    assert pose is not None
    assert pose['yaw'] == pytest.approx(yaw_sign * 25, abs=.001)
    assert pose['pitch'] == pytest.approx(pitch_sign * 25, abs=.001)
    assert pose['depth'] == pytest.approx(.8, abs=1e-5)
    assert pose['center'] == pytest.approx((.5, .5), abs=.006)
    hull = cv2.convexHull(obs.corners.astype(np.float32))
    assert pose['area'] == pytest.approx(cv2.contourArea(hull) / (1280 * 960))


@pytest.mark.parametrize('model', list(CameraModel))
def test_distorted_off_axis_pose(board, model):
    cal = intrinsics(model)
    cal.dist_coeffs[:] = [-.12, .025, -.001, .0003] if model == CameraModel.FISHEYE else [-.15, .03, .001, -.001, 0]
    obs = observation(board, model=model, angles=(-18, 22, 6), center=(.25, -.12))
    rv = np.radians((-18., 22., 6.)).reshape(3, 1)
    tv = np.array([.25, -.12, .8]) - cv2.Rodrigues(rv)[0] @ board.object_points.mean(axis=0)
    project = cv2.fisheye.projectPoints if model == CameraModel.FISHEYE else cv2.projectPoints
    obs.corners = project(board.object_points.astype(float).reshape(-1, 1, 3), rv, tv,
                          cal.camera_matrix, cal.dist_coeffs)[0]
    pose = ag.measure_pose(board, obs, cal)
    assert pose is not None
    assert pose['yaw'] > 15 and pose['pitch'] > 12
    assert pose['depth'] == pytest.approx(.8, abs=1e-5)


def test_initial_and_camera_count_contract(board):
    result = analyze(board, {'L': [], 'R': []})
    r = result.recommendation
    assert (r.zone_row, r.zone_col, r.tilt, r.scale) == (1, 1, 'front', 'medium')
    assert r.sample_count == 0 and not r.pose_available
    assert '未验证' in r.reason and '不代表标定完成' in result.summary
    for obs in ({}, {'L': []}, {'L': [], 'R': [], 'X': []}):
        with pytest.raises(ValueError):
            analyze(board, obs)


def test_matching_ids_not_list_position(board):
    data = diverse(board, count=4)
    data['R'].reverse()
    data['L'].append(observation(board, 'left-only', center=(.3, 0)))
    data['R'].append(observation(board, 'right-only', center=(-.3, 0)))
    assert analyze(board, data).recommendation.sample_count == 4
    assert analyze(board, {'L': data['L'], 'R': []}).recommendation.sample_count == 0


@pytest.mark.parametrize('kind', ['nan', 'ids', 'repeated-id', 'few', 'shape', 'outside',
                                  'size', 'fractional-id', 'collinear', 'partial'])
def test_invalid_observations_do_not_count(board, kind):
    data = paired(board)
    obs = data['L'][0]
    if kind == 'nan':
        obs.corners[0] = np.nan
    elif kind == 'ids':
        obs.ids[0] = board.num_corners
    elif kind == 'repeated-id':
        obs.ids[0] = obs.ids[1]
    elif kind == 'few':
        obs.corners, obs.ids = obs.corners[:4], obs.ids[:4]
    elif kind == 'shape':
        obs.corners = obs.corners.reshape(-1)
    elif kind == 'outside':
        obs.corners[0] = -1
    elif kind == 'size':
        obs.image_size = (0, 960)
    elif kind == 'fractional-id':
        obs.ids = obs.ids.astype(float)
    else:
        n = 8 if kind == 'collinear' else 16
        obs.corners, obs.ids = obs.corners[:n], obs.ids[:n]
    assert analyze(board, data).recommendation.sample_count == 0
    assert ag.measure_pose(board, obs, intrinsics()) is None


def test_partial_board_center_uses_homography(board):
    full = observation(board, angles=(-12, 24, 0))
    # Retain six of eight columns: a substantial but asymmetric partial board.
    keep = full.ids % 8 < 6
    partial = ag.GuidanceObservation(full.corners[keep], full.ids[keep], full.image_size, 'p')
    a = ag.measure_pose(board, full, intrinsics())
    b = ag.measure_pose(board, partial, intrinsics())
    assert a is not None and b is not None
    np.testing.assert_allclose(a['center'], b['center'], atol=1e-6)
    assert b['area'] < a['area']
    assert np.linalg.norm(partial.corners.reshape(-1, 2).mean(axis=0) / full.image_size
                          - b['center']) > .01


def test_duplicate_compositions_and_ids_are_stable(board):
    data = paired(board)
    first = analyze(board, data).recommendation
    for cam in data:
        duplicate = deepcopy(data[cam][0])
        duplicate.frame_id = 'new-id-same-composition'
        duplicate.corners = duplicate.corners[::-1].copy()
        duplicate.ids = duplicate.ids[::-1].copy()
        data[cam].extend([deepcopy(data[cam][0]), duplicate])
    assert analyze(board, data).recommendation == first
    assert first.sample_count == 1
    data = {cam: seq[:1] for cam, seq in data.items()}
    data['L'].append(observation(board, '0', center=(.2, .1)))
    assert analyze(board, data).recommendation.sample_count == 0


def test_new_matching_samples_change_target_and_bootstrap_tilts(board):
    data = diverse(board, count=5)
    previous = analyze(board, {'L': [], 'R': []}).recommendation
    tilts = set()
    for n in range(1, 6):
        result = analyze(board, {cam: seq[:n] for cam, seq in data.items()})
        r = result.recommendation
        assert (r.zone_row, r.zone_col, r.tilt, r.scale) != (
            previous.zone_row, previous.zone_col, previous.tilt, previous.scale)
        assert not r.pose_available and '未验证' in r.reason
        tilts.add(r.tilt)
        previous = r
    assert len(tilts) >= 4


@pytest.mark.parametrize('provided', [False, True])
def test_new_target_across_bootstrap_transition(board, provided):
    data = diverse(board)
    calibrations = {c: intrinsics() for c in data} if provided else {}
    targets = []
    for n in range(13):
        r = analyze(board, {c: seq[:n] for c, seq in data.items()}, calibrations).recommendation
        target = (r.zone_row, r.zone_col, r.tilt, r.scale)
        if targets:
            assert target != targets[-1]
        targets.append(target)


def test_both_cameras_affect_shortages(board):
    a = paired(board)
    b = deepcopy(a)
    b['R'] = [observation(board, center=(-.25, -.2), depth=.5)]
    # Occupancy and scale must be obtained from each camera, not just the left.
    ga = [ag._geometry(board, a[c][0]) for c in ('L', 'R')]
    gb = [ag._geometry(board, b[c][0]) for c in ('L', 'R')]
    assert ag._zone(ga[1][2]) != ag._zone(gb[1][2])
    assert ag._scale(ga[1][3]) != ag._scale(gb[1][3])
    ra, rb = analyze(board, a).recommendation, analyze(board, b).recommendation
    # Bootstrap keeps the same physical height; the second eye still changes target size.
    assert (ra.zone_row, ra.zone_col) == (rb.zone_row, rb.zone_col) == (1, 1)
    assert ra.scale != rb.scale
    # Swap roles: a symmetric aggregate must give the same next target.
    swapped = {'L': b['R'], 'R': b['L']}
    assert analyze(board, b).recommendation == analyze(board, swapped).recommendation


@pytest.mark.parametrize('area,scale', [(.039, 'small'), (.04, 'medium'), (.119, 'medium'),
                                       (.12, 'large')])
def test_scale_thresholds(area, scale):
    assert ag._scale(area) == scale


@pytest.mark.parametrize('model', list(CameraModel))
@pytest.mark.parametrize('count', [6, 12])
def test_bootstrap_recovers_only_temporary_intrinsics(board, model, count):
    data = diverse(board, model, count)
    original = deepcopy(data)
    result = analyze(board, data, model=model)
    assert set(result.intrinsics) == {'L', 'R'}
    assert result.recommendation.pose_available
    for camera, estimate in result.intrinsics.items():
        np.testing.assert_allclose(estimate.camera_matrix, intrinsics(model).camera_matrix,
                                   rtol=.015, atol=.5)
        assert estimate.rms_error < .01
        for a, b in zip(data[camera], original[camera]):
            np.testing.assert_array_equal(a.corners, b.corners)
            np.testing.assert_array_equal(a.ids, b.ids)


def test_front_parallel_and_repeated_views_cannot_bootstrap(board, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('Degenerate observations must not invoke calibration')
    monkeypatch.setattr(ag.IntrinsicCalibrator, 'calibrate', fail)
    data = {cam: [observation(board, i, center=((i % 3 - 1) * .12, 0),
                              depth=.6 + .05 * i) for i in range(9)] for cam in ('L', 'R')}
    assert not analyze(board, data).intrinsics
    assert not analyze(board, paired(board, count=12)).intrinsics


@pytest.mark.parametrize('failure', ['exception', 'nan', 'focal', 'pp', 'dist', 'rms', 'model', 'size'])
def test_unreliable_bootstrap_is_discarded(board, monkeypatch, failure):
    def bad_result(self):
        if failure == 'exception':
            raise cv2.error('synthetic solver failure')
        cal = intrinsics()
        if failure == 'nan':
            cal.camera_matrix[0, 0] = np.nan
        elif failure == 'focal':
            cal.camera_matrix[0, 0] = 1e8
        elif failure == 'pp':
            cal.camera_matrix[0, 2] = -10000
        elif failure == 'dist':
            cal.dist_coeffs[0] = np.inf
        elif failure == 'rms':
            cal.rms_error = 3.
        elif failure == 'model':
            cal.model = CameraModel.FISHEYE
        elif failure == 'size':
            cal.image_size = (640, 480)
        return cal
    monkeypatch.setattr(ag.IntrinsicCalibrator, 'calibrate', bad_result)
    result = analyze(board, diverse(board))
    assert not result.intrinsics and not result.recommendation.pose_available


def test_provided_intrinsics_preferred_copied_and_checked(board, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('Valid provided intrinsics must be preferred')
    monkeypatch.setattr(ag.IntrinsicCalibrator, 'calibrate', fail)
    provided = {'L': intrinsics(), 'R': intrinsics()}
    result = analyze(board, diverse(board), provided)
    assert result.recommendation.pose_available
    result.intrinsics['L'].camera_matrix[0, 0] = 123
    assert provided['L'].camera_matrix[0, 0] == 650
    for kind in ('size', 'model', 'nan'):
        bad = intrinsics()
        if kind == 'size':
            bad.image_size = (640, 480)
        elif kind == 'model':
            bad.model = CameraModel.FISHEYE
        else:
            bad.camera_matrix[0, 0] = np.nan
        assert ag.measure_pose(board, observation(board), bad) is None
        assert not analyze(board, paired(board), {'L': bad, 'R': bad}).intrinsics


def test_invalid_provided_intrinsics_can_fall_back_to_temporary_estimate(board):
    bad = intrinsics()
    bad.image_size = (640, 480)
    result = analyze(board, diverse(board, count=6), {'L': bad, 'R': intrinsics()})
    assert result.recommendation.pose_available
    assert result.intrinsics['L'].image_size == (1280, 960)
    assert bad.image_size == (640, 480)


def test_invalid_intrinsic_size_without_frames_is_rejected(board):
    bad = intrinsics()
    bad.image_size = (1280., 960.)
    result = analyze(board, {'L': [], 'R': []}, {'L': bad, 'R': bad})
    assert not result.intrinsics


def test_pose_rejects_bad_reprojection_and_behind_camera(board, monkeypatch):
    obs = observation(board)
    obs.corners[::2, 0, 0] += 25
    assert ag.measure_pose(board, obs, intrinsics()) is None
    monkeypatch.setattr(cv2, 'solvePnP', lambda *a, **k: (True, np.zeros((3, 1)),
                                                       np.array([[0.], [0.], [-1.]])))
    assert ag.measure_pose(board, observation(board), intrinsics()) is None


def test_inconsistent_sizes_do_not_mix_or_mutate(board):
    data = diverse(board, count=4)
    original = board.object_points.copy()
    for cam in data:
        data[cam][-1].image_size = (2560, 1920)
        data[cam][-1].corners *= 2
    assert analyze(board, data).recommendation.sample_count == 3
    np.testing.assert_array_equal(board.object_points, original)


def test_bootstrap_view_limit(board, monkeypatch):
    seen = []
    def record(self):
        seen.append((self.num_frames, self.min_frames))
        raise ValueError('bounded synthetic calibration')
    monkeypatch.setattr(ag.IntrinsicCalibrator, 'calibrate', record)
    data = diverse(board, count=45)
    # Ensure the repeated angle pattern still has genuinely different compositions.
    for cam in data:
        for i, obs in enumerate(data[cam]):
            obs.corners[:, :, 0] += i * 3
    analyze(board, data)
    assert seen == [(30, 6), (30, 6)]


@pytest.mark.parametrize('models', [CameraModel.PINHOLE, CameraModel.FISHEYE])
def test_pose_capability_without_observations(board, models):
    supplied = {cam: intrinsics(models) for cam in ('L', 'R')}
    result = analyze(board, {'L': [], 'R': []}, supplied, model=models)
    rec = result.recommendation
    assert rec.pose_available and rec.sample_count == 0
    assert '已有内参支持' in rec.reason and '倾角可估计' in rec.reason
    assert '已验证' not in rec.reason
    assert '姿态可解 0 组' in result.summary
    assert not analyze(board, {'L': [], 'R': []}, {'L': supplied['L']},
                       model=models).recommendation.pose_available


def test_failed_latest_pose_does_not_change_intrinsic_capability(board, monkeypatch):
    data = paired(board)
    monkeypatch.setattr(ag, 'measure_pose', lambda *args: None)
    result = analyze(board, data, {'L': intrinsics(), 'R': intrinsics()})
    assert result.recommendation.pose_available
    assert '已验证' not in result.recommendation.reason
    assert '姿态可解 0 组' in result.summary


@pytest.mark.parametrize('tilt', ag._TILTS)
def test_corner_footprint_reduces_size(board, tilt):
    sizes = [(1280, 960), (1280, 720)]
    for row, col in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        assert not ag._candidate_feasible(board, (row, col, tilt, 'large'), sizes)
        assert not ag._candidate_feasible(board, (row, col, tilt, 'medium'), sizes)
        assert ag._candidate_feasible(board, (row, col, tilt, 'small'), sizes)
    assert ag._candidate_feasible(board, (1, 1, tilt, 'large'), sizes)


def test_bootstrap_keeps_center_before_requesting_corner_coverage(board):
    result = analyze(board, paired(board))
    rec = result.recommendation
    assert (rec.zone_row, rec.zone_col) == (1, 1)
    assert rec.scale == 'medium'
    assert '中央已有观测' in rec.reason
    assert '中画面占比' in rec.reason and '冷启动' in rec.reason
    assert len(rec.reason) < 65
    assert '几何近似' in result.summary and '不保证双目共视' in result.summary


def test_extreme_second_camera_aspect_filters_candidate(board):
    candidate = (1, 0, 'front', 'small')
    assert ag._candidate_feasible(board, candidate, [(1280, 960)])
    assert not ag._candidate_feasible(board, candidate, [(1280, 960), (480, 1280)])
    assert not ag._candidate_feasible(board, candidate, [(480, 1280), (1280, 960)])
    for cam_sizes in ([(1280, 960), (480, 1280)], [(480, 1280), (1280, 960)]):
        supplied = {}
        for cam, size in zip(('L', 'R'), cam_sizes):
            cal = intrinsics()
            cal.image_size = size
            cal.camera_matrix[:2, 2] = np.array(size) / 2
            supplied[cam] = cal
        result = analyze(board, paired(board), supplied)
        # Observations, not mismatched provided intrinsics, determine actual image sizes.
        assert ag._candidate_feasible(board, (
            result.recommendation.zone_row, result.recommendation.zone_col,
            result.recommendation.tilt, result.recommendation.scale), [(1280, 960)] * 2)
        empty = analyze(board, {'L': [], 'R': []}, supplied)
        rec = empty.recommendation
        assert ag._candidate_feasible(board, (rec.zone_row, rec.zone_col, rec.tilt, rec.scale),
                                      cam_sizes)


def test_minimum_corner_spacing_is_checked_on_both_cameras(board):
    candidate = (1, 1, 'front', 'small')
    assert ag._candidate_feasible(board, candidate, [(1280, 960)])
    assert not ag._candidate_feasible(board, candidate, [(1280, 960), (160, 120)])
    assert not ag._candidate_feasible(board, candidate, [(160, 120), (1280, 960)])
    # Increasing target area can make the spacing usable without violating the margin.
    assert ag._candidate_feasible(board, (1, 1, 'front', 'large'), [(160, 120)])


def test_no_feasible_candidate_has_explicit_warning(board):
    supplied = {}
    for cam in ('L', 'R'):
        cal = intrinsics()
        cal.image_size = (32, 24)
        cal.camera_matrix = np.array([[20., 0, 16], [0, 20., 12], [0, 0, 1.]])
        supplied[cam] = cal
    result = analyze(board, {'L': [], 'R': []}, supplied)
    assert not result.recommendation.feasible
    assert result.recommendation.pose_available
    assert '难以兼顾' in result.recommendation.reason
    assert '不保证双目共视' in result.summary


@pytest.mark.parametrize('provided', [False, True])
def test_readonly_inputs_and_board_points_are_never_mutated(board, monkeypatch, provided):
    data = diverse(board, count=6)
    supplied = {cam: intrinsics() for cam in data} if provided else {}
    points = board.object_points.copy()
    expected_points = points.copy()
    points.setflags(write=False)
    monkeypatch.setattr(CharucoBoard, 'object_points', property(lambda self: points))
    original = deepcopy(data)
    for observations in data.values():
        for obs in observations:
            obs.corners.setflags(write=False)
            obs.ids.setflags(write=False)
    for cal in supplied.values():
        cal.camera_matrix.setflags(write=False)
        cal.dist_coeffs.setflags(write=False)
    result = analyze(board, data, supplied)
    assert result.recommendation.pose_available
    for cam, observations in data.items():
        for obs, saved in zip(observations, original[cam]):
            np.testing.assert_array_equal(obs.corners, saved.corners)
            np.testing.assert_array_equal(obs.ids, saved.ids)
            assert not obs.corners.flags.writeable and not obs.ids.flags.writeable
            pose = ag.measure_pose(board, obs, result.intrinsics[cam])
            assert pose is not None
            assert not np.shares_memory(pose['center'], obs.corners)
        if provided:
            assert not np.shares_memory(result.intrinsics[cam].camera_matrix,
                                        supplied[cam].camera_matrix)
            np.testing.assert_array_equal(supplied[cam].camera_matrix, intrinsics().camera_matrix)
    np.testing.assert_array_equal(points, expected_points)
    assert not points.flags.writeable
