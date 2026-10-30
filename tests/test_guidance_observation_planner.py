"""Observation-driven candidates, measured coverage and ergonomic scheduling."""

from copy import deepcopy

import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration import adaptive_guidance as ag
from multicamera.calibration.models import CameraModel
from test_adaptive_guidance import intrinsics, observation


@pytest.fixture
def board():
    return CharucoBoard(CharucoBoardConfig())


def analyze(board, data):
    return ag.analyze_guidance(board, CameraModel.PINHOLE, data,
                               {c: intrinsics() for c in data})


@pytest.mark.parametrize('tilt', ag._TILTS)
@pytest.mark.parametrize('zone', [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_medium_edges_fit_with_inward_shift_without_forcing_far(board, tilt, zone):
    center, area, predicted = ag._candidate_geometry(board, (*zone, tilt, 'medium'),
                                                    [(1280, 960), (1280, 720)])
    assert .04 <= area < .12
    assert ag._zone(center) == zone
    assert np.linalg.norm(np.asarray(center) - ((zone[1] + .5) / 3, (zone[0] + .5) / 3)) > .01
    points = board.object_points[:, :2]
    outer_ratio = (np.ptp(points, axis=0) + 2 * board.config.square_length) / np.ptp(points, axis=0)
    for points in predicted:
        half = np.ptp(points, axis=0) * outer_ratio / 2
        half *= 1 if tilt == 'front' else ag.TILTED_FOOTPRINT_PADDING
        assert np.all(np.asarray(center) - half >= ag.CANDIDATE_IMAGE_MARGIN - 1e-8)
        assert np.all(np.asarray(center) + half <= 1 - ag.CANDIDATE_IMAGE_MARGIN + 1e-8)


@pytest.mark.parametrize('sizes', [[(32, 24)], [(1280, 720), (32, 24)], [(0, 720)], [(1.5, 720)]])
def test_invalid_or_tiny_eye_is_not_bypassed(board, sizes):
    assert ag._candidate_geometry(board, (1, 1, 'up', 'medium'), sizes) is None


def test_coverage_uses_detected_points_not_board_center_or_inferred_hull(board):
    data = {c: [observation(board, depth=.8)] for c in ('L', 'R')}
    result = analyze(board, data)
    for cid in data:
        visits = np.asarray(result.image_coverage[cid])
        cells = ag._coverage_cells(data[cid][0].corners, data[cid][0].image_size)
        assert np.count_nonzero(visits) == len(cells) > 1
        assert visits.max() == 1
    duplicate = {c: seq + [deepcopy(seq[0])] for c, seq in data.items()}
    assert analyze(board, duplicate).image_coverage == result.image_coverage
    # A missing interior strip must not be filled by a convex-hull rasterization.
    for seq in data.values():
        obs = seq[0]
        points = obs.corners.reshape(-1, 2)
        keep = (points[:, 0] < 480) | (points[:, 0] >= 640)
        obs.corners, obs.ids = obs.corners[keep], obs.ids[keep]
    partial = analyze(board, data)
    assert sum(map(sum, partial.image_coverage['L'])) < sum(map(sum, result.image_coverage['L']))
    assert analyze(board, {'L': [], 'R': []}).image_coverage == {
        c: np.zeros(ag.COVERAGE_SHAPE, int).tolist() for c in data}


def test_suggested_or_unreliable_tilts_never_close_actual_gap(board):
    data = {c: [observation(board, i, center=(.004 * i, 0), depth=1.2)
                for i in range(10)] for c in ('L', 'R')}
    result = analyze(board, data)
    assert result.tilt_counts == {'front': 10}
    assert set(result.recommendation.missing_tilts) == set(ag._TILTS[1:])
    data = {c: [observation(board, angles=(25, 0, 0))] for c in data}
    result = analyze(board, data)
    assert result.tilt_counts == {'up': 1}
    data['R'] = [observation(board, angles=(-25, 0, 0))]
    assert not analyze(board, data).tilt_counts
    data['R'] = deepcopy(data['L'])
    bad = intrinsics()
    bad.rms_error = 3.
    result = ag.analyze_guidance(board, CameraModel.PINHOLE, data, {c: bad for c in data})
    assert not result.tilt_counts


def test_utility_ties_choose_less_vertical_and_depth_motion(board):
    candidates = [(0, 1, 'up', 'medium'), (1, 1, 'up', 'medium')]
    geometries = {c: ag._candidate_geometry(board, c, [(1280, 960)] * 2) for c in candidates}
    # Equal predicted benefit isolates the movement tie-break from grid aliasing.
    a, b = candidates
    geometries[a] = (*geometries[a][:2], geometries[b][2])
    coverage = [np.zeros(ag.COVERAGE_SHAPE, int) for _ in range(2)]
    from collections import Counter
    chosen = ag._choose_candidate(candidates, geometries, coverage, Counter(), Counter(),
                                  Counter(), np.array([.5, .5]), .07, 'up', None)
    assert chosen == candidates[1]
    costs = [ag._move_cost(c, geometries[c], np.array([.5, .5]), .07, 'up') for c in candidates]
    assert costs[1] < costs[0]


def test_far_heavy_history_requests_near_or_medium_and_retains_real_tilt_gaps(board):
    data = {c: [observation(board, i, depth=1.7, center=(.009 * (i % 7), .006 * i))
                for i in range(18)] for c in ('L', 'R')}
    result = analyze(board, data)
    assert result.recommendation.sample_count == 18
    assert result.recommendation.scale in ('large', 'medium')
    assert result.recommendation.tilt != 'front'
    assert set(result.recommendation.missing_tilts) == set(ag._TILTS[1:])
    assert analyze(board, data).recommendation == result.recommendation
    assert analyze(board, {c: seq[:-1] for c, seq in data.items()}).recommendation.sample_count == 17


def test_replay_and_target_use_dynamic_center_not_cell_midpoint():
    from multicamera.calibration.guidance_replay import _matches
    from multicamera.ui.widgets.capture_guide import BoardTilt, CaptureTarget
    rec = ag.GuidanceRecommendation(0, 0, 'front', 'medium', '', True, 9,
                                   target_center=(.32, .32), target_area=.055)
    target = CaptureTarget(0, 0, BoardTilt.FRONT, target_center=rec.target_center,
                           target_area=rec.target_area)
    assert (target.norm_cx, target.norm_cy) == rec.target_center
    pose = dict(yaw=0, pitch=0, depth=1., distance_area=.055, distance_reliable=True)
    measurements = [dict(geometry_valid=True, center=rec.target_center, pose=pose) for _ in range(2)]
    assert _matches(rec, measurements)['composition_match']
    for m in measurements:
        m['center'] = (1 / 6, 1 / 6)
    assert not _matches(rec, measurements)['composition_match']


def test_following_the_plan_does_not_produce_a_far_frontal_majority(board):
    data = {c: [] for c in ('L', 'R')}
    cal = intrinsics()
    factor = (ag._hull_area(board.object_points[:, :2]) * np.prod(np.diag(cal.camera_matrix)[:2])
              / np.prod(cal.image_size))
    targets = []
    angles = dict(front=(0, 0, 0), left=(0, -25, 0), right=(0, 25, 0),
                  up=(25, 0, 0), down=(-25, 0, 0))
    for n in range(ag.INITIAL_SAMPLE_BUDGET):
        rec = analyze(board, data).recommendation
        assert rec.phase == 'initial' and rec.sample_count == n
        targets.append(rec)
        depth = np.sqrt(factor / rec.target_area)
        center = ((rec.target_center[0] - .5) * cal.image_size[0] * depth / cal.camera_matrix[0, 0],
                  (rec.target_center[1] - .5) * cal.image_size[1] * depth / cal.camera_matrix[1, 1])
        for c, shift in (('L', 0.), ('R', -.04)):
            data[c].append(observation(board, n, angles=angles[rec.tilt], depth=depth,
                                       center=center, camera_shift=shift))
    result = analyze(board, data)
    assert sum(t.scale == 'small' for t in targets) <= 6
    assert sum(t.scale == 'large' for t in targets) >= 6
    assert sum(t.tilt == 'front' for t in targets) < len(targets) / 2
    assert all(result.tilt_counts.get(t, 0) >= 1 for t in ag._TILTS[1:])
    assert not result.recommendation.missing_tilts
    assert result.recommendation.phase == 'review'


def test_next_hint_is_conditional_and_does_not_advance_on_reanalysis(board):
    empty = {'L': [], 'R': []}
    first = analyze(board, empty)
    assert '验收并保存后' in first.recommendation.next_hint
    assert '未通过则不推进' in first.recommendation.next_hint
    assert analyze(board, empty).recommendation == first.recommendation
