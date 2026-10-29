"""Metric distance, conditional confidence and evidence-based distance scheduling."""

from copy import deepcopy

import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration import adaptive_guidance as ag
from multicamera.calibration.models import CameraModel
from test_adaptive_guidance import intrinsics, observation


@pytest.fixture
def board():
    return CharucoBoard(CharucoBoardConfig(cols=9, rows=7, square_length=.035,
                                          marker_length=.025))


def measurements(pose, expected=True):
    return [dict(valid=True, pose=deepcopy(pose), pose_expected=expected,
                 distance_area=.07) for _ in range(2)]


def at_scale(board, scale, frame, angle=(0, 0, 0)):
    cal = intrinsics()
    factor = (ag._hull_area(board.object_points[:, :2])
              * np.prod(np.diag(cal.camera_matrix)[:2]) / np.prod(cal.image_size))
    return observation(board, frame, depth=np.sqrt(factor / ag.SCALE_AREA_TARGETS[scale]),
                       angles=angle)


@pytest.mark.parametrize('model', list(CameraModel))
@pytest.mark.parametrize('angle', [(0, 0, 0), (25, -20, 0)])
def test_center_depth_range_and_local_uncertainty(board, model, angle):
    obs = observation(board, model=model, depth=.8, center=(.15, -.1), angles=angle)
    pose = ag.measure_pose(board, obs, intrinsics(model))
    assert pose['distance_reliable'], pose
    assert pose['depth'] == pytest.approx(.8, abs=1e-5)
    assert pose['range'] == pytest.approx(np.linalg.norm((.15, -.1, .8)), abs=1e-5)
    assert 0 < pose['depth_std'] < .01
    assert pose['reprojection_rms'] < .001


@pytest.mark.parametrize('model', list(CameraModel))
def test_far_board_has_larger_depth_uncertainty(board, model):
    near = ag.measure_pose(board, observation(board, model=model, depth=.6), intrinsics(model))
    far = ag.measure_pose(board, observation(board, model=model, depth=1.2), intrinsics(model))
    assert far['depth_std'] > near['depth_std'] * 3


def test_bad_intrinsic_fit_and_tiny_corners_cannot_claim_metric_distance(board):
    cal = intrinsics()
    cal.rms_error = 3.
    bad = ag.measure_pose(board, observation(board), cal)
    assert bad is not None and not bad['distance_reliable']
    assert '内参' in bad['distance_reason']
    tiny = ag.measure_pose(board, observation(board, depth=5.), intrinsics())
    assert not tiny['distance_reliable'] and '间距' in tiny['distance_reason']
    for pose in (bad, tiny):
        ready, text, _, source = ag.evaluate_distance('medium', measurements(pose))
        assert not ready and source == 'uncertain'
        assert '距离待确认' in text and '非实测距离' in text
        assert 'm）' not in text


def test_relative_reprojection_error_rejects_noisy_distant_board(board):
    obs = observation(board, depth=3.)
    obs.corners += np.random.default_rng(123).normal(0, 1.1, obs.corners.shape)
    pose = ag.measure_pose(board, obs, intrinsics())
    assert pose is not None and not pose['distance_reliable']
    assert '重投影' in pose['distance_reason']


def test_rank_deficient_quality_is_rejected_and_json_serializable(board, monkeypatch):
    import json
    monkeypatch.setattr(ag.np.linalg, 'svd', lambda *a, **k:
                        (None, np.zeros(6), np.eye(6)))
    pose = ag.measure_pose(board, observation(board), intrinsics())
    assert not pose['distance_reliable'] and '退化' in pose['distance_reason']
    assert pose['depth_std'] is None
    json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in pose.items()},
               allow_nan=False)


def test_relative_depth_uncertainty_threshold_is_enforced(board, monkeypatch):
    monkeypatch.setattr(ag, 'MAX_DISTANCE_RELATIVE_STD', 1e-6)
    pose = ag.measure_pose(board, observation(board), intrinsics())
    assert not pose['distance_reliable'] and '深度估计不稳定' in pose['distance_reason']


def test_metric_targets_and_single_eye_failure(board):
    pose = ag.measure_pose(board, at_scale(board, 'medium', 'a'), intrinsics())
    data = measurements(pose)
    ready, text, areas, source = ag.evaluate_distance('medium', data)
    assert ready and source == 'pnp' and len(areas) == 2
    assert '板中心深度' in text and '目标' in text and 'm）' in text
    data[1]['pose'] = None
    ready, text, _, source = ag.evaluate_distance('medium', data)
    assert not ready and source == 'uncertain' and '暂不自动验收' in text
    assert ag.evaluate_distance('medium', measurements(None, expected=False))[3] == 'composition'


def test_distance_boundary_waits_instead_of_approach_retreat_oscillation(board):
    pose = ag.measure_pose(board, at_scale(board, 'medium', 'a'), intrinsics())
    pose['distance_area'] = .04001
    pose['depth_std'] = .005
    ready, text, _, source = ag.evaluate_distance('medium', measurements(pose))
    assert not ready and source == 'pnp' and '距离临界' in text


def test_partial_points_keep_full_board_center_and_distance(board):
    full = observation(board, angles=(20, 25, 0), depth=.8)
    keep = full.ids % 8 < 6
    partial = deepcopy(full)
    partial.corners, partial.ids = partial.corners[keep], partial.ids[keep]
    pose = ag.measure_pose(board, partial, intrinsics())
    assert pose['distance_reliable'] and pose['depth'] == pytest.approx(.8, abs=1e-5)


def test_distribution_skips_filled_distance_and_undo_reopens_gap(board):
    data = {c: [] for c in ('L', 'R')}
    provided = {c: intrinsics() for c in data}
    angles = [(0, 0, 0), (0, 25, 0), (25, 0, 0), (0, -25, 0), (-25, 0, 0)]
    for i in range(5):
        scale = 'medium' if i < 3 else 'large'
        for seq in data.values():
            seq.append(at_scale(board, scale, str(i), angles[i]))
        result = ag.analyze_guidance(board, CameraModel.PINHOLE, data, provided)
        assert result.distance_counts[scale] == (i + 1 if i < 3 else i - 2)
        if i == 2:
            assert result.recommendation.scale == 'large'
        if i == 4:
            assert result.recommendation.scale == 'small'
    shorter = {c: seq[:-1] for c, seq in data.items()}
    result = ag.analyze_guidance(board, CameraModel.PINHOLE, shorter, provided)
    assert result.recommendation.scale == 'large'
    assert 'large' in result.recommendation.missing_distances
    duplicate = {c: seq + [deepcopy(seq[-1])] for c, seq in shorter.items()}
    assert ag.analyze_guidance(board, CameraModel.PINHOLE, duplicate, provided).distance_counts == result.distance_counts


def test_cold_start_counts_are_provisional_and_recomputed_with_intrinsics(board):
    data = {c: [at_scale(board, 'medium', 'a')] for c in ('L', 'R')}
    cold = ag.analyze_guidance(board, CameraModel.PINHOLE, data, {})
    assert cold.distance_counts == {} and cold.approximate_distance_counts == {'medium': 1}
    warm = ag.analyze_guidance(board, CameraModel.PINHOLE, data,
                               {c: intrinsics() for c in data})
    assert warm.distance_counts == {'medium': 1} and not warm.approximate_distance_counts


def test_uncertain_history_does_not_close_distance_gap(board):
    cal = intrinsics()
    cal.rms_error = 3.
    data = {c: [at_scale(board, 'medium', 'a')] for c in ('L', 'R')}
    result = ag.analyze_guidance(board, CameraModel.PINHOLE, data, {c: cal for c in data})
    assert not result.distance_counts
    assert set(result.recommendation.missing_distances) == {'small', 'medium', 'large'}


def test_replay_reports_separate_distance_evidence(board):
    from multicamera.calibration.guidance_replay import replay_observations
    obs = at_scale(board, 'medium', 'a')
    result = replay_observations(board, CameraModel.PINHOLE,
                                 [('a', {'left': obs, 'right': deepcopy(obs)})])
    assert result['final']['distance_counts'] == {}
    assert result['final']['approximate_distance_counts'] == {'medium': 1}
