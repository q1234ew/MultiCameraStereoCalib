"""Distance stages, tilt-independent depth feedback and practical height ordering."""

from copy import deepcopy

import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration import adaptive_guidance as ag
from multicamera.calibration.models import CameraModel
from test_adaptive_guidance import observation, intrinsics, diverse
from test_adaptive_capture_ui import activate, settle
from test_multimodal_preview import LEFT, RIGHT
from test_stereo_capture_ui import sample, publish_stable

pytest_plugins = ['test_stereo_capture_ui']


@pytest.fixture
def board():
    return CharucoBoard(CharucoBoardConfig(cols=9, rows=7, square_length=.035,
                                          marker_length=.025))


def test_explicit_middle_near_far_then_height_groups(board):
    data = diverse(board, count=22)
    for sequence in data.values():
        for i, obs in enumerate(sequence):
            obs.corners[:, :, 0] += 3 * i
    recs = [ag.analyze_guidance(board, CameraModel.PINHOLE,
            {c: seq[:n] for c, seq in data.items()},
            {c: intrinsics() for c in data}).recommendation for n in range(23)]
    # These observations are all far away: actual evidence supersedes request index.
    assert tuple(r.scale for r in recs[:7]) == ('medium',) * 7
    assert all((r.zone_row, r.zone_col) == (1, 1) for r in recs[:7])
    heights = [r.zone_row for r in recs]
    assert [h for i, h in enumerate(heights) if i == 0 or h != heights[i - 1]] == [1, 0, 2]
    assert len({r.tilt for r in recs[:7]}) >= 4
    # Re-analysis and duplicate frames cannot consume scheduling slots.
    once = ag.analyze_guidance(board, CameraModel.PINHOLE, data, {})
    duplicate = {c: [*seq, deepcopy(seq[-1])] for c, seq in data.items()}
    assert ag.analyze_guidance(board, CameraModel.PINHOLE, duplicate, {}).recommendation == once.recommendation


@pytest.mark.parametrize('model', list(CameraModel))
def test_equivalent_distance_not_confused_by_tilt_or_partial_points(board, model):
    cal = intrinsics(model)
    front = observation(board, model=model, depth=.8)
    tilted = observation(board, model=model, depth=.8, angles=(25, -25, 0))
    p_front = ag.measure_pose(board, front, cal)
    p_tilt = ag.measure_pose(board, tilted, cal)
    assert p_front is not None and p_tilt is not None
    assert p_tilt['distance_area'] == pytest.approx(p_front['distance_area'], rel=1e-4)
    partial = deepcopy(tilted)
    partial.corners, partial.ids = partial.corners[8:], partial.ids[8:]
    p_partial = ag.measure_pose(board, partial, cal)
    assert p_partial['distance_area'] == pytest.approx(p_front['distance_area'], rel=1e-4)
    farther = ag.measure_pose(board, observation(board, model=model, depth=1.6), cal)
    assert farther['distance_area'] == pytest.approx(p_front['distance_area'] / 4, rel=1e-4)


@pytest.mark.parametrize('areas,term,ready', [([.01, .02], '靠近', False),
    ([.15, .2], '远离', False), ([.06, .08], '距离到位', True),
    ([.01, .2], '判断冲突', False), ([], '等待', False), ([np.nan, .07], '等待', False)])
def test_both_eye_distance_feedback(areas, term, ready):
    ok, message = ag.distance_feedback('medium', areas)
    assert ok is ready and term in message


def test_infeasible_far_distance_never_selected(board):
    cal = intrinsics()
    cal.image_size = (160, 120)
    cal.camera_matrix = np.array([[80., 0., 80.], [0., 80., 60.], [0., 0., 1.]])
    assert not ag._candidate_feasible(board, (1, 1, 'front', 'small'), [cal.image_size])
    data = {c: [] for c in ('L', 'R')}
    result = ag.analyze_guidance(board, CameraModel.PINHOLE, data, {c: cal for c in data})
    rec = result.recommendation
    assert rec.feasible and rec.scale != 'small'
    assert '受限：远距离' in result.summary and '不计为已完成' in result.summary


def test_live_distance_changes_text_and_auto_gate(stereo_bench, app, monkeypatch):
    win = stereo_bench
    guide = activate(win, app)
    det, frame, intr = sample(win)
    win._calib_panel._intrinsics.update({LEFT: intr, RIGHT: intr})
    publish_stable(win, det, frame)
    settle(guide, app)
    pair, samples = guide._current_samples()
    target = guide._target()
    assert samples
    for area, message, ready in ((.02, '靠近', False), (.2, '远离', False),
                                 (.07, '距离到位', True)):
        monkeypatch.setattr(guide, '_pose_measurement', lambda *args, a=area:
                            {'yaw': 0., 'pitch': 0., 'depth': 1., 'distance_area': a,
                             'distance_reliable': True})
        ok, text = guide._distance_status(pair, samples, target)
        assert ok is ready and message in text and 'PnP深度等效' in text
        aligned, _ = guide._adaptive_aligned(pair, samples, target)
        assert aligned is ready
        guide.refresh()
        assert message in guide.reason.toolTip()
        if not ready:
            assert message in guide.instruction.text()
    monkeypatch.setattr(guide, '_pose_measurement', lambda *args: None)
    _, text = guide._distance_status(pair, samples, target)
    assert '非实测距离' in text
    assert not guide._adaptive_aligned(pair, samples, target)[0]


def test_distance_animation_moves_fixed_size_board_along_camera_axis():
    from multicamera.ui.widgets.capture_guide import CaptureTarget, BoardTilt
    from multicamera.ui.widgets.stereo_schematic import board_point, motion_description
    for scale, sign, text in (('large', -1, '靠近'), ('small', 1, '远离')):
        target = CaptureTarget(1, 1, BoardTilt.FRONT, scale=scale)
        start = np.array(board_point(0, 0, 0, target))
        end = np.array(board_point(0, 0, 1, target))
        assert sign * (end[2] - start[2]) > 0
        np.testing.assert_allclose(end[:2], start[:2])
        width_start = np.subtract(board_point(1, 0, 0, target), board_point(-1, 0, 0, target))
        width_end = np.subtract(board_point(1, 0, 1, target), board_point(-1, 0, 1, target))
        np.testing.assert_allclose(width_end, width_start)
        assert text in motion_description(target)


def test_missing_distances_get_bounded_repairs_and_remain_visible(board):
    data = {c: [observation(board, i, depth=.6, center=(.003 * i, 0)) for i in range(25)]
            for c in ('L', 'R')}
    provided = {c: intrinsics() for c in data}
    recs = [ag.analyze_guidance(board, CameraModel.PINHOLE,
            {c: seq[:n] for c, seq in data.items()}, provided).recommendation for n in range(26)]
    repairs = [r for r in recs if r.repairing_distance]
    assert len(repairs) == ag.MAX_DISTANCE_REPAIR_REQUESTS
    assert [r.scale for r in repairs] == ['large', 'large', 'small', 'small']
    assert {r.scale for r in repairs} == {'large', 'small'}
    assert all(r.zone_row == r.zone_col == 1 for r in repairs)
    assert all(r.missing_distances == ('large', 'small') for r in recs[7:])
    assert not recs[-1].repairing_distance
    summary = ag.analyze_guidance(board, CameraModel.PINHOLE, data, provided).summary
    assert '距离待补' in summary and '已达上限' in summary
    assert '不代表距离采集完成' in summary
    # Undo reconstructs the same pending repair from the remaining evidence.
    shorter = ag.analyze_guidance(board, CameraModel.PINHOLE,
                                 {c: seq[:7] for c, seq in data.items()}, provided)
    assert shorter.recommendation == recs[7]


def test_actual_distance_evidence_clears_gaps_without_repair(board):
    data = {c: [] for c in ('L', 'R')}
    provided = {c: intrinsics() for c in data}
    area = ag._hull_area(board.object_points[:, :2])
    cal = intrinsics()
    angles = [(0, 0, 0), (0, 25, 0), (25, 0, 0), (0, -25, 0),
              (-25, 0, 0), (0, 20, 0), (20, 0, 0)]
    for i, scale in enumerate(ag.CENTRAL_DISTANCE_PLAN):
        depth = np.sqrt(area * cal.camera_matrix[0, 0] * cal.camera_matrix[1, 1]
                        / (np.prod(cal.image_size) * ag.SCALE_AREA_TARGETS[scale]))
        for c in data:
            data[c].append(observation(board, i, depth=depth, angles=angles[i]))
    result = ag.analyze_guidance(board, CameraModel.PINHOLE, data, provided)
    assert not result.recommendation.missing_distances
    assert not result.recommendation.repairing_distance


@pytest.mark.parametrize('scale,area,ready', [
    ('small', .04, False), ('medium', .04, True),
    ('medium', .12, False), ('large', .12, True),
    ('small', .011, False), ('large', .46, False), ('large', .45, True),
])
def test_distance_band_boundaries_match_coverage(scale, area, ready):
    assert ag.distance_feedback(scale, [area, area])[0] is ready
    if ready:
        assert ag._scale(area) == scale


def test_single_ignored_gap_reports_per_distance_retry_limit(board):
    data = {c: [] for c in ('L', 'R')}
    angles = [(0, 0, 0), (0, 25, 0), (25, 0, 0), (0, -25, 0), (-25, 0, 0)]
    for i in range(12):
        for c in data:
            data[c].append(observation(board, i, depth=.8 if i in (5, 6) else .6,
                                       angles=angles[i % len(angles)], center=(.003 * i, 0)))
    result = ag.analyze_guidance(board, CameraModel.PINHOLE, data,
                                 {c: intrinsics() for c in data})
    assert result.recommendation.missing_distances == ('large',)
    assert not result.recommendation.repairing_distance
    assert '已达上限' in result.summary


@pytest.mark.parametrize('both_poses', [False, True])
def test_replay_and_client_share_distance_gate(stereo_bench, both_poses, monkeypatch):
    from multicamera.calibration.guidance_replay import _matches
    from multicamera.ui.widgets.capture_guide import CaptureTarget, BoardTilt

    guide = stereo_bench._stereo_capture
    pair = stereo_bench._calib_panel._sm.stereo_pairs['rgb']
    pose = {'yaw': 0., 'pitch': 0., 'depth': 1., 'distance_area': .07,
            'distance_reliable': True}
    poses = {LEFT: pose, RIGHT: pose if both_poses else None}
    monkeypatch.setattr(guide, '_pose_measurement', lambda cid, sample: poses[cid])
    samples = [{'valid': True, 'center': np.array([.5, .5]), 'area': .025,
                'distance_area': .025} for _ in range(2)]
    target = CaptureTarget(1, 1, BoardTilt.FRONT, scale='medium')
    measurements = [dict(s, pose=poses[cid], geometry_valid=True)
                    for s, cid in zip(samples, (LEFT, RIGHT))]
    expected, message, _, source = ag.evaluate_distance('medium', measurements)
    assert guide._distance_status([pair], samples, target) == (expected, message)
    rec = ag.GuidanceRecommendation(1, 1, 'front', 'medium', '', both_poses, 0)
    verdict = _matches(rec, measurements)
    assert verdict['composition_match'] is both_poses
    assert expected is both_poses
    assert source == ('pnp' if both_poses else 'uncertain')


def test_live_metric_distance_disappears_when_pose_is_untrusted(stereo_bench, app, tmp_path):
    win = stereo_bench
    guide = activate(win, app)
    det, frame, intr = sample(win)
    win._calib_panel._intrinsics.update({LEFT: intr, RIGHT: intr})
    publish_stable(win, det, frame)
    settle(guide, app)
    assert '板中心深度' in guide.reason.text() and 'm' in guide.pose.text()
    assert '目标' in guide.reason.toolTip()
    assert '目标' not in guide.compact_summary.text()
    app.processEvents()
    win.grab().save(str(tmp_path / 'pnp-distance-credible.png'))
    bad = deepcopy(intr)
    bad.rms_error = 3.
    win._calib_panel._intrinsics.update({LEFT: bad, RIGHT: bad})
    publish_stable(win, deepcopy(det), frame)
    settle(guide, app)
    assert '板中心深度' not in guide.reason.text()
    assert '不显示米数' in guide.pose.text()
    assert '内参拟合误差偏大' in guide.reason.toolTip()
    assert guide.schematic.motion == 'hold'
    assert not guide.auto_ready()
    app.processEvents()
    win.grab().save(str(tmp_path / 'pnp-distance-uncertain.png'))


def test_uncertain_distance_blocks_raw_gate_as_well_as_preview(stereo_bench, app, monkeypatch):
    guide = activate(stereo_bench, app)
    det, frame, intr = sample(stereo_bench)
    intr.rms_error = 3.
    stereo_bench._calib_panel._intrinsics.update({LEFT: intr, RIGHT: intr})
    publish_stable(stereo_bench, det, frame)
    settle(guide, app)
    pairs, samples = guide._current_samples()
    ready, text = guide._adaptive_aligned(pairs, samples, guide._target())
    assert not ready and '距离待确认' in text
    # Simulate the preview gate having passed before the raw recheck.
    monkeypatch.setattr(guide, 'auto_ready', lambda: True)
    ready, text = guide.validate_capture('rgb', frame, frame, det, det)
    assert not ready and '距离待确认' in text
