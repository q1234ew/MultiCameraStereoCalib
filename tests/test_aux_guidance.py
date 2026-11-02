"""Planar first pass, bounded repairs and IR-specific recommendations."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from multicamera.calibration.aux_capture_quality import AuxFrameQuality
from multicamera.calibration.aux_guidance import (
    AREA_RANGES, AREA_TARGETS, AuxTarget, analyze_aux_guidance, feasible_target, fit_target,
    guidance_action, initial_plan,
)
from multicamera.calibration.models import CameraIntrinsics, CameraModel
from multicamera.calibration.planar import (
    PatternBoardConfig, PlanarDetectionResult, PlanarPatternDetector,
)


def observations(count, *, intrinsics=True):
    board = PatternBoardConfig.ir_chessboard()
    obj = PlanarPatternDetector._make_object_points(board)
    intr = CameraIntrinsics(np.array([[210., 0., 128.], [0., 210., 96.], [0., 0., 1.]]),
                            np.zeros(5), (256, 192), rms_error=.2)
    rows = []
    for i in range(count):
        rv = np.array([.05 + i * .009, -.2 + i * .012, 0.])
        points = cv2.projectPoints(obj, rv, np.array([-.20 + i * .008, -.12, 1.2]),
                                   intr.camera_matrix, intr.dist_coeffs)[0].reshape(-1, 2)
        det = PlanarDetectionResult(True, obj.copy(), points.astype(np.float32),
                                    intr.image_size, board.pattern_type)
        rows.append((str(i), det, det))
    return board, rows, (intr, intr) if intrinsics else (None, None)


def test_first_pass_has_distances_before_same_height_edge_groups():
    plan = initial_plan()
    assert len(plan) == 17
    assert [t.scale for t in plan[:9]] == ['medium'] * 5 + ['large'] * 2 + ['small'] * 2
    assert [t.row for t in plan[9:]] == [1, 1, 0, 0, 0, 2, 2, 2]
    assert all(t.scale == 'small' for t in plan[9:])  # Leave room for full outer squares.


@pytest.mark.parametrize('count', [0, 8, 16])
def test_no_early_repair_or_reordering_after_intrinsics_appear(count, monkeypatch):
    board, rows, intr = observations(count)
    monkeypatch.setattr('multicamera.calibration.aux_guidance._bootstrap', lambda *args: None)
    cold = analyze_aux_guidance(board, CameraModel.PINHOLE, rows)
    warm = analyze_aux_guidance(board, CameraModel.PINHOLE, rows, intr)
    assert cold.phase == warm.phase == 'initial'
    assert cold.target == warm.target == initial_plan()[count]


def test_repair_is_bounded_even_when_distance_never_becomes_verified(monkeypatch):
    monkeypatch.setattr('multicamera.calibration.aux_guidance._bootstrap', lambda *args: None)
    for count, expected in ((17, 'repair'), (18, 'repair'), (20, 'repair'), (21, 'complete'),
                             (24, 'complete')):
        board, rows, _ = observations(count)
        result = analyze_aux_guidance(board, CameraModel.PINHOLE, rows)
        assert result.unique_count == count
        assert result.phase == expected
        assert result.repair_count <= 4
        assert result.missing  # No model is never misreported as distance success.
        assert (result.target is None) == (expected == 'complete')


def test_duplicates_invalid_sizes_and_grid_do_not_advance(monkeypatch):
    monkeypatch.setattr('multicamera.calibration.aux_guidance._bootstrap', lambda *args: None)
    board, rows, intr = observations(1)
    _, a, b = rows[0]
    incoming = rows + [('repeat', a, b), ('bad-size', replace(a, image_size=(640, 480)), b),
                       ('bad-grid', replace(a, image_points=a.image_points[:-1]), b)]
    before = a.image_points.copy()
    result = analyze_aux_guidance(board, CameraModel.PINHOLE, incoming, intr)
    assert result.unique_count == 1 and result.duplicate_count == 1
    assert result.sample_decisions == {'0': 'included', 'repeat': 'duplicate',
                                      'bad-size': 'size_mismatch', 'bad-grid': 'invalid_detection'}
    assert result.target == initial_plan()[1]
    np.testing.assert_array_equal(a.image_points, before)


def test_only_matching_sensor_model_and_size_supply_guidance_intrinsics(monkeypatch):
    monkeypatch.setattr('multicamera.calibration.aux_guidance._bootstrap', lambda *args: None)
    board, rows, intr = observations(3)
    result = analyze_aux_guidance(board, CameraModel.PINHOLE, rows,
                                   (intr[0], replace(intr[1], image_size=(1280, 720))))
    assert result.intrinsics[0] is intr[0] and result.intrinsics[1] is None
    assert result.pose_count == 0 and not result.distance_counts


def test_auxiliary_footprint_controls_distance_not_rgb_field_of_view():
    board = PatternBoardConfig.ir_chessboard()
    q_rgb = AuxFrameQuality(True, (), {'area': .01, 'center': [.5, .5]}, None)
    q_ir = AuxFrameQuality(True, (), {'area': .10, 'center': [.5, .5]}, None)
    stage, message = guidance_action(board, AuxTarget(), [q_rgb, q_ir])
    assert stage == 'tilt_unverified' and '人工确认' in message
    for area, word in ((.02, '靠近'), (.3, '远离')):
        stage, message = guidance_action(board, AuxTarget(tilt='up'),
                                         [q_rgb, replace(q_ir, metrics={'area': area})])
        assert stage == 'distance' and word in message and '正对' in message
        assert '上边' not in message


def test_pose_is_checked_in_rgb_reference_and_ir_distance_is_separate():
    rgb = AuxFrameQuality(True, (), {'area': .01, 'center': [.5, .5]},
                          {'yaw': 0, 'pitch': -25, 'distance_reliable': True})
    ir = AuxFrameQuality(True, (), {'area': .01, 'center': [.5, .5]},
                         {'distance_area': .1, 'distance_reliable': True})
    assert guidance_action(PatternBoardConfig.ir_chessboard(), AuxTarget(tilt='up'),
                            [rgb, ir])[0] == 'ready'
    assert guidance_action(PatternBoardConfig.ir_chessboard(), AuxTarget(tilt='down'),
                            [rgb, ir])[0] == 'tilt'


def test_low_resolution_or_large_envelope_never_claims_completion():
    board = PatternBoardConfig.ir_chessboard()
    assert feasible_target(board, AuxTarget(scale='large'), (256, 192))
    assert feasible_target(board, AuxTarget(0, 0, scale='small'), (256, 192))
    assert not feasible_target(board, AuxTarget(0, 0, scale='large'), (256, 192))
    assert not feasible_target(board, AuxTarget(scale='small'), (48, 32))


def test_wide_board_fits_near_band_without_requiring_nominal_area(monkeypatch):
    board = PatternBoardConfig(cols=12, rows=4, square_size=.08)
    obj = PlanarPatternDetector._make_object_points(board)
    points = obj[:, :2] / board.square_size * 17 + [34., 70.]
    rows = []
    for i in range(5):
        det = PlanarDetectionResult(True, obj.copy(), (points + [2 * i, 0]).astype(np.float32),
                                    (256, 192), board.pattern_type)
        rows.append((str(i), det, det))
    monkeypatch.setattr('multicamera.calibration.aux_guidance._bootstrap', lambda *args: None)
    result = analyze_aux_guidance(board, CameraModel.PINHOLE, rows)
    assert result.unique_count == 5 and result.phase == 'initial'
    assert result.target.scale == 'large'
    area = result.target.target_area
    assert AREA_RANGES['large'][0] <= area < AREA_TARGETS['large']
    span = np.sqrt(area * 256 * 192 * np.array([11 / 3, 3 / 11]))
    spacing = span / [11, 3]
    assert spacing.min() >= 4
    half = (span + 2 * spacing) / 2
    assert np.all([128, 96] - half >= 2 - 1e-8)
    assert np.all([128, 96] + half <= np.array([256, 192]) - 3 + 1e-8)
    # The target area is only a drawing reference, not a new narrow capture gate.
    qualities = [AuxFrameQuality(True, (), {'area': .194, 'center': [.5, .5]}, None)] * 2
    assert guidance_action(board, result.target, qualities)[0] == 'tilt_unverified'
    assert rows[0][1].image_points[0, 0] == 34


@pytest.mark.parametrize('size', [(0, 192), (256, 0), (True, 192), (256., 192), (24, 16)])
def test_aux_fit_rejects_invalid_or_unresolvable_sizes(size):
    assert fit_target(PatternBoardConfig.ir_chessboard(), AuxTarget(), size) is None


def test_aux_fit_searches_larger_area_when_nominal_spacing_is_too_small():
    board = PatternBoardConfig.ir_chessboard()
    fitted = fit_target(board, AuxTarget(scale='small'), (96, 64))
    assert fitted is not None
    assert AREA_TARGETS['small'] < fitted.target_area < AREA_RANGES['small'][1]


def test_truly_infeasible_aux_target_stays_blocked(monkeypatch):
    board, rows, _ = observations(1, intrinsics=False)
    rows = [(key, replace(a, image_size=(32, 24), image_points=a.image_points / 8),
             replace(b, image_size=(32, 24), image_points=b.image_points / 8))
            for key, a, b in rows]
    monkeypatch.setattr('multicamera.calibration.aux_guidance._bootstrap', lambda *args: None)
    result = analyze_aux_guidance(board, CameraModel.PINHOLE, rows)
    assert result.phase == 'blocked' and result.target is None
    assert result.unique_count == 1


def test_quality_feedback_translates_machine_codes():
    q = AuxFrameQuality(False, ('low_board_contrast',), {}, None)
    stage, text = guidance_action(PatternBoardConfig.ir_chessboard(), AuxTarget(), [q, q])
    assert stage == 'quality' and '对比不足' in text and 'low_board' not in text


def test_reverse_enumeration_is_duplicate_without_reordering_observations():
    board, rows, intr = observations(1)
    _, a, b = rows[0]
    reverse = replace(b, image_points=b.image_points[::-1].copy())
    original = reverse.image_points.copy()
    result = analyze_aux_guidance(board, CameraModel.PINHOLE,
                                   rows + [('reverse-ir', a, reverse)], intr)
    assert result.unique_count == 1 and result.duplicate_count == 1
    np.testing.assert_array_equal(reverse.image_points, original)
