"""Partial ChArUco views must not inherit fisheye-only training constraints."""

from copy import deepcopy
import json

import numpy as np
import pytest

from multicamera.board.detector import DetectionResult
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.model_selection import assess_camera_models
from multicamera.calibration.models import CameraModel
from multicamera.calibration.stereo import StereoCalibrator
from multicamera.calibration import validated_selection as selection
from test_model_selection import calibrators


def partial_views(count=12, *, small_common_subset=False):
    left, right, stereo = calibrators(count)
    board = left.board
    result = (IntrinsicCalibrator(board, min_frames=6),
              IntrinsicCalibrator(board, min_frames=6), StereoCalibrator(board, min_frames=6))
    for i in range(count):
        detections = []
        for cal in (left, right):
            ids = cal._all_ids[i].ravel()
            # Every view has a broad 2D grid, but their global intersection is empty.
            keep = ids != i if small_common_subset else ids % 4 != i % 4
            if not small_common_subset and i % 2:
                keep[np.flatnonzero(keep)[0]] = False
            indices = np.flatnonzero(keep)
            if cal is right:
                indices = indices[::-1]
            detections.append(DetectionResult(cal._all_corners[i][indices],
                                              cal._all_ids[i][indices], [], None, cal._image_size))
        fid = f'take/{i * 2 + 3:04d}'
        assert result[0].add_detection(detections[0], frame_id=fid)
        assert result[1].add_detection(detections[1], frame_id=fid)
        assert result[2].add_detection_pair(*detections, frame_id=fid)
    return result


def snapshot(calibrators):
    return [[deepcopy(cal.frame_ids), cal.model,
             [value.copy() for value in (cal._all_corners if isinstance(cal, IntrinsicCalibrator)
                                         else cal._obj_points)],
             [value.copy() for value in (cal._all_ids if isinstance(cal, IntrinsicCalibrator)
                                         else cal._img_points_left)]] for cal in calibrators]


def assert_unchanged(calibrators, before):
    after = snapshot(calibrators)
    for old, new in zip(before, after):
        assert old[:2] == new[:2]
        for old_values, new_values in zip(old[2:], new[2:]):
            assert len(old_values) == len(new_values)
            for old_array, new_array in zip(old_values, new_values):
                np.testing.assert_array_equal(old_array, new_array)


def test_pinhole_fold_preserves_each_pair_ids_counts_points_and_detaches_input():
    cals = partial_views()
    before = snapshot(cals)
    stereo = cals[2]
    indices = [i for i in range(stereo.num_frames) if i % 3 != 0]
    train = [stereo.frame_ids[i] for i in indices]
    assert not set.intersection(*(set(ids) for ids in stereo.frame_corner_ids))
    left, right, copied = selection._fold_calibrators(*cals, train)
    assert copied.frame_ids == train
    assert left.frame_ids == right.frame_ids == train
    assert copied.frame_common_corner_counts == [stereo.frame_common_corner_counts[i] for i in indices]
    for row, i in enumerate(indices):
        np.testing.assert_array_equal(copied.frame_corner_ids[row], stereo.frame_corner_ids[i])
        for attr in ('_obj_points', '_img_points_left', '_img_points_right'):
            original, clone = getattr(stereo, attr)[i], getattr(copied, attr)[row]
            np.testing.assert_array_equal(original, clone)
            assert not np.shares_memory(original, clone)
    assert_unchanged(cals, before)


def test_fisheye_fold_still_selects_identical_cross_view_ids():
    cals = partial_views(small_common_subset=True)
    for cal in cals:
        cal.model = CameraModel.FISHEYE
    before = snapshot(cals)
    train = cals[2].frame_ids[:9]
    copied = selection._fold_calibrators(*cals, train)[2]
    expected = set.intersection(*(set(ids) for ids in cals[2].frame_corner_ids[:9]))
    assert len(expected) >= 6
    assert all(set(ids) == expected for ids in copied.frame_corner_ids)
    assert all(count == len(expected) for count in copied.frame_common_corner_counts)
    assert_unchanged(cals, before)


def test_fisheye_insufficient_global_ids_is_not_silently_relaxed():
    cals = partial_views()
    for cal in cals:
        cal.model = CameraModel.FISHEYE
    before = snapshot(cals)
    with pytest.raises(ValueError, match='跨帧共同角点仅 0 个'):
        selection._fold_calibrators(*cals, cals[2].frame_ids[:9])
    assert_unchanged(cals, before)


@pytest.mark.parametrize('count,fold_count', [(12, 3), (15, 5)])
def test_actual_pinhole_validation_handles_partial_views_without_holdout_leakage(count, fold_count):
    cals = partial_views(count)
    before = snapshot(cals)
    logs = []
    *_, report = selection.calibrate_validated(*cals, log=logs.append)
    assert report['selected'] == 'standard'
    assert report['validation_status'] == 'passed'
    assert report['policy_version'] == 3
    assert report['training_corner_policy'] == 'per_pair_common_ids'
    assert any('允许不同图像对使用不同角点' in line for line in logs)
    validation = report['candidates']['standard']['validation']
    assert validation['rms_px'] < .01
    assert len(validation['folds']) == fold_count
    counts = dict(zip(cals[2].frame_ids, cals[2].frame_common_corner_counts))
    tests = []
    for fold in validation['folds']:
        assert fold['health']['status'] == 'ok'
        assert fold['training_corner_policy'] == 'per_pair_common_ids'
        assert fold['training_corner_counts'] == [counts[key] for key in fold['train_ids']]
        assert not set(fold['train_ids']) & set(fold['test_ids'])
        assert set(fold['train_ids']) | set(fold['test_ids']) == set(report['evaluated_ids'])
        tests.extend(fold['test_ids'])
    assert sorted(tests) == sorted(report['evaluated_ids'])
    assert_unchanged(cals, before)
    json.dumps(report, allow_nan=False)


def test_rational_candidate_can_validate_partial_views_but_keeps_health_gates(monkeypatch):
    cals = partial_views()
    before = snapshot(cals)
    health = selection.check_stereo_health

    def reject_plain(left, right, stereo):
        if left.dist_coeffs.size < 8:
            return {'status': 'failed', 'reasons': ['plain model rejected for test']}
        return health(left, right, stereo)

    monkeypatch.setattr(selection, 'check_stereo_health', reject_plain)
    *_, report = selection.calibrate_validated(*cals)
    assert report['selected'] == 'rational'
    assert report['candidates']['standard']['status'] == 'failed'
    assert report['candidates']['constrained']['status'] == 'failed'
    folds = report['candidates']['rational']['validation']['folds']
    assert len(folds) == 3
    assert all(fold['health']['status'] == 'ok' for fold in folds)
    assert all(fold['training_corner_policy'] == 'per_pair_common_ids' for fold in folds)
    assert_unchanged(cals, before)


def test_partial_views_still_fail_when_all_candidates_fail_health(monkeypatch):
    monkeypatch.setattr(selection, 'check_stereo_health',
                        lambda *args: {'status': 'failed', 'reasons': ['invalid lens']})
    with pytest.raises(ValueError, match='没有通过有效性及留出验证的候选模型'):
        selection.calibrate_validated(*partial_views())


def test_projection_comparison_only_intersects_fisheye_and_reports_truthful_training_ids(monkeypatch):
    cals = partial_views()
    before = snapshot(cals)
    calls = []
    original = StereoCalibrator.copy_with_common_ids

    def track(self):
        calls.append(self.model)
        return original(self)

    monkeypatch.setattr(StereoCalibrator, 'copy_with_common_ids', track)
    report = assess_camera_models(*cals)
    pinhole = report['candidates']['pinhole']
    assert pinhole['status'] == 'ok'
    assert pinhole['rms_px'] < .01
    assert calls == [CameraModel.FISHEYE] * 3
    assert report['candidates']['fisheye']['status'] == 'failed'
    assert all('跨帧共同角点' in row['error'] for row in report['candidates']['fisheye']['folds'])
    counts = dict(zip(cals[2].frame_ids, cals[2].frame_common_corner_counts))
    for row in pinhole['folds']:
        assert row['shared_training_ids'] is None
        assert row['training_corner_policy'] == 'per_pair_common_ids'
        assert row['training_corner_counts'] == [counts[key] for key in row['train_ids']]
    for pin, fish in zip(pinhole['folds'], report['candidates']['fisheye']['folds']):
        assert pin['test_ids'] == fish['test_ids']
        assert pin['train_ids'] == fish['train_ids']
    assert_unchanged(cals, before)
