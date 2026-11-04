"""Automatic discrete correspondence is independent per fold and never a physical claim."""

from copy import deepcopy

import cv2
import numpy as np
import pytest

from multicamera.calibration import joint_aux
from multicamera.calibration.offline_aux import detection_fingerprint, reordered_detection
from multicamera.calibration.grid_order import canonicalize_detection
from multicamera.calibration.models import CameraModel
from test_offline_aux_loop import BOARD, synthetic_pairs


ORDERS = ['identity', 'reverse_rows', 'reverse_cols', 'reverse_both']


def scrambled_pairs(model=CameraModel.PINHOLE, *, rotation=None):
    rgb, aux, expected, records = synthetic_pairs(model)
    if rotation is not None:
        expected[:3, :3] = cv2.Rodrigues(np.array([.01, -.02, rotation]))[0]
        for _, left, right in records:
            xyz = joint_aux._pose(left, rgb)
            right.image_points = joint_aux.project(
                (xyz - expected[:3, 3]) @ expected[:3, :3], aux)
    changed, orders = [], {}
    for i, (key, left, right) in enumerate(records):
        a, b = (i + 1) % 4, (i * 3) % 4
        changed.append((key, reordered_detection(left, BOARD, ORDERS[a]),
                        reordered_detection(right, BOARD, ORDERS[b])))
        orders[key] = ORDERS[a ^ b]
    return rgb, aux, expected, changed, orders


@pytest.mark.parametrize('model,key', [(CameraModel.PINHOLE, 'pinhole_constrained'),
                                      (CameraModel.FISHEYE, 'fisheye_2')])
def test_raw_mixed_orders_recover_rig_without_manual_relabeling(model, key):
    rgb, _, expected, records, expected_orders = scrambled_pairs(model)
    originals = deepcopy(records)
    rgb_before = deepcopy(rgb.to_dict())
    _, result = joint_aux.calibrate_offline_aux(records, BOARD, rgb, key)
    np.testing.assert_allclose(result.T_rgb_left_aux, expected, atol=2e-4)
    assert rgb.to_dict() == rgb_before
    info = result.calibration_info
    assert info['numbering_policy'] == 'automatic_legal_grid_correspondence_per_training_fold'
    assert info['heldout']['rgb_transfer_rms_px'] < .03
    assert info['heldout']['aux_transfer_rms_px'] < .01
    for fit in [f['fit'] for f in info['folds']] + [info['fit']]:
        resolution = fit['numbering_resolution']
        assert resolution['physical_correspondence_verified'] is False
        assert len(resolution['attempts']) <= 4
        assert {row['source_id']: row['aux_order'] for row in resolution['frames']} == {
            key: expected_orders[key] for key in fit['source_ids']}
    for old, new in zip(originals, records):
        for before, after in zip(old[1:], new[1:]):
            np.testing.assert_array_equal(before.image_points, after.image_points)
            np.testing.assert_array_equal(before.object_points, after.object_points)


@pytest.mark.parametrize('angle', [1.7, 3.0])
def test_image_canonical_origins_do_not_assume_same_physical_origin_for_rotated_sensor(angle):
    rgb, _, expected, records, _ = scrambled_pairs(rotation=angle)
    canonical = [(key, canonicalize_detection(left, BOARD), canonicalize_detection(right, BOARD))
                 for key, left, right in records]
    _, result = joint_aux.calibrate_offline_aux(canonical, BOARD, rgb, 'pinhole_constrained')
    np.testing.assert_allclose(result.T_rgb_left_aux, expected, atol=2e-4)
    assert any(row['aux_order'] != 'identity'
               for row in result.calibration_info['fit']['numbering_resolution']['frames'])


def test_each_fold_owns_correspondence_initialization_and_holdout_only_uses_frozen_model(monkeypatch):
    rgb, _, _, records, expected_orders = scrambled_pairs()
    original_fit = joint_aux._fit
    calls = []

    def track(training, *args):
        calls.append([row[0] for row in training])
        return original_fit(training, *args)

    monkeypatch.setattr(joint_aux, '_fit', track)
    _, result = joint_aux.calibrate_offline_aux(records, BOARD, rgb, 'pinhole_constrained')
    info = result.calibration_info
    assert len(calls) == 4
    heldout = []
    for fit_ids, fold in zip(calls, info['folds']):
        assert fit_ids == fold['train_ids'] == fold['fit']['numbering_resolution']['source_ids']
        assert not set(fit_ids) & set(fold['heldout_ids'])
        discrete = fold['numbering_resolution']
        assert discrete['parameter_refit'] is False
        assert discrete['physical_correspondence_verified'] is False
        assert {row['source_id']: row['aux_order'] for row in discrete['frames']} == {
            key: expected_orders[key] for key in fold['heldout_ids']}
        heldout.extend(fold['heldout_ids'])
    assert calls[-1] == [row[0] for row in records]
    assert sorted(heldout) == [row[0] for row in records]


def test_frozen_transform_assignment_never_runs_intrinsic_or_extrinsic_calibration(monkeypatch):
    rgb, aux, transform, records, expected_orders = scrambled_pairs()
    before = deepcopy((rgb.to_dict(), aux.to_dict(), records))
    monkeypatch.setattr(joint_aux.AuxIntrinsicCalibrator, 'calibrate',
                        lambda *a: pytest.fail('Held-out assignment must not fit camera parameters'))
    aligned, decisions = joint_aux._resolve_for_transform(records, BOARD, rgb, aux, transform)
    assert {row['source_id']: row['aux_order'] for row in decisions} == expected_orders
    assert joint_aux.transfer_metrics(aligned, rgb, aux, transform)['aux_transfer_rms_px'] < 1e-6
    assert (rgb.to_dict(), aux.to_dict()) == before[:2]
    for old, new in zip(before[2], records):
        np.testing.assert_array_equal(old[2].image_points, new[2].image_points)


def test_weak_permutation_margin_cannot_be_mislabeled_as_verified():
    rgb, aux, transform, records = synthetic_pairs()
    key, left, right = records[0]
    small = deepcopy(aux)
    small.camera_matrix[0, 0] *= .005
    small.camera_matrix[1, 1] *= .005
    xyz = (joint_aux._pose(left, rgb) - transform[:3, 3]) @ transform[:3, :3]
    right.image_points = joint_aux.project(xyz, small)
    with pytest.raises(ValueError, match='候选差距不足'):
        joint_aux._resolve_for_transform([(key, left, right)], BOARD, rgb, small, transform)


def test_repeated_pose_never_runs_joint_optimizer_to_invent_a_unique_label(monkeypatch):
    rgb, aux, _, records = synthetic_pairs()
    repeated = [(str(i), *deepcopy(records[0][1:])) for i in range(9)]
    monkeypatch.setattr(joint_aux.AuxIntrinsicCalibrator, 'calibrate', lambda *a: deepcopy(aux))
    monkeypatch.setattr(joint_aux, '_fit_fixed_orders',
                        lambda *a, **k: pytest.fail('Repeated observations must remain ambiguous'))
    with pytest.raises(ValueError, match='共识不足.*重复或共面'):
        joint_aux._fit(repeated, BOARD, rgb, 'pinhole_constrained')


def test_wrong_photo_pairs_are_not_silently_repaired_by_corner_permutations():
    rgb, _, _, records = synthetic_pairs()
    permutation = [4, 0, 8, 2, 7, 1, 5, 3, 6]
    wrong = [(key, left, records[permutation[i]][2]) for i, (key, left, _) in enumerate(records)]
    with pytest.raises(ValueError, match='对应|共识|留出|深度|有效性'):
        joint_aux.calibrate_offline_aux(wrong, BOARD, rgb, 'pinhole_constrained')


def test_numbering_resolution_records_exact_solver_fingerprints():
    rgb, _, _, records, _ = scrambled_pairs()
    _, _, fit = joint_aux._fit(records, BOARD, rgb, 'pinhole_constrained')
    for decision, (key, left, right) in zip(fit['numbering_resolution']['frames'], records):
        assert decision['source_id'] == key
        assert decision['rgb_input_fingerprint'] == detection_fingerprint(left)
        assert decision['aux_input_fingerprint'] == detection_fingerprint(right)
        actual = reordered_detection(right, BOARD, decision['aux_order'])
        assert decision['aux_solver_fingerprint'] == detection_fingerprint(actual)


@pytest.mark.parametrize('factor,method', [(1.13, 'rigid_consensus'),
    (1.3, 'bounded_rotation_hypotheses_joint_refinement')])
def test_biased_mono_focal_seed_is_refined_without_weakening_final_consensus(monkeypatch, factor, method):
    rgb, aux, expected, records, _ = scrambled_pairs(CameraModel.FISHEYE)
    biased = deepcopy(aux)
    biased.camera_matrix[0, 0] *= factor
    biased.camera_matrix[1, 1] *= factor
    monkeypatch.setattr(joint_aux.AuxIntrinsicCalibrator, 'calibrate', lambda *a: deepcopy(biased))
    intr, transform, fit = joint_aux._fit(records, BOARD, rgb, 'fisheye_2')
    np.testing.assert_allclose(transform, expected, atol=2e-4)
    np.testing.assert_allclose(intr.camera_matrix, aux.camera_matrix, atol=.02)
    info = fit['numbering_resolution']
    assert info['method'] == method
    assert info['final_pose_consensus']['checked_views'] == len(records)
    assert not info['final_pose_consensus']['unresolved_ids']


def test_rotation_initialization_does_not_discard_uncovered_views():
    rgb, aux, _, records = synthetic_pairs()
    # A wrong photo has no compatible rotation within the unchanged 8-degree gate.
    wrong = deepcopy(records)
    rotation = cv2.Rodrigues(np.array([.5, .4, .3]))[0]
    xyz = wrong[-1][2].object_points @ rotation.T + [-.2, -.1, 1.]
    wrong[-1][2].image_points = joint_aux.project(xyz, aux)
    assert not joint_aux._rotation_hypotheses(wrong, BOARD, rgb, aux)


def test_coplanar_translation_cannot_calibrate_optics_despite_unique_numbering(monkeypatch):
    rgb, aux, transform, records = synthetic_pairs()
    for i, (_, left, right) in enumerate(records):
        xyz = left.object_points + [-.4 + (i % 3) * .3, -.3 + (i // 3) * .2, 1.]
        left.image_points = joint_aux.project(xyz, rgb)
        right.image_points = joint_aux.project((xyz - transform[:3, 3]) @ transform[:3, :3], aux)
    assert joint_aux._correspondence_report(records, BOARD, rgb, aux)['status'] == 'consistent'
    monkeypatch.setattr(joint_aux.AuxIntrinsicCalibrator, 'calibrate',
                        lambda *a: pytest.fail('A single global plane does not identify the camera'))
    with pytest.raises(ValueError, match='共面观测'):
        joint_aux._fit(records, BOARD, rgb, 'pinhole_constrained')


def weak_noisy_pairs():
    rgb, aux, transform, records = synthetic_pairs(CameraModel.FISHEYE)
    objects = records[0][1].object_points
    rng = np.random.default_rng(13)
    changed = []
    for i in range(12):
        rotation = cv2.Rodrigues(np.array([
            .02 * np.sin(i), .025 * np.cos(i * .7), .01 * np.sin(i * .3)]))[0]
        xyz = objects @ rotation.T + [-.25 + (i % 3) * .04,
                                      -.15 + (i % 2) * .02, .6 + (i % 5) * .15]
        left, right = deepcopy(records[0][1:])
        left.image_points = joint_aux.project(xyz, rgb)
        right.image_points = joint_aux.project(
            (xyz - transform[:3, 3]) @ transform[:3, :3], aux) + rng.normal(0, .2, (24, 2))
        changed.append((f'{i:04d}', left, right))
    return rgb, aux, transform, changed


def test_noisy_planar_pnp_does_not_veto_unique_healthy_joint_projection(caplog):
    rgb, _, expected, records = weak_noisy_pairs()
    with caplog.at_level('INFO', logger=joint_aux.__name__):
        intr, transform, fit = joint_aux._fit(records, BOARD, rgb, 'fisheye_2')
    assert fit['converged'] and fit['health']['status'] == 'ok'
    assert np.linalg.norm(transform[:3, 3] - expected[:3, 3]) < .002
    resolution = fit['numbering_resolution']
    # The PnP diagnostic remains visible instead of claiming unanimous pose evidence.
    assert resolution['final_pose_consensus']['unresolved_ids']
    assert resolution['pose_consensus_role'] == 'diagnostic_only_after_joint_fit'
    assert resolution['physical_correspondence_verified'] is False
    assert any(attempt.get('pose_diagnostic_disagreement_ids') for attempt in resolution['attempts'])
    assert 'Single-view PnP diagnostic differs' in caplog.text
    assert len(resolution['frames']) == len(records)
    for frame in resolution['frames']:
        assert frame['aux_order'] == 'identity'
        assert frame['runner_aux_transfer_rms_px'] - frame['best_aux_transfer_rms_px'] >= 2.
    metrics = joint_aux.transfer_metrics(records, rgb, intr, transform)
    assert metrics['rgb_transfer_rms_px'] < joint_aux.LIMITS['rgb_transfer_rms_px']
    assert metrics['aux_transfer_rms_px'] < joint_aux.LIMITS['aux_transfer_rms_px']


def test_noisy_joint_projection_still_requires_disjoint_heldout_validation():
    rgb, _, expected, records = weak_noisy_pairs()
    _, result = joint_aux.calibrate_offline_aux(records, BOARD, rgb, 'fisheye_2')
    report = result.calibration_info
    assert len(report['folds']) == 3
    assert report['status'] == 'offline_candidate_pending_physical_validation'
    assert report['heldout']['rgb_transfer_rms_px'] < joint_aux.LIMITS['rgb_transfer_rms_px']
    assert report['heldout']['aux_transfer_rms_px'] < joint_aux.LIMITS['aux_transfer_rms_px']
    assert np.linalg.norm(result.T_rgb_left_aux[:3, 3] - expected[:3, 3]) < .002
    for fold in report['folds']:
        assert not set(fold['train_ids']) & set(fold['heldout_ids'])
        assert fold['fit']['converged'] and fold['fit']['health']['status'] == 'ok'
        assert fold['numbering_resolution']['parameter_refit'] is False


@pytest.mark.parametrize('failure', ['margin', 'assignment', 'pixel_limit', 'equivalent_solutions'])
def test_diagnostic_pnp_never_bypasses_projection_or_multiple_solution_gates(monkeypatch, failure):
    rgb, aux, transform, records = synthetic_pairs()
    assignment = {row[0]: 'identity' for row in records}
    monkeypatch.setattr(joint_aux.AuxIntrinsicCalibrator, 'calibrate', lambda *a: deepcopy(aux))
    monkeypatch.setattr(joint_aux, '_correspondence_report', lambda *a: dict(
        status='ambiguous', frames=[], unresolved_ids=list(assignment)))
    hypotheses = [assignment, dict(assignment)] if failure == 'equivalent_solutions' else [assignment]
    monkeypatch.setattr(joint_aux, '_rotation_hypotheses', lambda *a: hypotheses)
    monkeypatch.setattr(joint_aux, '_fit_fixed_orders', lambda *a, **k:
        (deepcopy(aux), transform.copy(), {'converged': True, 'nfev': 1}))
    original = joint_aux._resolve_for_transform

    def resolve(*args):
        if failure == 'margin':
            raise ValueError('候选差距不足')
        aligned, frames = original(*args)
        if failure == 'assignment':
            frames[0]['aux_order'] = 'reverse_both'
        return aligned, frames

    monkeypatch.setattr(joint_aux, '_resolve_for_transform', resolve)
    if failure == 'pixel_limit':
        monkeypatch.setattr(joint_aux, 'transfer_metrics', lambda *a: dict(
            rgb_transfer_rms_px=2.01, aux_transfer_rms_px=.1))
    message = {'margin': '候选差距不足', 'assignment': '投影不支持',
               'pixel_limit': '既有像素门槛', 'equivalent_solutions': '近似等价解'}[failure]
    with pytest.raises(ValueError, match=message):
        joint_aux._fit(records, BOARD, rgb, 'pinhole_constrained')
