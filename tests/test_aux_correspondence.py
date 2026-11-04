"""Diagnose symmetric-grid disagreement without silently rewriting observations."""

from copy import deepcopy
from types import SimpleNamespace
import cv2

import numpy as np
import pytest
from PySide6.QtWidgets import QDialogButtonBox
from PySide6.QtCore import QObject, Signal

from multicamera.calibration import joint_aux
from multicamera.calibration.offline_aux import reordered_detection
from multicamera.calibration.planar import PatternBoardConfig, PatternType
from multicamera.ui.threads.calibration_worker import AuxCorrespondenceWorker
from multicamera.ui.widgets.aux_offline_review import AuxOfflineReviewDialog
from test_offline_aux_loop import BOARD, review_fixture, synthetic_pairs


def mixed_pairs():
    rgb, aux, transform, records = synthetic_pairs()
    suspects = {'0000', '0004', '0007'}
    mixed = [(key, left, reordered_detection(right, BOARD, 'reverse_both')
              if key in suspects else right) for key, left, right in records]
    return rgb, aux, transform, mixed, suspects


def test_mixed_orders_identified_without_modifying_input():
    rgb, aux, _, records, suspects = mixed_pairs()
    before = deepcopy(records)
    report = joint_aux._correspondence_report(records, BOARD, rgb, aux)
    assert report['status'] == 'inconsistent'
    assert set(report['suspect_ids']) == suspects
    assert report['physical_correspondence_verified'] is False
    for old, new in zip(before, records):
        for a, b in zip(old[1:], new[1:]):
            np.testing.assert_array_equal(a.image_points, b.image_points)


def test_inspector_mono_seed_is_diagnostic_only():
    rgb, _, _, records, suspects = mixed_pairs()
    before = deepcopy(rgb.to_dict())
    report = joint_aux.inspect_aux_correspondences(records, BOARD, rgb, 'pinhole_constrained')
    assert set(report['suspect_ids']) == suspects
    assert report['annotation_scope'] == 'full_review_dataset'
    assert report['source_ids'] == [row[0] for row in records]
    assert rgb.to_dict() == before


def test_repeated_pose_is_ambiguous_not_an_automatic_relabel():
    rgb, aux, _, records = synthetic_pairs()
    repeated = [(str(i), *deepcopy(records[0][1:])) for i in range(9)]
    report = joint_aux._correspondence_report(repeated, BOARD, rgb, aux)
    assert report['status'] == 'ambiguous'
    assert not report['suspect_ids'] and not report['frames']


def test_rotated_sensor_does_not_assume_common_image_top_left():
    rgb, aux, _, records = synthetic_pairs()
    rotation = cv2.Rodrigues(np.array([0., 0., 1.7]))[0]
    translation = np.array([.025, .001, .004])
    for index, (_, left, right) in enumerate(records):
        pose = joint_aux._solve_board_pose(left, rgb)
        xyz = left.object_points @ cv2.Rodrigues(pose[0])[0].T + pose[1].ravel()
        right.image_points = joint_aux.project((xyz - translation) @ rotation, aux)
        if index % 2:
            right.image_points = right.image_points[::-1].copy()
    report = joint_aux._correspondence_report(records, BOARD, rgb, aux)
    assert report['status'] == 'inconsistent'
    assert report['suspect_ids'] == ['0001', '0003', '0005', '0007']


@pytest.mark.parametrize('rows,expected', [(4, ['identity', 'reverse_both']),
                                          (5, ['identity', 'reverse_rows'])])
def test_only_rigid_symmetries_of_asymmetric_grid_are_allowed(rows, expected):
    board = PatternBoardConfig(cols=6, rows=rows, square_size=.08,
                               pattern_type=PatternType.CIRCLES_GRID, asymmetric=True)
    assert joint_aux._rigid_grid_orders(board) == expected
    assert len(joint_aux._rigid_grid_orders(BOARD)) == 4


@pytest.mark.parametrize('failure', ['missing_pose', 'projection_error', 'opencv_error'])
def test_unresolved_pnp_is_reported_without_claiming_complete_consistency(monkeypatch, failure):
    rgb, aux, _, records = synthetic_pairs()
    missing = records[-1][1]
    solve, error = joint_aux._solve_board_pose, joint_aux._reprojection_error

    def pose(det, intr):
        if det is missing and failure == 'missing_pose':
            return None
        if det is missing and failure == 'opencv_error':
            raise cv2.error('synthetic pose error')
        return solve(det, intr)

    monkeypatch.setattr(joint_aux, '_solve_board_pose', pose)
    monkeypatch.setattr(joint_aux, '_reprojection_error', lambda det, *a:
        100. if det is missing and failure == 'projection_error' else error(det, *a))
    report = joint_aux._correspondence_report(records, BOARD, rgb, aux)
    assert report['status'] == 'partial' and report['checked_views'] == 8
    assert report['unresolved_ids'] == ['0008']
    assert not report['suspect_ids']


def test_disagreement_stops_before_unproductive_optimizer(monkeypatch):
    import scipy.optimize
    rgb, _, _, records, _ = mixed_pairs()
    monkeypatch.setattr(scipy.optimize, 'least_squares',
                        lambda *a, **k: pytest.fail('Wrong correspondence must stop before fitting'))
    with pytest.raises(ValueError, match='0000.*0004.*0007.*未自动翻转'):
        joint_aux._fit(records, BOARD, rgb, 'pinhole_constrained', resolve_correspondence=False)


def test_diagnostic_distinguishes_corner_numbering_from_image_pairing():
    rgb, _, _, records, _ = mixed_pairs()
    with pytest.raises(ValueError, match='角点编号方向待核对（不代表RGB/IR图像配错）'):
        joint_aux._fit(records, BOARD, rgb, 'pinhole_constrained', resolve_correspondence=False)


def test_explicit_reordering_recovers_full_holdout_pipeline():
    rgb, aux, expected, records, _ = mixed_pairs()
    report = joint_aux._correspondence_report(records, BOARD, rgb, aux)
    candidates = {row['source_id']: row['aux_order'] for row in report['frames']}
    corrected = [(key, left, reordered_detection(right, BOARD, candidates[key]))
                 for key, left, right in records]
    before = deepcopy(rgb.to_dict())
    _, result = joint_aux.calibrate_offline_aux(corrected, BOARD, rgb, 'pinhole_constrained')
    assert rgb.to_dict() == before
    np.testing.assert_allclose(result.T_rgb_left_aux, expected, atol=2e-4)
    heldout = []
    for fold in result.calibration_info['folds']:
        assert not set(fold['train_ids']) & set(fold['heldout_ids'])
        assert fold['fit']['converged'] and fold['fit']['health']['status'] == 'ok'
        assert fold['fit']['source_ids'] == fold['train_ids']
        heldout.extend(fold['heldout_ids'])
    assert sorted(heldout) == [row[0] for row in records]


@pytest.mark.parametrize('finite', [True, False])
def test_budget_exhaustion_and_nonfinite_result_never_accepted(monkeypatch, caplog, finite):
    import scipy.optimize
    rgb, _, _, records = synthetic_pairs()
    def failed(fun, initial, **kwargs):
        assert kwargs['max_nfev'] == 300
        assert kwargs['ftol'] == kwargs['xtol'] == kwargs['gtol'] == 1e-9
        assert kwargs['loss'] == 'soft_l1' and kwargs['f_scale'] == .5
        return SimpleNamespace(success=not finite, x=initial if finite else initial * np.nan,
            status=0, nfev=300, njev=290, cost=10., optimality=.2,
            message='The maximum number of function evaluations is exceeded.')
    monkeypatch.setattr(scipy.optimize, 'least_squares', failed)
    with pytest.raises(ValueError, match='未收敛.*nfev=300.*图像对 000'):
        joint_aux._fit(records, BOARD, rgb, 'pinhole_constrained')
    assert 'worst RGB/AUX RMS' in caplog.text


def test_candidate_requires_explicit_confirmation_and_cannot_bypass_quality(app, tmp_path):
    image, det, review, _ = review_fixture(tmp_path)
    dialog = AuxOfflineReviewDialog([('0000', image, image, det, det)], {'0000': review}, BOARD)
    dialog.confirm.click()
    assert dialog.decisions
    dialog.set_correspondence_report(dict(status='inconsistent', suspect_ids=['0000'],
        frames=[dict(source_id='0000', aux_order='reverse_both', suspect=True)]))
    assert dialog.orders[1].currentData() == 'identity'  # No automatic relabel or invalidation.
    assert dialog.decisions
    dialog.apply_candidate.click()
    assert dialog.orders[1].currentData() == 'reverse_both'
    assert not dialog.decisions
    dialog.confirm.click()
    assert dialog.decisions['0000']['corner_orders']['aux'] == 'reverse_both'
    assert dialog.decisions['0000']['correspondence_aid']['candidate']['aux_order'] == 'reverse_both'
    dialog.orders[1].setCurrentIndex(dialog.orders[1].findData('identity'))
    dialog.confirm.click()
    assert 'correspondence_aid' not in dialog.decisions['0000']
    dialog.set_checking(True)
    dialog.binding.setChecked(True)
    assert not dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    dialog.set_checking(False)
    assert not dialog.buttons.button(QDialogButtonBox.Save).isEnabled()  # Known conflict remains.
    dialog.apply_candidate.click()
    dialog.confirm.click()
    assert dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    dialog.reviews['0000']['image_accepted'] = False
    dialog._render()
    assert not dialog.confirm.isEnabled()


def test_ambiguous_and_failed_inspection_leave_no_candidate(app, tmp_path):
    image, det, review, _ = review_fixture(tmp_path)
    dialog = AuxOfflineReviewDialog([('0000', image, image, det, det)], {'0000': review}, BOARD)
    dialog.set_correspondence_report(dict(status='ambiguous', frames=[], suspect_ids=[]))
    assert not dialog.apply_candidate.isEnabled() and not dialog.decisions
    assert '不明确' in dialog.candidate_status.text()
    dialog.set_check_error('missing intrinsics')
    assert 'missing intrinsics' in dialog.candidate_status.text()
    assert not dialog.apply_candidate.isEnabled()


def test_worker_snapshots_inputs_and_reports_failure(app, monkeypatch):
    rgb, _, _, records, _ = mixed_pairs()
    worker = AuxCorrespondenceWorker(records, BOARD, rgb, 'pinhole_constrained')
    rgb.camera_matrix[0, 0] = 1.
    assert worker.inputs[2].camera_matrix[0, 0] != 1.
    results, errors = [], []
    worker.finished_ok.connect(results.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert results and not errors
    monkeypatch.setattr(joint_aux, 'inspect_aux_correspondences',
                        lambda *args: (_ for _ in ()).throw(ValueError('bad model')))
    worker.run()
    assert errors == ['bad model']


@pytest.mark.parametrize('action', ['success', 'error', 'closed', 'changed'])
def test_panel_ignores_closed_or_stale_inspection_and_tracks_busy(
        app, tmp_path, monkeypatch, action):
    from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
    from multicamera.streaming.stream_manager import (AuxiliaryCameraConfig, CameraConfig,
        StereoPairConfig, StreamManager)
    from multicamera.ui.widgets import calib_panel

    class ManualWorker(QObject):
        finished_ok = Signal(object)
        failed = Signal(str)
        finished = Signal()

        def __init__(self, *args):
            super().__init__(args[-1])

        def start(self):
            pass

    monkeypatch.setattr(calib_panel, 'AuxCorrespondenceWorker', ManualWorker)
    manager = StreamManager()
    manager.add_virtual_stereo_pair(StereoPairConfig('rgb', CameraConfig('left', ''),
                                                    CameraConfig('right', '')))
    manager.set_auxiliary_camera(AuxiliaryCameraConfig('ir', ''))
    panel = calib_panel.CalibrationPanel(manager, CharucoBoard(CharucoBoardConfig()))
    image, det, review, _ = review_fixture(tmp_path)
    panel._aux_offline_records = [('0000', image, image, det, det)]
    panel._aux_offline_reviews = {'0000': review}
    panel._intrinsics['left'] = synthetic_pairs()[0]
    panel._open_aux_offline_review()
    dialog = panel._aux_offline_dialog
    dialog.check.click()
    worker = panel._aux_order_worker
    assert worker is not None and panel.has_active_operations and panel._aux_operation_busy()
    if action == 'closed':
        dialog.reject()
    elif action == 'changed':
        panel._aux_dataset_revision += 1
    report = dict(status='inconsistent', suspect_ids=['0000'],
                  frames=[dict(source_id='0000', aux_order='reverse_both', suspect=True)])
    if action == 'error':
        worker.failed.emit('synthetic inspection failure')
    else:
        worker.finished_ok.emit(report)
    assert panel._aux_order_worker is worker  # Keep blocking until QThread.finished.
    worker.finished.emit()
    assert panel._aux_order_worker is None
    if action == 'success':
        assert dialog.apply_candidate.isEnabled()
    else:
        assert not dialog._candidates
        if action != 'closed':
            assert dialog.check.isEnabled() and '未完成' in dialog.candidate_status.text()
    assert not dialog.decisions and not panel._aux_pairs
    panel.close()
