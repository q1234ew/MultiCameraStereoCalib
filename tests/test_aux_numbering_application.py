"""Current numbering gates review application without altering raw observations."""

from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtWidgets import QDialogButtonBox

from multicamera.calibration.offline_aux import (
    ORDER_OPTIONS, confirm_offline_pair, detection_fingerprint, review_order_check,
)
from multicamera.calibration.planar import PatternBoardConfig, PatternType
from multicamera.ui.widgets.aux_offline_review import AuxOfflineReviewDialog
import test_aux_review_state as review_state
from test_aux_review_state import file_hashes, install_result, two_pair_dialog
from test_offline_aux_loop import BOARD, review_fixture

reviewed_panel = review_state.reviewed_panel


def report(order='reverse_both'):
    return dict(status='inconsistent', suspect_ids=['0000'],
        physical_correspondence_verified=False,
        frames=[dict(source_id='0000', aux_order=order, suspect=True)])


@pytest.mark.parametrize('candidate', ORDER_OPTIONS)
@pytest.mark.parametrize('rgb', ORDER_OPTIONS)
@pytest.mark.parametrize('aux', ORDER_OPTIONS)
def test_relative_numbering_checks_all_compositions(candidate, rgb, aux):
    flips = {'identity': 0, 'reverse_rows': 1, 'reverse_cols': 2, 'reverse_both': 3}
    decisions = {'0000': {'corner_orders': dict(rgb=rgb, aux=aux)}}
    state = review_order_check(decisions, report(candidate), BOARD)
    expected = (flips[rgb] ^ flips[aux]) == flips[candidate]
    assert state['matched_ids'] == (['0000'] if expected else [])
    assert state['conflict_ids'] == ([] if expected else ['0000'])
    assert not state['unchecked_ids']


@pytest.mark.parametrize('status', ['ambiguous', 'consistent', 'partial'])
def test_missing_candidate_never_claims_geometric_or_physical_validation(status):
    state = review_order_check({'0000': {'corner_orders': dict(rgb='identity', aux='identity')}},
                               dict(status=status, frames=[]), BOARD)
    assert state == dict(matched_ids=[], conflict_ids=[], unchecked_ids=['0000'],
                         invalid_order_ids=[])


@pytest.mark.parametrize('rows,legal', [(4, 'reverse_both'), (5, 'reverse_rows')])
def test_asymmetric_grid_cannot_use_nonrigid_common_flips(rows, legal):
    board = PatternBoardConfig(cols=6, rows=rows, square_size=.08,
                               pattern_type=PatternType.CIRCLES_GRID, asymmetric=True)
    for order in ORDER_OPTIONS:
        state = review_order_check({'0000': {'corner_orders': dict(rgb=order, aux=order)}},
                                   report('identity'), board)
        allowed = order in ('identity', legal)
        assert state['invalid_order_ids'] == ([] if allowed else ['0000'])
        assert state['matched_ids'] == (['0000'] if allowed else [])


def test_check_only_and_both_reversals_cannot_apply_conflicting_decisions(app, tmp_path):
    image, det, review, _ = review_fixture(tmp_path)
    decision = confirm_offline_pair(review, 'identity', 'identity')
    dialog = AuxOfflineReviewDialog([('0000', image, image, det, det)], {'0000': review},
                                    BOARD, decisions={'0000': decision})
    original = det.image_points.copy()
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.binding.setChecked(True)
    dialog.set_correspondence_report(report())
    assert dialog.decisions['0000'] == decision
    assert '当前编号冲突' in dialog.list.item(0).text()
    assert '当前编号冲突 1' in dialog.selection_status.text()
    assert not dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    dialog.accept()
    assert not accepted and '仅点击检查不会调整编号' in dialog.candidate_status.text()
    for order in dialog.orders:
        order.setCurrentIndex(order.findData('reverse_both'))
    dialog.confirm.click()
    dialog.accept()
    assert not accepted
    assert '与几何候选冲突' in dialog.details.text()
    # Equivalent choice: relabel RGB only, not necessarily AUX.
    dialog.orders[1].setCurrentIndex(dialog.orders[1].findData('identity'))
    dialog.confirm.click()
    assert dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    assert '与几何候选一致' in dialog.details.text()
    dialog.accept()
    assert accepted == [True]
    np.testing.assert_array_equal(det.image_points, original)


def test_excluding_conflict_allows_remaining_manual_pair_without_auto_admission(app, tmp_path):
    dialog = two_pair_dialog(tmp_path)
    dialog.confirm.click()  # Conflicting raw identity, retained but blocked.
    dialog.confirm.click()  # Pair 0001 has no reliable candidate.
    dialog.binding.setChecked(True)
    assert not dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    dialog.list.setCurrentRow(0)
    dialog.exclude.click()
    assert list(dialog.decisions) == ['0001']
    assert dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    assert '无可靠几何候选 1' in dialog.selection_status.text()


def test_failed_recheck_keeps_conflict_and_candidate_but_never_auto_changes_orders(app, tmp_path):
    dialog = two_pair_dialog(tmp_path)
    dialog.confirm.click()
    dialog.binding.setChecked(True)
    dialog.set_checking(True)
    dialog.set_check_error('synthetic model failure')
    assert not dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    dialog.list.setCurrentRow(0)
    assert dialog.apply_candidate.isEnabled()
    assert dialog.orders[1].currentData() == 'identity'
    assert '已有检查和编号冲突保留' in dialog.candidate_status.text()


def test_three_complete_pairs_require_inspection_before_application(app, tmp_path):
    image, det, review, _ = review_fixture(tmp_path)
    reviews = {str(i): {**deepcopy(review), 'source_id': str(i)} for i in range(3)}
    dialog = AuxOfflineReviewDialog([(key, image, image, det, det) for key in reviews],
                                    reviews, BOARD)
    for i in range(3):
        dialog.list.setCurrentRow(i)
        dialog.confirm.click()
    dialog.binding.setChecked(True)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    assert not dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    dialog.accept()
    assert not accepted and '请先点击' in dialog.candidate_status.text()
    dialog.set_check_error('failed')
    assert not dialog.buttons.button(QDialogButtonBox.Save).isEnabled()
    # Ambiguous consensus is explicitly manual, never labeled as a geometric match.
    dialog.set_correspondence_report(dict(status='ambiguous', frames=[], suspect_ids=[]))
    assert '无可靠几何候选 3' in dialog.selection_status.text()
    dialog.accept()
    assert accepted == [True]


def inspection(panel, order='reverse_both'):
    checked = dict(report(order), aux_model=panel._combo_aux_model.currentData(),
                   board=panel._aux_board_config().to_dict(),
                   fixed_rgb_intrinsics=panel._intrinsics[panel._rgb_left_camera_id()].to_dict(),
                   source_ids=[str(row[0]) for row in panel._aux_offline_records
                               if panel._aux_offline_reviews[str(row[0])]['image_accepted']])
    return dict(report=checked, input_fingerprints={
        str(key): [detection_fingerprint(rgb), detection_fingerprint(aux)]
        for key, _, _, rgb, aux in panel._aux_offline_records})


@pytest.mark.parametrize('failure', ['conflict', 'image_points', 'object_points'])
def test_panel_guard_rejects_conflict_or_stale_check_without_touching_result(
        reviewed_panel, failure):
    panel, source = reviewed_panel
    aux = install_result(panel)
    evidence = inspection(panel)
    decision = confirm_offline_pair(panel._aux_offline_reviews['0000'], 'identity', 'identity')
    if failure != 'conflict':
        decision = confirm_offline_pair(panel._aux_offline_reviews['0000'],
                                       'identity', 'reverse_both')
        getattr(panel._aux_offline_records[0][4], failure)[0, 0] += .01
    before = file_hashes(panel._session.session_dir) | file_hashes(source)
    revision = panel._aux_dataset_revision
    panel._apply_aux_offline_review({'0000': decision}, correspondence_check=evidence)
    assert panel._rig.aux_cameras['ir'] is aux and not panel._aux_pairs
    assert panel._aux_dataset_revision == revision and not panel._aux_offline_confirmations
    assert '离线复核未应用，保留当前数据' in panel._log_text.toPlainText()
    assert before == file_hashes(panel._session.session_dir) | file_hashes(source)


def test_panel_requires_check_before_applying_batch(reviewed_panel):
    panel, _ = reviewed_panel
    row = panel._aux_offline_records[0]
    panel._aux_offline_records = [(str(i), *deepcopy(row[1:])) for i in range(3)]
    original = panel._aux_offline_reviews['0000']
    panel._aux_offline_reviews = {str(i): {**deepcopy(original), 'source_id': str(i)}
                                  for i in range(3)}
    decisions = {key: confirm_offline_pair(review, 'identity', 'identity')
                 for key, review in panel._aux_offline_reviews.items()}
    panel._apply_aux_offline_review(decisions)
    assert not panel._aux_pairs
    assert '请先完成本批' in panel._log_text.toPlainText()


@pytest.mark.parametrize('rgb,aux', [('identity', 'reverse_both'), ('reverse_both', 'identity')])
def test_panel_stores_exact_selected_fingerprints_and_equivalent_orders(
        reviewed_panel, finish_aux_tasks, rgb, aux):
    panel, _ = reviewed_panel
    decisions = {'0000': confirm_offline_pair(panel._aux_offline_reviews['0000'], rgb, aux)}
    evidence = inspection(panel)
    panel._apply_aux_offline_review(decisions, correspondence_check=evidence)
    finish_aux_tasks(panel)
    assert len(panel._aux_pairs) == 1
    review = panel._aux_offline_confirmations['0000']
    assert review['corner_orders'] == dict(rgb=rgb, aux=aux)
    check = review['numbering_check']
    assert check['status'] == 'matches_geometric_candidate'
    assert check['physical_correspondence_verified'] is False
    assert check['raw_fingerprints'] == evidence['input_fingerprints']['0000']
    assert check['selected_fingerprints'] == [detection_fingerprint(det)
                                              for det in panel._aux_pairs[0][1:]]
    assert f'RGB={rgb} / IR辅助={aux}' in panel._log_text.toPlainText()
    for digest in check['selected_fingerprints']:
        assert digest[:12] in panel._log_text.toPlainText()
    assert 'numbering_check' not in decisions['0000']  # No mutation of dialog draft.


def test_fingerprint_is_portable_and_tracks_numbering_and_geometry(tmp_path):
    _, det, _, _ = review_fixture(tmp_path)
    before = detection_fingerprint(det)
    other = deepcopy(det)
    other.image_points = other.image_points.astype('>f8')
    other.object_points = other.object_points.astype('>f8')
    assert detection_fingerprint(other) == before
    other.image_points = other.image_points[::-1].copy()
    assert detection_fingerprint(other) != before


@pytest.mark.parametrize('change', ['model', 'intrinsics', 'board', 'source_ids'])
def test_panel_rejects_report_from_other_solver_binding_even_with_current_context(
        reviewed_panel, change):
    panel, _ = reviewed_panel
    aux = install_result(panel)
    decision = confirm_offline_pair(panel._aux_offline_reviews['0000'], 'identity', 'reverse_both')
    evidence = inspection(panel)
    if change == 'model':
        evidence['report']['aux_model'] = 'different_model'
    elif change == 'intrinsics':
        evidence['report']['fixed_rgb_intrinsics']['camera_matrix'][0][0] += 1.
    elif change == 'board':
        evidence['report']['board']['square_size'] *= 2
    else:
        evidence['report']['source_ids'] = ['another_source']
    panel._apply_aux_offline_review({'0000': decision}, panel._aux_solver_context(),
                                    correspondence_check=evidence)
    assert panel._rig.aux_cameras['ir'] is aux and not panel._aux_pairs
    assert '检查的模型、固定RGB内参、板配置或观测批次已变化' in panel._log_text.toPlainText()


def test_unchanged_application_updates_workspace_evidence_not_historical_result(
        reviewed_panel, finish_aux_tasks):
    panel, source = reviewed_panel
    decision = confirm_offline_pair(panel._aux_offline_reviews['0000'], 'identity', 'reverse_both')
    evidence = inspection(panel)
    evidence['report'].update(status='ambiguous', frames=[])
    panel._apply_aux_offline_review({'0000': decision}, correspondence_check=evidence)
    finish_aux_tasks(panel)
    assert panel._aux_offline_confirmations['0000']['numbering_check']['status'] == 'no_reliable_candidate'
    aux = install_result(panel)
    result_snapshot = deepcopy(aux.to_dict())
    before = file_hashes(panel._session.session_dir) | file_hashes(source)
    revision = panel._aux_dataset_revision
    evidence = inspection(panel)
    panel._apply_aux_offline_review({'0000': decision}, correspondence_check=evidence)
    check = panel._aux_offline_confirmations['0000']['numbering_check']
    assert check['status'] == 'matches_geometric_candidate'
    assert check['report'] == evidence['report']
    assert panel._rig.aux_cameras['ir'] is aux and panel._aux_dataset_revision == revision
    assert aux.to_dict() == result_snapshot
    assert before == file_hashes(panel._session.session_dir) | file_hashes(source)
