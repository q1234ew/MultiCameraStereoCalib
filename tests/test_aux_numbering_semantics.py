"""Input attestations remain historical when the final algorithm changes numbering."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from multicamera.board.charuco_board import CharucoBoard
from multicamera.calibration.models import CameraModel
from multicamera.calibration.offline_aux import (
    confirm_offline_pair, detection_fingerprint, file_identity, numbering_provenance_summary,
    reordered_detection,
)
from multicamera.io.export import export_rig_json
from multicamera.streaming.stream_manager import AuxiliaryCameraConfig, StreamManager
from multicamera.ui.threads.calibration_worker import AuxCalibrationWorker
from multicamera.ui.widgets.calib_panel import CalibrationPanel
from test_offline_aux_loop import BOARD, synthetic_pairs
from test_session_restore_workspace import hashes, make_session


@pytest.fixture
def final_numbering_case(app, tmp_path):
    session, pair, rig = make_session(tmp_path / 'session', auxiliary=True)
    manager = StreamManager()
    manager.add_virtual_stereo_pair(pair)
    config = AuxiliaryCameraConfig('thermal', '', board_config=BOARD)
    manager.set_auxiliary_camera(config)
    board = CharucoBoard.from_dict(session.metadata['board'])
    panel = CalibrationPanel(manager, board)
    panel.restore_session_result(session, rig, board=board, auxiliary=config)
    rgb, _, _, records = synthetic_pairs(CameraModel.FISHEYE)
    records = [(key, left, reordered_detection(right, BOARD, 'reverse_both')
                if key in ('0000', '0004', '0007') else right) for key, left, right in records]
    reviews = []
    for key, left, right in records:
        sources = []
        for side, det in (('rgb', left), ('aux', right)):
            path = tmp_path / f'{key}_{side}.points'
            path.write_bytes(det.image_points.tobytes())
            sources.append(file_identity(path))
        review = confirm_offline_pair(dict(source_id=key, sources=sources,
            board=BOARD.to_dict(), image_accepted=True), 'identity', 'identity')
        review['numbering_check'] = dict(status='no_reliable_candidate',
            physical_correspondence_verified=False,
            selected_fingerprints=[detection_fingerprint(left), detection_fingerprint(right)])
        reviews.append(review)
    panel._aux_pairs = records
    panel._aux_offline_records = [(key, None, None, left, right) for key, left, right in records]
    panel._aux_offline_confirmations = {row['source_id']: deepcopy(row) for row in reviews}
    panel._combo_aux_model.setCurrentIndex(panel._combo_aux_model.findData('fisheye_2'))
    panel._pending_aux_context = ('thermal', 'ir', BOARD, 'left')
    panel._pending_aux_solver_key = panel._aux_solver_context()
    outcome, errors = {}, []

    def save_result(result):
        panel.set_result_model(result)
        outcome['saved'], outcome['save_errors'] = session.save_calibration_results(result)

    panel.calibration_finished.connect(save_result)
    worker = AuxCalibrationWorker(aux_id='thermal', board_config=BOARD,
        camera_model=CameraModel.PINHOLE, detections=records, rgb_left_intrinsics=rgb,
        min_intrinsic_frames=3, offline_model_key='fisheye_2', offline_reviews=reviews)
    worker.finished_ok.connect(lambda intr, extr: outcome.update(extrinsics=extr))
    worker.finished_ok.connect(panel._on_aux_calibration_finished)
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors, errors
    assert 'saved' in outcome and not outcome['save_errors']
    try:
        yield panel, manager, session, config, reviews, outcome
    finally:
        manager.stop_all()
        panel.close()


def test_actual_panel_success_saves_input_and_final_scopes_without_rewriting_history(
        final_numbering_case, tmp_path):
    panel, _, session, _, original_reviews, outcome = final_numbering_case
    aux = panel.rig.aux_cameras['thermal']
    info = aux.calibration_info
    summary = info['numbering_provenance']
    assert info['offline_reviews'] == original_reviews
    assert list(panel._aux_offline_confirmations.values()) == original_reviews
    assert 'numbering_provenance' not in outcome['extrinsics'].calibration_info
    assert summary['physical_correspondence_verified'] is False
    assert info['fit']['numbering_resolution']['physical_correspondence_verified'] is False
    assert info['status'] == 'offline_candidate_pending_physical_validation'
    assert summary['final_numbering_status'] == 'recorded'
    assert summary['expected_source_ids'] == [row['source_id'] for row in original_reviews]
    assert summary['automatically_reordered_source_ids'] == ['0000', '0004', '0007']
    for row in summary['frames']:
        assert row['input_corner_orders'] == dict(rgb='identity', aux='identity')
        assert row['input_review_declaration'] == 'manually_confirmed_per_pair'
        if row['source_id'] in summary['automatically_reordered_source_ids']:
            assert row['algorithm_aux_order_relative_to_input'] == 'reverse_both'
            assert row['input_aux_fingerprint'] != row['final_aux_fingerprint']
    assert '仅输入声明' in panel._lbl_result_metrics.text()
    assert '自动重排 3 组' in panel._lbl_result_metrics.text()
    assert '逐对排列见操作日志' in panel._lbl_result_metrics.text()
    assert '0000=reverse_both' not in panel._lbl_result_metrics.text()
    assert '最终物理对应未验证' in panel._lbl_aux_status.text()
    logs = panel._log_text.toPlainText()
    assert '用户输入声明 RGB=identity / AUX=identity' in logs
    assert '算法最终相对输入 AUX=reverse_both' in logs
    assert '算法最终相对输入 AUX=identity' in logs
    assert '不表示用户确认最终对应' in logs
    loaded_info = session.load_rig().aux_cameras['thermal'].calibration_info
    assert loaded_info['numbering_provenance'] == summary
    assert loaded_info['offline_reviews'] == original_reviews
    path = tmp_path / 'export.json'
    export_rig_json(panel.rig, path)
    exported = json.loads(path.read_text())['aux_cameras']['thermal']['calibration_info']
    assert exported['numbering_provenance'] == summary
    assert exported['offline_reviews'] == original_reviews


def test_restore_rebuilds_display_from_final_orders_without_new_admission_or_disk_writes(
        final_numbering_case):
    _, manager, session, config, reviews, _ = final_numbering_case
    before = hashes(session)
    restored = CalibrationPanel(manager, CharucoBoard.from_dict(session.metadata['board']))
    try:
        restored.restore_session_result(session, session.load_rig(), auxiliary=config)
        aux = restored.rig.aux_cameras['thermal']
        assert aux.calibration_info['offline_reviews'] == reviews
        assert '自动重排 3 组' in restored._lbl_result_metrics.text()
        assert '仅输入声明' in restored._lbl_aux_status.text()
        assert '最终物理对应未验证' in restored._lbl_aux_status.text()
        assert '不表示用户确认最终对应' in restored._log_text.toPlainText()
        restored._refresh_aux_capture_state()
        assert '自动重排 3 组' in restored._lbl_aux_status.text()
        assert '最终编号分层 0007:' in restored._log_text.toPlainText()
        assert not restored._aux_offline_confirmations and not restored._aux_pairs
        assert hashes(session) == before
    finally:
        restored.close()


@pytest.mark.parametrize('orders', ['identity', 'reverse_both'])
def test_summary_is_derived_not_a_physical_confirmation_or_a_mutation(orders):
    info = dict(offline_reviews=[dict(source_id='pair', corner_orders={'rgb': 'reverse_rows',
        'aux': 'reverse_rows'}, physical_corner_order='manually_confirmed_per_pair')],
        fit={'numbering_resolution': {'frames': [dict(source_id='pair', aux_order=orders)]}},
        numbering_provenance={'physical_correspondence_verified': True})
    before = deepcopy(info)
    summary = numbering_provenance_summary(info)
    assert info == before
    assert summary['physical_correspondence_verified'] is False
    assert summary['frames'][0]['input_corner_orders'] == before['offline_reviews'][0]['corner_orders']
    assert summary['automatically_reordered_source_ids'] == ([] if orders == 'identity' else ['pair'])
    summary['frames'][0]['input_corner_orders']['rgb'] = 'identity'
    assert info == before


def test_legacy_missing_final_numbering_is_unknown_not_claimed_as_confirmed():
    info = dict(offline_reviews=[dict(source_id='pair',
        physical_corner_order='manually_confirmed_per_pair')])
    summary = numbering_provenance_summary(info)
    assert summary['final_numbering_status'] == 'not_recorded'
    text = CalibrationPanel._aux_numbering_text(SimpleNamespace(calibration_info=info))
    assert '历史结果未记录' in text and '不推断为沿用输入编号' in text
    assert '最终物理对应未验证' in text
    empty = CalibrationPanel._aux_numbering_text(SimpleNamespace(calibration_info={}))
    assert '未记录用户逐对声明' in empty and '历史结果未记录' in empty


@pytest.mark.parametrize('value', [None, [], {'unknown': True}])
def test_malformed_optional_numbering_does_not_invent_an_order_or_break_display(value):
    info = dict(offline_reviews=None,
                fit={'numbering_resolution': {'frames': [{'source_id': 'old', 'aux_order': value}]}})
    summary = numbering_provenance_summary(info)
    assert summary['final_numbering_status'] == 'not_recorded'
    assert not summary['automatically_reordered_source_ids']
    assert summary['physical_correspondence_verified'] is False


@pytest.mark.parametrize('count', [1, 17, 100])
def test_numbering_label_stays_bounded_while_report_retains_every_order(count):
    ids = [f'long_source_identifier_{index:04d}' for index in range(count)]
    info = dict(source_ids=ids, fit=dict(source_ids=ids, numbering_resolution={
        'source_ids': ids, 'frames': [dict(source_id=key, aux_order='reverse_both') for key in ids]}))
    summary = numbering_provenance_summary(info)
    assert summary['final_numbering_status'] == 'recorded'
    assert summary['automatically_reordered_source_ids'] == ids
    text = CalibrationPanel._aux_numbering_text(SimpleNamespace(calibration_info=info))
    assert f'自动重排 {count} 组' in text and '逐对排列见操作日志' in text
    assert len(text) < 150
    assert all(key not in text for key in ids) and 'reverse_both' not in text


@pytest.mark.parametrize('basis', ['report', 'fit', 'both', 'resolution_only', 'none'])
def test_partial_frames_and_legacy_metadata_never_claim_complete_numbering(basis):
    row = dict(source_id='first', aux_order='reverse_both')
    info = dict(fit={'numbering_resolution': {'frames': [row]}})
    if basis in ('report', 'both'):
        info['source_ids'] = ['first', 'second']
    if basis in ('fit', 'both'):
        info['fit']['source_ids'] = ['first', 'second']
    if basis == 'resolution_only':
        info['fit']['numbering_resolution']['source_ids'] = ['first']
    summary = numbering_provenance_summary(info)
    assert summary['final_numbering_status'] == 'partial'
    if basis in ('report', 'fit', 'both'):
        assert summary['missing_source_ids'] == ['second']
    text = CalibrationPanel._aux_numbering_text(SimpleNamespace(calibration_info=info))
    assert '记录不完整' in text and '已知辅助相对输入自动重排 1 组' in text
    assert '最终物理对应未验证' in text


@pytest.mark.parametrize('case', ['contradictory_batches', 'duplicate_frames', 'unexpected_frame',
                                 'invalid_order', 'malformed_batch', 'duplicate_batch'])
def test_inconsistent_or_malformed_coverage_is_partial_not_complete(case):
    ids = ['first', 'second']
    info = dict(source_ids=ids, fit=dict(source_ids=ids, numbering_resolution={
        'source_ids': ids, 'frames': [dict(source_id=key, aux_order='identity') for key in ids]}))
    if case == 'contradictory_batches':
        info['fit']['source_ids'] = ['first']
    elif case == 'duplicate_frames':
        info['fit']['numbering_resolution']['frames'][1]['source_id'] = 'first'
    elif case == 'unexpected_frame':
        info['fit']['numbering_resolution']['frames'][1]['source_id'] = 'other'
    elif case == 'invalid_order':
        info['fit']['numbering_resolution']['frames'][1]['aux_order'] = None
    elif case == 'malformed_batch':
        info['fit']['source_ids'] = [dict(bad=True)]
    else:
        info['fit']['source_ids'] = ['first', 'first']
    before = deepcopy(info)
    summary = numbering_provenance_summary(info)
    assert summary['final_numbering_status'] == 'partial'
    assert summary['physical_correspondence_verified'] is False
    assert info == before
