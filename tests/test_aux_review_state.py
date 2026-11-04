"""Review drafts, non-destructive application, and saved auxiliary solver state."""

from copy import deepcopy
import hashlib
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QDialogButtonBox, QMessageBox

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration.models import AuxCameraCalibration, CameraModel
from multicamera.calibration.offline_aux import confirm_offline_pair, offline_pair_review
from multicamera.streaming.stream_manager import AuxiliaryCameraConfig, StreamManager
from multicamera.ui.widgets.aux_offline_review import AuxOfflineReviewDialog
from multicamera.ui.widgets.calib_panel import CalibrationPanel
from multicamera.ui import main_window
from test_offline_aux_loop import BOARD, make_session, review_fixture, synthetic_pairs
from test_session_restore_workspace import make_session as make_restore_session, hashes


def two_pair_dialog(tmp_path):
    image, det, review, _ = review_fixture(tmp_path)
    reviews = {'0000': review, '0001': {**deepcopy(review), 'source_id': '0001'}}
    dialog = AuxOfflineReviewDialog([(key, image, image, det, det) for key in reviews],
                                    reviews, BOARD)
    dialog.set_correspondence_report(dict(status='inconsistent', suspect_ids=['0000'],
        physical_correspondence_verified=False, source_ids=list(reviews),
        frames=[dict(source_id='0000', aux_order='reverse_both', suspect=True)]))
    return dialog


def test_candidate_draft_and_lineage_survive_navigation_without_admission(app, tmp_path):
    dialog = two_pair_dialog(tmp_path)
    dialog.apply_candidate.click()
    assert not dialog.decisions
    dialog.list.setCurrentRow(1)
    dialog.orders[0].setCurrentIndex(dialog.orders[0].findData('reverse_rows'))
    dialog.list.setCurrentRow(0)
    assert dialog.orders[1].currentData() == 'reverse_both'
    assert not dialog.decisions
    dialog.confirm.click()
    decision = dialog.decisions['0000']
    assert decision['corner_orders'] == {'rgb': 'identity', 'aux': 'reverse_both'}
    assert decision['correspondence_aid']['candidate']['aux_order'] == 'reverse_both'
    assert dialog.orders[0].currentData() == 'reverse_rows'
    assert '0001' not in dialog.decisions
    dialog.confirm.click()
    assert 'correspondence_aid' not in dialog.decisions['0001']


def test_exclusion_retains_matching_draft_aid_but_never_admits_automatically(app, tmp_path):
    dialog = two_pair_dialog(tmp_path)
    dialog.apply_candidate.click()
    dialog.confirm.click()
    dialog.list.setCurrentRow(0)
    dialog.exclude.click()
    dialog.list.setCurrentRow(1)
    dialog.list.setCurrentRow(0)
    assert dialog.orders[1].currentData() == 'reverse_both' and not dialog.decisions
    dialog.confirm.click()
    assert 'correspondence_aid' in dialog.decisions['0000']
    dialog.list.setCurrentRow(0)
    dialog.orders[1].setCurrentIndex(dialog.orders[1].findData('identity'))
    dialog.list.setCurrentRow(1)
    dialog.list.setCurrentRow(0)
    dialog.confirm.click()
    assert 'correspondence_aid' not in dialog.decisions['0000']


def test_cancel_does_not_publish_drafts_or_change_originals(app, tmp_path):
    dialog = two_pair_dialog(tmp_path)
    originals = [det.image_points.copy() for row in dialog.records for det in row[3:]]
    dialog.apply_candidate.click()
    dialog.reject()
    reopened = AuxOfflineReviewDialog(dialog.records, dialog.reviews, BOARD)
    assert reopened.orders[1].currentData() == 'identity' and not reopened.decisions
    for before, det in zip(originals, [det for row in dialog.records for det in row[3:]]):
        np.testing.assert_array_equal(before, det.image_points)


def test_empty_or_busy_review_cannot_be_applied_even_via_accept(app, tmp_path):
    dialog = two_pair_dialog(tmp_path)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.binding.setChecked(True)
    button = dialog.buttons.button(QDialogButtonBox.Save)
    assert not button.isEnabled()
    dialog.accept()
    assert not accepted
    dialog.apply_candidate.click()
    dialog.confirm.click()
    assert button.isEnabled()
    dialog.set_checking(True)
    dialog.accept()
    assert not accepted and not button.isEnabled()
    dialog.set_checking(False)
    dialog.list.setCurrentRow(0)
    dialog.exclude.click()
    assert not button.isEnabled()
    dialog.accept()
    assert not accepted


@pytest.fixture
def reviewed_panel(app, tmp_path, finish_aux_tasks):
    session, configs, rig = make_session(tmp_path / 'workspace')
    manager = StreamManager()
    manager.add_virtual_stereo_pair(configs[0])
    manager.set_auxiliary_camera(AuxiliaryCameraConfig('ir', ''))
    panel = CalibrationPanel(manager, CharucoBoard(CharucoBoardConfig()))
    panel.set_session(session)
    panel._pair_calibs, panel._rig = deepcopy(rig.pairs), rig
    panel._intrinsics['left'] = deepcopy(rig.pairs['rgb'].left_intrinsics)
    source = tmp_path / 'source'
    source.mkdir()
    review_fixture(source)
    panel._start_aux_image_import(str(source))
    finish_aux_tasks(panel)
    yield panel, source
    finish_aux_tasks(panel)
    manager.stop_all()
    panel.close()


def install_result(panel):
    _, intr, transform, _ = synthetic_pairs(CameraModel.FISHEYE)
    aux = AuxCameraCalibration('ir', 'ir', intr, transform, frame_count=6,
        board_config=BOARD.to_dict(), calibration_info=dict(
            status='offline_candidate_pending_physical_validation', aux_model='fisheye_2',
            heldout={'rgb_transfer_rms_px': .4, 'aux_transfer_rms_px': .3}))
    panel._rig.aux_cameras['ir'] = aux
    panel._aux_intrinsics = deepcopy(intr)
    files, errors = panel._session.save_calibration_results(panel._rig)
    panel.set_result_model(panel._rig)
    panel.set_result_storage(panel._session.result_directory, files, errors)
    return aux


def file_hashes(folder):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}


@pytest.mark.parametrize('decisions', [{}, {'unknown': {'included': True}}])
def test_empty_review_keeps_result_and_storage(reviewed_panel, decisions):
    panel, source = reviewed_panel
    aux = install_result(panel)
    before = file_hashes(panel._session.session_dir) | file_hashes(source)
    storage, revision = panel._result_directory, panel._aux_dataset_revision
    panel._apply_aux_offline_review(decisions)
    assert panel._rig.aux_cameras['ir'] is aux and panel._aux_intrinsics is not None
    assert panel._result_directory == storage and panel._aux_dataset_revision == revision
    assert not panel._aux_pairs
    assert before == file_hashes(panel._session.session_dir) | file_hashes(source)


def test_unchanged_review_keeps_result_and_original_confirmation(reviewed_panel, finish_aux_tasks):
    panel, source = reviewed_panel
    decision = confirm_offline_pair(panel._aux_offline_reviews['0000'], 'identity', 'identity')
    panel._apply_aux_offline_review({'0000': decision})
    finish_aux_tasks(panel)
    aux = install_result(panel)
    revision, confirmations = panel._aux_dataset_revision, deepcopy(panel._aux_offline_confirmations)
    before = file_hashes(panel._session.session_dir) | file_hashes(source)
    repeated = deepcopy(decision)
    repeated['confirmed'] = 'later confirmation, same observations'
    panel._apply_aux_offline_review({'0000': repeated})
    assert panel._rig.aux_cameras['ir'] is aux
    assert panel._aux_dataset_revision == revision
    assert panel._aux_offline_confirmations == confirmations
    assert before == file_hashes(panel._session.session_dir) | file_hashes(source)


@pytest.mark.parametrize('changed', ['order', 'points', 'source'])
def test_changed_observations_still_invalidate_result(reviewed_panel, finish_aux_tasks, changed):
    panel, source = reviewed_panel
    decision = confirm_offline_pair(panel._aux_offline_reviews['0000'], 'identity', 'identity')
    panel._apply_aux_offline_review({'0000': decision})
    finish_aux_tasks(panel)
    install_result(panel)
    if changed == 'order':
        decision = confirm_offline_pair(panel._aux_offline_reviews['0000'], 'identity', 'reverse_both')
    elif changed == 'points':
        panel._aux_offline_records[0][4].image_points += .01
    else:
        path = source / 'aux_0000.png'
        path.write_bytes(path.read_bytes() + b'different source bytes, same pixels')
        key, left, right, ld, rd = panel._aux_offline_records[0]
        panel._aux_offline_reviews[key] = offline_pair_review(key, (left, right), (ld, rd), BOARD,
            paths=(source / 'rgb_left_0000.png', path))
        decision = confirm_offline_pair(panel._aux_offline_reviews[key], 'identity', 'identity')
    revision = panel._aux_dataset_revision
    panel._apply_aux_offline_review({'0000': decision})
    finish_aux_tasks(panel)
    assert panel._aux_intrinsics is None and not panel._rig.aux_cameras
    assert panel._aux_dataset_revision == revision + 1


def test_noop_check_does_not_bypass_source_hash_validation(reviewed_panel, finish_aux_tasks):
    panel, source = reviewed_panel
    decision = confirm_offline_pair(panel._aux_offline_reviews['0000'], 'identity', 'identity')
    panel._apply_aux_offline_review({'0000': decision})
    finish_aux_tasks(panel)
    aux = install_result(panel)
    (source / 'aux_0000.png').write_bytes(b'externally modified')
    panel._apply_aux_offline_review({'0000': decision})
    assert panel._rig.aux_cameras['ir'] is aux
    assert '原图已变化' in panel._log_text.toPlainText()


def wait_panel(app, panel):
    deadline = time.monotonic() + 30
    while panel.has_active_background_tasks():
        app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(.002)
    app.processEvents()


@pytest.mark.parametrize('key,model', [('pinhole_constrained', CameraModel.PINHOLE),
    ('pinhole', CameraModel.PINHOLE), ('fisheye_2', CameraModel.FISHEYE),
    (None, CameraModel.FISHEYE), ('unknown', CameraModel.PINHOLE),
    ('pinhole', CameraModel.FISHEYE), (['invalid'], CameraModel.FISHEYE)])
def test_main_window_restores_aux_model_and_state_without_admission(
        app, tmp_path, monkeypatch, key, model):
    session, _, rig = make_restore_session(tmp_path / 'source', auxiliary=True)
    _, intr, transform, _ = synthetic_pairs(model)
    info = dict(status='offline_candidate_pending_physical_validation', aux_model=key,
                heldout={'rgb_transfer_rms_px': .4, 'aux_transfer_rms_px': .3})
    rig.aux_cameras['thermal'] = AuxCameraCalibration('thermal', 'ir', intr, transform,
        frame_count=6, board_config=BOARD.to_dict(), calibration_info=info)
    session.save_calibration_results(rig)
    before = hashes(session)
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'local')
    monkeypatch.setattr(main_window.QFileDialog, 'getExistingDirectory', lambda *a: str(session.session_dir))
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: QMessageBox.Yes)
    win = main_window.MainWindow()
    panel = win._calib_panel
    # Session restore must not consult the previous workspace's auxiliary ID.
    win._stream_manager.set_auxiliary_camera(AuxiliaryCameraConfig('previous_ir', ''))
    win._on_load_session()
    panel = win._calib_panel
    wait_panel(app, panel)
    expected = key if isinstance(key, str) and key in ('pinhole_constrained', 'pinhole', 'fisheye_2') and (
        (key == 'fisheye_2') == (model == CameraModel.FISHEYE)) else None
    assert panel._combo_aux_model.currentData() == expected
    assert panel._aux_intrinsics.to_dict() == intr.to_dict()
    assert panel._aux_intrinsics is not panel.rig.aux_cameras['thermal'].intrinsics
    assert panel.rig.aux_cameras['thermal'].to_dict() == rig.aux_cameras['thermal'].to_dict()
    assert not panel._aux_pairs and not panel._aux_offline_confirmations
    assert not panel._check_aux_order.isChecked()
    assert '已有融合结果' in panel._aux_task_values['calibration'].text()
    assert not panel._btn_calib_aux.isEnabled()
    assert before == hashes(session)
    rgb = panel._intrinsics.pop('left')
    panel._refresh_aux_readiness()
    assert not panel._btn_calib_aux.isEnabled()
    assert '待就绪' in panel._sections[2]._header.text()
    assert '请先加载已完成的双目会话' in panel._lbl_aux_readiness.text()
    assert '已有融合结果' not in panel._aux_task_values['calibration'].text()
    panel._intrinsics['left'] = rgb
    panel._refresh_aux_readiness()
    assert '已有融合结果' in panel._aux_task_values['calibration'].text()
    if expected is None:
        records = synthetic_pairs()[3][:6]
        panel._aux_offline_records = [('raw',)]
        panel._aux_pairs = records
        panel._aux_offline_confirmations = {row[0]: {} for row in records}
        assert '请手动选择' in panel._aux_calibration_problem()
    win.close()


@pytest.mark.parametrize('mismatch', ['id', 'modality', 'board', 'reference', 'rgb_intrinsics'])
def test_incompatible_aux_is_historical_only_and_does_not_restore_cache(
        app, tmp_path, monkeypatch, mismatch):
    session, _, rig = make_restore_session(tmp_path / 'source', auxiliary=True)
    _, intr, transform, _ = synthetic_pairs(CameraModel.FISHEYE)
    config = AuxiliaryCameraConfig('thermal', '', board_config=BOARD)
    board = BOARD.to_dict()
    binding = {'left_id': 'left', 'fixed_intrinsics': rig.pairs['rgb'].left_intrinsics.to_dict()}
    if mismatch == 'board':
        board['square_size'] = .04
    elif mismatch == 'reference':
        binding['left_id'] = 'another_left'
    elif mismatch == 'rgb_intrinsics':
        binding['fixed_intrinsics']['camera_matrix'][0][0] += 1
    rig.aux_cameras['thermal'] = AuxCameraCalibration('thermal',
        'acoustic_rgb' if mismatch == 'modality' else 'ir', intr, transform,
        board_config=board, calibration_info={'aux_model': 'fisheye_2', 'rgb_binding': binding})
    # Historical incompatible input is deliberately written without the export gate.
    session.save_rig(rig, directory=session.result_directory)
    if mismatch == 'id':
        config.camera_id = 'different_ir'
    before = hashes(session)
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'local')
    win = main_window.MainWindow()
    panel = win._calib_panel
    panel.restore_session_result(session, rig, auxiliary=config)
    assert panel._aux_intrinsics is None
    assert panel._combo_aux_model.currentData() == 'pinhole_constrained'
    assert 'thermal' not in panel.rig.aux_cameras
    assert panel._historical_aux_results['thermal']['calibration'].to_dict() == rig.aux_cameras['thermal'].to_dict()
    assert before == hashes(session)
    win.close()


def test_stereo_only_session_resets_previous_auxiliary_model(app, tmp_path, monkeypatch):
    session, _, rig = make_restore_session(tmp_path / 'source')
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'local')
    win = main_window.MainWindow()
    panel = win._calib_panel
    panel._combo_aux_model.setCurrentIndex(panel._combo_aux_model.findData('fisheye_2'))
    panel.restore_session_result(session, rig)
    assert panel._combo_aux_model.currentData() == 'pinhole_constrained'
    assert panel._aux_intrinsics is None and not panel.rig.aux_cameras
    win.close()
