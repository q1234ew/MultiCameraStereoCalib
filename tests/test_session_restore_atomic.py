"""Rejected session preparation cannot publish a half-restored workspace."""

from copy import deepcopy
import json

import numpy as np
import pytest
from PySide6.QtWidgets import QDialogButtonBox, QMessageBox
from PySide6.QtCore import QCoreApplication, QEvent
from shiboken6 import isValid

from multicamera.calibration.models import AuxCameraCalibration, CameraExtrinsics, CameraModel
from multicamera.io import session_restore
from multicamera.io.export import (export_rig_json, export_rig_opencv_yaml,
                                   export_rig_kalibr_yaml)
from multicamera.streaming.stream_manager import FrameEntry
from multicamera.ui import main_window
from multicamera.ui.widgets.export_dialog import ExportDialog
from test_offline_aux_loop import BOARD, review_fixture, synthetic_pairs
from test_session_restore_workspace import make_session, hashes
from test_aux_review_state import wait_panel


def with_aux(tmp_path):
    session, _, rig = make_session(tmp_path, auxiliary=True)
    _, intr, transform, _ = synthetic_pairs(CameraModel.FISHEYE)
    rig.aux_cameras['thermal'] = AuxCameraCalibration('thermal', 'ir', intr, transform,
        frame_count=6, board_config=BOARD.to_dict(), calibration_info={
            'aux_model': 'fisheye_2', 'rgb_binding': {'left_id': 'left',
                'right_id': 'right', 'fixed_intrinsics': rig.pairs['rgb'].left_intrinsics.to_dict()}})
    session.save_calibration_results(rig)
    return session, rig


@pytest.fixture
def workspace(app, tmp_path, monkeypatch):
    current, _ = with_aux(tmp_path / 'current')
    incoming, rig = with_aux(tmp_path / 'incoming')
    incoming._metadata['auxiliary_camera']['camera_id'] = 'incoming_ir'
    incoming._save_metadata()
    rig.aux_cameras['incoming_ir'] = rig.aux_cameras.pop('thermal')
    rig.aux_cameras['incoming_ir'].camera_id = 'incoming_ir'
    rig.pairs['rgb'].calibration_info['selection'] = {
        'filter_key': 'all', 'warnings': [], 'input_count': 17, 'left': {}, 'right': {}, 'stereo': {}}
    incoming.save_calibration_results(rig)
    selected = [current.session_dir]
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'local')
    monkeypatch.setattr(main_window.QFileDialog, 'getExistingDirectory', lambda *a: str(selected[0]))
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: QMessageBox.Yes)
    win = main_window.MainWindow()
    win._on_load_session()
    panel, manager = win._calib_panel, win._stream_manager
    wait_panel(app, panel)
    image, det, review, _ = review_fixture(tmp_path)
    panel._aux_offline_records = [('0000', image, image, det, det)]
    panel._aux_offline_reviews = {'0000': review}
    panel._aux_pairs = [('0000', det, det)]
    panel._open_aux_offline_review()
    dialog = panel._aux_offline_dialog
    dialog.orders[1].setCurrentIndex(dialog.orders[1].findData('reverse_both'))
    manager._latest_frames['left'] = FrameEntry(np.zeros((720, 1280, 3), np.uint8), 1.)
    selected[0] = incoming.session_dir
    snapshot = dict(session=win._current_session, panel_session=panel._session, rig=win._rig,
        intrinsics=panel._intrinsics.copy(), aux=panel._aux_intrinsics,
        records=list(panel._aux_offline_records), pairs=list(panel._aux_pairs),
        sequence=panel._capture_sequence, auxiliary_sequence=panel._aux_capture_sequence,
        configuration=deepcopy(manager.stereo_pairs), auxiliary=deepcopy(manager.auxiliary_camera),
        frame=manager._latest_frames['left'], storage=panel._result_directory,
        files=list(panel._result_files), model=panel._combo_aux_model.currentData(),
        tab=win._workflow_tabs.currentIndex(), workflow=(panel._workflow_phase, panel._active_step),
        board=win._board, export=win._act_export.isEnabled(), current_hashes=hashes(current),
        dialog=dialog, session_text=win._session_button.text())
    resets = []
    panel.calibration_reset.connect(lambda: resets.append(True))
    yield win, incoming, rig, snapshot, resets
    if isValid(dialog):
        dialog.reject()
    win.close()


def assert_original_workspace(win, before, resets):
    panel, manager = win._calib_panel, win._stream_manager
    assert win._current_session is before['session']
    assert panel._session is before['panel_session']
    assert win._rig is panel.rig is before['rig']
    assert panel._intrinsics.keys() == before['intrinsics'].keys()
    for cid, intr in before['intrinsics'].items():
        assert panel._intrinsics[cid] is intr
    assert panel._aux_intrinsics is before['aux']
    assert panel._aux_offline_records == before['records']
    assert panel._aux_pairs == before['pairs']
    assert panel._capture_sequence is before['sequence']
    assert panel._aux_capture_sequence is before['auxiliary_sequence']
    assert manager.stereo_pairs == before['configuration']
    assert manager.auxiliary_camera == before['auxiliary']
    assert manager._latest_frames['left'] is before['frame']
    assert panel._result_directory == before['storage'] and panel._result_files == before['files']
    assert panel._combo_aux_model.currentData() == before['model']
    assert win._workflow_tabs.currentIndex() == before['tab']
    assert (panel._workflow_phase, panel._active_step) == before['workflow']
    assert win._board is before['board']
    assert win._act_export.isEnabled() == before['export']
    assert panel._aux_offline_dialog is before['dialog']
    assert before['dialog'].orders[1].currentData() == 'reverse_both'
    assert win._session_button.text() == before['session_text']
    assert hashes(before['session']) == before['current_hashes']
    assert not resets


@pytest.mark.parametrize('broken', ['null_binding', 'list_binding', 'string_binding',
    'null_fixed', 'incomplete_fixed', 'bad_model', 'bad_size', 'bad_manifest', 'bad_errors',
    'bad_resume', 'bad_rms', 'bad_selection', 'bad_excluded_ids', 'missing_heldout', 'read_error'])
def test_preparation_failure_keeps_entire_previous_workspace(workspace, monkeypatch, broken):
    win, session, rig, before, resets = workspace
    aux = rig.aux_cameras['incoming_ir']
    info, binding = aux.calibration_info, aux.calibration_info['rgb_binding']
    if broken in ('null_binding', 'list_binding', 'string_binding'):
        info['rgb_binding'] = {'null_binding': None, 'list_binding': [], 'string_binding': 'bad'}[broken]
    elif broken == 'null_fixed':
        binding['fixed_intrinsics'] = None
    elif broken == 'incomplete_fixed':
        binding['fixed_intrinsics'] = {'camera_matrix': []}
    elif broken == 'bad_model':
        binding['model'] = 'not_a_model'
    elif broken == 'bad_size':
        binding['image_size'] = [1280]
    elif broken == 'bad_rms':
        rig.pairs['rgb'].stereo.rms_error = 'invalid'
    elif broken == 'bad_selection':
        rig.pairs['rgb'].calibration_info['selection'] = []
    elif broken == 'bad_excluded_ids':
        rig.pairs['rgb'].calibration_info['selection'] = {'left': {'excluded_ids': 3}}
    elif broken == 'missing_heldout':
        info['status'] = 'offline_candidate_pending_physical_validation'
    elif broken == 'read_error':
        original = session_restore.file_identity
        def fail(path):
            if path == session.result_path:
                raise OSError('injected result read failure')
            return original(path)
        monkeypatch.setattr(session_restore, 'file_identity', fail)
    session.save_rig(rig, directory=session.result_directory)
    if broken == 'bad_manifest':
        (session.result_directory / 'calibration_manifest.json').write_text('{bad json', encoding='utf-8')
    elif broken == 'bad_errors':
        (session.result_directory / 'calibration_manifest.json').write_text('{"errors":null}', encoding='utf-8')
    elif broken == 'bad_resume':
        session._metadata['multimodal_frame_count'] = 'invalid'
        session._save_metadata()
    incoming_hashes = hashes(session)
    win._on_load_session()
    assert '加载会话失败，已保留当前会话' in win.statusBar().currentMessage()
    assert_original_workspace(win, before, resets)
    assert incoming_hashes == hashes(session)


@pytest.mark.parametrize('change', ['modify_result', 'remove_result', 'modify_manifest', 'cancel'])
def test_files_changed_during_confirmation_or_cancel_never_replace_workspace(workspace, monkeypatch, change):
    win, session, _, before, resets = workspace
    def confirm(*args):
        if change == 'cancel':
            return QMessageBox.No
        if change == 'remove_result':
            session.result_path.unlink()
        elif change == 'modify_result':
            path = session.result_path
            path.write_bytes(path.read_bytes() + b'\n')
        else:
            path = session.result_directory / 'calibration_manifest.json'
            path.write_bytes(path.read_bytes() + b'\n')
        return QMessageBox.Yes
    monkeypatch.setattr(main_window.QMessageBox, 'question', confirm)
    win._on_load_session()
    assert_original_workspace(win, before, resets)


def test_success_commits_one_workspace_and_closes_only_old_review(workspace, app):
    win, session, _, before, _ = workspace
    win._on_load_session()
    wait_panel(app, win._calib_panel)
    assert win._current_session is win._calib_panel._session
    assert win._current_session.session_dir == session.session_dir
    assert win._rig is win._calib_panel.rig
    assert win._stream_manager.auxiliary_camera.camera_id == 'incoming_ir'
    assert win._calib_panel._aux_intrinsics is not None
    assert win._calib_panel._aux_offline_dialog is None
    assert not isValid(before['dialog']) or not before['dialog'].isVisible()
    assert not win._calib_panel._aux_offline_confirmations
    assert hashes(before['session']) == before['current_hashes']


@pytest.mark.parametrize('stage', ['preview', 'result', 'rows', 'publish'])
def test_staged_failure_preserves_live_workspace_and_unsaved_review(workspace, monkeypatch, app, stage):
    win, session, _, before, resets = workspace
    old_panel, old_manager, central = win._calib_panel, win._stream_manager, win.centralWidget()
    old_timer = win._capture_ui_timer
    incoming_hashes = hashes(session)
    stopped = []
    monkeypatch.setattr(old_manager, 'stop_all', lambda: stopped.append(True))
    if stage == 'preview':
        owner, name = main_window.MultiStreamView, 'update_board'
    elif stage == 'result':
        owner, name = main_window.CalibrationPanel, 'restore_session_result'
    elif stage == 'rows':
        owner, name = main_window.CalibrationPanel, 'refresh_pairs'
    else:
        owner, name = win, 'setCentralWidget'
    original = getattr(owner, name)
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected staged ' + stage)
    monkeypatch.setattr(owner, name, fail)
    win._on_load_session()
    app.processEvents()
    assert 'injected staged ' + stage in win.statusBar().currentMessage()
    assert win.centralWidget() is central and win._calib_panel is old_panel
    assert win._stream_manager is old_manager and not stopped
    assert old_timer.isActive()
    assert_original_workspace(win, before, resets)
    assert before['dialog'].isVisible()
    assert incoming_hashes == hashes(session)


def test_visible_workbench_publish_failure_does_not_reset_live_guide(workspace, monkeypatch, app):
    win, _, _, _, _ = workspace
    win._calib_panel.show_workflow('capture', False)
    win.show()
    app.processEvents()
    guide = win._stereo_capture
    guide.samples['left'] = {'sentinel': 'stable unsaved sample'}
    previous = dict(guide.samples)
    central = win.centralWidget()
    original = win.setCentralWidget
    def fail(widget):
        original(widget)
        raise RuntimeError('publish visible workbench failed')
    monkeypatch.setattr(win, 'setCentralWidget', fail)
    win._on_load_session()
    assert win.centralWidget() is central and central.isVisible()
    assert guide.samples == previous


@pytest.mark.parametrize('stage', ['construct', 'signals'])
def test_partial_workbench_construction_keeps_old_timers_and_manager(workspace, monkeypatch, app, stage):
    win, _, _, before, resets = workspace
    central, timer, manager = win.centralWidget(), win._capture_ui_timer, win._stream_manager
    if stage == 'construct':
        owner, name = main_window.StereoCaptureWorkspace, '__init__'
    else:
        owner, name = win, '_connect_workspace_signals'
    original = getattr(owner, name)
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected partial workbench')
    monkeypatch.setattr(owner, name, fail)
    win._on_load_session()
    app.processEvents()
    assert win.centralWidget() is central and win._stream_manager is manager
    assert timer.isActive()
    assert_original_workspace(win, before, resets)


def test_earliest_constructor_failure_cannot_retire_existing_floating_logs(workspace, monkeypatch, app):
    win, _, _, before, resets = workspace
    win.resize(1200, 800)
    win._toggle_logs.setChecked(True)
    win._sync_auxiliary_panels()
    dialog, logs = win._aux_panel_dialogs['logs'], win._logs_widget
    def fail(*args, **kwargs):
        raise RuntimeError('injected earliest constructor failure')
    monkeypatch.setattr(main_window.MultiStreamView, '__init__', fail)
    win._on_load_session()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    assert isValid(dialog) and isValid(logs)
    assert win._aux_panel_dialogs['logs'] is dialog and win._logs_widget is logs
    assert_original_workspace(win, before, resets)


def test_uncomputed_fisheye_session_restores_solver_model(app, tmp_path, monkeypatch):
    from multicamera.io.session import CalibrationSession
    from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
    from multicamera.streaming.stream_manager import StereoPairConfig, CameraConfig
    pair = StereoPairConfig('rgb', CameraConfig('left', ''), CameraConfig('right', ''))
    session = CalibrationSession.create(tmp_path / 'source', CharucoBoard(CharucoBoardConfig()),
                                        [pair], camera_model='fisheye')
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'local')
    monkeypatch.setattr(main_window.QFileDialog, 'getExistingDirectory', lambda *a: str(session.session_dir))
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: QMessageBox.Yes)
    win = main_window.MainWindow()
    win._on_load_session()
    assert win._current_session.session_dir == session.session_dir
    assert win._calib_panel._camera_model is win._camera_model is CameraModel.FISHEYE
    assert not win._rig
    win.close()


def test_success_keeps_user_capture_preferences_separate_from_session_result(workspace):
    win, _, _, _, _ = workspace
    panel = win._calib_panel
    for name, key in (('_combo_calib_strategy', 'robust'), ('_combo_data_filter', 'quality'),
                      ('_combo_capture_quality', 'strict'), ('_combo_wb_mode', 'locked')):
        combo = getattr(panel, name)
        combo.setCurrentIndex(combo.findData(key))
    panel._spin_min.setValue(12)
    panel._check_wb.setChecked(False)
    win._stereo_capture.guide_mode.setCurrentIndex(win._stereo_capture.guide_mode.findData('fixed'))
    win._on_load_session()
    restored = win._calib_panel
    assert restored is not panel
    for name in ('_combo_calib_strategy', '_combo_data_filter', '_combo_capture_quality', '_combo_wb_mode'):
        assert getattr(restored, name).currentData() == getattr(panel, name).currentData()
    assert restored._spin_min.value() == 12 and not restored._check_wb.isChecked()
    assert win._stereo_capture.guide_mode.currentData() == 'fixed'
    assert restored._combo_aux_model.currentData() == 'fisheye_2'
    assert not restored._aux_offline_confirmations


def test_post_commit_header_failure_does_not_claim_rollback(workspace, monkeypatch, caplog):
    win, session, _, _, _ = workspace
    def fail():
        raise RuntimeError('connection header failed')
    monkeypatch.setattr(win, '_update_connection_summary', fail)
    win._on_load_session()
    assert win._current_session.session_dir == session.session_dir
    assert win._calib_panel._session is win._current_session
    assert win._rig is win._calib_panel.rig and win._act_export.isEnabled()
    assert '加载会话失败' not in win.statusBar().currentMessage()
    assert 'Session committed; connection header refresh failed' in caplog.text


@pytest.mark.parametrize('moment', ['before_plan', 'before_result', 'during_confirmation', 'after_render'])
def test_metadata_bytes_cannot_change_under_cached_configuration(workspace, monkeypatch, moment):
    win, session, _, before, resets = workspace
    def change():
        path = session.session_dir / session.METADATA_FILE
        data = json.loads(path.read_text(encoding='utf-8'))
        data['auxiliary_camera']['camera_id'] = 'modified_on_disk'
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    if moment in ('before_plan', 'before_result'):
        name = 'prepare_session_restore' if moment == 'before_plan' else 'prepare_session_result'
        original = getattr(session_restore, name)
        def changed(*args, **kwargs):
            change()
            return original(*args, **kwargs)
        monkeypatch.setattr(session_restore, name, changed)
    elif moment == 'during_confirmation':
        def confirm(*args):
            change()
            return QMessageBox.Yes
        monkeypatch.setattr(main_window.QMessageBox, 'question', confirm)
    else:
        original = main_window.CalibrationPanel.show_workflow
        def render(*args, **kwargs):
            result = original(*args, **kwargs)
            change()
            return result
        monkeypatch.setattr(main_window.CalibrationPanel, 'show_workflow', render)
    win._on_load_session()
    assert '加载会话失败，已保留当前会话' in win.statusBar().currentMessage()
    assert_original_workspace(win, before, resets)


def test_loaded_metadata_snapshot_accepts_own_configuration_write(tmp_path):
    session, _ = with_aux(tmp_path)
    loaded = type(session).load(session.session_dir)
    plan = session_restore.prepare_session_restore(loaded)
    loaded.record_configuration(plan.board, plan.pairs, plan.auxiliary, plan.model.value)
    loaded.verify_loaded_metadata()
    plan = session_restore.prepare_session_restore(loaded)
    plan.verify_sources()


@pytest.mark.parametrize('mismatch', ['fixed_k', 'fixed_d', 'reference', 'right', 'model', 'size',
                                    'modality', 'board', 'rgb_role_collision'])
def test_incompatible_aux_is_quarantined_and_absent_from_current_exports(app, tmp_path, monkeypatch, mismatch):
    session, rig = with_aux(tmp_path / 'source')
    info = rig.aux_cameras['thermal'].calibration_info
    binding = info['rgb_binding']
    if mismatch == 'fixed_k':
        binding['fixed_intrinsics']['camera_matrix'][0][0] += 1
    elif mismatch == 'fixed_d':
        binding['fixed_intrinsics']['dist_coeffs'][0] += .001
    elif mismatch == 'reference':
        binding['left_id'] = 'different_left'
    elif mismatch == 'right':
        binding['right_id'] = 'different_right'
    elif mismatch == 'model':
        binding['model'] = 'fisheye'
    elif mismatch == 'size':
        binding['image_size'] = [640, 480]
    elif mismatch == 'modality':
        rig.aux_cameras['thermal'].modality = 'acoustic_rgb'
    elif mismatch == 'board':
        rig.aux_cameras['thermal'].board_config['square_size'] = .04
    quarantined_id = 'thermal'
    if mismatch == 'rgb_role_collision':
        quarantined_id = 'right'
        rig.aux_cameras['right'] = rig.aux_cameras.pop('thermal')
        rig.aux_cameras['right'].camera_id = 'right'
    rig.extrinsics[quarantined_id] = CameraExtrinsics(np.eye(3), np.zeros((3, 1)))
    session.save_rig(rig, directory=session.result_directory)
    before = hashes(session)
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'local')
    monkeypatch.setattr(main_window.QFileDialog, 'getExistingDirectory', lambda *a: str(session.session_dir))
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: QMessageBox.Yes)
    win = main_window.MainWindow()
    win._on_load_session()
    wait_panel(app, win._calib_panel)
    panel = win._calib_panel
    assert win._current_session.session_dir == session.session_dir
    assert win._rig is panel.rig
    assert not panel.rig.aux_cameras
    if mismatch == 'rgb_role_collision':
        assert 'right' in panel.rig.extrinsics
    else:
        assert 'thermal' not in panel.rig.extrinsics
    assert panel._aux_intrinsics is None and not panel._aux_offline_confirmations
    assert panel._historical_aux_results[quarantined_id]['calibration'].to_dict() == rig.aux_cameras[quarantined_id].to_dict()
    assert '已隔离' in panel._lbl_result_risks.text()
    assert '当前仅有RGB结果' in panel._lbl_result_decision.text()
    reasons = {cid: value['reason'] for cid, value in panel._historical_aux_results.items()}
    dialog = ExportDialog(win._rig, excluded_aux=reasons)
    assert '已隔离' in dialog._summary.toPlainText()
    assert all(dialog._scope.itemData(i)[0] != 'fusion' for i in range(dialog._scope.count()))
    target = dialog.export_to(tmp_path / 'current.json')
    assert not json.loads(target.read_text(encoding='utf-8'))['aux_cameras']
    export_rig_opencv_yaml(panel.rig, tmp_path / 'current.yml')
    export_rig_kalibr_yaml(panel.rig, tmp_path / 'current.yaml')
    assert before == hashes(session)
    dialog.close()
    win.close()


@pytest.mark.parametrize('exporter,suffix', [(export_rig_json, '.json'),
    (export_rig_opencv_yaml, '.yml'), (export_rig_kalibr_yaml, '.yaml')])
def test_direct_export_rejects_mismatching_binding_without_replacing_file(tmp_path, exporter, suffix):
    _, rig = with_aux(tmp_path / 'source')
    rig.aux_cameras['thermal'].calibration_info['rgb_binding']['fixed_intrinsics']['camera_matrix'][0][0] += 1
    target = tmp_path / ('result' + suffix)
    target.write_bytes(b'previous export')
    with pytest.raises(ValueError, match='不允许参与当前融合导出'):
        exporter(rig, target)
    assert target.read_bytes() == b'previous export'


def test_export_dialog_blocks_all_and_fusion_but_allows_isolated_rgb(app, tmp_path):
    _, rig = with_aux(tmp_path / 'source')
    rig.aux_cameras['thermal'].calibration_info['rgb_binding']['left_id'] = 'different'
    dialog = ExportDialog(rig)
    save = dialog._buttons.button(QDialogButtonBox.Save)
    assert not save.isEnabled()
    dialog._scope.setCurrentIndex(next(i for i in range(dialog._scope.count())
        if tuple(dialog._scope.itemData(i)) == ('fusion', 'thermal')))
    assert not save.isEnabled()
    target = tmp_path / 'blocked.json'
    with pytest.raises(ValueError):
        dialog.export_to(target)
    assert not target.exists()
    dialog._scope.setCurrentIndex(next(i for i in range(dialog._scope.count())
        if tuple(dialog._scope.itemData(i)) == ('pair', 'rgb')))
    assert save.isEnabled()
    dialog.export_to(tmp_path / 'rgb.json')
    dialog.close()
