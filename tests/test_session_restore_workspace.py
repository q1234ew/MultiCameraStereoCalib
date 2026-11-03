"""Opening a historical result must not require an already configured device."""

from copy import deepcopy
import hashlib
import json
from unittest.mock import Mock
import time

import numpy as np
import cv2
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration.models import (
    CameraIntrinsics, MultiCameraRig, StereoCalibResult, StereoPairCalibration,
)
from multicamera.calibration.offline_aux import validate_session_rig
from multicamera.calibration.planar import PatternBoardConfig
from multicamera.io.session import CalibrationSession
from multicamera.io.session_restore import prepare_session_restore, inspect_session_files
from multicamera.streaming.stream_manager import (
    AuxiliaryCameraConfig, CameraConfig, FrameEntry, StereoPairConfig,
)
from multicamera.ui import main_window


def make_session(tmp_path, *, offline=False, legacy=False, auxiliary=False):
    name, lid, rid = ('offline', 'offline_L', 'offline_R') if offline else ('rgb', 'left', 'right')
    pair = StereoPairConfig(name, CameraConfig(lid, 'http://device.test/left'),
                           CameraConfig(rid, 'http://device.test/right'))
    board = CharucoBoard(CharucoBoardConfig(cols=9, rows=7))
    aux = (AuxiliaryCameraConfig('thermal', 'http://device.test/ir',
           board_config=PatternBoardConfig.ir_chessboard()) if auxiliary else None)
    session = CalibrationSession.create(tmp_path, board, [pair], auxiliary=aux)
    intr = CameraIntrinsics(np.array([[740., 0., 640.], [0., 745., 360.], [0., 0., 1.]]),
                            np.zeros(5), (1280, 720))
    stereo = StereoCalibResult(np.eye(3), np.array([[-.05], [0.], [0.]]),
                               np.eye(3), np.eye(3), .2)
    rig = MultiCameraRig(pairs={name: StereoPairCalibration(name, lid, rid, intr, intr, stereo)},
                         reference_camera=lid)
    session.save_calibration_results(rig)
    if legacy:
        session._metadata['pairs'] = []
        session._save_metadata()
    return session, pair, rig


def hashes(session):
    return {str(path.relative_to(session.session_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in session.session_dir.rglob('*') if path.is_file()}


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'workspace')
    monkeypatch.setattr(main_window.QMessageBox, 'warning', lambda *a: None)
    monkeypatch.setattr(main_window.QMessageBox, 'information', lambda *a: None)
    win = main_window.MainWindow()
    yield win
    win.close()


def choose_session(monkeypatch, session):
    monkeypatch.setattr(main_window.QFileDialog, 'getExistingDirectory',
                         lambda *a: str(session.session_dir))


def test_prepare_uses_session_board_and_offline_bindings_without_source_writes(tmp_path):
    session, _, _ = make_session(tmp_path, auxiliary=True)
    before = hashes(session)
    plan = prepare_session_restore(session)
    assert plan.board.config.cols == 9 and plan.board.config.rows == 7
    assert plan.pairs[0].left.camera_id == 'left'
    assert all(not c.url for p in plan.pairs for c in (p.left, p.right))
    assert plan.auxiliary.camera_id == 'thermal' and not plan.auxiliary.url
    assert plan.auxiliary.pattern_board_config().square_size == .08
    assert plan.auxiliary.pattern_board_config().cols == 6
    assert hashes(session) == before


def test_legacy_canonical_offline_pair_needs_explicit_compatibility(tmp_path):
    session, pair, _ = make_session(tmp_path, offline=True, legacy=True)
    with pytest.raises(ValueError, match='绑定'):
        validate_session_rig(session, [pair])
    plan = prepare_session_restore(session)
    assert plan.legacy_offline and plan.pairs[0].left.camera_id == 'offline_L'
    assert not plan.pairs[0].left.url


def test_preparation_retains_generic_stereo_stream_type(tmp_path):
    session, _, _ = make_session(tmp_path)
    session._metadata['pairs'][0]['stream_type'] = 'ir'
    plan = prepare_session_restore(session)
    assert plan.pairs[0].left.stream_type == plan.pairs[0].right.stream_type == 'ir'


@pytest.mark.parametrize('kind', ['swapped', 'missing', 'renamed', 'duplicate', 'invalid_health'])
def test_offline_open_does_not_waive_actual_result_corruption(tmp_path, kind):
    session, _, rig = make_session(tmp_path)
    if kind == 'swapped':
        session._metadata['pairs'][0]['left_id'] = 'right'
        session._metadata['pairs'][0]['right_id'] = 'left'
    elif kind == 'missing':
        session._metadata['pairs'] = []
    elif kind == 'renamed':
        rig.pairs['rgb'].left_id = 'offline_L'
        rig.pairs['rgb'].right_id = 'offline_R'
        session.save_rig(rig, directory=session.result_directory)
    elif kind == 'duplicate':
        rig.pairs['rgb'].left_id = rig.pairs['rgb'].right_id
        session.save_rig(rig, directory=session.result_directory)
    else:
        rig.pairs['rgb'].left_intrinsics.camera_matrix[0, 0] = -1
        session.save_rig(rig, directory=session.result_directory)
    with pytest.raises(ValueError):
        prepare_session_restore(session)


def test_legacy_exception_never_overrides_explicit_offline_binding_conflict(tmp_path):
    session, _, _ = make_session(tmp_path, offline=True)
    session._metadata['pairs'][0]['left_id'] = 'other'
    with pytest.raises(ValueError, match='ID 不一致'):
        prepare_session_restore(session)


@pytest.mark.parametrize('legacy', [False, True])
def test_fresh_application_opens_result_as_confirmed_offline_workspace(
        window, tmp_path, monkeypatch, legacy):
    session, _, _ = make_session(tmp_path / 'source', offline=legacy, legacy=legacy, auxiliary=True)
    before = hashes(session)
    choose_session(monkeypatch, session)
    prompts = []
    def approve(*args):
        prompts.append(args[2])
        return main_window.QMessageBox.Yes
    monkeypatch.setattr(main_window.QMessageBox, 'question', approve)
    monkeypatch.setattr(window._stream_manager, 'start_all',
                         lambda: pytest.fail('Loading must never connect historical devices'))
    window._on_load_session()
    lid = 'offline_L' if legacy else 'left'
    assert window._current_session.session_dir == session.session_dir
    assert window._calib_panel._intrinsics[lid].image_size == (1280, 720)
    assert window._calib_panel._board is window._board
    assert window._board.to_dict() == session.metadata['board']
    assert window._act_export.isEnabled()
    assert not window._stream_manager.configured_camera_urls()
    assert not window._stream_manager._grabbers
    assert all(window._stream_manager.is_virtual_pair(n) for n in window._rig.pairs)
    assert prompts and '不自动连接' in prompts[0]
    assert window._combo_task.currentData() is True
    assert hashes(session) == before
    assert '离线打开' in window._calib_panel._log_text.toPlainText()
    if legacy:
        assert '旧版' in prompts[0] and '不映射' in prompts[0]


def test_cancel_preserves_current_session_config_results_and_streams(window, tmp_path, monkeypatch):
    session, _, _ = make_session(tmp_path / 'source')
    old, pair, rig = make_session(tmp_path / 'current')
    pair.left.camera_id = 'another_device'
    window._stream_manager.add_virtual_stereo_pair(pair)
    window._calib_panel.refresh_pairs()
    window._current_session = old
    window._calib_panel.set_session(old)
    window._rig = rig
    before, source_before = window._session_configuration(), hashes(session)
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.No)
    stop = Mock()
    monkeypatch.setattr(window._stream_manager, 'stop_all', stop)
    window._on_load_session()
    assert window._current_session is old and window._calib_panel._session is old
    assert window._rig is rig and window._session_configuration() == before
    assert '已取消' in window.statusBar().currentMessage()
    assert hashes(session) == source_before
    stop.assert_not_called()


def test_matching_session_needs_no_config_confirmation(window, tmp_path, monkeypatch):
    session, pair, _ = make_session(tmp_path / 'source')
    window._stream_manager.add_virtual_stereo_pair(pair)
    window._board = CharucoBoard.from_dict(session.metadata['board'])
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question',
                         lambda *a: pytest.fail('Matching workspace must load directly'))
    window._on_load_session()
    assert window._current_session.session_dir == session.session_dir
    assert window._stream_manager.stereo_pairs['rgb'] is pair


def test_same_ids_with_different_device_url_still_require_offline_confirmation(
        window, tmp_path, monkeypatch):
    session, pair, _ = make_session(tmp_path / 'source')
    pair.left.url = 'http://another-device.test/left'
    window._stream_manager.add_virtual_stereo_pair(pair)
    window._stream_manager._virtual_pairs.discard(pair.name)
    window._board = CharucoBoard.from_dict(session.metadata['board'])
    choose_session(monkeypatch, session)
    question = Mock(return_value=main_window.QMessageBox.No)
    monkeypatch.setattr(main_window.QMessageBox, 'question', question)
    window._on_load_session()
    question.assert_called_once()
    assert window._current_session is None
    assert window._stream_manager.stereo_pairs['rgb'] is pair


def test_invalid_session_does_not_even_offer_rebinding(window, tmp_path, monkeypatch):
    session, _, _ = make_session(tmp_path / 'source')
    session._metadata['pairs'][0]['left_id'] = 'wrong'
    session._save_metadata()
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question',
                         lambda *a: pytest.fail('Corruption is not a configuration choice'))
    window._on_load_session()
    assert window._current_session is None
    assert not window._stream_manager.stereo_pairs
    assert 'ID 不一致' in window.statusBar().currentMessage()


def test_current_resolution_mismatch_opens_offline_without_reusing_live_frames(
        window, tmp_path, monkeypatch):
    session, pair, _ = make_session(tmp_path / 'source')
    manager = window._stream_manager
    manager.add_virtual_stereo_pair(pair)
    window._board = CharucoBoard.from_dict(session.metadata['board'])
    manager._latest_frames['left'] = FrameEntry(np.zeros((480, 640, 3), np.uint8), time.monotonic())
    choose_session(monkeypatch, session)
    question = Mock(return_value=main_window.QMessageBox.Yes)
    monkeypatch.setattr(main_window.QMessageBox, 'question', question)
    window._on_load_session()
    question.assert_called_once()
    assert window._current_session.session_dir == session.session_dir
    assert not manager._latest_frames and not manager.configured_camera_urls()
    assert window._calib_panel._intrinsics['left'].image_size == (1280, 720)


def test_corrupt_auxiliary_board_is_rejected_before_context_changes(window, tmp_path, monkeypatch):
    session, _, _ = make_session(tmp_path / 'source', auxiliary=True)
    session._metadata['auxiliary_camera']['board_config']['square_size'] = -1
    session._save_metadata()
    before = window._session_configuration()
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question',
                         lambda *a: pytest.fail('Invalid board must be rejected in preflight'))
    window._on_load_session()
    assert window._current_session is None and window._session_configuration() == before


def test_session_backstop_rejection_never_replaces_stream_configuration(window, tmp_path, monkeypatch):
    session, _, _ = make_session(tmp_path / 'source')
    choose_session(monkeypatch, session)
    before = window._session_configuration()
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    def reject(*args, **kwargs):
        raise RuntimeError('Session ownership changed')
    monkeypatch.setattr(main_window.CalibrationPanel, 'set_session', reject)
    window._on_load_session()
    assert window._current_session is None and window._session_configuration() == before
    assert 'ownership changed' in window.statusBar().currentMessage()


def test_new_offline_result_persists_virtual_bindings_for_next_load(window, tmp_path):
    session, pair, rig = make_session(tmp_path / 'source', offline=True, legacy=True)
    pair.left.url = pair.right.url = ''
    window._stream_manager.add_virtual_stereo_pair(pair)
    window._calib_panel.refresh_pairs()
    window._board = CharucoBoard.from_dict(session.metadata['board'])
    window._current_session = session
    window._calib_panel.set_session(session)
    original = deepcopy(session.metadata)
    window._on_calib_finished(rig)
    loaded = CalibrationSession.load(session.session_dir)
    assert validate_session_rig(loaded, [pair]).pairs['offline'].left_id == 'offline_L'
    assert loaded.metadata['configuration_history'][-1]['previous']['pairs'] == original['pairs']
    assert not prepare_session_restore(loaded).legacy_offline


def wait_restore(app, panel):
    deadline = time.monotonic() + 20
    while panel.has_active_background_tasks():
        app.processEvents()
        assert time.monotonic() < deadline, panel._log_text.toPlainText()
        time.sleep(.002)
    app.processEvents()


def add_saved_images(session, *, stereo=2, aux=2):
    board = CharucoBoard.from_dict(session.metadata['board'])
    image = cv2.cvtColor(board.generate_image(pixel_size=50), cv2.COLOR_GRAY2BGR)
    for index in range(stereo):
        session.save_frame_pair('rgb', image, image, frame_idx=index)
    thermal = np.full((192, 256, 3), 255, np.uint8)
    for row in range(5):
        for col in range(7):
            if (row + col) % 2 == 0:
                thermal[40+row*20:60+row*20, 40+col*20:60+col*20] = 0
    for _ in range(aux):
        session.save_multimodal_pair(thermal, thermal)


def test_inventory_reports_missing_halves_and_does_not_require_optional_exports(tmp_path):
    session, _, _ = make_session(tmp_path, auxiliary=True)
    add_saved_images(session)
    folder = session.session_dir / 'multimodal'
    (folder / 'aux_0001.png').unlink()
    (session.session_dir / 'images/rgb/right_0001.png').unlink()
    before = hashes(session)
    report = inspect_session_files(session)
    assert report['aux_pairs'] == 1 and report['auto_import']
    assert any('aux_0001' in value for value in report['missing'])
    assert any('right_0001' in value for value in report['missing'])
    assert not any('Kalibr' in value or '.yml' in value for value in report['missing'])
    assert hashes(session) == before


def test_rgb_restore_decodes_unicode_paths_without_opencv_filename_io(app, tmp_path, monkeypatch):
    from multicamera.ui.threads.import_worker import StereoImageImportWorker
    session, _, _ = make_session(tmp_path / '标定 会话')
    add_saved_images(session, stereo=1, aux=0)
    monkeypatch.setattr(cv2, 'imread', lambda *a: pytest.fail('Use byte decoding for Unicode paths'))
    worker = StereoImageImportWorker(str(session.session_dir / 'images/rgb'),
                                     CharucoBoard.from_dict(session.metadata['board']))
    results, errors = [], []
    worker.finished_ok.connect(results.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors and len(results[0]) == 1
    assert results[0][0][3].num_corners == (9 - 1) * (7 - 1)


def test_legacy_complete_json_is_loaded_but_broken_latest_pointer_is_not_bypassed(tmp_path):
    session, _, rig = make_session(tmp_path)
    (session.session_dir / 'calibration_all_json.json').write_text(
        json.dumps(rig.to_dict()), encoding='utf-8')
    pointer = session.session_dir / 'results/latest.json'
    pointer.unlink()
    assert prepare_session_restore(session).rig.pairs['rgb'].left_id == 'left'
    assert session.result_path.name == 'calibration_all_json.json'
    pointer.write_text('{"directory":"calib_missing"}', encoding='utf-8')
    with pytest.raises(ValueError, match='缺少 rig.json'):
        prepare_session_restore(session)


def test_both_image_modes_are_recovered_without_resetting_existing_rgb_result(
        window, app, tmp_path, monkeypatch):
    session, _, rig = make_session(tmp_path, auxiliary=True)
    add_saved_images(session)
    before = hashes(session)
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    window._on_load_session()
    panel = window._calib_panel
    wait_restore(app, panel)
    assert panel._stereo_cals['rgb'].num_frames == 2
    assert panel._intr_cals['left'].frame_ids == ['0000', '0001']
    assert 'offline' not in window._stream_manager.stereo_pairs
    assert panel._pair_calibs['rgb'].to_dict() == rig.pairs['rgb'].to_dict()
    assert len(panel._aux_offline_records) == 2 and not panel._aux_pairs
    assert panel._aux_saved_count == 2 and panel._aux_capture_count == 2
    assert panel._aux_task_values['intrinsics'].text() == '已恢复'
    assert panel._aux_offline_dialog is None  # No unsolicited modal dialog during load.
    assert hashes(session) == before


def test_raw_session_without_results_restores_rgb_observations_for_first_calibration(
        window, app, tmp_path, monkeypatch):
    session, _, _ = make_session(tmp_path)
    add_saved_images(session, aux=0)
    (session.session_dir / 'results/latest.json').unlink()
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    window._on_load_session()
    wait_restore(app, window._calib_panel)
    assert window._current_session is not None and window._rig is None
    assert window._calib_panel._stereo_cals['rgb'].num_frames == 2
    assert not window._act_export.isEnabled()
    assert '双目结果 JSON' in window._calib_panel._log_text.toPlainText()


def test_resume_allows_transport_change_but_preserves_rgb_intrinsics(
        window, tmp_path, monkeypatch):
    from multicamera.ui.widgets.config_dialog import ConfigDialog
    session, _, rig = make_session(tmp_path, auxiliary=True)
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    window._on_load_session()
    original = deepcopy(window._calib_panel._intrinsics['left'].to_dict())
    started = Mock()
    monkeypatch.setattr(window._stream_manager, 'start_all', started)
    def configure(dialog):
        assert dialog._table.item(0, 3).text() == 'http://device.test/left'
        dialog._on_accept()
        return dialog.result()
    monkeypatch.setattr(ConfigDialog, 'exec', configure)
    window._on_continue_aux_capture()
    assert started.call_count == 1
    assert window._calib_panel._intrinsics['left'].to_dict() == original
    assert window._calib_panel._pair_calibs['rgb'].to_dict() == rig.pairs['rgb'].to_dict()
    assert window._calib_panel._active_step == 2
    assert window._calib_panel.workflow_phase == 'capture'
    assert window._current_session.session_dir == session.session_dir
    assert window._current_session.metadata['rgb_continuation_history'][-1]['bindings']['rgb']['left_id'] == 'left'


def test_resume_state_reserves_orphan_indices_and_never_overwrites_saved_images(tmp_path):
    session, _, _ = make_session(tmp_path, auxiliary=True)
    add_saved_images(session)
    before = hashes(session)
    (session.session_dir / 'multimodal/quality_0008.json').write_text('{}', encoding='utf-8')
    (session.session_dir / 'images/rgb/left_0010.png').write_bytes(b'orphan')
    session._metadata['multimodal_frame_count'] = 0
    session._metadata['frame_count'] = 0
    assert session.multimodal_resume_state() == {'saved_pairs': 2, 'next_index': 9}
    assert session.next_stereo_frame_index == 11
    image = np.zeros((16, 24, 3), np.uint8)
    assert session.save_multimodal_pair(image, image) == 9
    assert all(hashes(session)[key] == value for key, value in before.items() if key != 'session.json')


@pytest.mark.parametrize('change', ['ids', 'swap', 'board', 'model'])
def test_continuation_rejects_nontransport_rgb_changes(tmp_path, change):
    from multicamera.ui.configuration import continuation_impact
    from multicamera.calibration.models import CameraModel
    session, pair, _ = make_session(tmp_path)
    board = CharucoBoard.from_dict(session.metadata['board'])
    proposed, next_board, model = deepcopy(pair), board, CameraModel.PINHOLE
    if change == 'ids':
        proposed.left.camera_id = 'another'
    elif change == 'swap':
        proposed.left, proposed.right = proposed.right, proposed.left
    elif change == 'board':
        next_board = CharucoBoard(CharucoBoardConfig())
    else:
        model = CameraModel.FISHEYE
    with pytest.raises(ValueError):
        continuation_impact(board, next_board, CameraModel.PINHOLE, model,
                            {'rgb': pair}, [proposed], None, None, {'rgb'})


def test_confirmed_aux_url_change_keeps_historical_observations(tmp_path):
    from dataclasses import replace
    from multicamera.ui.configuration import continuation_impact
    from multicamera.calibration.models import CameraModel
    session, pair, _ = make_session(tmp_path)
    board = CharucoBoard.from_dict(session.metadata['board'])
    old = AuxiliaryCameraConfig('ir', 'http://old-device.test/stream')
    new = replace(old, url='http://new-address.test/stream')
    impact = continuation_impact(board, board, CameraModel.PINHOLE, CameraModel.PINHOLE,
                                  {'rgb': pair}, [pair], old, new, {'rgb'})
    assert impact.streams_changed and not impact.reset_aux and not impact.changed_pairs


def test_configuration_metadata_failure_rolls_back_memory_and_disk(tmp_path, monkeypatch):
    from pathlib import Path
    session, pair, _ = make_session(tmp_path)
    before, metadata = hashes(session), deepcopy(session.metadata)
    original = Path.write_text
    def fail(path, *args, **kwargs):
        if path.name == 'session.json':
            raise OSError('disk full')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', fail)
    with pytest.raises(OSError):
        session.record_configuration(CharucoBoard(CharucoBoardConfig()), [pair], None,
                                      continuation={'same_rgb_device_lens_settings': 'user_confirmed'})
    assert session.metadata == metadata and hashes(session) == before


@pytest.mark.parametrize('format_name', ['json', 'standard', 'complete'])
def test_stereo_file_import_native_formats_and_unchanged_source(tmp_path, format_name):
    from multicamera.io.calibration_import import load_stereo_calibration
    from multicamera.io.export import export_rig_json, export_rig_opencv_standard_yaml, export_rig_opencv_yaml
    _, _, rig = make_session(tmp_path / 'session')
    export = {'json': export_rig_json, 'standard': export_rig_opencv_standard_yaml,
              'complete': export_rig_opencv_yaml}[format_name]
    path = tmp_path / ('标定 文件.json' if format_name == 'json' else '标定 文件.yml')
    export(rig, path)
    before = path.read_bytes()
    loaded, provenance = load_stereo_calibration(path, translation_unit='m')
    np.testing.assert_array_equal(loaded.pairs['rgb'].left_intrinsics.camera_matrix,
                                  rig.pairs['rgb'].left_intrinsics.camera_matrix)
    np.testing.assert_array_equal(loaded.pairs['rgb'].stereo.T, rig.pairs['rgb'].stereo.T)
    assert loaded.pairs['rgb'].left_id == 'left' and path.read_bytes() == before
    assert provenance['source']['sha256'] == hashlib.sha256(before).hexdigest()


def test_opencv_import_requires_units_and_converts_mm_without_guessing(tmp_path):
    from multicamera.io.calibration_import import load_stereo_calibration
    from multicamera.io.export import export_rig_opencv_standard_yaml
    _, _, rig = make_session(tmp_path / 'session')
    rig.pairs['rgb'].stereo.T *= 1000
    path = tmp_path / 'millimetres.yml'
    export_rig_opencv_standard_yaml(rig, path)
    with pytest.raises(ValueError, match='长度单位'):
        load_stereo_calibration(path)
    loaded, _ = load_stereo_calibration(path, translation_unit='mm')
    assert np.linalg.norm(loaded.pairs['rgb'].stereo.T) == pytest.approx(.05)
    from multicamera.calibration.quality_summary import summarize_quality
    assert summarize_quality(loaded).title == '导入参数，精度待验证'


def test_missing_json_intrinsic_fields_are_reported_before_import(tmp_path):
    from multicamera.io.calibration_import import load_stereo_calibration
    _, _, rig = make_session(tmp_path / 'session')
    data = rig.to_dict()
    data['pairs']['rgb']['left_intrinsics'].pop('image_size')
    path = tmp_path / 'broken.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='image_size'):
        load_stereo_calibration(path)


def test_json_without_rms_marks_unknown_metrics_instead_of_measured_zero(tmp_path):
    from multicamera.io.calibration_import import load_stereo_calibration
    _, _, rig = make_session(tmp_path / 'source')
    data = rig.to_dict()
    data['pairs']['rgb']['left_intrinsics'].pop('rms_error')
    path = tmp_path / 'without-errors.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    loaded, _ = load_stereo_calibration(path)
    assert loaded.pairs['rgb'].calibration_info['missing_imported_metrics'] is True


def test_reading_standalone_file_creates_continuable_session_without_connecting(
        window, tmp_path, monkeypatch):
    from multicamera.io.export import export_rig_json
    _, _, rig = make_session(tmp_path / 'source')
    path = tmp_path / 'parameters.json'
    export_rig_json(rig, path)
    before = path.read_bytes()
    monkeypatch.setattr(main_window.QFileDialog, 'getOpenFileName', lambda *a: (str(path), ''))
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    warnings = []
    monkeypatch.setattr(main_window.QMessageBox, 'warning', lambda *a: warnings.append(a))
    started = Mock()
    monkeypatch.setattr(window._stream_manager, 'start_all', started)
    window._on_load_calibration_file()
    assert not warnings, warnings
    assert window._current_session is not None
    assert window._calib_panel._intrinsics['left'].image_size == (1280, 720)
    assert window._calib_panel._btn_continue_aux.isEnabled()
    assert window._rig.pairs['rgb'].calibration_info['imported_calibration']['source']['path'] == str(path.resolve())
    assert window._current_session.load_rig().pairs['rgb'].left_id == 'left'
    assert path.read_bytes() == before
    started.assert_not_called()


def test_parameter_file_can_reuse_compatible_session_and_restore_both_image_modes(
        window, app, tmp_path, monkeypatch):
    from multicamera.io.export import export_rig_json
    session, _, rig = make_session(tmp_path / 'source', auxiliary=True)
    add_saved_images(session)
    images_before = {key: value for key, value in hashes(session).items() if key.endswith('.png')}
    previous_batch = session.result_directory
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    window._on_load_session()
    wait_restore(app, window._calib_panel)
    path = tmp_path / 'parameters.json'
    export_rig_json(rig, path)
    monkeypatch.setattr(main_window.QFileDialog, 'getOpenFileName', lambda *a: (str(path), ''))
    warnings = []
    monkeypatch.setattr(main_window.QMessageBox, 'warning', lambda *a: warnings.append(a))
    window._on_load_calibration_file()
    wait_restore(app, window._calib_panel)
    assert not warnings, warnings
    assert window._current_session.session_dir == session.session_dir
    assert window._current_session.result_directory != previous_batch
    assert (previous_batch / 'rig.json').is_file()
    assert window._calib_panel._stereo_cals['rgb'].num_frames == 2
    assert len(window._calib_panel._aux_offline_records) == 2
    assert not window._calib_panel._aux_pairs
    assert all(hashes(session)[key] == value for key, value in images_before.items())


def test_cancel_standalone_import_does_not_create_a_session(window, tmp_path, monkeypatch):
    from multicamera.io.export import export_rig_json
    _, _, rig = make_session(tmp_path / 'source')
    path = tmp_path / 'parameters.json'
    export_rig_json(rig, path)
    before = list(window._session_manager.base_dir.iterdir())
    monkeypatch.setattr(main_window.QFileDialog, 'getOpenFileName', lambda *a: (str(path), ''))
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.No)
    window._on_load_calibration_file()
    assert window._current_session is None and window._rig is None
    assert list(window._session_manager.base_dir.iterdir()) == before


def test_failed_import_pointer_does_not_relabel_old_parameters(window, app, tmp_path, monkeypatch):
    from multicamera.io.export import export_rig_json
    session, _, rig = make_session(tmp_path / 'source')
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    window._on_load_session()
    wait_restore(app, window._calib_panel)
    before = deepcopy(window._rig.to_dict())
    source_before = deepcopy(window._calib_panel._rgb_result_provenance)
    rig.pairs['rgb'].stereo.T[0, 0] = -.09
    filename = tmp_path / 'different.json'
    export_rig_json(rig, filename)
    monkeypatch.setattr(main_window.QFileDialog, 'getOpenFileName', lambda *a: (str(filename), ''))
    original = window._current_session._atomic_json
    def fail_pointer(path, payload):
        if path.name == 'latest.json':
            raise OSError('cannot publish pointer')
        return original(path, payload)
    monkeypatch.setattr(window._current_session, '_atomic_json', fail_pointer)
    warnings = []
    monkeypatch.setattr(main_window.QMessageBox, 'warning', lambda *a: warnings.append(a))
    window._on_load_calibration_file()
    assert warnings and '索引' in warnings[0][-1]
    assert window._rig.to_dict() == before
    assert window._calib_panel._rgb_result_provenance == source_before


def test_imported_file_always_disconnects_before_reusing_current_image_session(
        window, app, tmp_path, monkeypatch):
    from multicamera.io.export import export_rig_json
    session, pair, rig = make_session(tmp_path / 'source')
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    window._on_load_session()
    wait_restore(app, window._calib_panel)
    window._stream_manager.add_virtual_stereo_pair(pair)
    window._stream_manager._virtual_pairs.discard(pair.name)
    filename = tmp_path / 'parameters.json'
    export_rig_json(rig, filename)
    monkeypatch.setattr(main_window.QFileDialog, 'getOpenFileName', lambda *a: (str(filename), ''))
    window._on_load_calibration_file()
    assert window._stream_manager.is_virtual_pair('rgb')
    assert not window._stream_manager.configured_camera_urls()
    assert window._rig.pairs['rgb'].calibration_info['imported_calibration']['physical_rgb_binding'] == 'user_confirmed_for_session'


def test_continuation_metadata_failure_preserves_configuration_and_reviews(
        window, app, tmp_path, monkeypatch):
    from multicamera.ui.widgets.config_dialog import ConfigDialog
    session, _, _ = make_session(tmp_path / 'source', auxiliary=True)
    add_saved_images(session)
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    monkeypatch.setattr(main_window.QMessageBox, 'warning', lambda *a: None)
    window._on_load_session()
    wait_restore(app, window._calib_panel)
    configuration = window._session_configuration()
    records = window._calib_panel._aux_offline_records
    metadata = deepcopy(window._current_session.metadata)
    def fail(*a, **kw):
        raise OSError('disk full')
    monkeypatch.setattr(window._current_session, 'record_configuration', fail)
    def configure(dialog):
        dialog._on_accept()
        return dialog.result()
    monkeypatch.setattr(ConfigDialog, 'exec', configure)
    start = Mock()
    monkeypatch.setattr(window._stream_manager, 'start_all', start)
    window._on_continue_aux_capture()
    assert window._session_configuration() == configuration
    assert window._current_session.metadata == metadata
    assert window._calib_panel._aux_offline_records is records
    start.assert_not_called()


def test_loading_ignores_current_group_order_for_fusion_reference(
        window, tmp_path, monkeypatch):
    session, pair, _ = make_session(tmp_path / 'source')
    other = StereoPairConfig('other', CameraConfig('other_L', ''), CameraConfig('other_R', ''))
    window._stream_manager.add_virtual_stereo_pair(other)
    window._stream_manager.add_virtual_stereo_pair(pair)
    window._board = CharucoBoard.from_dict(session.metadata['board'])
    choose_session(monkeypatch, session)
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    window._on_load_session()
    assert next(iter(window._stream_manager.stereo_pairs)) == 'other'
    assert window._calib_panel._rgb_left_camera_id() == 'left'
    from multicamera.ui.configuration import continuation_impact
    with pytest.raises(ValueError, match='参考相机'):
        continuation_impact(window._board, window._board, window._camera_model, window._camera_model,
                            {'rgb': pair, 'other': other}, [other, pair], None, None, {'rgb'})
