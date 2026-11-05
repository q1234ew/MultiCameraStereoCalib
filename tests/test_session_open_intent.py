"""Public loading separates offline intent from confirmed device continuation."""

from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QDialog

from multicamera.ui import main_window
from multicamera.ui.widgets.session_open_dialog import SessionOpenDialog
import test_session_restore_workspace as restore
from test_session_restore_workspace import make_session, choose_session, hashes, wait_restore

window = restore.window


@pytest.mark.parametrize('mode', ['offline', 'online', None])
def test_public_load_choice_preserves_files_and_cancel_keeps_workspace(
        window, app, tmp_path, monkeypatch, mode):
    session, _, _ = make_session(tmp_path, auxiliary=True)
    choose_session(monkeypatch, session)
    before, panel, manager = hashes(session), window._calib_panel, window._stream_manager
    monkeypatch.setattr(window, '_choose_session_open_mode', lambda *a: mode)
    monkeypatch.setattr(main_window.QMessageBox, 'question',
                        lambda *a: pytest.fail('No duplicate offline confirmation'))
    continuation = Mock()
    monkeypatch.setattr(window, '_on_continue_aux_capture', continuation)
    window._act_load_session.trigger()
    if mode is None:
        assert window._calib_panel is panel and window._stream_manager is manager
        assert window._current_session is None
    else:
        wait_restore(app, window._calib_panel)
        for _ in range(3):
            app.processEvents()
            if mode == 'online' and not continuation.called:
                from PySide6.QtTest import QTest
                QTest.qWait(120)
        assert window._current_session.session_dir == session.session_dir
        assert not window._stream_manager.configured_camera_urls()
        assert bool(continuation.called) is (mode == 'online')
        assert window._calib_panel._intrinsics['left'].image_size == (1280, 720)
    assert hashes(session) == before


def test_online_load_real_continuation_keeps_session_and_rgb(window, app, tmp_path, monkeypatch):
    from multicamera.ui.widgets.config_dialog import ConfigDialog
    session, _, rig = make_session(tmp_path, auxiliary=True)
    choose_session(monkeypatch, session)
    monkeypatch.setattr(window, '_choose_session_open_mode', lambda *a: 'online')
    prompts = []
    def approve(*args):
        prompts.append(args[1])
        return main_window.QMessageBox.Yes
    monkeypatch.setattr(main_window.QMessageBox, 'question', approve)
    def configure(dialog):
        dialog._on_accept()
        return dialog.result()
    monkeypatch.setattr(ConfigDialog, 'exec', configure)
    connected = Mock()
    monkeypatch.setattr(window, '_on_start_streams', connected)
    window._on_open_session()
    wait_restore(app, window._calib_panel)
    from PySide6.QtTest import QTest
    QTest.qWait(250)
    assert connected.call_count == 1
    assert prompts == ['确认沿用双目标定']
    assert window._calib_panel._pair_calibs['rgb'].to_dict() == rig.pairs['rgb'].to_dict()
    assert window._current_session.session_dir == session.session_dir
    assert window._calib_panel.workflow_phase == 'capture'


def test_continuation_waits_for_import_and_drops_stale_session(window, app, monkeypatch):
    from PySide6.QtTest import QTest
    session = object()
    window._current_session = session
    panel = window._calib_panel
    panel._session_restore_aux = True
    continuation = Mock()
    monkeypatch.setattr(window, '_on_continue_aux_capture', continuation)
    window._schedule_session_continuation(session)
    QTest.qWait(150)
    continuation.assert_not_called()
    window._current_session = None
    panel._session_restore_aux = False
    QTest.qWait(150)
    continuation.assert_not_called()


def test_dialog_defaults_offline_and_disables_online_without_rgb(app, tmp_path):
    dialog = SessionOpenDialog('calib_example', '已保存 17 组；待复核 17 组',
                               ['辅助相机：当前未配置'], can_continue=False)
    dialog.show()
    app.processEvents()
    assert dialog.offline.isDefault() and not dialog.online.isEnabled()
    dialog._choose('online')
    assert dialog.mode is None
    dialog.grab().save(str(tmp_path / 'session_open.png'))
    dialog.offline.click()
    assert dialog.result() == QDialog.Accepted and dialog.mode == 'offline'
    dialog.close()


def test_cancel_device_configuration_keeps_restored_data_without_connecting(
        window, app, tmp_path, monkeypatch):
    from PySide6.QtTest import QTest
    from multicamera.ui.widgets.config_dialog import ConfigDialog
    session, _, rig = make_session(tmp_path, auxiliary=True)
    choose_session(monkeypatch, session)
    before = hashes(session)
    monkeypatch.setattr(window, '_choose_session_open_mode', lambda *a: 'online')
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    monkeypatch.setattr(ConfigDialog, 'exec', lambda *a: QDialog.Rejected)
    connected = Mock()
    monkeypatch.setattr(window, '_on_start_streams', connected)
    window._on_open_session()
    wait_restore(app, window._calib_panel)
    QTest.qWait(250)
    connected.assert_not_called()
    assert hashes(session) == before
    assert window._calib_panel._pair_calibs['rgb'].to_dict() == rig.pairs['rgb'].to_dict()
    assert not window._stream_manager.configured_camera_urls()


def test_closing_cancels_pending_continuation(window, app, monkeypatch):
    from PySide6.QtTest import QTest
    window._current_session = object()
    connected = Mock()
    monkeypatch.setattr(window, '_on_continue_aux_capture', connected)
    window._schedule_session_continuation(window._current_session)
    window.close()
    QTest.qWait(150)
    connected.assert_not_called()
