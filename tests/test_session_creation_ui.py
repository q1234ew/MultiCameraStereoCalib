"""Session creation commits its visible identity once and preserves existing files."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QMessageBox

from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import CameraConfig, FrameEntry, StereoPairConfig
from multicamera.ui import main_window


@pytest.fixture
def window(app, tmp_path, monkeypatch, finish_aux_tasks):
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    window = main_window.MainWindow()
    window._stream_manager.add_virtual_stereo_pair(StereoPairConfig(
        'rgb', CameraConfig('left', ''), CameraConfig('right', '')))
    window._calib_panel.refresh_pairs()
    window.show()
    app.processEvents()
    yield window
    finish_aux_tasks(window._calib_panel)
    QThreadPool.globalInstance().waitForDone(15000)
    window._stream_manager.stop_all()
    window.close()


def test_new_action_immediately_updates_button_without_resize_or_connection_event(window):
    assert window._session_button.text() == '未创建会话'
    window._act_new_session.trigger()
    session = window._current_session
    assert session is not None and window._calib_panel._session is session
    assert window._session_button.text() != '未创建会话'
    assert session.name in window._session_button.toolTip()
    assert str(session.session_dir.resolve()) in window._session_button.toolTip()
    assert '尚未采图' in window.statusBar().currentMessage()
    assert window._act_new_session.isEnabled() and window._session_button.isEnabled()


def test_repeated_empty_creation_uses_one_directory(window):
    window._on_new_session()
    first = window._current_session
    for _ in range(5):
        window._on_new_session()
    assert window._current_session is first
    assert list(window._session_manager.base_dir.iterdir()) == [first.session_dir]
    assert '未重复创建目录' in window.statusBar().currentMessage()


def test_reentrant_creation_never_calls_factory_twice(window, monkeypatch):
    original = window._session_manager.create_session
    calls = []

    def create(*args, **kwargs):
        calls.append(True)
        window._on_new_session()
        return original(*args, **kwargs)

    monkeypatch.setattr(window._session_manager, 'create_session', create)
    window._on_new_session()
    assert len(calls) == 1 and window._current_session is not None


def test_unbound_creation_is_rolled_back_without_touching_prior_session(window, monkeypatch):
    window._on_new_session()
    old = window._current_session
    image = np.zeros((12, 16, 3), np.uint8)
    old.save_frame_pair('rgb', image, image)
    original = {p: p.read_bytes() for p in old.session_dir.rglob('*') if p.is_file()}
    created = []
    factory = window._session_manager.create_session

    def create(*args, **kwargs):
        session = factory(*args, **kwargs)
        created.append(session)
        return session

    def reject(session):
        raise RuntimeError('binding failed')

    monkeypatch.setattr(window._session_manager, 'create_session', create)
    monkeypatch.setattr(window._calib_panel, 'set_session', reject)
    monkeypatch.setattr(main_window.QMessageBox, 'warning', lambda *args: None)
    window._on_new_session()
    assert window._current_session is old and window._calib_panel._session is old
    assert len(created) == 1 and not created[0].session_dir.exists()
    assert original == {p: p.read_bytes() for p in old.session_dir.rglob('*') if p.is_file()}
    assert list(window._session_manager.base_dir.iterdir()) == [old.session_dir]
    assert window._act_new_session.isEnabled() and window._session_button.isEnabled()


def test_explicit_new_session_confirms_and_resets_observations_not_disk_history(window, monkeypatch):
    window._on_new_session()
    old = window._current_session
    panel = window._calib_panel
    panel._intr_cals['left'] = SimpleNamespace(num_frames=1)
    panel._capture_sequence.advance()
    panel._lbl_capture_decision.setText('old receipt')
    prior = (old.session_dir / old.METADATA_FILE).read_bytes()
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *args: QMessageBox.No)
    window._on_new_session()
    assert window._current_session is old and panel._intr_cals['left'].num_frames == 1
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *args: QMessageBox.Yes)
    window._on_new_session()
    assert window._current_session is not old
    assert not panel._intr_cals and not panel._pair_calibs
    assert panel._capture_sequence.done_count == 0 and not panel._lbl_capture_decision.text()
    assert (old.session_dir / old.METADATA_FILE).read_bytes() == prior


def test_first_capture_auto_binding_preserves_offline_observations(window):
    panel = window._calib_panel
    previous = SimpleNamespace(num_frames=2)
    panel._intr_cals['left'] = previous
    panel._aux_capture_sequence.advance()
    panel.create_session_requested.emit()
    assert window._current_session is not None
    assert panel._intr_cals['left'] is previous and panel._aux_capture_sequence.done_count == 1
    assert window._current_session.name in window._session_button.toolTip()
    panel.create_session_requested.emit()
    assert len(list(window._session_manager.base_dir.iterdir())) == 1


def test_loading_updates_visible_identity_immediately(window, monkeypatch, tmp_path):
    source = CalibrationSession.create(tmp_path / 'import', window._board,
                                       list(window._stream_manager.stereo_pairs.values()))
    monkeypatch.setattr(main_window.QFileDialog, 'getExistingDirectory', lambda *args: str(source.session_dir))
    window._on_load_session()
    assert window._current_session.name == source.name
    assert source.name in window._session_button.toolTip()
    assert window._session_button.text() != '未创建会话'


def test_uncreated_session_folder_button_opens_menu_without_allocating(window, monkeypatch):
    shown = []
    monkeypatch.setattr(window._session_button, 'showMenu', lambda: shown.append(True))
    window._session_button.click()
    assert shown == [True] and window._current_session is None
    assert not list(window._session_manager.base_dir.iterdir())


def test_pristine_cleanup_refuses_loaded_modified_or_nonempty_session(window, tmp_path):
    loaded = CalibrationSession(tmp_path / 'loaded')
    assert not loaded.discard_unpublished() and loaded.session_dir.exists()
    pairs = list(window._stream_manager.stereo_pairs.values())
    created = CalibrationSession.create(tmp_path / 'other', window._board, pairs)
    path = created.session_dir / 'images' / 'session.json'
    path.write_bytes(b'owned by user')
    assert not created.is_pristine() and not created.discard_unpublished()
    assert path.read_bytes() == b'owned by user'
    empty = CalibrationSession.create(tmp_path / 'other', window._board, pairs)
    assert empty.is_pristine() and empty.discard_unpublished() and not empty.session_dir.exists()


def test_initial_metadata_failure_does_not_leave_an_extra_directory(window, monkeypatch, tmp_path):
    base = tmp_path / 'failed-create'
    monkeypatch.setattr(CalibrationSession, '_save_metadata',
                        lambda self: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        CalibrationSession.create(base, window._board, list(window._stream_manager.stereo_pairs.values()))
    assert list(base.iterdir()) == []


def test_new_session_clears_offline_and_review_images_but_keeps_live_and_hidden_views(window):
    view, manager = window._stream_view, window._stream_manager
    image = np.zeros((120, 160, 3), np.uint8)
    view.show_static_images({'left': image, 'right': image})
    view.close_preview('right')
    view.show_review_images('old-frame', {'left': image})
    window._on_new_session()
    assert not view._static_images and view._review_images is None
    assert view._views['left']._last_frame is None
    assert 'right' not in view._views and view.has_closed_previews
    live = np.full_like(image, 127)
    manager.camera_connected.emit('left')
    manager._latest_frames['left'] = FrameEntry(live, time.time())
    view.show_review_images('another-old-frame', {'left': image})
    view.begin_capture_session()
    assert view._views['left']._last_frame is not None
    assert view._connection_state['left'][0]
    assert 'right' not in view._views
