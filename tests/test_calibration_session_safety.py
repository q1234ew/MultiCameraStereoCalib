"""Session ownership and nonblocking window lifecycle regression tests."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QCloseEvent

from multicamera.calibration.models import CameraIntrinsics, StereoCalibResult, StereoPairCalibration
from multicamera.io.export import export_rig_opencv_yaml
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import AuxiliaryCameraConfig, CameraConfig, StereoPairConfig
from multicamera.ui import main_window
from multicamera.ui.threads.calibration_worker import PairCalibrationWorker
from multicamera.ui.widgets.calib_panel import _UndoCaptureEntry


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setattr(main_window, "sessions_dir", lambda: tmp_path / "sessions")
    win = main_window.MainWindow()
    win._stream_manager.add_virtual_stereo_pair(StereoPairConfig(
        "rgb", CameraConfig("left", "", "left"), CameraConfig("right", "", "right"),
    ))
    win._calib_panel.refresh_pairs()
    session = CalibrationSession(tmp_path / "old")
    win._calib_panel.set_session(session)
    win._current_session = session
    yield win
    assert not win._calib_panel.has_active_background_tasks()
    win.close()


def wait_until(app, predicate):
    loop = QEventLoop()
    poll = QTimer()
    poll.setInterval(5)
    poll.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    poll.start()
    deadline.start(5000)
    if not predicate():
        loop.exec()
    poll.stop()
    deadline.stop()
    assert predicate(), "Background task did not reach the expected state"


def assert_refused(window, action):
    text = window.statusBar().currentMessage()
    assert action in text
    assert "拒绝" in text
    assert "后台任务" in text
    assert text in window._calib_panel._log_text.toPlainText()


@pytest.mark.parametrize("operation", ["new", "load", "close"])
def test_busy_operations_are_refused_without_side_effects(window, monkeypatch, operation):
    panel = window._calib_panel
    old = window._current_session
    create = Mock()
    choose = Mock()
    stop = Mock()
    shutdown = Mock()
    monkeypatch.setattr(window._session_manager, "create_session", create)
    monkeypatch.setattr(main_window.QFileDialog, "getExistingDirectory", choose)
    monkeypatch.setattr(window._stream_manager, "stop_all", stop)
    monkeypatch.setattr(panel._white_balance_guard, "shutdown", shutdown)
    with monkeypatch.context() as patch:
        patch.setattr(panel, "has_active_background_tasks", lambda: True)
        if operation == "new":
            window._on_new_session()
            assert_refused(window, "新建会话")
        elif operation == "load":
            window._on_load_session()
            assert_refused(window, "加载会话")
        else:
            event = QCloseEvent()
            window.closeEvent(event)
            assert not event.isAccepted()
            assert_refused(window, "关闭窗口")
    assert window._current_session is old and panel._session is old
    create.assert_not_called()
    choose.assert_not_called()
    stop.assert_not_called()
    shutdown.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("_calib_worker", object()),
    ("_aux_worker", object()),
    ("_stereo_import_worker", object()),
    ("_aux_import_worker", object()),
    ("_save_workers", {"token": object()}),
    ("_save_queue", [object()]),
    ("_pending_save_tokens", {"token"}),
    ("_calib_queue", ["rgb"]),
    ("_aux_ui_busy", True),
    ("_aux_import_busy", True),
])
def test_background_api_covers_tasks_and_pending_work(window, monkeypatch, field, value):
    panel = window._calib_panel
    assert not panel.has_active_background_tasks()
    with monkeypatch.context() as patch:
        patch.setattr(panel, field, value)
        assert panel.has_active_background_tasks()
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()
    assert not panel.has_active_background_tasks()


def test_background_api_covers_white_balance_and_auto_capture(window, monkeypatch):
    panel = window._calib_panel
    with monkeypatch.context() as patch:
        patch.setattr(panel._white_balance_guard, "busy", True)
        assert panel.has_active_background_tasks()
    panel._auto_timer.start()
    try:
        assert panel.has_active_background_tasks()
    finally:
        panel._auto_timer.stop()
    assert not panel.has_active_background_tasks()


def test_loading_rechecks_busy_after_directory_dialog(window, monkeypatch, tmp_path):
    panel = window._calib_panel
    old = window._current_session
    load = Mock()

    def choose(*args):
        panel._pending_save_tokens.add("new-task")
        return str(tmp_path / "target")

    monkeypatch.setattr(main_window.QFileDialog, "getExistingDirectory", choose)
    monkeypatch.setattr(CalibrationSession, "load", load)
    try:
        window._on_load_session()
        assert_refused(window, "加载会话")
        assert window._current_session is old and panel._session is old
        load.assert_not_called()
    finally:
        panel._pending_save_tokens.clear()


@pytest.mark.parametrize("failure", ["corrupt", "permission"])
def test_load_failure_preserves_current_session(window, monkeypatch, tmp_path, failure):
    panel = window._calib_panel
    old = window._current_session
    target = CalibrationSession(tmp_path / "broken")
    metadata = target.session_dir / target.METADATA_FILE
    metadata.write_text("{invalid json", encoding="utf-8")
    if failure == "permission":
        read_text = Path.read_text

        def denied(path, *args, **kwargs):
            if path == metadata:
                raise PermissionError("session metadata is not readable")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr(main_window.QFileDialog, "getExistingDirectory", lambda *a: str(target.session_dir))
    undo = ([], False)
    panel._undo_stack.append(undo)
    window._on_load_session()
    assert window._current_session is old and panel._session is old
    assert panel._undo_stack == [undo]
    text = window.statusBar().currentMessage()
    assert "加载会话失败" in text and "已保留当前会话" in text
    assert text in panel._log_text.toPlainText()


@pytest.mark.parametrize("operation", ["new", "load"])
def test_idle_session_switch_clears_previous_undo(window, monkeypatch, tmp_path, operation):
    panel = window._calib_panel
    target = CalibrationSession(tmp_path / "target")
    panel._undo_stack.append(([], False))
    panel._refresh_undo_btn()
    assert panel._btn_undo.isEnabled()
    if operation == "new":
        monkeypatch.setattr(window._session_manager, "create_session", lambda *a, **kw: target)
        window._on_new_session()
    else:
        monkeypatch.setattr(main_window.QFileDialog, "getExistingDirectory", lambda *a: str(target.session_dir))
        monkeypatch.setattr(CalibrationSession, "load", lambda *a: target)
        window._on_load_session()
    assert window._current_session is target and panel._session is target
    assert not panel._undo_stack and not panel._btn_undo.isEnabled()
    assert panel._next_session_frame_idx == target.frame_count
    assert not panel._save_sessions


def test_set_session_backstop_and_first_automatic_session(window, tmp_path):
    panel = window._calib_panel
    old = panel._session
    target = CalibrationSession(tmp_path / "target")
    panel._calib_queue.append("rgb")
    try:
        with pytest.raises(RuntimeError, match="后台任务"):
            panel.set_session(target)
        assert panel._session is old
        panel.set_session(old)
        panel._session = None
        panel.set_session(target)
        assert panel._session is target
    finally:
        panel._calib_queue.clear()


@pytest.mark.parametrize("operation", ["new", "load"])
def test_set_session_rejection_keeps_main_window_reference(window, monkeypatch, tmp_path, operation):
    panel = window._calib_panel
    old = window._current_session
    target = CalibrationSession(tmp_path / "target")

    def reject(session):
        raise RuntimeError("background ownership changed")

    monkeypatch.setattr(panel, "set_session", reject)
    monkeypatch.setattr(main_window.QMessageBox, "warning", Mock())
    if operation == "new":
        monkeypatch.setattr(window._session_manager, "create_session", lambda *a, **kw: target)
        window._on_new_session()
    else:
        monkeypatch.setattr(main_window.QFileDialog, "getExistingDirectory", lambda *a: str(target.session_dir))
        monkeypatch.setattr(CalibrationSession, "load", lambda *a: target)
        window._on_load_session()
    assert window._current_session is old and panel._session is old
    assert "background ownership changed" in window.statusBar().currentMessage()


def test_streaming_alone_does_not_block_switch_or_close(window, monkeypatch, tmp_path):
    panel = window._calib_panel
    target = CalibrationSession(tmp_path / "target")
    window._act_stop_streams.setEnabled(True)
    stop = Mock()
    shutdown = Mock(return_value=True)
    monkeypatch.setattr(window._stream_manager, "stop_all", stop)
    monkeypatch.setattr(panel._white_balance_guard, "shutdown", shutdown)
    monkeypatch.setattr(window._session_manager, "create_session", lambda *a, **kw: target)
    assert not panel.has_active_background_tasks()
    window._on_new_session()
    assert window._current_session is target
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    shutdown.assert_called_once_with()
    stop.assert_called_once_with()


def test_idle_close_refused_if_white_balance_cleanup_is_not_complete(window, monkeypatch):
    shutdown = Mock(return_value=False)
    stop = Mock()
    with monkeypatch.context() as patch:
        patch.setattr(window._calib_panel._white_balance_guard, "shutdown", shutdown)
        patch.setattr(window._stream_manager, "stop_all", stop)
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()
        shutdown.assert_called_once_with()
        stop.assert_not_called()
        text = window.statusBar().currentMessage()
        assert "白平衡资源尚未释放" in text
        assert text in window._calib_panel._log_text.toPlainText()


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_result_signal_before_worker_exit_still_blocks_lifecycle(app, window, monkeypatch, outcome):
    panel = window._calib_panel
    release = threading.Event()
    result_emitted = threading.Event()
    intr = CameraIntrinsics(np.eye(3), np.zeros((1, 5)), (640, 480), rms_error=.1)
    stereo = StereoCalibResult(
        np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3), rms_error=.1,
    )

    class TailWorker(PairCalibrationWorker):
        def run(self):
            if outcome == 'success':
                pair = StereoPairCalibration('rgb', 'left', 'right', intr, intr, stereo)
                self.finished_ok.emit('rgb', 'left', 'right', intr, intr, pair)
            else:
                self.failed.emit('rgb', 'simulated calibration failure')
            result_emitted.set()
            release.wait(10)

    worker = TailWorker(
        "rgb", "left", "right", None, None,
        None,
        left_intrinsics=intr, right_intrinsics=intr, parent=panel,
    )
    worker.finished_ok.connect(panel._on_pair_calibration_finished)
    worker.failed.connect(panel._on_pair_calibration_failed)
    panel._calib_worker = worker
    panel._set_calibration_busy(True)
    stop = Mock()
    shutdown = Mock(return_value=True)
    monkeypatch.setattr(window._stream_manager, "stop_all", stop)
    monkeypatch.setattr(panel._white_balance_guard, "shutdown", shutdown)
    worker.start()
    try:
        assert result_emitted.wait(5)
        wait_until(app, lambda: panel._calib_worker is None)
        assert (window._current_session.load_rig() is not None) == (outcome == "success")
        assert worker.isRunning() and panel.has_active_background_tasks()
        assert not panel.has_active_operations
        window._on_new_session()
        assert_refused(window, "新建会话")
        window._on_load_session()
        assert_refused(window, "加载会话")
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()
        stop.assert_not_called()
        shutdown.assert_not_called()
    finally:
        release.set()
        assert worker.wait(5000)
        app.processEvents()
    assert not panel.has_active_background_tasks()
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    shutdown.assert_called_once_with()
    stop.assert_called_once_with()


def test_canceled_save_and_undo_keep_original_session_ownership(app, window, monkeypatch, tmp_path):
    panel = window._calib_panel
    old = panel._session
    target = CalibrationSession(tmp_path / "target")
    image = np.zeros((12, 12, 3), np.uint8)
    target.save_frame_pair("rgb", image, image, frame_idx=0)
    entered = threading.Event()
    release = threading.Event()
    save = old.save_frame_pair

    def blocked_save(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("Test did not release the save worker")
        return save(*args, **kwargs)

    monkeypatch.setattr(old, "save_frame_pair", blocked_save)
    index, token = panel._start_frame_save("rgb", image, image)
    worker = panel._save_workers[token]
    try:
        assert entered.wait(5)
        assert panel._save_sessions[token] is old
        with pytest.raises(RuntimeError):
            panel.set_session(target)
        entry = _UndoCaptureEntry("rgb", "left", "right", index, token, False, False, False, old)
        panel._undo_apply_entry(entry)
        # Force a late-callback context change to verify ownership independently of the guard.
        panel._session = target
    finally:
        release.set()
        assert worker.wait(5000)
        wait_until(app, lambda: not panel.has_active_background_tasks())
    assert not (old.session_dir / "images/rgb/left_0000.png").exists()
    assert (target.session_dir / "images/rgb/left_0000.png").exists()
    assert not panel._save_sessions and not panel._pending_save_tokens
    save("rgb", image, image, frame_idx=0)
    panel._undo_apply_entry(entry)
    assert not (old.session_dir / "images/rgb/left_0000.png").exists()
    assert (target.session_dir / "images/rgb/left_0000.png").exists()


def test_aux_metadata_permission_error_keeps_result_exportable(app, window, monkeypatch, tmp_path):
    panel = window._calib_panel
    window._stream_manager.set_auxiliary_camera(AuxiliaryCameraConfig("aux", ""))
    intr = CameraIntrinsics(
        np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
        np.zeros((1, 5)), (640, 480), rms_error=.1,
    )
    stereo = StereoCalibResult(
        np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3), rms_error=.1,
    )
    panel._pair_calibs["rgb"] = StereoPairCalibration("rgb", "left", "right", intr, intr, stereo)
    panel._intrinsics.update(left=intr, right=intr)
    panel._pending_aux_context = (
        "aux", panel._aux_modality(), panel._aux_board_config(), "left",
    )
    record = Mock(side_effect=PermissionError("multimodal metadata is read-only"))
    monkeypatch.setattr(panel._session, "record_multimodal_result", record)
    emitted = []
    panel.calibration_finished.connect(emitted.append)
    panel._set_aux_busy(True)
    assert panel.has_active_background_tasks()
    extr = SimpleNamespace(
        T_rgb_left_aux=np.eye(4), rms_error=.2, frame_count=3, per_frame_errors=[.2] * 3,
    )
    panel._on_aux_calibration_finished(intr, extr)
    wait_until(app, lambda: not panel.has_active_background_tasks())
    record.assert_called_once()
    assert not panel._aux_ui_busy and panel._aux_worker is None
    assert not panel._ui_operation_busy
    assert panel._btn_import.isEnabled() and panel._combo_data_filter.isEnabled()
    assert len(emitted) == 1 and emitted[0] is window._rig
    assert panel._aux_intrinsics is intr
    np.testing.assert_array_equal(window._rig.aux_cameras["aux"].T_rgb_left_aux, extr.T_rgb_left_aux)
    assert window._act_export.isEnabled() and panel._btn_export_results.isEnabled()
    assert "aux" in window._current_session.load_rig().aux_cameras
    export_path = tmp_path / "manual_export.yml"
    export_rig_opencv_yaml(window._rig, export_path)
    assert export_path.is_file() and export_path.stat().st_size > 0
    log = panel._log_text.toPlainText()
    assert "会话记录保存失败" in log and "结果仍可导出" in log
    assert "multimodal metadata is read-only" in log
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
