"""Regression checks for automatic result storage and repeat calibration."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QMessageBox

from multicamera.board.detector import DetectionResult
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.models import CameraIntrinsics, StereoCalibResult, StereoPairCalibration
from multicamera.calibration.multiview import MultiViewCalibrator
from multicamera.calibration.stereo import StereoCalibrator
from multicamera.io import export
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import CameraConfig, StereoPairConfig
from multicamera.ui import main_window
from multicamera.ui.theme import configure_application_style


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setattr(main_window, "sessions_dir", lambda: tmp_path / "sessions")
    win = main_window.MainWindow()
    yield win
    for worker in list(win._calib_panel._save_workers.values()):
        worker.wait(10000)
    if win._calib_panel._calib_worker is not None:
        win._calib_panel._calib_worker.wait(10000)
    app.processEvents()
    win.close()


@pytest.fixture
def rig():
    intr = CameraIntrinsics(
        np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
        np.zeros((1, 5)), (640, 480), rms_error=0.1,
    )
    stereo = StereoCalibResult(
        np.eye(3), np.array([[-0.1], [0.], [0.]]), np.eye(3), np.eye(3), rms_error=0.1,
    )
    pair = StereoPairCalibration("rgb", "left", "right", intr, intr, stereo)
    cal = MultiViewCalibrator()
    cal.add_pair_calibration(pair)
    return cal.calibrate()


def wait_until(app, predicate):
    loop = QEventLoop()
    poll = QTimer()
    poll.setInterval(10)
    poll.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    poll.start()
    deadline.start(30000)
    if not predicate():
        loop.exec()
    poll.stop()
    deadline.stop()
    app.processEvents()
    assert predicate(), "Background calibration/save did not finish"


def seed_observations(window, count=8):
    panel = window._calib_panel
    pair = StereoPairConfig(
        "rgb", CameraConfig("left", "", "left"), CameraConfig("right", "", "right"),
    )
    window._stream_manager.add_virtual_stereo_pair(pair)
    panel._spin_min.setValue(5)
    panel._intr_cals = {
        cid: IntrinsicCalibrator(panel._board, min_frames=5) for cid in ("left", "right")
    }
    panel._stereo_cals["rgb"] = StereoCalibrator(panel._board, min_frames=5)
    panel._fqs = {"left": [], "right": []}
    # Solver detections are simulated below; captures still need a nonblank photo fixture.
    texture = cv2.resize(panel._board.generate_image(pixel_size=40), (640, 480))
    image = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    k = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    ids = np.arange(panel._board.num_corners, dtype=np.int32).reshape(-1, 1)
    records = []
    for i in range(count + 1):
        rv = np.array([0.12 * np.sin(i), 0.2 * np.cos(i), 0.04 * i])
        tv = np.array([[-0.3 + .03 * i], [-0.2 + .01 * i], [1.2 + .04 * i]])
        dl, dr = [
            DetectionResult(
                cv2.projectPoints(
                    panel._board.object_points, rv, tv + offset, k, np.zeros(5),
                )[0].astype(np.float32), ids.copy(), [], None, (640, 480),
            )
            for offset in (np.zeros((3, 1)), np.array([[-.1], [0.], [0.]]))
        ]
        records.append((dl, dr))
        if i < count:
            panel._intr_cals["left"].add_detection(dl, image=image)
            panel._intr_cals["right"].add_detection(dr, image=image)
            panel._stereo_cals["rgb"].add_detection_pair(dl, dr)
    panel.refresh_pairs()
    return image, records[-1]


def test_completion_without_session_saves_readable_files(window, rig):
    window._on_calib_finished(rig)
    session = window._current_session
    assert session is not None
    assert session.load_rig().reference_camera == "left"
    results = session.result_directory
    for name in ("rig_opencv.yml", "rig_opencv_standard.yml"):
        fs = cv2.FileStorage(str(results / name), cv2.FILE_STORAGE_READ)
        try:
            assert fs.isOpened()
            np.testing.assert_allclose(fs.getNode("M1").mat(), rig.pairs["rgb"].left_intrinsics.camera_matrix)
        finally:
            fs.release()
    assert "cam1:" in (results / "rig_kalibr.yaml").read_text()
    assert str(results.resolve()) in window.statusBar().currentMessage()


def test_repeat_completion_updates_existing_session(window, rig):
    window._on_calib_finished(rig)
    session = window._current_session
    previous = session.result_directory
    previous_json = (previous / 'rig.json').read_bytes()
    rig.pairs["rgb"].stereo.rms_error = 0.025
    window._on_calib_finished(rig)
    assert window._current_session is session
    assert session.load_rig().pairs["rgb"].stereo.rms_error == 0.025
    assert session.result_directory != previous
    assert (previous / 'rig.json').read_bytes() == previous_json


def test_disconnected_pairs_each_get_standard_and_kalibr_files(window, rig):
    first = rig.pairs["rgb"]
    second = StereoPairCalibration(
        "rgb2", "left2", "right2", first.left_intrinsics, first.right_intrinsics, first.stereo,
    )
    rig.pairs["rgb2"] = second
    window._on_calib_finished(rig)
    results = window._current_session.result_directory
    assert (results / "rig.json").is_file()
    for index, pair_name in enumerate(("rgb", "rgb2"), start=1):
        fs = cv2.FileStorage(
            str(results / f"rig_pair_{index:02d}_opencv_standard.yml"), cv2.FILE_STORAGE_READ,
        )
        try:
            assert fs.getNode("pair_name").string() == pair_name
        finally:
            fs.release()
        assert "cam1:" in (results / f"rig_pair_{index:02d}_kalibr.yaml").read_text()
    assert "rig_kalibr.yaml" in window._calib_panel._log_text.toPlainText()


def test_failed_format_does_not_lose_other_results(window, rig, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated Kalibr failure")

    monkeypatch.setattr(export, "export_rig_kalibr_yaml", fail)
    window._on_calib_finished(rig)
    results = window._current_session.result_directory
    assert (results / "rig.json").is_file()
    assert (results / "rig_opencv_standard.yml").is_file()
    assert "simulated Kalibr failure" in window._calib_panel._log_text.toPlainText()


def test_repeat_partial_export_never_exposes_previous_format_as_current(window, rig, monkeypatch):
    window._on_calib_finished(rig)
    old = window._current_session.result_directory
    old_yaml = (old / 'rig_kalibr.yaml').read_bytes()

    def fail(*args, **kwargs):
        raise OSError('Kalibr unavailable this generation')

    monkeypatch.setattr(export, 'export_rig_kalibr_yaml', fail)
    window._on_calib_finished(rig)
    current = window._current_session.result_directory
    panel = window._calib_panel
    assert current == panel._result_directory != old
    assert not (current / 'rig_kalibr.yaml').exists()
    assert (old / 'rig_kalibr.yaml').read_bytes() == old_yaml
    assert (current / 'rig.json').is_file()
    assert (current / 'rig_opencv_standard.yml').is_file()
    assert '部分格式未生成' in panel._lbl_result_storage.text()
    assert str(current.resolve()) in panel._lbl_result_storage.toolTip()
    assert str(old.resolve()) not in panel._lbl_result_storage.toolTip()
    assert panel._btn_open_results.isEnabled() and panel._btn_copy_results.isEnabled()


@pytest.mark.parametrize('strategy', ['filter', 'optimization'])
def test_strategy_change_keeps_outdated_history_until_new_result(window, rig, strategy):
    window._on_calib_finished(rig)
    panel = window._calib_panel
    previous = panel._result_directory
    if strategy == 'filter':
        panel._combo_data_filter.setCurrentIndex(panel._combo_data_filter.findData('quality'))
    else:
        panel._combo_calib_strategy.setCurrentIndex(panel._combo_calib_strategy.findData('joint_refine'))
    assert window._rig is panel._rig is None
    assert not window._act_export.isEnabled()
    assert panel._results_outdated
    assert panel._previous_result_directory == previous
    assert '旧结果已过期' in panel._lbl_result_storage.text()
    assert str(previous.resolve()) in panel._lbl_result_storage.text()
    panel._clear_calibration_results()  # Repeated changes must not forget the original path.
    assert panel._previous_result_directory == previous
    assert (previous / 'rig.json').is_file()
    window._on_calib_finished(rig)
    assert not panel._results_outdated
    assert panel._previous_result_directory is None
    assert panel._result_directory != previous


def test_save_error_keeps_manual_export_available(window, rig, monkeypatch):
    messages = []

    def fail(*args):
        raise PermissionError("read-only result directory")

    monkeypatch.setattr(CalibrationSession, "save_calibration_results", fail)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[-1]))
    window._on_calib_finished(rig)
    assert window._act_export.isEnabled()
    assert "read-only result directory" in messages[0]
    assert "自动保存失败" in window.statusBar().currentMessage()


def test_failed_json_write_keeps_previous_file(tmp_path, rig, monkeypatch):
    session = CalibrationSession(tmp_path)
    path = session.save_rig(rig)
    previous = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError, match="disk full"):
        session.save_rig(rig)
    assert path.read_bytes() == previous


@pytest.mark.parametrize('undo_disk_error', [False, True])
def test_recalibrate_after_supplement_uses_all_observations(app, window, monkeypatch, undo_disk_error):
    image, (dl, dr) = seed_observations(window)
    panel = window._calib_panel
    used_counts = []
    calibrate = IntrinsicCalibrator.calibrate

    def tracked(self, flags=0):
        used_counts.append(self.num_frames)
        return calibrate(self, flags=flags)

    monkeypatch.setattr(IntrinsicCalibrator, "calibrate", tracked)
    panel._calibrate_pair("rgb")
    wait_until(app, lambda: panel._calib_worker is None)
    assert used_counts == [8, 8]
    assert panel._pair_rows["rgb"].btn.isEnabled()
    assert panel._btn_calib_all.isEnabled()
    session = window._current_session
    monkeypatch.setattr(window._stream_manager, "get_sync_pair", lambda name: (image, image))
    monkeypatch.setattr(panel._intr_cals["left"].detector, "detect", lambda frame: dl)
    monkeypatch.setattr(panel._intr_cals["right"].detector, "detect", lambda frame: dr)
    panel._capture_impl()
    wait_until(app, lambda: not panel._save_workers and not panel._save_queue)
    assert "rgb" not in panel._pair_calibs
    assert not window._act_export.isEnabled()
    assert panel._pair_rows["rgb"].btn.isEnabled()
    panel._calibrate_pair("rgb")
    wait_until(app, lambda: panel._calib_worker is None)
    assert used_counts == [8, 8, 9, 9] + [6] * 6
    choice = panel._pair_calibs['rgb'].calibration_info['model_selection']
    assert choice['validation_status'] == 'passed'
    assert window._act_export.isEnabled()
    assert window._current_session is session
    np.testing.assert_allclose(
        session.load_rig().pairs["rgb"].left_intrinsics.camera_matrix[0, 0], 800., atol=.2,
    )
    if undo_disk_error:
        def fail_delete(*args):
            raise PermissionError('capture directory is read-only')
        monkeypatch.setattr(session, 'delete_saved_pair_frame', fail_delete)
    panel._on_undo_capture()
    assert panel._stereo_cals["rgb"].num_frames == 8
    assert "rgb" not in panel._pair_calibs
    assert panel._pair_rows["rgb"].btn.isEnabled()
    assert not window._act_export.isEnabled()
    if undo_disk_error:
        assert '磁盘图片删除失败' in panel._log_text.toPlainText()


def test_batch_recalibrates_completed_pairs_without_new_capture(app, window, monkeypatch):
    monkeypatch.delattr(cv2.aruco, "calibrateCameraCharuco", raising=False)
    seed_observations(window)
    panel = window._calib_panel
    panel._calibrate_pair("rgb")
    wait_until(app, lambda: panel._calib_worker is None)
    calls = []
    calibrate = IntrinsicCalibrator.calibrate

    def tracked(self, flags=0):
        calls.append(self.num_frames)
        return calibrate(self, flags=flags)

    monkeypatch.setattr(IntrinsicCalibrator, "calibrate", tracked)
    panel._on_calibrate_all()
    wait_until(app, lambda: panel._calib_worker is None and not panel._calib_queue)
    assert calls == [8, 8]
    assert window._current_session.load_rig() is not None


def test_preview_failure_does_not_block_save_or_buttons(app, window, monkeypatch):
    seed_observations(window)
    panel = window._calib_panel

    def fail(*args):
        raise RuntimeError("simulated preview failure")

    monkeypatch.setattr(panel, "_show_rect_preview", fail)
    panel._calibrate_pair("rgb")
    wait_until(app, lambda: panel._calib_worker is None)
    assert window._current_session.load_rig() is not None
    assert panel._pair_rows["rgb"].btn.isEnabled()
    assert panel._btn_import.isEnabled() and panel._btn_to_calibration.isEnabled()
    # This fixture is a virtual pair: live capture stays unavailable, not busy.
    assert not panel._btn_capture.isEnabled()
    assert "simulated preview failure" in panel._log_text.toPlainText()


@pytest.mark.parametrize('action', ['filter', 'strategy', 'model', 'reset'])
def test_invalidating_results_removes_old_rectification(window, action):
    panel = window._calib_panel
    panel._lbl_rect_preview.setPixmap(QPixmap(80, 40))
    panel._lbl_rect_preview.show()
    panel._lbl_rect_context.setText('旧针孔结果')
    panel._lbl_rect_context.show()
    if action == 'filter':
        panel._combo_data_filter.setCurrentIndex(panel._combo_data_filter.findData('quality'))
    elif action == 'strategy':
        panel._combo_calib_strategy.setCurrentIndex(panel._combo_calib_strategy.findData('joint_refine'))
    elif action == 'model':
        panel.apply_recommended_model('fisheye')
    else:
        panel._reset_all()
    assert panel._lbl_rect_preview.pixmap().isNull()
    assert panel._lbl_rect_preview.isHidden()
    assert panel._lbl_rect_context.isHidden()
    assert not panel._lbl_rect_context.text()


def test_rectification_preview_uses_only_own_pair_and_rejects_wrong_size(window, rig, monkeypatch):
    panel = window._calib_panel
    pair = rig.pairs['rgb']
    image = np.zeros((480, 640, 3), np.uint8)
    panel._import_last_pair = (image, image)  # A different historical pair must not be reused.
    calls = []

    def generate(left, right, stereo, **kwargs):
        calls.append((left, right))
        return np.hstack((left, right))

    monkeypatch.setattr(panel._evaluator, 'generate_rectification_preview', generate)
    panel._show_rect_preview('rgb', pair)
    assert not calls
    assert panel._lbl_rect_preview.isHidden()
    panel._imported_pair_frames = {'rgb': [(image, image)]}
    panel._show_rect_preview('rgb', pair)
    assert len(calls) == 1
    assert not panel._lbl_rect_preview.pixmap().isNull()
    assert 'rgb · 针孔模型 · 该组最后导入配对' in panel._lbl_rect_context.text()
    panel._imported_pair_frames['rgb'] = [(image[:100], image)]
    with pytest.raises(ValueError, match='尺寸'):
        panel._show_rect_preview('rgb', pair)
    assert len(calls) == 1
    assert panel._lbl_rect_preview.pixmap().isNull()
    assert panel._lbl_rect_context.isHidden()


@pytest.mark.parametrize('size', [(1000, 680), (1280, 800), (1440, 1024)])
def test_toolbar_tools_and_session_remain_accessible_at_all_sizes(window, app, size, tmp_path):
    configure_application_style(app)
    window._current_session = CalibrationSession.create(tmp_path, window._board, [])
    long_name = 'calib_' + 'LongSessionName' * 20
    window._current_session._metadata['name'] = long_name
    window.resize(*size)
    window._sync_toolbar_density()
    window.show()
    app.processEvents()
    toolbar = window._toolbar
    buttons = [toolbar.widgetForAction(action) for action in (
        window._act_start_streams, window._act_stop_streams,
        window._act_config_tb, window._act_rgb_settings_tb,
    )] + [window._session_button]
    for button in buttons:
        assert button.isVisible()
        assert toolbar.rect().contains(button.geometry())
        assert button.toolTip()
    for action in (window._act_config_tb, window._act_rgb_settings_tb):
        expected = Qt.ToolButtonIconOnly if size[0] < 1200 else Qt.ToolButtonTextBesideIcon
        assert toolbar.widgetForAction(action).toolButtonStyle() == expected
    assert window._session_menu.actions() == [window._act_new_session, window._act_load_session]
    assert long_name in window._session_button.toolTip()
    assert len(window._session_button.text()) < len(long_name)
    message = '导出失败：' + '/very_long_directory/' * 30
    window.statusBar().showMessage(message)
    assert window.statusBar().currentMessage() == window.statusBar().toolTip() == message
    window.statusBar().clearMessage()
    assert window.statusBar().toolTip() == ''


@pytest.mark.parametrize('control_metrics', ['default', 'wide'])
@pytest.mark.parametrize('size', [(1000, 680), (1001, 681), (1280, 800), (1440, 1024)])
@pytest.mark.parametrize('phase', ['calibration', 'results'])
def test_long_result_text_wraps_without_clipping_or_horizontal_scroll(
        window, rig, app, size, phase, control_metrics):
    configure_application_style(app)
    seed_observations(window)
    panel = window._calib_panel
    if control_metrics == 'wide':
        # Exercise larger platform/font metrics without depending on installed Windows fonts.
        panel.setStyleSheet('QToolButton { font-size:16px; padding-left:9px; padding-right:10px; }')
    panel._pair_calibs = dict(rig.pairs)
    panel._sync_pair_states()
    panel.set_result_model(rig)
    long_directory = Path('/result_root') / ('calib_' + 'a' * 180)
    panel.set_result_storage(long_directory, [long_directory / 'rig.json'],
                             ['rig_kalibr.yaml: ' + '长错误说明' * 40])
    window.resize(*size)
    window.show()
    panel.show_workflow(phase)
    app.processEvents()
    app.processEvents()
    scroll = panel._inspector_scroll
    assert scroll.horizontalScrollBar().maximum() == 0, {
        'phase': phase, 'size': size, 'metrics': control_metrics,
        'viewport': scroll.viewport().size().toTuple(),
        'content': scroll.widget().size().toTuple(),
        'minimum': scroll.widget().minimumSizeHint().toTuple(),
    }
    labels = ([panel._lbl_result_storage, panel._lbl_result_decision] if phase == 'results'
              else [panel._pair_rows['rgb']._lbl_status, panel._lbl_calib_hint,
                    panel._lbl_model_choice, panel._lbl_calibration_options_summary])
    for label in labels:
        assert label.isVisible()
        assert label.hasHeightForWidth()
        assert label.height() >= label.heightForWidth(label.width())
    assert str(long_directory.resolve()) in panel._lbl_result_storage.toolTip()
    assert str(long_directory) not in panel._lbl_result_storage.text()
