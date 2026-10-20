"""Regression checks for automatic result storage and repeat calibration."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from multicamera.board.detector import DetectionResult
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.models import CameraIntrinsics, StereoCalibResult, StereoPairCalibration
from multicamera.calibration.multiview import MultiViewCalibrator
from multicamera.calibration.stereo import StereoCalibrator
from multicamera.io import export
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import CameraConfig, StereoPairConfig
from multicamera.ui import main_window


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


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
    image = np.zeros((480, 640, 3), dtype=np.uint8)
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
    results = session.session_dir / "results"
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
    rig.pairs["rgb"].stereo.rms_error = 0.025
    window._on_calib_finished(rig)
    assert window._current_session is session
    assert session.load_rig().pairs["rgb"].stereo.rms_error == 0.025


def test_disconnected_pairs_each_get_standard_and_kalibr_files(window, rig):
    first = rig.pairs["rgb"]
    second = StereoPairCalibration(
        "rgb2", "left2", "right2", first.left_intrinsics, first.right_intrinsics, first.stereo,
    )
    rig.pairs["rgb2"] = second
    window._on_calib_finished(rig)
    results = window._current_session.session_dir / "results"
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
    results = window._current_session.session_dir / "results"
    assert (results / "rig.json").is_file()
    assert (results / "rig_opencv_standard.yml").is_file()
    assert "simulated Kalibr failure" in window._calib_panel._log_text.toPlainText()


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


def test_recalibrate_after_supplement_uses_all_observations(app, window, monkeypatch):
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
    assert used_counts == [8, 8, 9, 9]
    assert window._act_export.isEnabled()
    assert window._current_session is session
    np.testing.assert_allclose(
        session.load_rig().pairs["rgb"].left_intrinsics.camera_matrix[0, 0], 800., atol=.2,
    )
    panel._on_undo_capture()
    assert panel._stereo_cals["rgb"].num_frames == 8
    assert "rgb" not in panel._pair_calibs
    assert panel._pair_rows["rgb"].btn.isEnabled()


def test_batch_recalibrates_completed_pairs_without_new_capture(app, window, monkeypatch):
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
    assert panel._btn_capture.isEnabled()
    assert "simulated preview failure" in panel._log_text.toPlainText()
