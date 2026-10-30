"""Configuration protection, export scope and multimodal readiness regressions."""

from dataclasses import replace
import json
import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QMessageBox

from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.models import (
    AuxCameraCalibration, CameraIntrinsics, CameraModel, StereoCalibResult, StereoPairCalibration,
)
from multicamera.calibration.multiview import MultiViewCalibrator
from multicamera.calibration.planar import PatternBoardConfig, PlanarDetectionResult
from multicamera.calibration.stereo import StereoCalibrator
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import AuxiliaryCameraConfig, CameraConfig, FrameEntry, StereoPairConfig
from multicamera.ui import main_window
from multicamera.ui.widgets.config_dialog import ConfigDialog
from multicamera.ui.widgets.export_dialog import ExportDialog
from multicamera.ui.configuration import configuration_impact
from multicamera.ui.widgets.stream_view import CoverageTracker


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setattr(main_window, "sessions_dir", lambda: tmp_path / "sessions")
    win = main_window.MainWindow()
    yield win
    win._calib_panel._auto_timer.stop()
    win._stream_manager.stop_all()
    win.close()
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


@pytest.fixture
def calibrated(window):
    sm, panel = window._stream_manager, window._calib_panel
    intr = CameraIntrinsics(np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
                            np.zeros((1, 5)), (640, 480), rms_error=.1)
    stereo = StereoCalibResult(np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3))
    mv = MultiViewCalibrator()
    for name, left, right in (("rgb", "left", "right"), ("rgb2", "left2", "right2")):
        sm.add_stereo_pair(StereoPairConfig(name, CameraConfig(left, f"http://device/{left}"),
                                           CameraConfig(right, f"http://device/{right}")))
        pc = StereoPairCalibration(name, left, right, intr, intr, stereo)
        panel._pair_calibs[name] = pc
        mv.add_pair_calibration(pc)
        panel._stereo_cals[name] = StereoCalibrator(panel._board)
        for cid in (left, right):
            panel._intrinsics[cid] = intr
            panel._intr_cals[cid] = IntrinsicCalibrator(panel._board)
    sm.set_auxiliary_camera(AuxiliaryCameraConfig("ir", "http://device/ir"))
    rig = mv.calibrate()
    transform = np.eye(4)
    transform[0, 3] = .04
    rig.aux_cameras["ir"] = AuxCameraCalibration("ir", "ir", intr, transform)
    panel._rig = rig
    panel._aux_intrinsics = intr
    board = PatternBoardConfig.ir_chessboard()
    panel._aux_observation_board = board
    obj = np.zeros((24, 3), np.float32)
    points = np.zeros((24, 1, 2), np.float32)
    det = PlanarDetectionResult(True, obj, points, (640, 480), board.pattern_type)
    panel._aux_pairs = [(str(i), det, det) for i in range(3)]
    panel._aux_images = [(np.zeros((480, 640, 3), np.uint8),) * 2]
    panel._check_aux_order.setChecked(True)  # This fixture represents an already reviewed dataset.
    window._rig = rig
    window._act_export.setEnabled(True)
    panel.refresh_pairs()
    return rig


def run_config(window, monkeypatch, edit, answer=QMessageBox.Yes):
    questions = []

    def question(*args):
        questions.append(args[2])
        return answer

    def execute(dialog):
        edit(dialog)
        dialog._on_accept()
        return dialog.result()

    monkeypatch.setattr(QMessageBox, "question", question)
    monkeypatch.setattr(ConfigDialog, "exec", execute)
    window._on_config()
    return questions


def test_unchanged_config_preserves_calibration_and_streams(window, calibrated, monkeypatch):
    original = window._calib_panel._intr_cals["left"]
    stopped = []
    monkeypatch.setattr(window._stream_manager, "stop_all", lambda: stopped.append(True))
    assert not run_config(window, monkeypatch, lambda dlg: None)
    assert not stopped
    assert window._rig is calibrated
    assert window._calib_panel._intr_cals["left"] is original
    assert window._act_export.isEnabled()


def test_aux_only_change_preserves_rgb_and_invalidates_only_aux(window, calibrated, monkeypatch):
    original = window._calib_panel._intr_cals["left"]
    questions = run_config(window, monkeypatch, lambda dlg: dlg._edit_aux_url.setText("http://device/new-ir"))
    assert len(questions) == 1
    assert "辅助相机" in questions[0]
    assert "以下组" not in questions[0]
    assert set(window._calib_panel._pair_calibs) == {"rgb", "rgb2"}
    assert window._calib_panel._intr_cals["left"] is original
    assert not window._calib_panel._aux_pairs
    assert not window._rig.aux_cameras
    assert window._act_export.isEnabled()


def test_cancelled_config_does_not_mutate_streams_data_or_files(window, calibrated, monkeypatch, tmp_path):
    grabber = window._stream_manager._grabbers["ir"]
    path = tmp_path / "old-result.json"
    path.write_text("old result", encoding="utf-8")
    stopped = []
    monkeypatch.setattr(window._stream_manager, "stop_all", lambda: stopped.append(True))
    questions = run_config(window, monkeypatch, lambda dlg: dlg._edit_aux_url.setText("http://device/new-ir"),
                           QMessageBox.No)
    assert questions
    assert not stopped
    assert window._stream_manager._grabbers["ir"] is grabber
    assert window._stream_manager.auxiliary_camera.url == "http://device/ir"
    assert window._rig is calibrated
    assert len(window._calib_panel._aux_pairs) == 3
    assert path.read_text() == "old result"


@pytest.mark.parametrize("row,remaining,aux_kept", [(1, "rgb", True), (0, "rgb2", False)])
def test_pair_change_invalidates_only_its_dependencies(window, calibrated, monkeypatch, row, remaining, aux_kept):
    original = window._calib_panel._pair_calibs[remaining]
    run_config(window, monkeypatch, lambda dlg: dlg._table.item(row, 3).setText("http://device/changed"))
    assert set(window._calib_panel._pair_calibs) == {remaining}
    assert window._calib_panel._pair_calibs[remaining] is original
    assert bool(window._rig.aux_cameras) == aux_kept
    assert bool(window._calib_panel._aux_pairs) == aux_kept
    assert window._rig.reference_camera in {cid for pair in window._rig.pairs.values()
                                           for cid in (pair.left_id, pair.right_id)}


@pytest.mark.parametrize("change", ["board", "model"])
def test_global_geometry_change_confirms_before_clearing(window, calibrated, monkeypatch, tmp_path, change):
    session = CalibrationSession.create(tmp_path, window._board, list(window._stream_manager.stereo_pairs.values()))
    image = np.zeros((10, 10, 3), np.uint8)
    session.save_frame_pair("rgb", image, image)
    window._current_session = session
    window._calib_panel.set_session(session)

    def edit(dlg):
        if change == "board":
            dlg._spin_cols.setValue(dlg._spin_cols.value() + 1)
        else:
            dlg._combo_model.setCurrentIndex(dlg._combo_model.findData(CameraModel.FISHEYE))

    assert run_config(window, monkeypatch, edit)
    assert window._rig is None
    assert not window._act_export.isEnabled()
    assert not window._calib_panel._intr_cals
    assert (session.session_dir / "images/rgb/left_0000.png").is_file()
    metadata = json.loads((session.session_dir / "session.json").read_text())
    assert metadata["configuration_history"]
    assert metadata["camera_model"] == window._camera_model.value


def test_config_blocked_during_background_or_auto_capture(window, monkeypatch):
    opened, messages = [], []
    monkeypatch.setattr(ConfigDialog, "exec", lambda self: opened.append(True))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args[2]))
    window._calib_panel._auto_timer.start()
    window._on_config()
    assert not opened
    assert messages


def test_offline_pair_survives_configuration_reopen(window, monkeypatch):
    sm = window._stream_manager
    sm.add_virtual_stereo_pair(StereoPairConfig("offline", CameraConfig("offline_L", ""),
                                              CameraConfig("offline_R", "")))
    run_config(window, monkeypatch, lambda dlg: None)
    assert "offline" in sm.stereo_pairs
    assert not sm.configured_camera_urls()


def test_session_configuration_history_preserves_files(window, calibrated, tmp_path, monkeypatch):
    sm = window._stream_manager
    session = CalibrationSession.create(tmp_path, window._board, list(sm.stereo_pairs.values()), sm.auxiliary_camera)
    image = np.zeros((10, 10, 3), np.uint8)
    session.save_multimodal_pair(image, image)
    before = (session.session_dir / "multimodal/aux_0000.png").read_bytes()
    window._current_session = session
    window._calib_panel.set_session(session)
    run_config(window, monkeypatch, lambda dlg: dlg._edit_aux_url.setText("http://device/new-ir"))
    metadata = json.loads((session.session_dir / "session.json").read_text())
    assert metadata["auxiliary_camera"]["url"] == "http://device/new-ir"
    assert metadata["configuration_history"][-1]["previous"]["auxiliary_camera"]["url"] == "http://device/ir"
    assert metadata["multimodal_frame_count"] == 1
    assert (session.session_dir / "multimodal/aux_0000.png").read_bytes() == before


def test_readiness_requires_rgb_intrinsics_but_offline_does_not_require_live_stream(window, calibrated):
    panel = window._calib_panel
    intr = panel._intrinsics.pop("left")
    panel._refresh_aux_readiness()
    assert not panel._btn_calib_aux.isEnabled()
    assert "请先完成 RGB" in panel._lbl_aux_readiness.text()
    assert "待就绪" in panel._sections[2]._header.text()
    panel._intrinsics["left"] = intr
    panel._refresh_aux_readiness()
    assert panel._btn_calib_aux.isEnabled()
    assert not panel._btn_capture_aux.isEnabled()
    assert "标定就绪" in panel._lbl_aux_readiness.text()


@pytest.mark.parametrize("problem", ["stale", "skew", "missing", "nonfinite", "disconnected"])
def test_live_readiness_blocks_invalid_frames_and_capture(window, calibrated, problem, monkeypatch):
    sm, panel = window._stream_manager, window._calib_panel
    image = np.zeros((480, 640, 3), np.uint8)
    now = time.time()
    sm._latest_frames = {cid: FrameEntry(image, now) for cid in ("left", "ir")}
    panel._refresh_aux_readiness()
    assert panel._btn_capture_aux.isEnabled()
    if problem == "missing":
        sm._latest_frames.pop("ir")
    elif problem == "disconnected":
        sm.camera_disconnected.emit("ir", "已断开")
    else:
        timestamp = {"stale": now - 2, "skew": now - .6, "nonfinite": float("nan")}[problem]
        sm._latest_frames["ir"] = FrameEntry(image, timestamp)
    panel._refresh_aux_readiness()
    assert not panel._btn_capture_aux.isEnabled()
    count = len(panel._aux_pairs)
    if panel._session:
        monkeypatch.setattr(panel._session, "save_multimodal_pair", lambda *args: pytest.fail("must not save"))
    panel._on_capture_aux()
    assert len(panel._aux_pairs) == count
    assert "采图未就绪" in panel._lbl_aux_readiness.text()


def test_readiness_rejects_board_or_image_size_mismatch_and_busy(window, calibrated):
    panel = window._calib_panel
    board = panel._aux_observation_board
    panel._aux_observation_board = replace(board, square_size=.04)
    panel._refresh_aux_readiness()
    assert not panel._btn_calib_aux.isEnabled()
    panel._aux_observation_board = board
    rgb = panel._aux_pairs[0][1]
    panel._aux_pairs[0] = ("bad", replace(rgb, image_size=(320, 240)), rgb)
    panel._refresh_aux_readiness()
    assert "分辨率" in panel._lbl_aux_readiness.text()
    assert not panel._btn_calib_aux.isEnabled()
    panel._aux_pairs[0] = ("ok", rgb, rgb)
    panel._set_aux_busy(True)
    assert not panel._btn_calib_aux.isEnabled()
    assert "后台任务" in panel._lbl_aux_readiness.text()
    panel._set_aux_busy(False)
    assert panel._btn_calib_aux.isEnabled()


@pytest.fixture
def export_dialog(app, calibrated):
    dialog = ExportDialog(calibrated)
    yield dialog
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def choose(dialog, scope, key="", format_key="json"):
    index = next(i for i in range(dialog._scope.count())
                 if tuple(dialog._scope.itemData(i)) == (scope, key))
    dialog._scope.setCurrentIndex(index)
    dialog._format.setCurrentIndex(dialog._format.findData(format_key))


def test_export_scope_lists_exact_content_and_standard_requires_one_pair(export_dialog):
    dlg = export_dialog
    assert not dlg._format.model().item(dlg._format.findData("standard")).isEnabled()
    assert set(dlg.selected_rig().pairs) == {"rgb", "rgb2"}
    choose(dlg, "pair", "rgb2", "standard")
    rig = dlg.selected_rig()
    assert set(rig.pairs) == {"rgb2"}
    assert not rig.aux_cameras
    assert rig.reference_camera == "left2"
    assert set(rig.extrinsics) == {"left2", "right2"}
    assert "不包含任何红外" in dlg._note.text()
    assert "ir" in dlg._summary.toPlainText()
    choose(dlg, "fusion", "ir")
    assert set(dlg.selected_rig().pairs) == {"rgb"}
    assert set(dlg.selected_rig().aux_cameras) == {"ir"}
    assert "含 RGB 右目" in dlg._note.text()


@pytest.mark.parametrize("scope,key,format_key", [
    ("all", "", "json"), ("all", "", "opencv"), ("pair", "rgb2", "standard"),
    ("pair", "rgb2", "kalibr"), ("fusion", "ir", "json"), ("fusion", "ir", "opencv"),
    ("fusion", "ir", "kalibr"),
])
def test_exported_files_match_selected_scope(export_dialog, tmp_path, scope, key, format_key):
    dlg = export_dialog
    choose(dlg, scope, key, format_key)
    path = dlg.export_to(tmp_path / dlg.suggested_filename)
    assert path.is_file()
    if format_key == "json":
        data = json.loads(path.read_text())
        assert set(data["pairs"]) == set(dlg.selected_rig().pairs)
        assert set(data["aux_cameras"]) == set(dlg.selected_rig().aux_cameras)
    elif format_key == "standard":
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            assert fs.getNode("pair_name").string() == "rgb2"
            assert fs.getNode("left_camera_id").string() == "left2"
            assert not fs.getNode("M1").empty()
            assert fs.getNode("aux_camera_count").empty()
        finally:
            fs.release()
    elif format_key == "opencv":
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            assert int(fs.getNode("aux_camera_count").real()) == len(dlg.selected_rig().aux_cameras)
        finally:
            fs.release()
    else:
        text = path.read_text()
        assert ("cam2:" in text) == (scope == "fusion")
        assert ("/ir/image_raw" in text) == (scope == "fusion")


def test_disconnected_kalibr_scope_is_blocked_before_save(export_dialog, tmp_path):
    dlg = export_dialog
    choose(dlg, "all", format_key="kalibr")
    assert not dlg._buttons.button(QDialogButtonBox.Save).isEnabled()
    assert "外参链不完整" in dlg._error.text()
    with pytest.raises(ValueError):
        dlg.export_to(tmp_path / "chain.yaml")
    assert not (tmp_path / "chain.yaml").exists()


def test_export_failure_and_wrong_extension_keep_previous_file(export_dialog, tmp_path, monkeypatch):
    dlg = export_dialog
    target = tmp_path / "old.json"
    target.write_text("previous", encoding="utf-8")

    def fail(rig, path):
        path.write_text("partial", encoding="utf-8")
        raise OSError("disk full")

    label, file_filter, suffix, _ = dlg.FORMATS["json"]
    monkeypatch.setitem(dlg.FORMATS, "json", (label, file_filter, suffix, fail))
    with pytest.raises(OSError, match="disk full"):
        dlg.export_to(target)
    assert target.read_text() == "previous"
    with pytest.raises(ValueError, match="扩展名"):
        dlg.export_to(tmp_path / "wrong.yml")
    assert not (tmp_path / "wrong.yml").exists()
    monkeypatch.setitem(dlg.FORMATS, "json", (label, file_filter, suffix, lambda rig, path: path.touch()))
    with pytest.raises(OSError, match="有效文件"):
        dlg.export_to(target)
    assert target.read_text() == "previous"


def test_manual_export_errors_are_visible_and_do_not_report_success(window, calibrated, tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(ExportDialog, "exec", lambda self: 1)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path / "wrong.yml"), ""))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[2]))
    window._on_export()
    assert messages and "扩展名" in messages[0]
    assert "已导出" not in window.statusBar().currentMessage()


def test_manual_export_routes_selected_second_pair(window, calibrated, tmp_path, monkeypatch):
    def execute(dlg):
        choose(dlg, "pair", "rgb2", "standard")
        return 1

    path = tmp_path / "selected.yml"
    monkeypatch.setattr(ExportDialog, "exec", execute)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    window._on_export()
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        assert fs.getNode("pair_name").string() == "rgb2"
    finally:
        fs.release()
    assert "已导出" in window.statusBar().currentMessage()


def test_configuration_reconnect_preserves_unaffected_preview_coverage(window, calibrated, monkeypatch):
    tracker = CoverageTracker((640, 480))
    tracker.total_frames = 8
    window._stream_view._trackers["left"] = tracker
    window._act_stop_streams.setEnabled(True)
    calls = []
    monkeypatch.setattr(window._stream_manager, "start_all", lambda: calls.append(True))
    run_config(window, monkeypatch, lambda dlg: dlg._edit_aux_url.setText("http://device/new-ir"))
    assert calls == [True]
    assert window._stream_view._trackers["left"] is tracker
    assert tracker.total_frames == 8


def test_shared_camera_groups_are_invalidated_together(window):
    board = window._board
    old = {
        "a": StereoPairConfig("a", CameraConfig("left", "url-left"), CameraConfig("shared", "url-shared")),
        "b": StereoPairConfig("b", CameraConfig("shared", "url-shared"), CameraConfig("right", "url-right")),
    }
    proposed = [replace(old["a"], left=CameraConfig("left", "new-left")), old["b"]]
    impact = configuration_impact(board, board, CameraModel.PINHOLE, CameraModel.PINHOLE, old, proposed, None, None)
    assert impact.changed_pairs == {"a", "b"}


def test_standard_scope_switch_and_backstop_never_export_first_pair_implicitly(export_dialog, tmp_path):
    dlg = export_dialog
    choose(dlg, "pair", "rgb2", "standard")
    dlg._scope.setCurrentIndex(0)
    assert dlg.format_key == "opencv"
    dlg._format.setCurrentIndex(dlg._format.findData("standard"))
    assert not dlg._buttons.button(QDialogButtonBox.Save).isEnabled()
    with pytest.raises(ValueError, match="明确选择"):
        dlg.export_to(tmp_path / "forbidden.yml")
    assert not (tmp_path / "forbidden.yml").exists()


def test_new_aux_data_invalidates_only_aux_result(window, calibrated):
    panel = window._calib_panel
    _, rgb, aux = panel._aux_pairs[0]
    frame = np.zeros((480, 640, 3), np.uint8)
    panel._on_import_aux_images_ready([("1", frame, frame, rgb, aux)] * 3)
    assert not window._rig.aux_cameras
    assert set(window._rig.pairs) == {"rgb", "rgb2"}
    assert window._act_export.isEnabled()
    assert panel._aux_intrinsics is None


def test_suffix_completion_requires_confirmation_before_overwriting(window, calibrated, tmp_path, monkeypatch):
    target = tmp_path / "existing.json"
    target.write_text("previous", encoding="utf-8")
    monkeypatch.setattr(ExportDialog, "exec", lambda self: 1)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target.with_suffix("")), ""))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
    window._on_export()
    assert target.read_text() == "previous"
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    window._on_export()
    assert set(json.loads(target.read_text())["pairs"]) == {"rgb", "rgb2"}


def test_aux_task_locks_stereo_mutations_until_completion(window, calibrated, monkeypatch):
    panel = window._calib_panel
    pairs = dict(panel._pair_calibs)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: pytest.fail("must not open importer"))
    panel._set_aux_busy(True)
    assert not panel._btn_capture.isEnabled()
    assert not panel._btn_import.isEnabled()
    assert not panel._combo_calib_strategy.isEnabled()
    panel._capture_impl()
    panel._on_import_images()
    panel._calibrate_pair("rgb")
    panel._on_calibrate_all()
    assert panel._calib_worker is None
    assert panel._stereo_import_worker is None
    assert panel._pair_calibs == pairs
    panel._set_aux_busy(False)
    assert panel._btn_import.isEnabled()
    assert panel._combo_calib_strategy.isEnabled()


def test_empty_calibrators_do_not_trigger_destructive_change_warning(window, monkeypatch):
    panel = window._calib_panel
    window._stream_manager.add_stereo_pair(StereoPairConfig(
        "rgb", CameraConfig("left", "http://device/left"), CameraConfig("right", "http://device/right"),
    ))
    panel._intr_cals["left"] = IntrinsicCalibrator(panel._board)
    panel._stereo_cals["rgb"] = StereoCalibrator(panel._board)
    questions = run_config(window, monkeypatch, lambda dlg: dlg._table.item(0, 3).setText("http://device/changed"))
    assert not questions


def test_invalid_offline_photos_still_require_change_confirmation(window, monkeypatch):
    panel = window._calib_panel
    window._stream_manager.add_virtual_stereo_pair(StereoPairConfig(
        "offline", CameraConfig("offline_L", ""), CameraConfig("offline_R", ""),
    ))
    frame = np.zeros((20, 20, 3), np.uint8)
    records = [("0001", frame, frame, None, None)]
    panel._offline_import_records = records
    questions = run_config(window, monkeypatch,
                           lambda dlg: dlg._spin_cols.setValue(dlg._spin_cols.value() + 1), QMessageBox.No)
    assert questions
    assert panel._offline_import_records is records


def test_configuration_metadata_write_failure_keeps_previous_file(window, calibrated, tmp_path, monkeypatch):
    sm = window._stream_manager
    session = CalibrationSession.create(tmp_path, window._board, list(sm.stereo_pairs.values()), sm.auxiliary_camera)
    metadata = session.session_dir / "session.json"
    previous = metadata.read_bytes()
    window._current_session = session
    window._calib_panel.set_session(session)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(metadata), "write_text", fail)
    run_config(window, monkeypatch, lambda dlg: dlg._edit_aux_url.setText("http://device/new-ir"))
    assert metadata.read_bytes() == previous
    assert session.metadata["auxiliary_camera"]["url"] == "http://device/new-ir"
    assert "配置记录失败" in window._calib_panel._log_text.toPlainText()
    assert set(window._rig.pairs) == {"rgb", "rgb2"}
