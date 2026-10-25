"""Independent 7 x 5 / 80 mm IR-board configuration, import and calibration."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration.models import AuxCameraCalibration, CameraIntrinsics, MultiCameraRig
from multicamera.calibration.planar import (
    PatternBoardConfig, PatternType, PlanarDetectionResult, PlanarPatternDetector,
)
from multicamera.io.board_config import load_pattern_board_config
from multicamera.io.export import export_rig_opencv_yaml
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import (
    AuxiliaryCameraConfig, CameraConfig, StereoPairConfig, StreamManager,
)
from multicamera.ui.threads.calibration_worker import AuxCalibrationWorker
from multicamera.ui.threads.import_worker import AuxImageImportWorker
from multicamera.ui.widgets.calib_panel import CalibrationPanel
from multicamera.ui.widgets.config_dialog import ConfigDialog


@pytest.fixture
def dialog(app):
    manager = StreamManager()
    board = CharucoBoard(CharucoBoardConfig())
    dlg = ConfigDialog(manager, board)
    dlg._combo_fusion_mode.setCurrentIndex(dlg._combo_fusion_mode.findData("ir"))
    yield dlg
    dlg.close()
    manager.stop_all()


def chessboard_image():
    image = np.full((192, 256), 255, dtype=np.uint8)
    for row in range(5):
        for col in range(7):
            if (row + col) % 2 == 0:
                image[40 + row * 20:60 + row * 20, 40 + col * 20:60 + col * 20] = 0
    return image


def test_ir_board_dimensions_and_native_resolution_detection():
    board = PatternBoardConfig.ir_chessboard()
    detector = PlanarPatternDetector(board)
    result = detector.detect(chessboard_image())
    assert result.valid
    assert result.num_points == 24
    assert result.image_size == (256, 192)
    np.testing.assert_allclose(result.object_points[-1], [.4, .24, 0], atol=1e-7)
    assert PatternBoardConfig.from_dict(board.to_dict()) == board


def test_packaged_board_template():
    path = Path(__file__).resolve().parents[1] / "assets/boards/ir_chessboard_7x5_80mm.json"
    assert load_pattern_board_config(path) == PatternBoardConfig.ir_chessboard()


def test_explicit_board_is_authoritative_over_legacy_pattern_field():
    board = PatternBoardConfig(cols=8, rows=5, square_size=.025,
                               pattern_type=PatternType.CIRCLES_GRID, asymmetric=True)
    cfg = AuxiliaryCameraConfig("aux", "", board_config=board)
    assert cfg.pattern_board_config() == board
    assert cfg.to_dict()["board_pattern"] == "circles_grid"


def test_ustreamer_opencv_yaml_import_ignores_camera_matrices(tmp_path):
    path = tmp_path / "ir_intrinsics.yaml"
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("board_cols", 6)
    fs.write("board_rows", 4)
    fs.write("square_mm", 80.)
    fs.write("K", np.eye(3))
    fs.write("D", np.zeros((1, 5)))
    fs.release()
    assert load_pattern_board_config(path) == PatternBoardConfig.ir_chessboard()


@pytest.mark.parametrize("data", [
    {"cols": 6, "rows": 4},
    {"cols": 6.5, "rows": 4, "square_size": .08},
    {"cols": 6, "rows": 2, "square_size": .08},
    {"cols": True, "rows": 4, "square_size": .08},
    {"cols": 6, "rows": 4, "square_size": 0},
    {"cols": 6, "rows": 4, "square_size": float("nan")},
    {"cols": 6, "rows": 4, "square_size": .08, "square_mm": 40},
    {"cols": 6, "board_cols": 7, "rows": 4, "square_mm": 80},
    {"cols": 6, "rows": 4, "square_mm": 80, "asymmetric": True},
])
def test_invalid_or_ambiguous_board_rejected(data):
    with pytest.raises(ValueError):
        PatternBoardConfig.from_dict(data)


def test_json_import_and_reopening_dialog_preserve_independent_settings(dialog, tmp_path, monkeypatch):
    rgb_before = dialog.board.to_dict()
    custom = PatternBoardConfig(cols=8, rows=5, square_size=.025,
                                pattern_type=PatternType.CIRCLES_GRID, asymmetric=True)
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(custom.to_dict()), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: (str(path), ""))
    dialog._on_import_aux_board()
    assert dialog._aux_board_values() == custom
    assert dialog.board.to_dict() == rgb_before
    assert dialog._spin_aux_spacing.value() == 25
    dialog._on_accept()
    cfg = dialog._stream_manager.auxiliary_camera
    assert cfg.pattern_board_config() == custom
    assert cfg.url == ""
    assert dialog._stream_manager.camera_ids == []
    assert dialog._stream_manager.configured_camera_urls() == {}
    dialog._stream_manager.start_all()
    assert dialog._stream_manager.camera_ids == []
    reopened = ConfigDialog(dialog._stream_manager, dialog.board)
    try:
        assert reopened._aux_board_values() == custom
    finally:
        reopened.close()


def test_bad_import_does_not_overwrite_current_values(dialog, tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text('{"board_cols":6,"board_rows":4}', encoding="utf-8")
    errors = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: (str(path), ""))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: errors.append(a))
    dialog._on_import_aux_board()
    assert errors
    assert dialog._aux_board_values() == PatternBoardConfig.ir_chessboard()


def test_import_does_not_silently_clamp_spacing(dialog):
    before = dialog._aux_board_values()
    for size in (1e-8, .0800004):
        with pytest.raises(ValueError):
            dialog._set_aux_board_values(PatternBoardConfig(cols=8, rows=5, square_size=size))
        assert dialog._aux_board_values() == before


def test_invalid_manual_values_do_not_replace_configuration(dialog, monkeypatch):
    dialog._spin_aux_rows.setValue(2)
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: errors.append(a))
    dialog._on_accept()
    assert errors
    assert dialog._stream_manager.auxiliary_camera is None


def test_panel_never_derives_auxiliary_geometry_from_rgb(app):
    manager = StreamManager()
    manager.set_auxiliary_camera(AuxiliaryCameraConfig("aux", ""))
    rgb_board = CharucoBoard(CharucoBoardConfig(cols=15, rows=10, square_length=.04))
    panel = CalibrationPanel(manager, rgb_board)
    try:
        assert panel._aux_board_config() == PatternBoardConfig.ir_chessboard()
        assert "24" in panel._lbl_aux_board.text()
        panel.update_board(CharucoBoard(CharucoBoardConfig(
            cols=12, rows=8, square_length=.03, marker_length=.02,
        )))
        assert panel._aux_board_config() == PatternBoardConfig.ir_chessboard()
        assert manager.auxiliary_camera.pattern_board_config().square_size == .08
    finally:
        panel.close()


def test_changed_board_cannot_reuse_pending_import_or_existing_detections(app):
    manager = StreamManager()
    old_board = PatternBoardConfig.ir_chessboard()
    manager.set_auxiliary_camera(AuxiliaryCameraConfig("aux", "", board_config=old_board))
    manager.add_virtual_stereo_pair(StereoPairConfig(
        "rgb", CameraConfig("left", "", stream_type="rgb"), CameraConfig("right", ""),
    ))
    panel = CalibrationPanel(manager, CharucoBoard(CharucoBoardConfig()))
    try:
        existing = [("existing", None, None)] * 3
        panel._aux_pairs = existing.copy()
        panel._aux_observation_board = old_board
        panel._aux_import_worker = SimpleNamespace(board_config=old_board)
        manager.set_auxiliary_camera(AuxiliaryCameraConfig(
            "aux", "", board_config=PatternBoardConfig(cols=6, rows=4, square_size=.04),
        ))
        panel._on_import_aux_images_ready([])
        assert panel._aux_pairs == existing
        assert "已忽略旧检测结果" in panel._log_text.toPlainText()
        panel._intrinsics["left"] = CameraIntrinsics(np.eye(3), np.zeros((1, 5)), (640, 480))
        panel._on_calibrate_aux()
        assert panel._aux_worker is None
        assert "参数与已检测图像不一致" in panel._log_text.toPlainText()
        panel._pending_aux_context = ("aux", "ir", old_board, "left")
        panel._on_aux_calibration_finished(None, None)
        assert panel._rig is None
        assert "未应用旧结果" in panel._log_text.toPlainText()
    finally:
        panel.close()


def test_session_metadata_import_and_old_aux_result_compatibility(tmp_path):
    cfg = AuxiliaryCameraConfig("ir", "")
    board = PatternBoardConfig.ir_chessboard()
    session = CalibrationSession.create(tmp_path, CharucoBoard(CharucoBoardConfig()), [], cfg)
    session.record_multimodal_result("ir", "ir", "chessboard", 12, .2, board.to_dict())
    loaded = CalibrationSession.load(session.session_dir)
    assert loaded.metadata["auxiliary_camera"]["board_config"] == board.to_dict()
    assert loaded.metadata["multimodal_result"]["board_config"] == board.to_dict()
    assert load_pattern_board_config(session.session_dir / session.METADATA_FILE) == board
    intr = CameraIntrinsics(np.eye(3), np.zeros((1, 5)), (256, 192))
    old = AuxCameraCalibration("ir", "ir", intr, np.eye(4)).to_dict()
    assert "board_config" not in old
    assert AuxCameraCalibration.from_dict(old).board_config is None


def test_result_json_board_import_and_ambiguous_rig_rejected(tmp_path):
    path = tmp_path / "rig.json"
    data = {"aux_cameras": {"ir": {"board_config": PatternBoardConfig.ir_chessboard().to_dict()}}}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_pattern_board_config(path) == PatternBoardConfig.ir_chessboard()
    data["aux_cameras"]["another"] = data["aux_cameras"]["ir"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple"):
        load_pattern_board_config(path)


def test_offline_import_detects_same_independent_board_on_both_images(app, tmp_path):
    image = chessboard_image()
    assert cv2.imwrite(str(tmp_path / "rgb_left_0001.png"), image)
    assert cv2.imwrite(str(tmp_path / "aux_0001.png"), image)
    worker = AuxImageImportWorker(str(tmp_path), PatternBoardConfig.ir_chessboard())
    imported, failed = [], []
    worker.finished_ok.connect(imported.append)
    worker.failed.connect(failed.append)
    worker.run()
    assert not failed
    rgb_det, aux_det = imported[0][0][3:]
    assert rgb_det.num_points == aux_det.num_points == 24
    np.testing.assert_allclose(rgb_det.object_points[-1], [.4, .24, 0], atol=1e-7)


def test_aux_worker_recovers_intrinsics_and_metric_transform_with_80mm_board(app, tmp_path):
    board = PatternBoardConfig.ir_chessboard()
    obj = PlanarPatternDetector._make_object_points(board)
    rgb_k = np.array([[800., 0, 640], [0, 820., 360], [0, 0, 1.]])
    aux_k = np.array([[210., 0, 128], [0, 205., 96], [0, 0, 1.]])
    t_rgb_aux = np.array([[.035], [-.01], [.02]])
    detections = []
    for i in range(16):
        rv = np.array([.2 * np.sin(i), .3 * np.cos(i), .02 * i])
        tv = np.array([[-.2 + .01 * i], [-.1 + .02 * np.sin(i)], [1.2 + .02 * i]])
        pair = []
        for k, t, size in ((rgb_k, tv, (1280, 720)), (aux_k, tv - t_rgb_aux, (256, 192))):
            corners = cv2.projectPoints(obj, rv, t, k, np.zeros(5))[0].reshape(-1, 2)
            pair.append(PlanarDetectionResult(True, obj.copy(), corners.astype(np.float32),
                                              size, PatternType.CHESSBOARD))
        detections.append((str(i), *pair))
    rgb_intr = CameraIntrinsics(rgb_k, np.zeros((1, 5)), (1280, 720))
    worker = AuxCalibrationWorker("ir", board, rgb_intr.model, detections, rgb_intr, 10)
    results, failed = [], []
    worker.finished_ok.connect(lambda intr, extr: results.append((intr, extr)))
    worker.failed.connect(failed.append)
    worker.run()
    assert not failed
    intr, extr = results[0]
    np.testing.assert_allclose(intr.camera_matrix, aux_k, atol=.01, rtol=0)
    np.testing.assert_allclose(extr.T_rgb_left_aux[:3, 3], t_rgb_aux.flatten(), atol=.0001)
    aux = AuxCameraCalibration("ir", "ir", intr, extr.T_rgb_left_aux,
                               board_type="chessboard", board_config=board.to_dict())
    rig = MultiCameraRig(aux_cameras={"ir": aux})
    restored = MultiCameraRig.from_dict(rig.to_dict())
    assert restored.aux_cameras["ir"].board_config == board.to_dict()
    path = export_rig_opencv_yaml(rig, tmp_path / "rig.yml")
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        assert fs.getNode("aux_ir_board_cols").real() == 6
        assert fs.getNode("aux_ir_board_rows").real() == 4
        assert fs.getNode("aux_ir_board_square_size_m").real() == pytest.approx(.08)
    finally:
        fs.release()
