"""RGB-left + centre IR preview routing and 24-point planar detection regressions."""

import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QMessageBox

from multicamera.calibration.planar import PatternBoardConfig, PlanarPatternDetector
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import (
    AuxiliaryCameraConfig, CameraConfig, FrameEntry, StereoPairConfig,
)
from multicamera.ui import main_window
from multicamera.ui.widgets.config_dialog import ConfigDialog

LEFT, RIGHT, IR = "z_rgb_left", "rgb_right", "a_ir_middle"


@pytest.fixture
def window(app, tmp_path, monkeypatch, finish_aux_tasks):
    monkeypatch.setattr(main_window, "sessions_dir", lambda: tmp_path / "sessions")
    win = main_window.MainWindow()
    yield win
    finish_aux_tasks(win._calib_panel)
    win._calib_panel._auto_timer.stop()
    win._stream_manager.stop_all()
    win.close()
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def configure(window, *, auxiliary=True, offline_aux=False):
    manager = window._stream_manager
    pair = StereoPairConfig(
        "rgb", CameraConfig(LEFT, "http://device.test:8080/left", "left", "rgb", "rgb"),
        CameraConfig(RIGHT, "http://device.test:8080/right", "right", "rgb", "rgb"),
    )
    manager.add_stereo_pair(pair)
    if auxiliary:
        manager.set_auxiliary_camera(AuxiliaryCameraConfig(
            IR, "" if offline_aux else "http://device.test:8081/stream",
            board_config=PatternBoardConfig.ir_chessboard(),
        ))
    window._calib_panel.refresh_pairs()
    return manager


def board_image():
    image = np.full((192, 256, 3), 255, dtype=np.uint8)
    for row in range(5):
        for col in range(7):
            if (row + col) % 2 == 0:
                image[40 + row * 20:60 + row * 20, 40 + col * 20:60 + col * 20] = 0
    return image


def test_entering_multimodal_selects_left_ir_in_role_order_and_ignores_right(window):
    manager = configure(window)
    view, panel = window._stream_view, window._calib_panel
    view.show_connecting(manager.configured_camera_urls())
    assert set(view._views) == {LEFT, RIGHT}
    panel._activate_step(2)
    assert view._multimodal_mode
    assert set(view._views) == {LEFT, IR}
    assert view._layout.itemAtPosition(0, 0).widget().camera_id == LEFT
    assert view._layout.itemAtPosition(0, 1).widget().camera_id == IR
    assert "RGB 左目" in view._views[LEFT]._header.toolTip()
    assert "红外（中间）" in view._views[IR]._header.toolTip()
    assert view._capture_sequence is panel._aux_capture_sequence
    manager.camera_connected.emit(RIGHT)
    manager.frame_received.emit(RIGHT, board_image(), time.time())
    assert set(view._views) == {LEFT, IR}
    assert RIGHT not in view._last_detection


def test_fusion_preview_uses_24_point_board_for_both_native_and_resized_frames(window, monkeypatch):
    configure(window)
    view = window._stream_view
    window._calib_panel._activate_step(2)

    def no_charuco(*args):
        pytest.fail("Fusion preview must not use the RGB ChArUco detector")

    monkeypatch.setattr(view._detector, "detect", no_charuco)
    ir = board_image()
    rgb = cv2.resize(ir, (1280, 960), interpolation=cv2.INTER_NEAREST)
    for cid, frame in ((LEFT, rgb), (IR, ir)):
        window._stream_manager.camera_connected.emit(cid)
        view._last_render_ts.pop(cid, None)
        window._stream_manager.frame_received.emit(cid, frame, time.time())
        result = view._last_detection[cid]
        assert result.num_corners == 24
        assert result.image_size == (frame.shape[1], frame.shape[0])
        assert "24/24" in view._views[cid]._info.text()
    assert view._preview_corner_count == 24
    assert view._board.num_corners == 126


def test_missing_ir_has_visible_placeholder_instead_of_rgb_right(window):
    manager = configure(window, auxiliary=False)
    window._calib_panel._activate_step(2)
    view = window._stream_view
    assert set(view._views) == {LEFT, "aux_cam"}
    assert "未配置" in view._views["aux_cam"]._info.text()
    manager.frame_received.emit(RIGHT, board_image(), time.time())
    assert RIGHT not in view._views


def test_ir_failure_clears_image_and_does_not_fall_back_or_resurrect_cache(window, monkeypatch):
    manager = configure(window)
    failures = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: failures.append(args))
    window._calib_panel._activate_step(2)
    manager.camera_connected.emit(IR)
    frame = board_image()
    manager._latest_frames[IR] = FrameEntry(frame, time.time())
    manager.frame_received.emit(IR, frame, time.time())
    assert window._stream_view._views[IR]._last_frame is not None
    manager.camera_disconnected.emit(IR, "HTTP 503")
    window._calib_panel._activate_step(2)
    tile = window._stream_view._views[IR]
    assert tile._last_frame is None
    assert "HTTP 503" in tile._info.text()
    assert RIGHT not in window._stream_view._views
    assert failures


def test_offline_aux_preview_uses_actual_left_id_and_restores_each_workflow(window):
    configure(window, offline_aux=True)
    panel, view = window._calib_panel, window._stream_view
    image = board_image()
    view.show_static_images({LEFT: image, RIGHT: image})
    panel._activate_step(2)
    detection = PlanarPatternDetector(PatternBoardConfig.ir_chessboard()).detect(image)
    panel._on_import_aux_images_ready([("0001", image, image, detection, detection)])
    assert set(view._views) == {LEFT, IR}
    assert "RGB_L" not in view._views
    assert "24/24" in view._views[IR]._info.text()
    panel._activate_step(0)
    assert not view._multimodal_mode
    assert set(view._views) == {LEFT, RIGHT}
    assert view._views[LEFT]._last_frame is not None
    panel._activate_step(2)
    assert set(view._views) == {LEFT, IR}
    assert "24/24" in view._views[IR]._info.text()


def test_live_capture_saves_raw_left_and_ir_not_right(window, tmp_path, monkeypatch, finish_aux_tasks):
    manager = configure(window)
    panel, view = window._calib_panel, window._stream_view
    left = cv2.resize(board_image(), (768, 576), interpolation=cv2.INTER_NEAREST)
    ir = board_image()
    right = np.zeros_like(left)
    for cid, frame in ((LEFT, left), (RIGHT, right), (IR, ir)):
        manager._latest_frames[cid] = FrameEntry(frame, time.time())
        manager.camera_connected.emit(cid)
    session = CalibrationSession.create(
        tmp_path, window._board, list(manager.stereo_pairs.values()), manager.auxiliary_camera,
    )
    panel.set_session(session)
    monkeypatch.setattr(panel._aux_guide, 'stability_reasons', lambda detections, snapshot=None: [])
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert len(panel._aux_pairs) == 1
    assert set(view._views) == {LEFT, IR}
    assert view._trackers[LEFT].total_frames == view._trackers[IR].total_frames == 1
    assert panel._stereo_cals.get("rgb") is None
    np.testing.assert_array_equal(cv2.imread(str(session.session_dir / "multimodal/rgb_left_0000.png")), left)
    np.testing.assert_array_equal(cv2.imread(str(session.session_dir / "multimodal/aux_0000.png")), ir)
    assert True not in view._static_images


def test_section_switch_stops_stereo_auto_capture_and_changes_detector(window):
    configure(window)
    panel = window._calib_panel
    panel._btn_auto.setChecked(True)
    assert panel._auto_timer.isActive()
    panel._sections[2]._header.click()
    assert not panel._auto_timer.isActive()
    assert not panel._btn_auto.isChecked()
    assert window._stream_view._preview_corner_count == 24
    panel._sections[0]._header.click()
    assert not window._stream_view._multimodal_mode
    assert window._stream_view._preview_corner_count == 126


def test_configure_aux_button_opens_aux_configuration(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "_on_config", lambda auxiliary=False: calls.append(auxiliary))
    window._calib_panel._btn_configure_aux.click()
    assert calls == [True]


def test_applying_aux_configuration_resumes_previous_connections(window, monkeypatch):
    manager = configure(window, auxiliary=False)
    window._act_stop_streams.setEnabled(True)
    window._act_start_streams.setEnabled(False)
    started = []
    monkeypatch.setattr(manager, "start_all", lambda: started.append(manager.configured_camera_urls()))

    def accept_aux(self):
        self._combo_fusion_mode.setCurrentIndex(self._combo_fusion_mode.findData("ir"))
        self._edit_aux_id.setText(IR)
        self._edit_aux_url.setText("http://device.test:8081/stream")
        self._on_accept()
        return self.result()

    monkeypatch.setattr(ConfigDialog, "exec", accept_aux)
    window._on_config(True)
    assert len(started) == 1
    assert set(started[0]) == {LEFT, RIGHT, IR}
    assert window._stream_view._multimodal_mode
    assert set(window._stream_view._views) == {LEFT, IR}
    assert window._act_stop_streams.isEnabled()


@pytest.mark.parametrize("collision", ["id", "url"])
def test_aux_cannot_reuse_rgb_id_or_stream(window, monkeypatch, collision):
    manager = configure(window, auxiliary=False)
    dlg = ConfigDialog(manager, window._board)
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args))
    try:
        dlg.focus_auxiliary()
        dlg._edit_aux_id.setText(RIGHT if collision == "id" else IR)
        dlg._edit_aux_url.setText(
            "http://device.test:8080/right" if collision == "url" else "http://device.test:8081/stream"
        )
        dlg._on_accept()
        assert errors
        assert manager.auxiliary_camera is None
    finally:
        dlg.close()


def test_non_rgb_pair_is_never_used_as_rgb_reference(window):
    manager = window._stream_manager
    manager.add_virtual_stereo_pair(StereoPairConfig(
        "ir_stereo", CameraConfig("ir_left", "", stream_type="ir"),
        CameraConfig("ir_right", "", stream_type="ir"),
    ))
    assert window._calib_panel._rgb_left_camera_id() is None


@pytest.mark.parametrize("multimodal", [False, True])
def test_close_preview_button_hides_tile_until_explicit_restore(window, multimodal):
    manager = configure(window)
    view, panel = window._stream_view, window._calib_panel
    view.show_connecting(manager.configured_camera_urls())
    panel._activate_step(2 if multimodal else 0)
    manager.camera_connected.emit(LEFT)
    frame = board_image()
    manager._latest_frames[LEFT] = FrameEntry(frame, time.time())
    manager.frame_received.emit(LEFT, frame, time.time())
    original_grabber = manager._grabbers[LEFT]
    view._views[LEFT]._close_button.click()
    assert LEFT not in view._views
    assert window._act_restore_previews.isEnabled()
    assert manager._grabbers[LEFT] is original_grabber
    assert manager.get_latest_frame(LEFT) is not None
    for _ in range(2):
        manager.camera_connected.emit(LEFT)
        manager.frame_received.emit(LEFT, frame, time.time())
    panel._activate_step(2 if multimodal else 0)
    assert LEFT not in view._views
    window._act_restore_previews.trigger()
    assert LEFT in view._views
    assert view._views[LEFT]._last_frame is not None
    assert not view.has_closed_previews
    assert not window._act_restore_previews.isEnabled()


def test_closing_left_in_fusion_places_ir_in_first_cell(window):
    configure(window)
    view = window._stream_view
    window._calib_panel._activate_step(2)
    view._expanded_id = LEFT
    view.close_preview(LEFT)
    assert view._expanded_id is None
    assert view._layout.itemAtPosition(0, 0).widget().camera_id == IR
    window._calib_panel._activate_step(0)
    window._calib_panel._activate_step(2)
    assert LEFT not in view._views
    assert view._layout.itemAtPosition(0, 0).widget().camera_id == IR


def test_closed_offline_tiles_do_not_reappear_and_all_can_be_restored(window):
    configure(window, offline_aux=True)
    panel, view = window._calib_panel, window._stream_view
    panel._activate_step(2)
    images = {LEFT: board_image(), IR: board_image()}
    view.show_multimodal_images(images)
    view.close_preview(LEFT)
    view.close_preview(IR)
    assert not view._views
    assert view._empty.previews_closed
    view.show_multimodal_images(images)
    assert not view._views
    view.restore_closed_previews()
    assert set(view._views) == {LEFT, IR}
    assert all(tile._last_frame is not None for tile in view._views.values())


def test_fusion_capture_works_with_closed_previews_and_preserves_coverage(window, tmp_path, monkeypatch, finish_aux_tasks):
    manager = configure(window)
    panel, view = window._calib_panel, window._stream_view
    frame = board_image()
    for cid in (LEFT, IR):
        manager._latest_frames[cid] = FrameEntry(frame, time.time())
        manager.camera_connected.emit(cid)
    panel.set_session(CalibrationSession.create(
        tmp_path, window._board, list(manager.stereo_pairs.values()), manager.auxiliary_camera,
    ))
    monkeypatch.setattr(panel._aux_guide, 'stability_reasons', lambda detections, snapshot=None: [])
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    tracker = view._trackers[IR]
    view.close_preview(IR)
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert IR not in view._views
    assert tracker.total_frames == 2
    assert len(panel._aux_pairs) == 2
    assert (panel._session.session_dir / "multimodal/aux_0001.png").exists()
    view.restore_closed_previews()
    assert view._trackers[IR] is tracker
    assert IR in view._views


def test_reconnect_respects_hidden_previews_but_clear_resets_visibility(window):
    manager = configure(window)
    view = window._stream_view
    view.show_connecting(manager.configured_camera_urls())
    view.close_preview(RIGHT)
    view.show_connecting(manager.configured_camera_urls())
    assert RIGHT not in view._views
    view.clear_all()
    assert not view.has_closed_previews
    assert not window._act_restore_previews.isEnabled()
    view.show_connecting(manager.configured_camera_urls())
    assert RIGHT in view._views


def test_layout_and_deferred_removal_keep_qt_parent_ownership(window):
    manager = configure(window)
    view = window._stream_view
    view.show_connecting(manager.configured_camera_urls())
    view._on_camera_tile_clicked(LEFT)
    assert all(tile.parent() is view for tile in view._views.values())
    closed = view._views[LEFT]
    view.close_preview(LEFT)
    assert closed.parent() is view
    assert closed.isHidden()
    view.restore_closed_previews()
    assert all(tile.parent() is view for tile in view._views.values())
    assert view._empty.parent() is view
