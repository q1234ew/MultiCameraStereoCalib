"""Truthful stereo guide telemetry and reversible workbench placement."""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest

from multicamera.board.detector import DetectionResult
from multicamera.calibration.models import CameraIntrinsics
from multicamera.streaming.stream_manager import FrameEntry
from multicamera.ui import main_window
from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.pair_gallery import PreviewPair
from test_multimodal_preview import LEFT, RIGHT, configure


@pytest.fixture
def stereo_bench(app, monkeypatch, tmp_path):
    configure_application_style(app)
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    window = main_window.MainWindow()
    window._stereo_capture.guide_mode.setCurrentIndex(1)
    configure(window)
    window._calib_panel.show_workflow('capture', False)
    window.resize(1440, 1024)
    window.show()
    app.processEvents()
    yield window
    window._calib_panel._auto_timer.stop()
    window._stream_manager.stop_all()
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def sample(window):
    board = window._board
    K = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
    dist = np.zeros((1, 5))
    tvec = np.array([-.3, -.2, 1.])
    points, _ = cv2.projectPoints(board.object_points, np.zeros(3), tvec, K, dist)
    detection = DetectionResult(
        points.astype(np.float32), np.arange(board.num_corners, dtype=np.int32)[:, None],
        [], None, (1280, 720),
    )
    frame = np.full((720, 1280, 3), 220, np.uint8)
    image = board.generate_image(pixel_size=40, margin=0)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    frame[180:540, 370:910] = cv2.resize(image, (540, 360))
    return detection, frame, CameraIntrinsics(K, dist, (1280, 720))


def publish(window, detection, frame):
    guide = window._stereo_capture
    for cid in (LEFT, RIGHT):
        window._stream_manager._latest_frames[cid] = FrameEntry(frame, time.time())
        guide.observe(cid, detection, frame)


def publish_stable(window, detection, frame):
    """Three distinct preview detections spanning a real stability interval."""
    clock = [time.monotonic()]
    window._stereo_capture._clock = lambda: clock[0]
    window._stereo_capture.samples.clear()
    for _ in range(3):
        clock[0] += .35
        publish(window, detection, frame)


def test_layout_switch_restores_footer_and_capture_controls(stereo_bench, app, tmp_path):
    win = stereo_bench
    panel, guide = win._calib_panel, win._stereo_capture
    assert guide.isVisible() and not win._inspector.isVisible()
    assert panel._action_footer.parentWidget() is win._capture_actions
    assert guide.details.widget() is panel._sections[0]
    assert panel._btn_capture.isVisible()
    assert not guide.details.isVisible()
    guide.coverage_button.click()
    assert guide.details.isVisible()
    panel._combo_pair.setCurrentIndex(0)
    for _ in range(3):
        panel.show_workflow('calibration', False)
        assert not guide.isVisible() and win._inspector.isVisible()
        assert panel._action_footer.parentWidget() is panel
        assert guide.details.widget() is None
        panel.show_workflow('capture', True)
        assert panel._btn_capture_aux.isVisible()
        panel.show_workflow('capture', False)
        assert panel._btn_capture.isVisible()
    guide.coverage_button.setChecked(False)
    det, frame, intr = sample(win)
    panel._intrinsics[LEFT] = intr
    publish(win, det, frame)
    directory = Path(os.environ.get('MULTICAMERA_UI_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    for width, height in ((1440, 1024), (1280, 800), (1000, 680)):
        win.resize(width, height)
        for cid in (LEFT, RIGHT):
            win._stream_view._on_frame(cid, frame, time.time())
        app.processEvents()
        app.processEvents()
        button = panel._btn_capture
        assert button.isVisible()
        for control in (button, panel._btn_import, panel._btn_to_calibration, panel._btn_undo):
            assert win.rect().contains(control.mapTo(win, control.rect().topLeft()))
            assert win.rect().contains(control.mapTo(win, control.rect().bottomRight()))
            assert control.height() >= control.minimumSizeHint().height()
        assert guide.instruction.height() >= guide.instruction.fontMetrics().height()
        assert win._preview_scroll.horizontalScrollBar().maximum() == 0
        assert win._preview_scroll.verticalScrollBar().maximum() == 0
        assert win._guide_scroll.horizontalScrollBar().maximum() == 0
        viewport = win._preview_scroll.viewport()
        for tile in win._stream_view._views.values():
            if tile.isVisible():
                assert viewport.rect().contains(tile.mapTo(viewport, tile.rect().topLeft()))
                assert viewport.rect().contains(tile.mapTo(viewport, tile.rect().bottomRight()))
                assert tile._image_label.geometry().bottom() < tile._info.geometry().top()
                pixmap = tile._image_label.pixmap()
                if pixmap is not None and not pixmap.isNull():
                    assert pixmap.height() <= tile._image_label.height()
                    assert pixmap.width() <= tile._image_label.width()
        assert win._capture_bottom.height() <= (
            win._gallery.height() + panel._action_footer.height() + 12)
        if width == 1280:
            viewport = win._preview_scroll.viewport()
            canvas = guide.schematic.canvas
            assert viewport.rect().contains(canvas.mapTo(viewport, canvas.rect().topLeft()))
            assert viewport.rect().contains(canvas.mapTo(viewport, canvas.rect().bottomRight()))
            assert guide.compact_summary.isVisible() and not guide.pose.isVisible()
        win.grab().save(str(directory / f'stereo-capture-{width}.png'))
    panel._capture_sequence.jump_to(10)
    guide.refresh()
    guide.schematic.pause.setChecked(True)
    guide.schematic.progress = .65
    win._guide_scroll.ensureWidgetVisible(guide.schematic.canvas)
    app.processEvents()
    assert '右边缘靠近镜头' in guide.instruction.text()
    win.grab().save(str(directory / 'stereo-capture-motion-scrolled.png'))


def test_gallery_and_logs_keep_preview_visible_and_restore_inline(stereo_bench, app, tmp_path):
    win = stereo_bench
    _, frame, _ = sample(win)
    pairs = [PreviewPair(f'rgb {i:04d}', LEFT, RIGHT, frame, frame) for i in range(4)]
    win._on_preview_pairs_changed(pairs)
    for cid in (LEFT, RIGHT):
        win._stream_view._on_frame(cid, frame, time.time())
    directory = Path(os.environ.get('MULTICAMERA_UI_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    win._gallery._toggle.setChecked(True)
    win._toggle_logs.setChecked(True)
    for size in ((1000, 680), (1280, 800), (1440, 1024), (1000, 680)):
        win.resize(*size)
        app.processEvents()
        app.processEvents()
        assert win._preview_scroll.verticalScrollBar().maximum() == 0
        viewport = win._preview_scroll.viewport()
        for cid in (LEFT, RIGHT):
            tile = win._stream_view._views[cid]
            assert viewport.rect().contains(tile.mapTo(viewport, tile.rect().bottomRight()))
        assert win._calib_panel._footer_info.parentWidget().rect().contains(
            win._calib_panel._footer_info.geometry())
        for key, widget in (('gallery', win._gallery._body), ('logs', win._logs_widget)):
            dialog = win._aux_panel_dialogs[key]
            assert dialog.isVisible()
            assert widget.isVisible()
            assert widget.parentWidget() is dialog
            dialog.grab().save(str(directory / f'{key}-tool-{size[0]}.png'))
        win.grab().save(str(directory / f'capture-panels-{size[0]}.png'))
    QTest.keyClick(win._aux_panel_dialogs['gallery'], Qt.Key_Escape)
    assert not win._gallery._toggle.isChecked()
    assert win._toggle_logs.isChecked()
    win._gallery._toggle.setChecked(True)
    assert win._aux_panel_dialogs['gallery'].isVisible()
    assert len(win._gallery._pairs) == 4
    win._aux_panel_dialogs['logs'].close()
    assert not win._toggle_logs.isChecked()
    win.resize(1440, 1024)
    win._calib_panel.show_workflow('calibration', False)
    app.processEvents()
    assert win._gallery._body.parentWidget() is win._gallery
    assert win._logs_widget.parentWidget() is win._workspace


def test_expanded_schematic_fits_sidebar_without_horizontal_scroll(stereo_bench, app, tmp_path):
    win = stereo_bench
    guide = win._stereo_capture
    directory = Path(os.environ.get('MULTICAMERA_UI_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    for size in ((1000, 680), (1280, 800), (1440, 1024)):
        win.resize(*size)
        guide.schematic.expand.setChecked(True)
        app.processEvents()
        app.processEvents()
        assert win._guide_scroll.horizontalScrollBar().maximum() == 0
        win._guide_scroll.ensureWidgetVisible(guide.schematic.expand)
        app.processEvents()
        viewport = win._guide_scroll.viewport()
        button = guide.schematic.expand
        assert viewport.rect().contains(button.mapTo(viewport, button.rect().bottomRight()))
        assert '收起' in button.toolTip()
        assert not guide.schematic.view_options.isHidden()
        win.grab().save(str(directory / f'expanded-schematic-{size[0]}.png'))
        guide.schematic.expand.setChecked(False)
        assert '展开' in button.toolTip()


def test_pose_uses_calibrated_pnp_not_example_values(stereo_bench):
    win = stereo_bench
    guide = win._stereo_capture
    det, frame, intr = sample(win)
    publish(win, det, frame)
    assert '待内参' in guide.pose.text()
    win._calib_panel._intrinsics[LEFT] = intr
    guide.refresh()
    assert '1.00 m' in guide.pose.text()
    intr.image_size = (640, 480)
    guide.refresh()
    assert '不匹配' in guide.pose.text()
    win._calib_panel._intrinsics.clear()
    guide.refresh()
    assert '待内参' in guide.pose.text()


def test_auto_waits_for_stability_and_rejects_stale_or_review(stereo_bench, monkeypatch):
    win = stereo_bench
    guide = win._stereo_capture
    det, frame, _ = sample(win)
    publish(win, det, frame)
    assert not guide.auto_ready()
    publish_stable(win, det, frame)
    assert guide.auto_ready()
    win._calib_panel._capture_sequence.jump_to(9)
    assert not guide.auto_ready()
    win._calib_panel._capture_sequence.jump_to(0)
    win._stream_manager._latest_frames[RIGHT] = FrameEntry(frame, time.time() - .5)
    assert not guide.auto_ready()
    win._stream_manager._latest_frames[RIGHT] = FrameEntry(frame, time.time())
    events = []
    monkeypatch.setattr(win._calib_panel, '_on_capture', lambda: events.append('capture'))
    win._calib_panel._on_auto_capture()
    assert events == ['capture']
    guide.samples[LEFT]['time'] -= 2
    assert not guide.auto_ready()
    win._calib_panel._on_auto_capture()
    assert events == ['capture']
    publish_stable(win, det, frame)
    win._stream_view.show_review_images('test', {LEFT: frame, RIGHT: frame})
    assert not guide.auto_ready()
    guide.refresh()
    assert '回看' in guide.instruction.text()
    assert not guide.schematic.timer.isActive()


def test_auto_toggle_and_navigation_stop_timer(stereo_bench):
    panel = stereo_bench._calib_panel
    assert not panel._btn_auto.isChecked()
    panel._btn_auto.setChecked(True)
    assert panel._auto_timer.isActive()
    panel._btn_auto.setChecked(False)
    assert not panel._auto_timer.isActive()
    panel._btn_auto.setChecked(True)
    panel.show_workflow('calibration', False)
    assert not panel._auto_timer.isActive() and not panel._btn_auto.isChecked()
    panel.show_workflow('capture', False)
    count = panel._capture_sequence.done_count
    panel._btn_skip_stereo.click()
    assert panel._capture_sequence.current_index == 1
    assert panel._capture_sequence.done_count == count


def test_region_overlay_never_modifies_source_or_implies_depth(stereo_bench):
    win = stereo_bench
    det, frame, _ = sample(win)
    before = frame.copy()
    rendered = win._stream_view._draw_detection(frame, det)
    assert np.array_equal(frame, before)
    assert not np.array_equal(rendered, before)
    assert all('距' not in target.label for target in win._calib_panel._capture_sequence.targets)


def test_basic_checks_do_not_claim_pose_acceptance(stereo_bench):
    win = stereo_bench
    det, frame, _ = sample(win)
    win._calib_panel._capture_sequence.jump_to(10)
    publish(win, det, frame)
    publish(win, det, frame)
    guide = win._stereo_capture
    assert '不含姿态' in guide.status.text()
    assert '待人工确认' in guide.pose_status.text()
    assert not guide.auto_ready()


def test_duplicate_capture_is_advisory_and_undoable(stereo_bench, monkeypatch):
    win = stereo_bench
    panel = win._calib_panel
    det, frame, _ = sample(win)
    monkeypatch.setattr(panel, '_ensure_capture_session', lambda: True)
    monkeypatch.setattr(panel, '_prepare_white_balance', lambda *args: True)
    publish(win, det, frame)
    panel._capture_impl()
    count = panel._stereo_cals['rgb'].num_frames
    publish(win, det, frame)
    panel._capture_impl()
    assert panel._stereo_cals['rgb'].num_frames == count + 1
    assert '重复构图提醒' in panel._log_text.toPlainText()
    panel._on_undo_capture()
    assert panel._stereo_cals['rgb'].num_frames == count
