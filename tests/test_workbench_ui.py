"""Native Qt workbench layout, navigation, review and data-path regressions."""

import os
from pathlib import Path
import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer
from PySide6.QtGui import QDesktopServices

from multicamera.calibration.models import (
    AuxCameraCalibration, CameraIntrinsics, MultiCameraRig, StereoCalibResult, StereoPairCalibration,
)
from multicamera.calibration.planar import PlanarPatternDetector
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import FrameEntry
from multicamera.ui import main_window
from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.pair_gallery import PreviewPair

from test_multimodal_preview import LEFT, RIGHT, IR, board_image, configure


@pytest.fixture
def bench(app, monkeypatch, tmp_path, finish_aux_tasks):
    configure_application_style(app)
    monkeypatch.setattr(main_window, "sessions_dir", lambda: tmp_path / "sessions")
    window = main_window.MainWindow()
    manager = configure(window)
    panel = window._calib_panel
    monkeypatch.setattr(panel._aux_guide, 'stability_reasons', lambda detections, snapshot=None: [])
    rgb = np.full((720, 1280, 3), 245, np.uint8)
    rgb[:, 160:1120] = cv2.resize(board_image(), (960, 720), interpolation=cv2.INTER_NEAREST)
    ir = cv2.applyColorMap(cv2.cvtColor(board_image(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
    panel._intrinsics[LEFT] = CameraIntrinsics(
        np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]]), np.zeros((1, 5)), (1280, 720),
    )
    window._current_session = CalibrationSession.create(
        tmp_path, window._board, list(manager.stereo_pairs.values()), manager.auxiliary_camera, "ui_demo",
    )
    panel.set_session(window._current_session)
    window.resize(1440, 1024)
    window.show()
    app.processEvents()
    panel.show_workflow("capture", True)
    detector = PlanarPatternDetector(panel._aux_board_config())
    panel._on_import_aux_images_ready([
        (str(i), rgb, ir, detector.detect(rgb), detector.detect(ir)) for i in range(8)
    ])
    publish(window, rgb, ir)
    app.processEvents()
    finish_aux_tasks(panel)
    yield window, rgb, ir
    finish_aux_tasks(panel)
    manager.stop_all()
    panel._auto_timer.stop()
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def publish(window, rgb, ir):
    manager = window._stream_manager
    now = time.time()
    for cid, frame in ((LEFT, rgb), (IR, ir)):
        manager._latest_frames[cid] = FrameEntry(frame, now)
        manager.camera_connected.emit(cid)
        window._stream_view._last_render_ts.pop(cid, None)
        manager.frame_received.emit(cid, frame, time.time())


def test_repeated_window_toolbar_construction(app, tmp_path, monkeypatch):
    monkeypatch.setattr(main_window, "sessions_dir", lambda: tmp_path / "sessions")
    for _ in range(40):
        window = main_window.MainWindow()
        for action in (window._act_start_streams, window._act_stop_streams, window._act_config_tb):
            assert action.parent() is window
            assert not action.icon().isNull()
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()


def test_default_is_split_with_fixed_primary_action(bench):
    win, _, _ = bench
    panel, view = win._calib_panel, win._stream_view
    assert view.presentation_mode == "split"
    assert win._view_split.isChecked()
    assert not win._comparison_host.isVisible()
    assert win._workflow_tabs.currentIndex() == 0
    assert panel._footer_pages.currentIndex() == 2
    assert panel._btn_capture_aux.isVisible()
    assert panel._btn_capture_aux.isEnabled()
    assert not panel._btn_calib_aux.isVisible()
    assert not panel._log_text.isVisible()
    assert set(view._views) == {LEFT, IR}
    assert RIGHT not in view._visible_camera_ids()


def test_stereo_capture_empty_state_has_explicit_progress(bench, app, tmp_path):
    win, _, _ = bench
    panel = win._calib_panel
    panel.show_workflow('capture', False)
    panel._update_capture_stats()
    assert panel._lbl_capture_count.text() == '有效配对 0 / 20'
    assert panel._cap_progress.value() == 0
    assert panel._cap_progress.maximum() == 20
    assert panel._cap_progress.height() == 8
    assert not panel._cap_progress.isTextVisible()
    assert panel._cap_progress.isHidden()
    assert '--' not in panel._lbl_assess.text()
    assert panel._data_score.isHidden()
    assert panel._lbl_tips.isHidden()
    panel._spin_min.setValue(25)
    assert panel._lbl_capture_count.text() == '有效配对 0 / 25'
    directory = Path(os.environ.get('MULTICAMERA_UI_QA_DIR', tmp_path / 'ui'))
    directory.mkdir(parents=True, exist_ok=True)
    win.resize(1000, 680)
    app.processEvents()
    win.grab().save(str(directory / 'stereo-empty-1000x680.png'))


def test_stereo_progress_uses_valid_pairs_and_reset_clears_stale_quality(bench):
    from types import SimpleNamespace
    from multicamera.calibration.assessment import Grade

    win, _, _ = bench
    panel = win._calib_panel
    name = next(iter(panel._selected_pairs()))
    original = panel._stereo_cals
    try:
        panel._stereo_cals = {name: SimpleNamespace(num_frames=7, ready=False)}
        panel._pair_capture_press_count[name] = 10
        panel._data_score.set_grade(Grade.GOOD, 80)
        panel._lbl_tips.setText('old warning')
        panel._update_capture_stats()
        assert panel._lbl_capture_count.text() == '有效配对 7 / 20'
        assert panel._cap_progress.value() == 7
        assert not panel._cap_progress.isHidden()
        assert panel._data_score.isHidden()
        assert not panel._lbl_tips.text()
    finally:
        panel._stereo_cals = original
    panel._reset_all()
    assert panel._lbl_capture_count.text() == '有效配对 0 / 20'
    assert panel._lbl_assess.text() == '尚无质量评估'


def test_workflow_navigation_preserves_calibration_and_camera_roles(bench):
    win, _, _ = bench
    panel = win._calib_panel
    pairs = panel._aux_pairs
    intr = panel._intrinsics[LEFT]
    win._workflow_tabs.setCurrentIndex(1)
    assert panel.workflow_phase == "calibration"
    assert panel._footer_pages.currentIndex() == 3
    assert panel._btn_calib_aux.isVisible()
    assert not panel._btn_capture_aux.isVisible()
    win._workflow_tabs.setCurrentIndex(2)
    assert panel._result_body.isVisible()
    assert not panel._btn_export_results.isEnabled()
    win._workflow_tabs.setCurrentIndex(0)
    assert panel._aux_pairs is pairs
    assert panel._intrinsics[LEFT] is intr
    assert set(win._stream_view._visible_camera_ids()) == {LEFT, IR}
    win._combo_task.setCurrentIndex(0)
    assert not win._stream_view._multimodal_mode
    assert IR not in win._stream_view._visible_camera_ids()
    assert not win._view_aux.isEnabled()
    win._combo_task.setCurrentIndex(1)
    assert win._stream_view._multimodal_mode


def test_focus_mode_keeps_rgb_comparison_and_original_sources(bench, app):
    win, rgb, ir = bench
    manager, view = win._stream_manager, win._stream_view
    original = {cid: manager._grabbers[cid] for cid in (LEFT, IR)}
    win._view_aux.click()
    app.processEvents()
    assert view.presentation_mode == "aux_focus"
    assert win._comparison_host.isVisible()
    assert view._views[LEFT].parent() is win._comparison_host
    assert view._views[IR].parent() is view
    assert all(manager._grabbers[cid] is grabber for cid, grabber in original.items())
    np.testing.assert_array_equal(manager.get_latest_frame(LEFT)[0], rgb)
    np.testing.assert_array_equal(manager.get_latest_frame(IR)[0], ir)
    win._view_split.click()
    app.processEvents()
    assert not win._comparison_host.isVisible()
    assert all(view._views[cid].parent() is view for cid in (LEFT, IR))


def test_focus_close_restore_and_escape_are_coherent(bench, app):
    win, _, _ = bench
    view = win._stream_view
    win._view_aux.click()
    view.close_preview(LEFT)
    app.processEvents()
    assert view.presentation_mode == "split"
    assert not win._comparison_host.isVisible()
    assert not win._view_aux.isEnabled()
    view.restore_closed_previews()
    app.processEvents()
    assert view.presentation_mode == "aux_focus"
    assert set(view._views) == {LEFT, IR}
    view._on_camera_tile_clicked(IR)
    assert view._expanded_id == IR
    assert not win._comparison_host.isVisible()
    view._on_camera_tile_clicked(IR)
    assert view.presentation_mode == "aux_focus"


def test_gallery_review_is_explicit_and_does_not_replace_capture_source(
        bench, app, monkeypatch, finish_aux_tasks):
    win, rgb, ir = bench
    gallery, view, panel = win._gallery, win._stream_view, win._calib_panel
    guide = panel._aux_guide
    monkeypatch.setattr(guide, '_request_preview', lambda: None)
    pair = gallery._pairs[-1]
    index = len(gallery._pairs) - 1
    item = next(gallery._list.item(i) for i in range(gallery._list.count())
                if gallery._list.item(i).data(Qt.UserRole) == index)
    gallery._select_pair(item)
    assert view.review_key == pair.key
    assert win._review_bar.isVisible()
    assert '正在回看原图' in guide.phase.text()
    assert pair.key in guide.phase.text()
    assert '实时指引已暂停' in guide.action.text()
    assert not panel._btn_capture_aux.isEnabled()
    assert '先结束回看' in panel._btn_capture_aux.toolTip()
    assert '验收通过' in view._views[LEFT]._info.toolTip()
    assert '可采纳' not in view._views[LEFT]._info.toolTip()
    assert pair.quality_state == '可采纳'  # Display names do not migrate stored states.
    before = view._views[IR]._last_frame.copy()
    win._stream_manager.frame_received.emit(IR, np.zeros_like(ir), time.time())
    np.testing.assert_array_equal(view._views[IR]._last_frame, before)
    rgb_new, ir_new = rgb.copy(), ir.copy()
    rgb_new[:20, :20] = 70
    ir_new[:10, :10] = 80
    publish(win, rgb_new, ir_new)
    assert view.review_key == pair.key
    assert view.review_matches({LEFT: pair.left, IR: pair.right})
    count = len(panel._aux_images)
    folder = win._current_session.session_dir / 'multimodal'
    originals = {path.name: path.read_bytes() for path in folder.iterdir()}
    panel._btn_capture_aux.click()
    panel._on_capture_aux()  # The slot also rejects non-button calls during review.
    assert len(panel._aux_images) == count
    assert panel._aux_capture_worker is None
    assert view.review_key == pair.key
    assert originals == {path.name: path.read_bytes() for path in folder.iterdir()}
    assert panel._aux_saved_count == 0
    win._btn_end_review.click()
    assert not view.review_key
    assert not win._review_bar.isVisible()
    assert not guide.last_qualities and not guide._samples
    assert '正在回看' not in guide.phase.text()
    panel._refresh_aux_readiness()
    assert panel._btn_capture_aux.isEnabled()
    # Source routing is checked after the separately tested live stability gate.
    monkeypatch.setattr(panel._aux_live_gate, 'start', lambda **kw: panel._capture_aux_original())
    panel._btn_capture_aux.click()
    app.processEvents()
    finish_aux_tasks(panel)
    assert not view.review_key
    assert not win._review_bar.isVisible()
    session = win._current_session.session_dir
    np.testing.assert_array_equal(cv2.imread(str(session / "multimodal/rgb_left_0000.png")), rgb_new)
    np.testing.assert_array_equal(cv2.imread(str(session / "multimodal/aux_0000.png")), ir_new)


def test_review_pauses_pending_guidance_and_rejects_pre_review_preview(
        bench, monkeypatch, finish_aux_tasks):
    win, rgb, ir = bench
    panel = win._calib_panel
    guide = panel._aux_guide
    monkeypatch.setattr(guide, '_request_preview', lambda: None)
    pair = panel.preview_pairs()[0]
    guide.data_changed()
    panel._aux_seq_widget._btn_next.click()
    chosen = panel._aux_capture_sequence.current_index
    publish(win, rgb, ir)
    win._on_gallery_pair_selected(pair.key, {LEFT: pair.left, IR: pair.right})
    finish_aux_tasks(panel)
    assert '正在回看原图' in guide.phase.text()
    assert '实时指引已暂停' in guide.action.text()
    assert '引导采用 1 组' in guide.counts.text()
    assert not panel._btn_capture_aux.isEnabled()
    assert guide._manual and panel._aux_capture_sequence.current_index == chosen
    old_stamp = guide._clock() - .2
    win._btn_end_review.click()
    guide._preview_ready((guide.context_key(), old_stamp), None, '')
    assert not guide.last_qualities and not guide._preview_detections
    assert not guide._samples
    assert '可以采集' not in guide.action.text()
    assert f'浏览参考动作 {chosen + 1}/17' in guide.phase.text()


def test_review_status_uses_display_label_for_rejected_planar_images(bench):
    win, _, _ = bench
    pair = win._calib_panel.preview_pairs()[0]
    pair.quality_state = '待复核'
    win._show_review_pair(pair)
    assert '验收待复核' in win._stream_view._views[LEFT]._info.toolTip()
    assert pair.quality_state == '待复核'


def test_replacing_data_ends_stale_review_and_gallery_is_bounded(bench):
    win, rgb, ir = bench
    gallery, view = win._gallery, win._stream_view
    pair = gallery._pairs[-1]
    view.show_review_images(pair.key, {LEFT: pair.left, IR: pair.right})
    replacements = [PreviewPair(str(i), LEFT, IR, rgb, ir) for i in range(1000)]
    win._on_preview_pairs_changed(replacements)
    assert not view.review_key
    assert gallery._list.count() <= 6
    assert len(gallery._icon_cache) <= 6
    assert gallery._pairs[0].left is rgb
    gallery._toggle.setChecked(False)
    assert not gallery._body.isVisible()
    assert not gallery._icon_cache
    gallery._toggle.setChecked(True)
    assert gallery._list.count() <= 6
    gallery._page(-1)
    assert gallery._start < len(replacements) - gallery._page_size


def test_review_does_not_bring_closed_camera_back_or_lose_trackers(bench):
    win, rgb, ir = bench
    view = win._stream_view
    tracker = view._trackers[LEFT]
    view.close_preview(IR)
    view.show_review_images("sample", {LEFT: rgb, IR: ir})
    assert IR not in view._views
    assert view._trackers[LEFT] is tracker
    view.end_review()
    assert IR not in view._views
    assert view._trackers[LEFT] is tracker


def test_result_summary_and_native_actions_use_real_results(bench, monkeypatch, app, tmp_path):
    win, _, _ = bench
    panel = win._calib_panel
    intr = panel._intrinsics[LEFT]
    stereo = StereoCalibResult(np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3), rms_error=.15)
    pair = StereoPairCalibration("rgb", LEFT, RIGHT, intr, intr, stereo)
    aux_intr = CameraIntrinsics(np.eye(3), np.zeros((1, 5)), (256, 192), rms_error=.2)
    rig = MultiCameraRig(pairs={"rgb": pair}, reference_camera=LEFT, aux_cameras={
        IR: AuxCameraCalibration(IR, "ir", aux_intr, np.eye(4), rms_error=.22, frame_count=8),
    })
    win._on_calib_finished(rig)
    win._workflow_tabs.setCurrentIndex(2)
    assert "0.220" in panel._lbl_result_summary.text()
    assert "已保存" in panel._lbl_result_storage.text()
    assert '双目 RMS 0.150 px' in panel._lbl_result_metrics.text()
    assert '基线 100.00 mm' in panel._lbl_result_metrics.text()
    assert '融合 RMS 0.220 px' in panel._lbl_result_metrics.text()
    assert panel._lbl_result_metrics.isVisible()
    assert 'JSON：已生成' in panel._lbl_result_storage.text()
    assert panel._btn_export_results.isEnabled()
    assert panel._btn_open_results.isEnabled()
    app.processEvents()
    directory = Path(os.environ.get("MULTICAMERA_UI_QA_DIR", tmp_path / "ui"))
    directory.mkdir(parents=True, exist_ok=True)
    win.grab().save(str(directory / "results-1440x1024.png"))
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    panel._btn_open_results.click()
    assert [Path(path) for path in opened] == [panel.result_directory]
    panel._btn_copy_results.click()
    assert app.clipboard().text() == str(panel.result_directory.resolve())
    assert not panel._lbl_result_summary.isVisible()
    panel._btn_result_details.click()
    assert panel._lbl_result_summary.isVisible()
    assert '复核' in panel._lbl_result_decision.text()
    exports = []
    panel.export_requested.connect(lambda: exports.append(True))
    # Keep this assertion at the signal boundary; file-format dialogs have separate coverage.
    panel.export_requested.disconnect(win._on_export)
    panel._btn_export_results.click()
    assert exports == [True]
    panel._clear_calibration_results()
    assert panel.result_directory is None
    assert not panel._lbl_result_metrics.text()
    assert not panel._btn_export_results.isEnabled()
    assert not panel._btn_open_results.isEnabled()


def test_first_capture_creates_session_before_acquiring(bench, monkeypatch):
    win, _, _ = bench
    panel = win._calib_panel
    panel._session = None
    win._current_session = None
    panel._refresh_disk_hint()
    assert panel._btn_capture_aux.text() == '创建会话并采集'
    monkeypatch.setattr(panel, '_prepare_white_balance', lambda *args: False)
    panel._on_capture_aux()
    assert panel._session is not None
    assert win._current_session is panel._session
    assert panel._session.session_dir.exists()


def test_session_creation_failure_prevents_capture(bench, monkeypatch):
    win, _, _ = bench
    panel = win._calib_panel
    panel._session = None
    win._current_session = None
    messages = []

    def fail(*args, **kwargs):
        raise OSError('disk unavailable')

    monkeypatch.setattr(win._session_manager, 'create_session', fail)
    monkeypatch.setattr(main_window.QMessageBox, 'warning', lambda *args: messages.append(args))
    before = len(panel._aux_images)
    panel._on_capture_aux()
    assert panel._session is None and messages
    assert len(panel._aux_images) == before


def test_logs_are_collapsible_and_errors_remain_visible(bench):
    win, _, _ = bench
    assert not win._logs_widget.isVisible()
    win._calib_panel.append_log("完成一次采集")
    assert not win._logs_widget.isVisible()
    win._calib_panel.append_log("保存图片失败：磁盘不可写")
    assert win._toggle_logs.isChecked()
    assert win._logs_widget.isVisible()
    win._toggle_logs.setChecked(False)
    assert not win._logs_widget.isVisible()


def test_sidebar_summary_uses_actual_board_and_live_readiness(bench):
    win, _, _ = bench
    panel = win._calib_panel
    panel._refresh_aux_board_info()
    panel._refresh_aux_readiness()
    assert panel._lbl_task_title.text() == "当前任务"
    assert panel._lbl_task_phase.text() == "融合采集"
    assert panel._aux_task_values['sensor'].text() == 'RGB + 红外'
    assert panel._aux_task_values['intrinsics'].text() == '已就绪'
    assert panel._aux_task_values['capture'].text() == '视频已配对'
    assert panel._aux_board_values['geometry'].text() == '7 列 × 5 行格子'
    assert panel._aux_board_values['spacing'].text() == '80 mm'
    assert '24 点' in panel._aux_board_values['points'].text()
    assert panel._lbl_aux_pair_count.text() == '验收通过 8 组'
    assert panel._aux_seq_widget._title_lbl.text() == '参考动作 · 位置以红外画面为准'
    assert '不是已采数量' in panel._aux_seq_widget._title_lbl.toolTip()
    assert not panel._lbl_aux_readiness.isVisible()
    assert not panel._lbl_aux_board.isVisible()
    assert not panel._aux_seq_widget._dots.isVisible()
    assert not panel._aux_seq_widget._progress_lbl.isVisible()
    assert not panel._aux_cap_progress.isTextVisible()
    assert not panel._btn_rgb_prerequisite.isVisible()
    panel._aux_ui_busy = True
    panel._refresh_aux_readiness()
    assert panel._aux_task_values['capture'].text() == '处理中'
    assert not panel._btn_capture_aux.isEnabled()
    panel._aux_ui_busy = False
    panel._refresh_aux_readiness()


def test_sidebar_rgb_prerequisite_navigates_without_losing_auxiliary_data(bench, app):
    win, _, _ = bench
    panel = win._calib_panel
    pairs = panel._aux_pairs
    panel._intrinsics.pop(LEFT)
    panel._refresh_aux_readiness()
    assert not panel._btn_rgb_prerequisite.isVisible()  # RGB intrinsics are not a capture prerequisite.
    panel.show_workflow('calibration', True)
    assert panel._btn_rgb_prerequisite.isVisible()
    assert panel._aux_task_values['intrinsics'].text() == '未完成'
    assert panel._aux_task_values['calibration'].text() == '需先标定 RGB'
    assert not panel._btn_calib_aux.isEnabled()
    panel._btn_rgb_prerequisite.click()
    app.processEvents()
    assert panel.workflow_phase == 'calibration'
    assert panel._active_step == 1
    assert panel._aux_pairs is pairs
    assert win._combo_task.currentIndex() == 0


def test_sidebar_footer_sizes_to_current_actions_and_guide_keeps_navigation(bench, app):
    win, _, _ = bench
    panel = win._calib_panel
    guide = panel._aux_seq_widget
    index = panel._aux_capture_sequence.current_index
    assert not guide._btn_next.icon().isNull()
    guide._btn_next.click()
    assert panel._aux_capture_sequence.current_index == index + 1
    assert len(panel._aux_pairs) == 8
    for phase in ('capture', 'calibration', 'results'):
        panel.show_workflow(phase, True)
        app.processEvents()
        pages = panel._footer_pages
        assert pages.height() <= pages.currentWidget().sizeHint().height() + 4
    assert not panel._aux_progress_header.isVisible()


def test_live_aux_capture_requires_white_balance_for_both_rgb_sources(bench, monkeypatch):
    win, _, _ = bench
    panel, manager = win._calib_panel, win._stream_manager
    calls = []
    monkeypatch.setattr(manager, 'is_camera_running', lambda camera_id: camera_id in {LEFT, RIGHT})
    monkeypatch.setattr(panel._white_balance_guard, 'ensure', lambda urls, resume, **kw: calls.append((urls, resume)) or False)
    before = len(panel._aux_images)
    panel._on_capture_aux()
    assert len(panel._aux_images) == before
    assert len(calls) == 1
    assert set(calls[0][0]) == {'http://device.test:8080/left', 'http://device.test:8080/right'}
    assert all('ir' not in url for url in calls[0][0])
    assert panel._check_wb.isChecked()
    panel._check_wb.setChecked(False)
    assert panel._prepare_white_balance(manager.stereo_pairs, panel._on_capture)
    assert len(calls) == 1


def test_explicit_automatic_white_balance_policy_routes_both_rgb_sources(bench, monkeypatch):
    win, _, _ = bench
    panel, manager = win._calib_panel, win._stream_manager
    calls = []
    monkeypatch.setattr(manager, 'is_camera_running', lambda cid: cid in {LEFT, RIGHT})
    monkeypatch.setattr(panel._white_balance_guard, 'ensure',
                        lambda urls, resume, **kw: calls.append((urls, kw)) or False)
    assert panel._combo_wb_mode.currentData() == 'automatic'
    panel._on_capture_aux()
    assert len(calls) == 1 and calls[0][1] == {'lock': False}
    assert set(calls[0][0]) == {'http://device.test:8080/left', 'http://device.test:8080/right'}
    assert '锁定' not in panel._check_wb.text()
    panel._white_balance_guard.ready.add('http://device/')
    panel._combo_wb_mode.setCurrentIndex(panel._combo_wb_mode.findData('locked'))
    assert not panel._white_balance_guard.ready
    panel._on_white_balance_busy(True)
    assert not panel._combo_wb_mode.isEnabled()
    panel._on_white_balance_busy(False)
    panel._check_wb.setChecked(False)
    assert not panel._combo_wb_mode.isVisible()


def test_white_balance_busy_protects_configuration_and_session_invalidates_ready(bench):
    win, _, _ = bench
    panel = win._calib_panel
    guard = panel._white_balance_guard
    guard.busy = True
    assert panel.has_active_operations
    guard.ready.add('http://device.test:8080/')
    panel.set_session(win._current_session)
    assert not guard.ready
    assert not guard.busy


def test_white_balance_preparation_captures_only_new_live_frames(bench, app, monkeypatch):
    from multicamera.ui.threads import white_balance_worker

    win, rgb, ir = bench
    panel, manager = win._calib_panel, win._stream_manager
    monkeypatch.setattr(manager, 'is_camera_running', lambda cid: cid in {LEFT, RIGHT})
    monkeypatch.setattr(white_balance_worker, 'stabilize_and_lock', lambda urls, **kw: dict.fromkeys(urls, {}))
    rgb_new, ir_new = rgb.copy(), ir.copy()
    rgb_new[0, 0] = [12, 34, 56]
    ir_new[0, 0] = [23, 45, 67]
    initial = len(panel._aux_images)
    panel._on_capture_aux()
    assert panel._white_balance_guard.busy
    assert len(panel._aux_images) == initial
    assert panel._wb_progress.isVisible()
    assert panel._btn_capture_aux.text() == '白平衡准备中…'

    def fresh_frames():
        for cid, frame in ((LEFT, rgb_new), (IR, ir_new)):
            manager._latest_frames[cid] = FrameEntry(frame, time.time())
            manager.frame_received.emit(cid, frame, time.time())
    publisher = QTimer()
    publisher.setInterval(100)
    publisher.timeout.connect(fresh_frames)
    publisher.start()
    deadline = time.monotonic() + 8
    try:
        while len(panel._aux_images) == initial and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
    finally:
        publisher.stop()
    assert len(panel._aux_images) == initial + 1
    assert not panel._wb_progress.isVisible()
    np.testing.assert_array_equal(panel._aux_images[-1][0], rgb_new)
    saved = win._current_session.session_dir / 'multimodal/rgb_left_0000.png'
    np.testing.assert_array_equal(cv2.imread(str(saved)), rgb_new)


@pytest.mark.parametrize('size', [(1440, 1024), (1000, 680)])
def test_aux_wait_and_cancel_keep_preview_geometry(bench, app, monkeypatch, tmp_path, size):
    win, rgb, ir = bench
    panel = win._calib_panel
    win.resize(*size)
    publish(win, rgb, ir)
    monkeypatch.setattr(panel, '_prepare_white_balance', lambda *args: True)
    monkeypatch.setattr(panel._aux_guide, 'capture_problem', lambda: '保持整板可见，等待连续稳定检测')
    app.processEvents()
    geometry = {cid: tile.geometry() for cid, tile in win._stream_view._views.items()}
    before = panel._aux_saved_count
    panel._btn_capture_aux.click()
    app.processEvents()
    assert panel._aux_live_gate.waiting
    assert panel._btn_cancel_wb.isVisible() and panel._btn_cancel_wb.isEnabled()
    assert geometry == {cid: tile.geometry() for cid, tile in win._stream_view._views.items()}
    directory = Path(os.environ.get('MULTICAMERA_UI_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    win.grab().save(str(directory / f'aux-online-wait-{size[0]}x{size[1]}.png'))
    panel._btn_cancel_wb.click()
    app.processEvents()
    assert not panel._aux_live_gate.waiting and panel._aux_saved_count == before
    assert '未保存' in panel._aux_guide.receipt.text()
    assert geometry == {cid: tile.geometry() for cid, tile in win._stream_view._views.items()}


@pytest.mark.parametrize("size", [(1440, 1024), (1280, 800), (1000, 680)])
def test_aux_copy_transitions_keep_preview_and_fixed_receipt_geometry(bench, app, size):
    win, rgb, ir = bench
    win.resize(*size)
    publish(win, rgb, ir)
    for _ in range(4):
        app.processEvents()
    panel, guide = win._calib_panel, win._calib_panel._aux_guide
    content = panel._sections[2].content_layout
    assert content.indexOf(guide) < content.indexOf(panel._aux_seq_widget)
    assert content.indexOf(panel._aux_seq_widget) < content.indexOf(guide.details)

    def geometry():
        return tuple((tile.geometry(), tile._image_label.geometry())
                     for tile in win._stream_view._views.values() if tile.isVisible())

    before = geometry()
    receipt_height = guide.receipt.height()
    panel._aux_seq_widget._btn_next.click()
    for text in ('正在检查本次原图 · 尚未确认保存\n检查后显示验收结果',
                 'live_0001 · 原图保存失败\n未计入验收和引导，请重新采图',
                 'live_0001 · 原图已保存 · 验收待复核\n引导未采用 · 验收记录写入失败'):
        guide.show_receipt_message(text)
        guide.refresh()
        for _ in range(4):
            app.processEvents()
        assert geometry() == before
        assert guide.receipt.height() == receipt_height
        assert '\n' in guide.receipt.visible_text()
        assert panel._inspector_scroll.horizontalScrollBar().maximum() == 0
        assert '已记录' not in panel._aux_seq_widget._title_lbl.text()
        assert panel._btn_capture_aux.isVisible()


@pytest.mark.parametrize("size", [(1440, 1024), (1280, 800), (1000, 680)])
def test_layout_keeps_actions_visible_and_images_aspect_fitted(bench, app, size, tmp_path):
    win, rgb, ir = bench
    win.resize(*size)
    publish(win, rgb, ir)
    app.processEvents()
    panel, view = win._calib_panel, win._stream_view
    assert panel._inspector_scroll.horizontalScrollBar().maximum() == 0
    assert panel._btn_capture_aux.isVisible()
    button = panel._btn_capture_aux
    top = button.mapTo(win, button.rect().topLeft())
    assert 0 <= top.y() < win.height() - button.height()
    viewport = panel._inspector_scroll.viewport()
    for item in (panel._aux_seq_widget, panel._aux_cap_progress):
        panel._inspector_scroll.ensureWidgetVisible(item, 0, 8)
        app.processEvents()
        assert item.mapTo(viewport, item.rect().topLeft()).y() >= 0
        bottom = item.mapTo(viewport, item.rect().bottomRight())
        assert bottom.y() < viewport.height()
    for cid, frame in ((LEFT, rgb), (IR, ir)):
        label = view._views[cid]._image_label
        pixmap = label.pixmap()
        assert pixmap is not None and not pixmap.isNull()
        assert pixmap.width() <= label.width() and pixmap.height() <= label.height()
        assert abs(pixmap.width() / pixmap.height() - frame.shape[1] / frame.shape[0]) < .015
    directory = Path(os.environ.get("MULTICAMERA_UI_QA_DIR", tmp_path / "ui"))
    directory.mkdir(parents=True, exist_ok=True)
    win.grab().save(str(directory / f"split-{size[0]}x{size[1]}.png"))
    win._view_aux.click()
    app.processEvents()
    win.grab().save(str(directory / f"focus-{size[0]}x{size[1]}.png"))
    assert win._comparison_host.isVisible()
    win._workflow_tabs.setCurrentIndex(1)
    app.processEvents()
    assert panel._btn_calib_aux.isVisible()
    win.grab().save(str(directory / f"calibration-{size[0]}x{size[1]}.png"))
