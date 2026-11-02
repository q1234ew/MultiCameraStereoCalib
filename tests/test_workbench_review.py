"""Design review fixes remain presentation-only over existing capture state."""

import pytest
import cv2
import numpy as np
from threading import Event
from PySide6.QtCore import QRectF

from multicamera.streaming.stream_manager import CameraConfig, StereoPairConfig
from multicamera.ui.theme import BORDER, SUCCESS
from multicamera.ui.widgets.board_view import view_rects
from multicamera.ui.widgets.pair_gallery import PreviewPair
from test_multimodal_preview import LEFT, RIGHT
from test_stereo_capture_ui import sample, stereo_bench as stereo_bench
from test_adaptive_workflow_followup import prepare_raw_capture
from test_real_pairs_ui import wait_for


def flush(app):
    for _ in range(6):
        app.processEvents()


@pytest.mark.parametrize('width', [280, 360, 440])
def test_compact_board_and_spatial_panes_have_equal_width(width):
    front, spatial = view_rects(QRectF(0, 0, width, 150), True)
    assert front.width() == spatial.width()
    assert front.right() < spatial.left()
    assert front.left() == 0
    assert spatial.right() == width


def test_empty_gallery_has_feedback_without_moving_previews_on_first_pair(stereo_bench, app):
    win = stereo_bench
    win.resize(1440, 1024)
    flush(app)
    gallery = win._gallery
    assert gallery._empty.isVisible()
    assert gallery._empty.text() == '尚无配对图像'
    baseline = win._stream_view.geometry()
    _, image, _ = sample(win)
    win._on_preview_pairs_changed([PreviewPair('rgb 0001', LEFT, RIGHT, image, image)])
    flush(app)
    assert gallery._empty.isHidden()
    assert gallery._body.isVisible()
    assert win._stream_view.geometry() == baseline
    win._on_preview_pairs_changed([])
    flush(app)
    assert gallery._empty.isVisible()
    assert win._stream_view.geometry() == baseline


def test_empty_gallery_does_not_overflow_short_window(stereo_bench, app):
    win = stereo_bench
    win.resize(1000, 680)
    flush(app)
    assert win._gallery._empty.isHidden()
    assert win._gallery.height() == 38
    assert win._preview_scroll.verticalScrollBar().maximum() == 0


def test_preview_verdict_changes_emphasis_not_capture_eligibility(stereo_bench, monkeypatch):
    win = stereo_bench
    guide, panel = win._stereo_capture, win._calib_panel
    counts = dict(panel._pair_capture_press_count)
    monkeypatch.setattr(guide, 'auto_ready', lambda: True)
    guide._refresh_live_checks([], True, True, True, True, False, False)
    assert '预览条件满足' in guide.capture_verdict.text()
    assert '原图仍需验收' in guide.capture_verdict.text()
    assert SUCCESS in guide.capture_verdict.styleSheet()
    assert 'border:1px' in guide.capture_verdict.styleSheet()
    guide._refresh_live_checks([], True, True, True, True, True, False)
    assert '准备、保存或分析' in guide.capture_verdict.text()
    assert SUCCESS not in guide.capture_verdict.styleSheet()
    assert BORDER in guide.capture_verdict.styleSheet()
    assert panel._pair_capture_press_count == counts
    assert not panel.preview_pairs()


def test_original_review_action_has_specific_accessible_name(stereo_bench):
    button = stereo_bench._btn_receipt_details
    assert button.text() == '回看原图'
    assert button.accessibleName() == '回看当前回执原图'
    assert stereo_bench._calib_panel._btn_skip_stereo.accessibleName() == '跳过当前双目采图建议'


def add_receipts_for_two_groups(win):
    panel = win._calib_panel
    win._stream_manager.add_stereo_pair(StereoPairConfig(
        'rear', CameraConfig('rear_L', '', 'left', 'rear', 'rgb'),
        CameraConfig('rear_R', '', 'right', 'rear', 'rgb')))
    panel.refresh_pairs()
    detection, image, _ = sample(win)
    first = PreviewPair('rgb 0009', LEFT, RIGHT, image, image, detection, detection,
                        '采图原图', win._board.num_corners, save_state='保存失败',
                        quality_state='待复核')
    second = PreviewPair('rear 0002', 'rear_L', 'rear_R', image, image, detection, detection,
                         '采图原图', win._board.num_corners, save_state='已保存',
                         quality_state='可采纳')
    panel._capture_reviews = {'rgb': [first], 'rear': [second]}
    panel._combo_pair.setCurrentIndex(0)
    panel._refresh_preview_pairs()
    return first, second


def test_all_groups_review_opens_exact_failed_receipt_not_last_group(stereo_bench):
    win = stereo_bench
    first, second = add_receipts_for_two_groups(win)
    guide = win._stereo_capture
    assert guide.receipt_key == first.key
    assert first.key in guide.receipt.text()
    assert '保存失败' in guide.receipt.text()
    assert '不要重复采集' in guide.receipt.toolTip()
    assert '验收详情后重拍' not in guide.receipt.toolTip()
    assert win._btn_receipt_details.isEnabled()
    win._btn_receipt_details.click()
    assert win._stream_view.review_key == first.key
    assert win._stream_view.review_key != second.key
    assert '保存失败' in win._stream_view._views[LEFT]._info.text()
    win._btn_end_review.click()
    win._calib_panel._combo_pair.setCurrentText('rear')
    assert guide.receipt_key == second.key
    assert second.key in guide.receipt.text()
    win._btn_receipt_details.click()
    assert win._stream_view.review_key == second.key


def test_deleted_receipt_does_not_fall_back_to_another_group(stereo_bench):
    win = stereo_bench
    first, _ = add_receipts_for_two_groups(win)
    assert win._stereo_capture.receipt_key == first.key
    win._calib_panel._capture_reviews['rgb'].clear()
    win._show_latest_capture()
    assert not win._stream_view.review_key
    win._calib_panel._refresh_preview_pairs()
    assert win._stereo_capture.receipt_key is None
    assert not win._btn_receipt_details.isEnabled()


def test_import_source_updates_navigation_without_waiting_for_timer(stereo_bench):
    win = stereo_bench
    panel = win._calib_panel
    win._capture_ui_timer.stop()
    detection, image, _ = sample(win)
    panel._on_import_images_ready([('0001', image, image, detection, detection)])
    assert panel.offline_selection
    assert win._btn_receipt_details.isHidden()
    assert win._preview_note.text() == '离线原图 · 不修改源文件'
    assert win._stereo_capture.receipt_key is None
    panel._combo_pair.setCurrentText('rgb')
    assert not panel.offline_selection
    assert not win._btn_receipt_details.isHidden()
    assert win._preview_note.text() == '原图检测 · 软件配对'


def test_stereo_review_keeps_legacy_capture_using_fresh_frames(stereo_bench, monkeypatch, app):
    win = stereo_bench
    panel, detection, fresh = prepare_raw_capture(win, monkeypatch)
    for cid in (LEFT, RIGHT):
        monkeypatch.setattr(panel._intr_cals[cid].detector, 'detect', lambda image: detection)
    historical = np.zeros_like(fresh)
    win._stream_view.show_review_images('historical', {LEFT: historical, RIGHT: historical})
    assert '实时指引已暂停' in win._stereo_capture.capture_verdict.text()
    assert '实时采图已暂停' not in win._stereo_capture.capture_verdict.text()
    assert '实时新帧' in win._stereo_capture.next_step.text()
    panel._btn_capture.click()
    wait_for(app, lambda: not panel._pending_save_tokens)
    assert not win._stream_view.review_key
    assert panel._session.frame_count == 1
    pair = panel.preview_pairs()[-1]
    assert pair.save_state == '已保存'
    np.testing.assert_array_equal(pair.left, fresh)
    assert not np.array_equal(pair.left, historical)
    saved = panel._session.session_dir / 'images' / 'rgb' / 'left_0000.png'
    np.testing.assert_array_equal(cv2.imread(str(saved)), fresh)


def test_saved_callback_updates_guidance_without_timer_and_pending_holds_action(
        stereo_bench, monkeypatch, app):
    win = stereo_bench
    panel, detection, _ = prepare_raw_capture(win, monkeypatch)
    guide = win._stereo_capture
    guide.timer.stop()
    win._capture_ui_timer.stop()
    for cid in (LEFT, RIGHT):
        monkeypatch.setattr(panel._intr_cals[cid].detector, 'detect', lambda image: detection)
    entered, release = Event(), Event()
    save = panel._session.save_frame_pair

    def delayed_save(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError('Test did not release delayed save')
        return save(*args, **kwargs)

    monkeypatch.setattr(panel._session, 'save_frame_pair', delayed_save)
    try:
        panel._btn_capture.click()
        wait_for(app, entered.is_set)
        assert '等待原图保存完成' in guide.instruction.text()
        assert '参考目标：' in guide.reason.text()
        assert '当前目标：' not in guide.reason.text()
        assert '正在保存' in guide.next_step.text()
        assert '已保存' not in guide.receipt.text()
    finally:
        release.set()
        wait_for(app, lambda: not panel._pending_save_tokens)
    assert '已保存' in guide.receipt.text()
    assert '正在保存' not in guide.next_step.text()
    assert '等待原图保存完成' not in guide.instruction.text()
    assert guide._instruction_stage == 'fixed'
    assert panel._session.frame_count == 1


def test_fixed_plan_without_frames_explains_connection_before_motion(stereo_bench):
    win = stereo_bench
    guide = win._stereo_capture
    sequence = win._calib_panel._capture_sequence
    win._stream_manager._latest_frames.clear()
    guide.refresh()
    assert '连接左右相机' in guide.instruction.text()
    assert guide._instruction_stage == 'waiting'
    assert guide.schematic.target == sequence.current
    assert not guide.auto_ready()
    assert sequence.current_index == 0
