"""Approved capture layout keeps current actions, next steps and receipts independent."""

import pytest
from PySide6.QtCore import QPoint, QRect

from multicamera.calibration.adaptive_guidance import DISTANCE_LABELS
from multicamera.ui.theme import SUCCESS
from multicamera.ui.widgets.pair_gallery import PreviewPair
from test_multimodal_preview import LEFT, RIGHT
from test_stereo_capture_ui import sample, stereo_bench as stereo_bench


def flush(app):
    for _ in range(6):
        app.processEvents()


@pytest.mark.parametrize('size', [(1000, 680), (1280, 800), (1440, 1024)])
def test_next_step_is_visible_above_diagram_and_footer_is_independent(stereo_bench, app, size):
    win = stereo_bench
    guide, panel = win._stereo_capture, win._calib_panel
    win.resize(*size)
    flush(app)
    assert guide.banner.parentWidget() is guide
    viewport = win._guide_scroll.viewport()
    for widget in (guide.instruction, guide.next_title, guide.next_step):
        assert widget.isVisible()
        assert viewport.rect().contains(widget.mapTo(viewport, widget.rect().topLeft()))
        assert viewport.rect().contains(widget.mapTo(viewport, widget.rect().bottomRight()))
    assert guide.banner.mapTo(guide, QPoint()).y() < guide.schematic_actions.mapTo(guide, QPoint()).y()
    assert guide.receipt.parentWidget() is win._receipt_strip
    assert panel._action_footer.parentWidget() is win._capture_actions
    assert win._capture_actions.width() >= win.width() - 50
    assert win._preview_scroll.horizontalScrollBar().maximum() == 0
    assert win._preview_scroll.verticalScrollBar().maximum() == 0
    assert win._guide_scroll.horizontalScrollBar().maximum() == 0
    for button in (panel._btn_capture, panel._btn_to_calibration, win._btn_import_capture):
        assert button.isVisible()
        assert win.rect().contains(button.mapTo(win, button.rect().bottomRight()))


def test_save_failure_and_pending_tasks_override_future_pose(stereo_bench, app):
    win = stereo_bench
    panel, guide = win._calib_panel, win._stereo_capture
    flush(app)
    baseline = win._stream_view.geometry()
    try:
        panel._pending_save_tokens.add('pending')
        guide.refresh()
        assert '正在保存' in guide.next_step.text()
        assert '保存确认后' in guide.next_step.text()
        panel._failed_save_jobs['failure'] = object()
        guide.refresh()
        assert '重试保存' in guide.next_step.text()
        assert '不要重复采图' in guide.next_step.text()
        assert SUCCESS not in guide.capture_verdict.styleSheet()
        panel._failed_save_jobs.clear()
        panel._pending_save_tokens.clear()
        panel._white_balance_guard.busy = True
        guide.refresh()
        assert '白平衡准备完成' in guide.next_step.text()
        assert SUCCESS not in guide.capture_verdict.styleSheet()
        panel._white_balance_guard.busy = False
        guide.refresh()
        flush(app)
        assert win._stream_view.geometry() == baseline
    finally:
        panel._pending_save_tokens.clear()
        panel._failed_save_jobs.clear()
        panel._white_balance_guard.busy = False


def test_header_import_routes_to_current_task_and_rechecks_enabled(stereo_bench, monkeypatch):
    win = stereo_bench
    panel = win._calib_panel
    calls = []
    monkeypatch.setattr(panel._btn_import, 'click', lambda: calls.append('stereo'))
    monkeypatch.setattr(panel._btn_import_aux, 'click', lambda: calls.append('aux'))
    for auxiliary, expected in ((False, 'stereo'), (True, 'aux'), (False, 'stereo')):
        panel.show_workflow('capture', auxiliary)
        source = panel._btn_import_aux if auxiliary else panel._btn_import
        source.setEnabled(True)
        win._refresh_capture_navigation()
        assert win._btn_import_capture.isVisible()
        win._btn_import_capture.click()
        assert calls[-1] == expected
        count = len(calls)
        source.setEnabled(False)
        win._import_capture_images()
        assert len(calls) == count
    panel.show_workflow('results', False)
    assert win._btn_import_capture.isHidden()


def test_latest_receipt_details_enters_review_without_mutating_images(stereo_bench, app):
    win = stereo_bench
    panel, guide = win._calib_panel, win._stereo_capture
    detection, image, _ = sample(win)
    pair = PreviewPair('rgb 0007', LEFT, RIGHT, image, image.copy(), detection, detection,
                       '采图原图', win._board.num_corners, save_state='保存失败',
                       quality_state='待复核')
    checksum = image.tobytes()
    panel._capture_reviews['rgb'] = [pair]
    win._on_preview_pairs_changed(panel.preview_pairs())
    win._refresh_capture_navigation()
    guide.refresh()
    assert win._btn_receipt_details.isEnabled()
    assert '保存失败' in guide.receipt.text()
    win._btn_receipt_details.click()
    flush(app)
    assert win._stream_view.review_key == pair.key
    assert '回看' in guide.instruction.text()
    assert '结束' in guide.next_step.text()
    assert SUCCESS not in guide.capture_verdict.styleSheet()
    assert image.tobytes() == checksum
    win._btn_end_review.click()
    flush(app)
    assert not win._stream_view.review_key
    assert pair.quality_state == '待复核'


def test_short_window_does_not_open_gallery_just_because_it_was_resized(stereo_bench, app):
    win = stereo_bench
    _, image, _ = sample(win)
    pairs = [PreviewPair(str(i), LEFT, RIGHT, image, image) for i in range(10)]
    win._on_preview_pairs_changed(pairs)
    win.resize(1440, 1024)
    flush(app)
    assert win._gallery._body.parentWidget() is win._gallery
    win.resize(1000, 680)
    flush(app)
    assert not win._gallery._toggle.isChecked()
    dialog = win._aux_panel_dialogs.get('gallery')
    assert dialog is None or not dialog.isVisible()
    baseline = win._stream_view.geometry()
    win._gallery._toggle.click()
    flush(app)
    assert win._aux_panel_dialogs['gallery'].isVisible()
    assert win._stream_view.geometry() == baseline
    win._gallery._toggle.click()
    win.resize(1440, 1024)
    flush(app)
    assert not win._aux_panel_dialogs['gallery'].isVisible()


def test_fixed_next_step_uses_actual_next_distance_and_pose_without_advancing(stereo_bench):
    guide = stereo_bench._stereo_capture
    sequence = stereo_bench._calib_panel._capture_sequence
    for index in (0, 5, 10):
        sequence.jump_to(index)
        guide.refresh()
        following = sequence.targets[index + 1]
        assert following.label in guide.next_step.text()
        assert DISTANCE_LABELS[following.scale] in guide.next_step.text()
        assert sequence.current_index == index
        assert guide.schematic.target == sequence.current
    sequence.jump_to(sequence.total - 1)
    guide.refresh()
    assert '末项' in guide.next_step.text()
    assert '评估' in guide.next_step.text()
    assert '采集完成' not in guide.next_step.text()


@pytest.mark.parametrize('size', [(1000, 680), (1280, 800), (1440, 1024)])
def test_capture_footer_settings_align_without_stretching_controls(stereo_bench, app, size):
    win = stereo_bench
    panel = win._calib_panel
    win.resize(*size)
    flush(app)
    footer = panel._action_footer

    def bounds(widget):
        return QRect(widget.mapTo(footer, QPoint()), widget.size())

    controls = (panel._btn_capture, panel._btn_auto, panel._btn_undo,
                panel._btn_skip_stereo, panel._btn_to_calibration,
                panel._lbl_raw_save, panel._combo_capture_quality,
                panel._btn_capture_decision_details, panel._check_wb, panel._combo_wb_mode)
    for widget in controls:
        assert widget.isVisible()
        assert footer.rect().contains(bounds(widget))
        assert widget.height() >= widget.minimumSizeHint().height() or widget is panel._lbl_raw_save
    settings = bounds(panel._combo_capture_quality)
    balance = bounds(panel._combo_wb_mode)
    assert abs(settings.center().y() - balance.center().y()) <= 1
    assert abs(settings.center().y() - bounds(panel._lbl_raw_save).center().y()) <= 1
    assert bounds(panel._btn_capture).bottom() < settings.top()
    assert bounds(panel._btn_skip_stereo).right() < bounds(panel._btn_to_calibration).left()
    assert settings.width() <= 340
    assert balance.width() <= 280
    assert bounds(panel._combo_wb_mode).left() - bounds(panel._check_wb).right() <= 12
    assert bounds(panel._lbl_capture_decision).width() >= settings.width()
    assert panel._capture_footer_layout.spacing() == 0
    assert win._preview_scroll.verticalScrollBar().maximum() == 0
    baseline = win._stream_view.geometry()
    panel._combo_capture_quality.setCurrentIndex(1)
    panel._check_wb.setChecked(False)
    panel._lbl_raw_save.setText('本次已落盘 120 组 · 保存中 0 · 失败 0')
    flush(app)
    assert win._stream_view.geometry() == baseline
    panel._check_wb.setChecked(True)
    flush(app)
    assert win._stream_view.geometry() == baseline


@pytest.mark.parametrize('size', [(1000, 680), (1280, 800), (1440, 1024)])
def test_guidance_has_one_current_heading_and_compact_distinct_checks(stereo_bench, app, size):
    win = stereo_bench
    win.resize(*size)
    flush(app)
    guide = win._stereo_capture
    assert guide.current_title.text() == '当前动作'
    assert guide.banner.layout().indexOf(guide.next_title) >= 0
    assert guide.banner.height() < (guide.instruction.height() + guide.reason.height()
                                    + guide.next_step.height() + 90)
    checks = guide.live_checks.layout()
    for key, expected in (('points', (0, 1)), ('sharp', (0, 4)),
                          ('stable', (1, 1)), ('pair', (1, 4))):
        widget = guide.check_values[key]
        assert checks.getItemPosition(checks.indexOf(widget))[:2] == expected
        assert widget.isVisible()
        assert widget.width() >= widget.fontMetrics().horizontalAdvance(widget.text())
    assert checks.columnMinimumWidth(2) >= 16
    assert guide.schematic.legend.isVisible()
    before = (guide.banner.height(), win._stream_view.geometry())
    guide.btn_retry_guidance.setVisible(True)
    flush(app)
    assert (guide.banner.height(), win._stream_view.geometry()) == before
    assert guide.btn_retry_guidance.mapTo(guide.banner, QPoint()).y() < guide.instruction.y()
