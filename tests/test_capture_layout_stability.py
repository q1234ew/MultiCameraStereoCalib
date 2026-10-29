"""Live status transitions must not resize the video tiles or their image areas."""

import time
from copy import deepcopy

import pytest
import numpy as np
from PySide6.QtCore import QPoint

from multicamera.ui.widgets.pair_gallery import PreviewPair
from multicamera.ui.widgets.stable_text import StableTextLabel
from test_adaptive_capture_ui import activate, settle
from test_multimodal_preview import LEFT, RIGHT
from test_stereo_capture_ui import publish_stable, sample

pytest_plugins = ['test_stereo_capture_ui']


def flush(app):
    for _ in range(5):
        app.processEvents()


def geometry(window):
    result = {}
    for cid, tile in window._stream_view._views.items():
        if not tile.isVisible():
            continue
        for name, widget in (('tile', tile), ('image', tile._image_label)):
            point = widget.mapTo(window, QPoint(0, 0))
            result[cid, name] = (point.x(), point.y(), widget.width(), widget.height())
    return result


@pytest.mark.parametrize('size', [(1000, 680), (1280, 800), (1440, 1024)])
def test_live_status_and_expansion_do_not_move_previews(stereo_bench, app, size, tmp_path):
    win = stereo_bench
    panel, guide = win._calib_panel, activate(win, app)
    det, frame, intr = sample(win)
    panel._intrinsics.update({LEFT: intr, RIGHT: intr})
    publish_stable(win, det, frame)
    settle(guide, app)
    for cid in (LEFT, RIGHT):
        win._stream_view._on_frame(cid, frame, time.time())
    win.resize(*size)
    flush(app)
    baseline = geometry(win)
    assert baseline
    win.grab().save(str(tmp_path / 'baseline.png'))
    assert win._preview_scroll.verticalScrollBar().maximum() == 0, (
        win.size(), win._preview_scroll.height(), panel._action_footer.height(), guide.banner.height(),
        [(w.objectName(), w.height(), w.fontMetrics().lineSpacing()) for w in (
            panel._lbl_raw_save, panel._lbl_capture_readiness, panel._lbl_capture_decision,
            panel._capture_status_pages, panel._combo_wb_mode)],
        win._gallery.height(), win._capture_bottom.height())

    def unchanged(stage):
        flush(app)
        win.grab().save(str(tmp_path / f'{size[0]}-{stage}.png'))
        assert geometry(win) == baseline, (stage, baseline, geometry(win),
            [(t._info.height(), t._info.fontMetrics().lineSpacing(), t._info.styleSheet())
             for t in win._stream_view._views.values()])
        assert win._preview_scroll.verticalScrollBar().maximum() == 0

    panel._white_balance_guard.busy = True
    panel._white_balance_guard.started_at = time.monotonic()
    panel._white_balance_guard.stage = '检查左右RGB白平衡稳定，等待新帧'
    panel._on_white_balance_busy(True)
    guide.refresh()
    unchanged('preparing')
    panel._white_balance_guard.busy = False
    panel._on_white_balance_busy(False)
    guide.refresh()
    unchanged('prepared')
    panel._pending_save_tokens.add('test-save')
    panel._lbl_capture_decision.setText('保存中 · 原图左右角点独立验收\n等待磁盘写入完成')
    panel._refresh_disk_hint()
    guide.refresh()
    unchanged('saving')
    panel._pending_save_tokens.clear()
    panel._capture_status_pages.setCurrentWidget(panel._btn_retry_save)
    panel._lbl_capture_decision.setText('保存失败；请重试原图，不要重复采集' * 10)
    guide.refresh()
    unchanged('failed')
    panel._capture_status_pages.setCurrentWidget(panel._lbl_capture_readiness)
    panel._lbl_capture_decision.setText('本组已保存')
    panel._lbl_capture_readiness.setText('双路已就绪，等待下一次原图采集' * 10)
    for tile in win._stream_view._views.values():
        tile.update_info('1280×720 · 15fps · 预览角点126/126' * 5)
    guide.refresh()
    unchanged('saved')
    guide.schematic.expand.setChecked(True)
    guide.coverage_button.setChecked(True)
    unchanged('guide-expanded')
    win._on_preview_pairs_changed([PreviewPair(
        'rgb 0001', LEFT, RIGHT, frame, frame, det, det, '采图原图', win._board.num_corners,
        save_state='已保存', quality_state='可采纳')])
    win._gallery._toggle.setChecked(True)
    win._toggle_logs.setChecked(True)
    unchanged('tools-open')


def test_stable_status_keeps_full_text_and_elides_only_rendering(app):
    label = StableTextLabel(lines=2)
    label.resize(160, label.height())
    label.show()
    text = '目标：中距离\n保持当前高度，向后移动标定板；这是一条很长的完整诊断信息'
    height = label.height()
    label.setText(text)
    flush(app)
    assert label.height() == height
    assert label.text() == text
    assert '…' in label.visible_text()
    assert len(label.visible_text().splitlines()) <= 2
    label.setMinimumWidth(140)
    label.setStyleSheet('font-size:20px;')
    flush(app)
    assert label.minimumWidth() == 140


def test_persistent_receipt_distinguishes_partial_saved_original(stereo_bench, app):
    win = stereo_bench
    guide = activate(win, app)
    det, frame, _ = sample(win)
    partial = deepcopy(det)
    partial.charuco_corners = partial.charuco_corners[:-2]
    partial.charuco_ids = partial.charuco_ids[:-2]
    item = PreviewPair('rgb 0001', LEFT, RIGHT, frame, frame, det, partial,
                       '采图原图', win._board.num_corners, 'test-save',
                       save_state='保存中', quality_state='可采纳')
    win._calib_panel._capture_reviews['rgb'] = [item]
    win._calib_panel._saved_capture_tokens.add('test-save')
    guide.refresh()
    assert '已保存' in guide.receipt.text()
    assert '右124' in guide.receipt.text() and '部分角点可采纳' in guide.receipt.text()
    assert '下一步' in guide.receipt.toolTip()
    item.quality_state = '待复核'
    guide.refresh()
    assert '可采纳' not in guide.receipt.text()
    assert '未计入有效样本' in guide.receipt.toolTip()
    win._calib_panel._capture_reviews.clear()
    guide.refresh()
    assert '尚未采集' in guide.receipt.text()


def test_initial_guidance_separates_goal_action_and_manual_next_step(stereo_bench, app):
    win = stereo_bench
    guide = activate(win, app)
    det, frame, _ = sample(win)
    publish_stable(win, det, frame)
    _, samples = guide._current_samples()
    for measurement in samples:
        measurement.update(center=np.array([.5, .5]), area=.07, distance_area=.07)
    guide.refresh()
    assert guide.reason.text().startswith('目标：')
    assert '建立初始模型' in guide.reason.text()
    assert '冷启动' not in guide.reason.text()
    assert guide.instruction.text().startswith('操作：')
    assert '手动点击采集' in guide.instruction.text()
    assert guide.schematic.description.text() == guide.instruction.text()
