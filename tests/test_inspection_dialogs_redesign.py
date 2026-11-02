"""Inspection windows remain usable with long receipts and short desktops."""

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QToolButton

from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.capture_details import CaptureDetailsDialog
from multicamera.ui.widgets.rectification_viewer import RectificationViewer
from test_calibration_workbench_redesign import panel as panel


@pytest.mark.parametrize('size', [(360, 300), (800, 560), (1100, 720)])
def test_long_receipt_stays_scrollable_and_copy_is_exact(app, size):
    configure_application_style(app)
    dialog = CaptureDetailsDialog()
    text = '\n'.join(f'原图 {i:04d} <未通过>：双路完整原因，路径 C:\\样本\\原图.png'
                     for i in range(120))
    dialog.set_details('融合原图验收详情', text)
    dialog.resize(*size)
    dialog.show()
    app.processEvents()
    assert dialog.size().toTuple() == size
    assert dialog.content.isReadOnly()
    assert dialog.content.toPlainText() == text
    assert dialog.content.verticalScrollBar().maximum() > 0
    assert dialog.rect().contains(dialog.close_button.mapTo(dialog, dialog.close_button.rect().center()))
    dialog.copy_button.click()
    assert QApplication.clipboard().text() == text
    assert '已复制' in dialog.status.text()
    dialog.set_details('采图验收详情', '')
    assert not dialog.copy_button.isEnabled()
    assert dialog.content.toPlainText() == '暂无原图验收记录'
    dialog.close_button.click()
    assert not dialog.isVisible()


@pytest.mark.parametrize('size', [(360, 300), (800, 560), (1100, 720)])
def test_rectification_long_context_fit_pixels_and_empty_state(app, size):
    configure_application_style(app)
    viewer = RectificationViewer()
    viewer.resize(*size)
    viewer.show()
    app.processEvents()
    assert '暂无' in viewer.image.text()
    assert viewer.image.size() == viewer.scroll.viewport().size()
    assert all(not button.isEnabled() for button in viewer._image_buttons)
    image = QImage(2560, 720, QImage.Format_RGB32)
    image.fill(QColor('#70886b'))
    original = image.copy()
    context = '原图 rgb 0001\n' + '相机组 / 数据路径 / 标定模型说明\n' * 40
    viewer.set_image(image, context)
    app.processEvents()
    assert viewer.size().toTuple() == size
    assert viewer.context.text() == context
    assert viewer.context_scroll.height() <= 72
    assert viewer.context_scroll.verticalScrollBar().maximum() > 0
    viewport = viewer.scroll.viewport().size()
    assert viewer.image.width() <= viewport.width()
    assert viewer.image.height() <= viewport.height()
    assert viewer._image_buttons[0].isChecked()
    assert '2560 × 720' in viewer.image_info.text()
    viewer.actual_size()
    assert viewer.image.size() == image.size()
    assert viewer._image_buttons[3].isChecked()
    viewer.zoom(1.25)
    assert viewer._scale == 1.25
    assert not viewer._image_buttons[0].isChecked()
    assert not viewer._image_buttons[3].isChecked()
    assert viewer.scroll.horizontalScrollBar().maximum() > 0
    viewer.fit()
    app.processEvents()
    assert viewer._fit
    assert image == original
    viewer.clear_image()
    app.processEvents()
    assert viewer.image.size() == viewer.scroll.viewport().size()
    assert viewer._pixmap.isNull()
    pixmap = viewer.image.pixmap()
    assert pixmap is None or pixmap.isNull()
    assert viewer.context.text() == '' and viewer.context.toolTip() == ''
    assert all(not button.isEnabled() for button in viewer._image_buttons)
    close = next(button for button in viewer.findChildren(QToolButton)
                 if button.accessibleName() == '关闭校正图检查')
    assert viewer.rect().contains(close.mapTo(viewer, close.rect().center()))
    close.click()
    assert not viewer.isVisible()


def test_capture_details_reuse_snapshot_without_changing_workflow(panel, app):
    before = (panel._active_step, list(panel.preview_pairs()), len(panel._pending_save_tokens))
    stereo = 'rgb 0012 · 原图已保存 · 待复核\n原因 <完整保留>\nC:\\原图\\left_0012.png'
    auxiliary = 'live_0003 · 保存失败\nRGB左目 24 / 红外 24\n原图未落盘'
    panel._lbl_capture_decision.setToolTip(stereo)
    panel._btn_capture_decision_details.click()
    app.processEvents()
    dialog = panel._capture_details_dialog
    assert dialog.content.toPlainText() == stereo
    assert not dialog.isModal()
    panel._lbl_capture_decision.setToolTip('后续回执')
    assert dialog.content.toPlainText() == stereo
    panel._aux_guide.receipt.setToolTip(auxiliary)
    panel._btn_aux_quality_details.click()
    assert panel._capture_details_dialog is dialog
    assert dialog.windowTitle() == '融合原图验收详情'
    assert dialog.content.toPlainText() == auxiliary
    dialog.close_button.click()
    panel._btn_capture_decision_details.click()
    assert dialog.content.toPlainText() == '后续回执'
    assert before == (panel._active_step, list(panel.preview_pairs()), len(panel._pending_save_tokens))
    dialog.close()
