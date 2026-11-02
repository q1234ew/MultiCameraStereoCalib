"""Compact diagnostics keep the table, metric semantics and source actions intact."""

import copy

import numpy as np
import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtTest import QSignalSpy

from multicamera.calibration.models import CameraIntrinsics, StereoCalibResult, StereoPairCalibration
from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.calibration_diagnostics import CalibrationDiagnosticsDialog
from multicamera.ui.widgets.pair_gallery import PreviewPair


@pytest.fixture
def diagnostic_pair(app):
    configure_application_style(app)

    def intrinsic(errors):
        return CameraIntrinsics(
            np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
            np.zeros(5), (640, 480), frame_ids=["2", "10"], per_frame_errors=errors,
            per_frame_error_metric="euclidean_rms_px",
        )

    return StereoPairCalibration(
        "rgb", "left", "right", intrinsic([10., 2.]), intrinsic([.1, .2]),
        StereoCalibResult(np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3)),
        calibration_info={"selection": {"stereo": {"used_ids": ["10"]}},
                          "rectification": {"coordinate": "x", "per_frame": [
                              {"frame_id": "10", "rms_px": .35},
                          ]}},
    )


def bounds(dialog, widget):
    return QRect(widget.mapTo(dialog, widget.rect().topLeft()), widget.size())


def assert_layout(dialog, size):
    assert (dialog.width(), dialog.height()) == size
    widgets = [dialog._combo_pair, dialog._metric_summary, dialog._btn_details,
               dialog._table, dialog._footer, dialog._btn_preview, dialog._preview_status,
               dialog._btn_close]
    if dialog._btn_details.isChecked():
        widgets.append(dialog._details_scroll)
    for widget in widgets:
        assert widget.isVisible()
        assert dialog.rect().contains(bounds(dialog, widget)), widget.objectName()
    assert bounds(dialog, dialog._table).bottom() < bounds(dialog, dialog._footer).top()
    assert bounds(dialog, dialog._btn_preview).right() < bounds(dialog, dialog._preview_status).left()
    assert bounds(dialog, dialog._preview_status).right() < bounds(dialog, dialog._btn_close).left()
    assert dialog._table.height() >= 64
    assert dialog._table.viewport().height() >= 24
    summary = dialog._metric_summary
    assert summary.heightForWidth(summary.width()) <= summary.height()
    assert dialog._preview_status.fontMetrics().horizontalAdvance(
        dialog._preview_status.text()) <= dialog._preview_status.width()


@pytest.mark.parametrize("size", [(360, 300), (800, 560)])
@pytest.mark.parametrize("long_content", [False, True])
def test_table_and_footer_fit_with_collapsed_and_scrollable_details(
    app, diagnostic_pair, size, long_content, tmp_path,
):
    name = "双目组_" + "超长组名_with_no_spaces_" * 15 if long_content else "rgb"
    if long_content:
        diagnostic_pair.left_intrinsics.frame_ids = None
        diagnostic_pair.left_intrinsics.per_frame_error_metric = None
        diagnostic_pair.right_intrinsics.per_frame_error_metric = "<legacy_metric>" * 100
        diagnostic_pair.calibration_info["rectification"]["coordinate"] = "x 坐标说明" * 100
    dialog = CalibrationDiagnosticsDialog({name: diagnostic_pair}, [])
    dialog.resize(*size)
    dialog.show()
    app.processEvents()
    assert not dialog._details_scroll.isVisible()
    assert dialog._combo_pair.currentData() == name
    assert dialog._combo_pair.toolTip() == name
    assert dialog._combo_pair.itemData(0, Qt.ToolTipRole) == name
    assert_layout(dialog, size)
    assert "不是独立验证" in dialog._metric_summary.text()
    assert "不表示零误差" in dialog._metric_summary.text()
    assert dialog.grab().save(str(tmp_path / f"collapsed-{size[0]}-{long_content}.png"))
    collapsed_height = dialog._table.height()
    dialog._btn_details.click()
    app.processEvents()
    assert_layout(dialog, size)
    assert dialog._table.height() < collapsed_height
    assert dialog._details_scroll.height() <= dialog.height() // 6
    assert dialog._details_scroll.horizontalScrollBar().maximum() == 0
    assert dialog._notes.textFormat() == Qt.PlainText
    assert "不是独立验证" in dialog._notes.text()
    assert "不是零误差" in dialog._notes.text()
    assert "x 坐标" in dialog._notes.text()
    assert dialog.grab().save(str(tmp_path / f"expanded-top-{size[0]}-{long_content}.png"))
    if long_content:
        assert "不映射原图" in dialog._notes.text()
        assert "旧指标口径未记录" in dialog._notes.text()
        assert "<legacy_metric>" in dialog._notes.text()
        assert dialog._details_scroll.verticalScrollBar().maximum() > 0
    footer = bounds(dialog, dialog._footer)
    scroll = dialog._details_scroll.verticalScrollBar()
    scroll.setValue(scroll.maximum())
    app.processEvents()
    assert bounds(dialog, dialog._footer) == footer
    assert dialog.grab().save(str(tmp_path / f"diagnostics-{size[0]}-{long_content}.png"))
    dialog._btn_details.click()
    app.processEvents()
    assert not dialog._details_scroll.isVisible()
    assert dialog._table.height() == collapsed_height
    dialog._btn_close.click()
    assert not dialog.isVisible()


@pytest.mark.parametrize("with_pair", [False, True])
def test_empty_states_are_explicit_without_fabricated_rows(app, diagnostic_pair, with_pair):
    for intr in (diagnostic_pair.left_intrinsics, diagnostic_pair.right_intrinsics):
        intr.frame_ids = []
        intr.per_frame_errors = []
    diagnostic_pair.calibration_info = {}
    dialog = CalibrationDiagnosticsDialog({"rgb": diagnostic_pair} if with_pair else {}, [])
    dialog.resize(360, 300)
    dialog.show()
    app.processEvents()
    assert_layout(dialog, (360, 300))
    assert dialog._table.rowCount() == 0
    assert dialog._empty_state.isVisible()
    assert dialog._empty_state.text() == (
        "当前组暂无逐帧记录" if with_pair else "暂无双目标定结果")
    assert dialog._table.viewport().rect().contains(dialog._empty_state.geometry())
    assert dialog._combo_pair.isEnabled() == with_pair
    assert not dialog._btn_preview.isEnabled()
    assert dialog._preview_status.text() == "未选择帧"
    signals = QSignalSpy(dialog.preview_requested)
    dialog._request_preview()
    assert signals.count() == 0


def test_resizing_with_long_table_keeps_scrollbars_and_actions_visible(app, diagnostic_pair):
    for intr in (diagnostic_pair.left_intrinsics, diagnostic_pair.right_intrinsics):
        intr.frame_ids = [str(index) for index in range(60)]
        intr.per_frame_errors = [index / 10 for index in range(60)]
    dialog = CalibrationDiagnosticsDialog({"rgb": diagnostic_pair}, [])
    dialog.resize(800, 560)
    dialog.show()
    dialog._btn_details.click()
    for size in ((800, 560), (360, 300), (800, 560)):
        dialog.resize(*size)
        app.processEvents()
        assert_layout(dialog, size)
        assert dialog._table.verticalScrollBar().maximum() > 0
        assert dialog._table.horizontalScrollBar().maximum() > 0
        footer = bounds(dialog, dialog._btn_close)
        dialog._table.scrollToBottom()
        dialog._table.horizontalScrollBar().setValue(
            dialog._table.horizontalScrollBar().maximum())
        app.processEvents()
        assert bounds(dialog, dialog._btn_close) == footer
        assert dialog._table.rowCount() == 60


def test_selection_sort_details_and_group_switch_keep_preview_identity(app, diagnostic_pair):
    other = copy.deepcopy(diagnostic_pair)
    other.pair_name = "空组"
    for intr in (other.left_intrinsics, other.right_intrinsics):
        intr.frame_ids = []
        intr.per_frame_errors = []
    other.calibration_info = {}
    image = np.zeros((24, 32, 3), np.uint8)
    preview = PreviewPair("rgb 0010", "left", "right", image, image.copy())
    dialog = CalibrationDiagnosticsDialog({"rgb": diagnostic_pair, "空组": other}, [preview])
    dialog.resize(360, 300)
    dialog.show()
    app.processEvents()
    signals = QSignalSpy(dialog.preview_requested)
    assert dialog._preview_status.text() == "未选择帧"
    assert not dialog._btn_preview.isEnabled()
    dialog._table.setCurrentCell(0, 0)
    assert dialog._preview_status.text() == "无匹配原图"
    assert "无可可靠匹配" in dialog._btn_preview.toolTip()
    assert not dialog._btn_preview.isEnabled()
    dialog._table.sortItems(1, Qt.AscendingOrder)
    assert dialog._table.item(0, 0).text() == "10"
    dialog._table.setCurrentCell(0, 1)
    assert dialog._preview_status.text() == "原图可用"
    assert dialog._btn_preview.isEnabled()
    assert dialog._btn_preview.toolTip() == preview.key
    before = [[dialog._table.item(row, col).text() for col in range(8)]
              for row in range(dialog._table.rowCount())]
    dialog._btn_details.click()
    app.processEvents()
    assert before == [[dialog._table.item(row, col).text() for col in range(8)]
                      for row in range(dialog._table.rowCount())]
    assert dialog._table.item(0, 7).text() == "0.350"
    assert dialog._table.item(0, 5).text() == "是"
    dialog._btn_preview.click()
    assert signals.count() == 1
    key, images = signals.at(0)
    assert key == preview.key
    assert images["left"] is preview.left and images["right"] is preview.right
    dialog._table.clearSelection()
    assert not dialog._btn_preview.isEnabled()
    assert dialog._preview_status.text() == "未选择帧"
    dialog._request_preview()
    assert signals.count() == 1
    dialog._combo_pair.setCurrentIndex(1)
    app.processEvents()
    assert dialog._empty_state.isVisible()
    assert "x 坐标" not in dialog._notes.text()
    assert dialog._btn_details.isChecked()
    dialog._combo_pair.setCurrentIndex(0)
    app.processEvents()
    assert not dialog._empty_state.isVisible()
    assert dialog._table.currentRow() == -1
    assert not dialog._btn_preview.isEnabled()
    assert dialog._preview_status.text() == "未选择帧"
