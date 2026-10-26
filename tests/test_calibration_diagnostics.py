"""Source-ID joins, numeric sorting, and read-only diagnostic navigation."""

import copy

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QAbstractItemView

from multicamera.calibration.models import CameraIntrinsics, StereoCalibResult, StereoPairCalibration
from multicamera.ui.theme import PRIMARY, configure_application_style
from multicamera.ui.widgets.calibration_diagnostics import CalibrationDiagnosticsDialog
from multicamera.ui.widgets.pair_gallery import PreviewPair


@pytest.fixture
def pair():
    def intrinsic(ids, errors):
        return CameraIntrinsics(
            np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
            np.zeros(5), (640, 480), frame_ids=ids, per_frame_errors=errors,
            per_frame_error_metric="euclidean_rms_px",
        )

    return StereoPairCalibration(
        "rgb", "left", "right", intrinsic(["0002", "0010"], [.25, 12.5]),
        intrinsic(["0002", "0007"], [.5, 2.]),
        StereoCalibResult(np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3)),
        calibration_info={"selection": {
            "left": {"used_ids": ["0002", "0010"], "excluded_ids": ["0007"]},
            "right": {"used_ids": ["0002", "0007"], "excluded_ids": ["0010"]},
            "stereo": {"used_ids": ["0002"], "excluded_ids": ["0007", "0010", "0020"]},
        }},
    )


def preview(pair, frame_id="0002", name=None):
    return PreviewPair(
        f"{name or pair.pair_name} {frame_id}", pair.left_id, pair.right_id,
        np.full((24, 32, 3), 40, np.uint8), np.full((24, 32, 3), 90, np.uint8),
    )


def row_for(dialog, frame_id):
    return next(row for row in range(dialog._table.rowCount())
                if dialog._table.item(row, 0).text() == frame_id)


def texts(dialog, frame_id):
    row = row_for(dialog, frame_id)
    return [dialog._table.item(row, column).text() for column in range(7)]


def test_gapped_ids_join_errors_and_selection_without_inventing_frames(app, pair):
    dialog = CalibrationDiagnosticsDialog({"rgb": pair}, [])
    assert dialog.windowTitle() == "逐帧误差检查"
    assert dialog._table.rowCount() == 4
    assert texts(dialog, "0002")[1:6] == ["0.250", "0.500", "是", "是", "是"]
    assert texts(dialog, "0007")[1:6] == ["—", "2.000", "否", "是", "否"]
    assert texts(dialog, "0010")[1:6] == ["12.500", "—", "是", "否", "否"]
    assert texts(dialog, "0020")[1:6] == ["—", "—", "未记录", "未记录", "否"]
    assert "不是独立验证" in dialog._notes.text()
    assert "不是零误差" in dialog._notes.text()
    assert dialog._table.editTriggers() == QAbstractItemView.NoEditTriggers


@pytest.mark.parametrize("order,expected", [
    (Qt.AscendingOrder, ["0002", "0010"]),
    (Qt.DescendingOrder, ["0010", "0002"]),
])
def test_rms_sorts_numerically_and_preview_tracks_sorted_source(app, pair, order, expected):
    frames = [preview(pair, "0002"), preview(pair, "0010")]
    before = [(frame.left.copy(), frame.right.copy()) for frame in frames]
    dialog = CalibrationDiagnosticsDialog({"rgb": pair}, frames)
    dialog._table.sortItems(1, order)
    ordered = [dialog._table.item(row, 0).text() for row in range(dialog._table.rowCount())
               if dialog._table.item(row, 1).text() != "—"]
    assert ordered == expected
    signals = QSignalSpy(dialog.preview_requested)
    dialog._table.setCurrentCell(row_for(dialog, "0010"), 1)
    assert dialog._btn_preview.isEnabled()
    dialog._btn_preview.click()
    assert signals.count() == 1
    key, images = signals.at(0)
    assert key == "rgb 0010"
    assert images[frames[1].left_id] is frames[1].left and images[frames[1].right_id] is frames[1].right
    for frame, (left, right) in zip(frames, before):
        np.testing.assert_array_equal(frame.left, left)
        np.testing.assert_array_equal(frame.right, right)


def test_double_click_requests_matching_source_not_current_sort_position(app, pair):
    frame = preview(pair, "0007")
    dialog = CalibrationDiagnosticsDialog({"rgb": pair}, [frame])
    dialog.show()
    app.processEvents()
    dialog._table.sortItems(2, Qt.DescendingOrder)
    row = row_for(dialog, "0007")
    signals = QSignalSpy(dialog.preview_requested)
    rect = dialog._table.visualItemRect(dialog._table.item(row, 0))
    QTest.mouseClick(dialog._table.viewport(), Qt.LeftButton, pos=rect.center())
    QTest.mouseDClick(dialog._table.viewport(), Qt.LeftButton, pos=rect.center())
    assert signals.count() == 1
    assert signals.at(0)[0] == frame.key


def test_missing_preview_disables_button_and_double_click_does_not_emit(app, pair):
    dialog = CalibrationDiagnosticsDialog({"rgb": pair}, [preview(pair)])
    dialog._table.setCurrentCell(row_for(dialog, "0010"), 0)
    assert not dialog._btn_preview.isEnabled()
    signals = QSignalSpy(dialog.preview_requested)
    dialog._table.cellDoubleClicked.emit(row_for(dialog, "0010"), 0)
    dialog._btn_preview.click()
    assert signals.count() == 0


def test_short_error_array_and_nonfinite_values_are_not_zero(app, pair):
    pair.left_intrinsics.per_frame_errors = [float("nan")]
    pair.right_intrinsics.per_frame_errors = [0., float("inf")]
    dialog = CalibrationDiagnosticsDialog({"rgb": pair}, [])
    assert texts(dialog, "0002")[1:3] == ["—", "0.000"]
    assert texts(dialog, "0007")[2] == "—"
    assert texts(dialog, "0010")[1] == "—"
    assert "不表示零误差" in dialog._table.item(row_for(dialog, "0010"), 1).toolTip()


def test_multiple_groups_do_not_cross_match_identical_frame_ids(app, pair):
    other = copy.deepcopy(pair)
    other.pair_name, other.left_id, other.right_id = "ir", "ir_left", "ir_right"
    rgb, ir = preview(pair), preview(other)
    dialog = CalibrationDiagnosticsDialog({"rgb": pair, "ir": other}, [rgb, ir])
    signals = QSignalSpy(dialog.preview_requested)
    dialog._combo_pair.setCurrentIndex(1)
    assert not dialog._btn_preview.isEnabled()
    dialog._table.setCurrentCell(row_for(dialog, "0002"), 0)
    dialog._btn_preview.click()
    assert signals.at(0)[0] == "ir 0002"
    assert signals.at(0)[1][ir.left_id] is ir.left
    dialog._combo_pair.setCurrentIndex(0)
    dialog._table.setCurrentCell(row_for(dialog, "0002"), 0)
    dialog._btn_preview.click()
    assert signals.at(1)[0] == "rgb 0002"


def test_legacy_json_only_shows_ordinal_errors_never_maps_preview(app, pair):
    data = pair.to_dict()
    data.pop("calibration_info")
    for side in ("left_intrinsics", "right_intrinsics"):
        data[side].pop("frame_ids")
        data[side].pop("per_frame_error_metric")
    legacy = StereoPairCalibration.from_dict(data)
    dialog = CalibrationDiagnosticsDialog({"rgb": legacy}, [preview(pair, "0001")])
    assert dialog._table.rowCount() == 2
    assert texts(dialog, "参与序号 1")[1:6] == ["0.250", "0.500", "是", "是", "未记录"]
    assert "仅按参与序号" in dialog._notes.text()
    assert "不映射原图" in dialog._notes.text()
    assert "旧指标口径未记录" in dialog._notes.text()
    dialog._table.selectRow(0)
    assert not dialog._btn_preview.isEnabled()


def test_one_legacy_side_does_not_assign_ordinal_error_to_known_id(app, pair):
    pair.left_intrinsics.frame_ids = None
    dialog = CalibrationDiagnosticsDialog({"rgb": pair}, [preview(pair)])
    assert texts(dialog, "0002")[1] == "—"
    assert texts(dialog, "参与序号 1")[1] == "0.250"
    dialog._table.setCurrentCell(row_for(dialog, "0002"), 0)
    assert not dialog._btn_preview.isEnabled()


def test_live_padded_preview_matches_unpadded_source_id(app, pair):
    pair.left_intrinsics.frame_ids = ["2", "10"]
    pair.right_intrinsics.frame_ids = ["2", "7"]
    pair.calibration_info = {}
    frame = preview(pair, "0010")
    dialog = CalibrationDiagnosticsDialog({"rgb": pair}, [frame])
    assert [dialog._table.item(row, 0).text() for row in range(3)] == ["2", "7", "10"]
    dialog._table.setCurrentCell(row_for(dialog, "10"), 0)
    signals = QSignalSpy(dialog.preview_requested)
    dialog._btn_preview.click()
    assert signals.at(0)[0] == "rgb 0010"
    assert texts(dialog, "10")[5] == "未记录"


def test_distinct_padded_ids_and_ambiguous_previews_are_not_aliased(app, pair):
    pair.left_intrinsics.frame_ids = ["2", "0002"]
    pair.right_intrinsics.frame_ids = []
    pair.calibration_info = {}
    frame = preview(pair, "0002")
    dialog = CalibrationDiagnosticsDialog({"rgb": pair}, [frame])
    dialog._table.setCurrentCell(row_for(dialog, "2"), 0)
    assert not dialog._btn_preview.isEnabled()
    dialog._table.setCurrentCell(row_for(dialog, "0002"), 0)
    assert dialog._btn_preview.isEnabled()
    duplicate = CalibrationDiagnosticsDialog({"rgb": pair}, [frame, frame])
    duplicate._table.setCurrentCell(row_for(duplicate, "0002"), 0)
    assert not duplicate._btn_preview.isEnabled()


def test_wrong_camera_ids_and_empty_images_never_enable_preview(app, pair):
    wrong = preview(pair)
    wrong.left_id = "someone_else"
    empty = preview(pair)
    empty.left = np.zeros((0, 0, 3), np.uint8)
    dialog = CalibrationDiagnosticsDialog({"rgb": pair}, [wrong, empty])
    dialog._table.setCurrentCell(row_for(dialog, "0002"), 0)
    assert not dialog._btn_preview.isEnabled()


def test_empty_pairs_are_safe_and_close_button_works(app):
    dialog = CalibrationDiagnosticsDialog({}, [])
    dialog.show()
    assert dialog._table.rowCount() == 0
    assert not dialog._btn_preview.isEnabled()
    assert "暂无" in dialog._notes.text()
    dialog._btn_close.click()
    assert not dialog.isVisible()


def test_stereo_rectification_residuals_use_source_ids_and_numeric_sort(app, pair):
    pair.calibration_info['rectification'] = {
        'coordinate': 'x', 'per_frame': [
            {'frame_id': '0002', 'rms_px': 10.}, {'frame_id': '0007', 'rms_px': 2.},
        ],
    }
    dialog = CalibrationDiagnosticsDialog({'rgb': pair}, [])
    assert dialog._table.item(row_for(dialog, '0002'), 7).text() == '10.000'
    assert dialog._table.item(row_for(dialog, '0007'), 7).text() == '2.000'
    assert dialog._table.item(row_for(dialog, '0010'), 7).text() == '—'
    dialog._table.sortItems(7, Qt.AscendingOrder)
    assert dialog._table.item(0, 0).text() == '0007'
    assert 'x 坐标' in dialog._notes.text()


def test_selected_diagnostic_row_has_readable_primary_background(app, pair):
    configure_application_style(app)
    dialog = CalibrationDiagnosticsDialog({'rgb': pair}, [])
    dialog.show()
    dialog._table.setCurrentCell(0, 0)
    app.processEvents()
    rect = dialog._table.visualItemRect(dialog._table.item(0, 0))
    image = dialog._table.viewport().grab().toImage()
    scale = image.devicePixelRatio()
    color = image.pixelColor(round((rect.x() + 5) * scale), round((rect.y() + 5) * scale))
    assert color.name() == PRIMARY


@pytest.mark.parametrize("size", [(900, 600), (480, 400)])
def test_dialog_fits_small_screens_with_scrollable_table(app, pair, size):
    configure_application_style(app)
    dialog = CalibrationDiagnosticsDialog({"rgb": pair}, [])
    dialog.resize(*size)
    dialog.show()
    app.processEvents()
    assert dialog.width() <= size[0] and dialog.height() <= size[1]
    for widget in (dialog._combo_pair, dialog._notes, dialog._table,
                   dialog._btn_preview, dialog._btn_close):
        assert dialog.rect().contains(widget.geometry().topLeft())
        assert dialog.rect().contains(widget.geometry().bottomRight())
    assert dialog._notes.geometry().bottom() < dialog._table.geometry().top()
    assert dialog._table.geometry().bottom() < dialog._btn_close.geometry().top()
