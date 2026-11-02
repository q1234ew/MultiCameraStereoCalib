"""Composite configuration fields reflow instead of overflowing at platform font limits."""

import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QBoxLayout, QDialogButtonBox, QStyleFactory

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.streaming.stream_manager import StreamManager
from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.config_dialog import ConfigDialog


def settle(app):
    for _ in range(8):
        app.processEvents()


def use_style(dialog, style_name):
    if style_name not in QStyleFactory.keys():
        pytest.skip(f'{style_name} Qt style is unavailable')
    style = QStyleFactory.create(style_name)
    style.setParent(dialog)
    for widget in [dialog, *dialog.findChildren(type(dialog._board_columns))]:
        widget.setStyle(style)


def required_width(row):
    items = [row._box.itemAt(index) for index in range(row._box.count())]
    margins = row._box.contentsMargins()
    return (sum(item.minimumSize().width() for item in items)
            + max(0, len(items) - 1) * row._box.spacing()
            + margins.left() + margins.right())


def assert_reachable(dialog, row, controls):
    for control in controls:
        assert control.isVisible()
        rect = QRect(control.mapTo(row, QPoint()), control.size())
        assert row.rect().contains(rect), (control.objectName(), rect, row.rect())
        assert control.width() >= control.minimumSizeHint().width()
        assert control.height() >= control.minimumSizeHint().height()
    scroll = dialog._tabs.currentWidget()
    assert scroll.horizontalScrollBar().maximum() == 0
    save = dialog._buttons.button(QDialogButtonBox.Save)
    assert dialog.rect().contains(QRect(save.mapTo(dialog, QPoint()), save.size()))


@pytest.mark.parametrize('style_name', ['Windows', 'Fusion'])
@pytest.mark.parametrize('size', [(900, 480), (901, 641)])
def test_large_platform_controls_stack_then_restore_without_clipping(app, style_name, size):
    configure_application_style(app)
    dialog = ConfigDialog(StreamManager(), CharucoBoard(CharucoBoardConfig()))
    use_style(dialog, style_name)
    dialog.setStyleSheet('QWidget {font-size:18px;} QComboBox {font-size:19px;}')
    dialog._combo_left_camera.setMinimumContentsLength(22)
    dialog._combo_right_camera.setMinimumContentsLength(22)
    dialog._combo_aux_camera.setMinimumContentsLength(30)
    dialog._add_row()
    dialog.focus_auxiliary()
    original = dialog._proposed_configuration()
    dialog.resize(*size)
    dialog.show()
    settle(app)
    for page, row, controls in (
        (0, dialog._stereo_service_picker, (dialog._combo_stream_type,
                                          dialog._combo_left_camera, dialog._combo_right_camera)),
        (1, dialog._aux_service_picker, (dialog._combo_aux_camera,
                                       dialog._btn_use_aux_service, dialog._btn_discover_aux)),
    ):
        dialog._tabs.setCurrentIndex(page)
        settle(app)
        assert dialog.size().toTuple() == size
        assert row._box.direction() == QBoxLayout.TopToBottom
        assert_reachable(dialog, row, controls)
        before = controls[0].mapTo(row, QPoint()).y()
        for control in controls[1:]:
            assert control.mapTo(row, QPoint()).y() > before
            before = control.mapTo(row, QPoint()).y()
        dialog.resize(1800, 900)
        settle(app)
        assert row._box.direction() == QBoxLayout.LeftToRight
        assert_reachable(dialog, row, controls)
        dialog.resize(*size)
        settle(app)
        assert row._box.direction() == QBoxLayout.TopToBottom
        assert_reachable(dialog, row, controls)
    assert dialog._proposed_configuration() == original


@pytest.mark.parametrize('style_name', ['Windows', 'Fusion'])
@pytest.mark.parametrize('page', [0, 1])
@pytest.mark.parametrize('base_stylesheet', [
    '', 'QWidget {font-size:18px;} QComboBox {font-size:19px;}',
])
def test_font_change_reflows_existing_open_configuration(app, style_name, page, base_stylesheet):
    configure_application_style(app)
    dialog = ConfigDialog(StreamManager(), CharucoBoard(CharucoBoardConfig()))
    use_style(dialog, style_name)
    dialog._combo_left_camera.setMinimumContentsLength(22)
    dialog._combo_right_camera.setMinimumContentsLength(22)
    dialog._combo_aux_camera.setMinimumContentsLength(30)
    dialog._add_row()
    if page == 1:
        dialog.focus_auxiliary()
    dialog._tabs.setCurrentIndex(page)
    dialog.setStyleSheet(base_stylesheet)
    dialog.resize(2200, 800)
    dialog.show()
    settle(app)
    row = dialog._aux_service_picker if page == 1 else dialog._stereo_service_picker
    small = required_width(row)
    small_overhead = dialog.width() - row.width()
    large_stylesheet = 'QWidget {font-size:22px;} QComboBox {font-size:24px;}'
    dialog.setStyleSheet(large_stylesheet)
    settle(app)
    large = required_width(row)
    assert large > small
    large_overhead = dialog.width() - row.width()
    small_window = small + small_overhead
    large_window = large + large_overhead
    assert large_window > small_window
    # Select an actual font boundary, not an assumed macOS-sized 900px row.
    dialog.setStyleSheet(base_stylesheet)
    dialog.resize(max(dialog.minimumWidth(), (small_window + large_window) // 2), 640)
    settle(app)
    controls = ((dialog._combo_aux_camera, dialog._btn_use_aux_service, dialog._btn_discover_aux)
                if page == 1 else (dialog._combo_stream_type, dialog._combo_left_camera,
                                   dialog._combo_right_camera))
    original = dialog._proposed_configuration()
    size = dialog.size()
    assert small_window <= dialog.width() < large_window
    assert required_width(row) <= row.width()
    assert row._box.direction() == QBoxLayout.LeftToRight
    assert_reachable(dialog, row, controls)
    dialog.setStyleSheet(large_stylesheet)
    settle(app)
    assert dialog.size() == size
    assert required_width(row) > row.width()
    assert row._box.direction() == QBoxLayout.TopToBottom
    assert_reachable(dialog, row, controls)
    dialog.setStyleSheet(base_stylesheet)
    settle(app)
    assert dialog.size() == size
    assert required_width(row) <= row.width()
    assert row._box.direction() == QBoxLayout.LeftToRight
    assert_reachable(dialog, row, controls)
    assert dialog._proposed_configuration() == original
