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


def assert_reachable(dialog, row, controls):
    for control in controls:
        assert control.isVisible()
        rect = QRect(control.mapTo(row, QPoint()), control.size())
        assert row.rect().contains(rect), (control.objectName(), rect, row.rect())
        assert control.width() >= control.minimumSizeHint().width()
        assert control.height() >= control.minimumSizeHint().height()
    for page in range(2):
        scroll = dialog._tabs.widget(page)
        assert scroll.horizontalScrollBar().maximum() == 0
    save = dialog._buttons.button(QDialogButtonBox.Save)
    assert dialog.rect().contains(QRect(save.mapTo(dialog, QPoint()), save.size()))


@pytest.mark.parametrize('style_name', ['Windows', 'Fusion'])
@pytest.mark.parametrize('size', [(900, 480), (901, 641)])
def test_large_platform_controls_stack_then_restore_without_clipping(app, style_name, size):
    if style_name not in QStyleFactory.keys():
        pytest.skip(f'{style_name} Qt style is unavailable')
    configure_application_style(app)
    dialog = ConfigDialog(StreamManager(), CharucoBoard(CharucoBoardConfig()))
    style = QStyleFactory.create(style_name)
    style.setParent(dialog)
    for widget in [dialog, *dialog.findChildren(type(dialog._board_columns))]:
        widget.setStyle(style)
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


def test_font_change_reflows_existing_open_configuration(app):
    configure_application_style(app)
    dialog = ConfigDialog(StreamManager(), CharucoBoard(CharucoBoardConfig()))
    dialog._combo_left_camera.setMinimumContentsLength(22)
    dialog._combo_right_camera.setMinimumContentsLength(22)
    dialog.resize(900, 640)
    dialog.show()
    settle(app)
    row = dialog._stereo_service_picker
    assert row._box.direction() == QBoxLayout.LeftToRight
    dialog.setStyleSheet('QWidget {font-size:22px;} QComboBox {font-size:24px;}')
    settle(app)
    assert row._box.direction() == QBoxLayout.TopToBottom
    assert dialog._tabs.widget(0).horizontalScrollBar().maximum() == 0
    dialog.setStyleSheet('')
    settle(app)
    assert row._box.direction() == QBoxLayout.LeftToRight
    assert dialog._tabs.widget(0).horizontalScrollBar().maximum() == 0
