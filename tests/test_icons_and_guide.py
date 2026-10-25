"""Bundled icons and truthful, navigable capture guidance."""

from pathlib import Path
import os

from PySide6.QtCore import QSize, QPoint, QPointF, Qt
from PySide6.QtGui import QIcon
from PySide6.QtTest import QTest

from multicamera.ui import theme
from multicamera.ui.widgets.capture_guide import (
    CaptureSequence, CaptureSequenceWidget, CaptureStatus, build_multimodal_sequence,
)


def test_bundled_icons_render_all_states_and_high_dpi(app):
    directory = Path(theme.__file__).with_name('icons')
    assert (directory / 'LICENSE').exists()
    for source in directory.glob('*.svg'):
        icon = theme.app_icon(source.stem)
        assert not icon.isNull()
        assert icon is theme.app_icon(source.stem)
        for mode in (QIcon.Normal, QIcon.Disabled, QIcon.Active):
            pixmap = icon.pixmap(QSize(20, 20), 2., mode)
            assert pixmap.width() == 40
            image = pixmap.toImage()
            assert any(image.pixelColor(x, y).alpha() for x in range(40) for y in range(40))
    icon = theme.app_icon('camera', '#ffffff')
    assert icon.pixmap(20, 20, QIcon.Normal).toImage() != icon.pixmap(20, 20, QIcon.Disabled).toImage()


def test_guide_menu_jump_and_pending_search_do_not_capture(app):
    theme.configure_application_style(app)
    sequence = CaptureSequence(build_multimodal_sequence())
    widget = CaptureSequenceWidget()
    widget.set_sequence(sequence)
    events = []
    widget.user_navigated.connect(lambda: events.append(True))
    try:
        widget._step_actions[4].trigger()
        assert sequence.current_index == 4 and sequence.done_count == 0
        assert '05' in widget._step_actions[4].text()
        sequence.advance()
        widget.refresh()
        assert '已采' in widget._step_actions[4].text()
        widget._btn_skip.click()
        assert sequence.targets[5].status == CaptureStatus.SKIPPED
        assert '跳过' in widget._step_actions[5].text()
        widget._pending_action.trigger()
        assert sequence.targets[sequence.current_index].status == CaptureStatus.PENDING
        assert '当前' in widget._step_actions[sequence.current_index].text()
        assert sequence.done_count == 1 and events
        assert '已采 1' in widget._large_guide.accessibleName()
        directory = os.environ.get('MULTICAMERA_UI_QA_DIR')
        if directory:
            destination = Path(directory)
            destination.mkdir(parents=True, exist_ok=True)
            widget.set_compact_mode(True)
            widget.resize(360, widget.sizeHint().height())
            widget.show()
            app.processEvents()
            widget.grab().save(str(destination / 'guide-360.png'))
            widget._step_menu.popup(widget.mapToGlobal(widget.rect().bottomLeft()))
            app.processEvents()
            widget._step_menu.grab().save(str(destination / 'guide-step-menu.png'))
            widget._step_menu.hide()
    finally:
        widget.deleteLater()


def test_browsing_all_steps_is_not_completion(app):
    sequence = CaptureSequence(build_multimodal_sequence())
    widget = CaptureSequenceWidget()
    widget.set_sequence(sequence)
    try:
        for _ in range(sequence.total):
            widget._btn_next.click()
        assert sequence.finished and sequence.done_count == 0
        assert widget._large_guide.completion_title == '引导已浏览完'
        assert widget._pending_action.isEnabled()
        for target in sequence.targets:
            target.status = CaptureStatus.DONE
        widget.refresh()
        assert widget._large_guide.completion_title == '采图计划已完成'
        assert not widget._pending_action.isEnabled()
    finally:
        widget.deleteLater()


def test_grid_click_selects_pending_pose_without_changing_records(app):
    theme.configure_application_style(app)
    sequence = CaptureSequence()
    widget = CaptureSequenceWidget()
    widget.set_sequence(sequence)
    widget.set_compact_mode(True)
    widget.resize(300, widget.sizeHint().height())
    widget.show()
    app.processEvents()
    diagram = widget._large_guide
    sequence.targets[3].status = CaptureStatus.DONE
    widget.refresh()
    before = [target.status for target in sequence.targets]
    events = []
    widget.user_navigated.connect(lambda: events.append(True))
    try:
        rect = diagram.grid_rect()
        point = QPoint(int(rect.center().x()), int(rect.y() + rect.height() / 6))
        assert diagram.zone_at(QPointF(point)) == (0, 1)
        QTest.mouseClick(diagram, Qt.LeftButton, pos=point)
        assert sequence.current_index == 17
        assert [target.status for target in sequence.targets] == before
        assert events == [True]
        assert diagram.zone_at(QPointF(-1, -1)) is None
        directory = os.environ.get('MULTICAMERA_UI_QA_DIR')
        if directory:
            destination = Path(directory)
            destination.mkdir(parents=True, exist_ok=True)
            for width in (300, 360, 460):
                widget.resize(width, widget.height())
                app.processEvents()
                widget.grab().save(str(destination / f'guide-tilt-{width}.png'))
    finally:
        widget.close()
        widget.deleteLater()


def test_expanded_guide_shares_navigation_and_closes_on_task_hide(app):
    theme.configure_application_style(app)
    sequence = CaptureSequence()
    widget = CaptureSequenceWidget()
    widget.set_sequence(sequence)
    widget.set_compact_mode(True)
    widget.show()
    app.processEvents()
    try:
        assert widget._large_guide.minimumHeight() >= 200
        widget._btn_expand.click()
        app.processEvents()
        dialog = widget._detail_dialog
        assert dialog is not None and dialog.isVisible()
        assert dialog.canvas.width() > 600
        dialog.buttons[1].click()
        assert sequence.current_index == 1 and sequence.done_count == 0
        assert dialog.canvas._target is sequence.current
        sequence.advance()
        widget.refresh()
        assert dialog.canvas._target is sequence.current
        assert dialog.canvas._targets is sequence.targets
        directory = os.environ.get('MULTICAMERA_UI_QA_DIR')
        if directory:
            destination = Path(directory)
            destination.mkdir(parents=True, exist_ok=True)
            dialog.grab().save(str(destination / 'guide-expanded.png'))
        widget.hide()
        assert widget._detail_dialog is None and not dialog.isVisible()
        widget.show()
        widget._btn_expand.click()
        assert widget._detail_dialog is not dialog
    finally:
        widget.close()
        widget.deleteLater()
