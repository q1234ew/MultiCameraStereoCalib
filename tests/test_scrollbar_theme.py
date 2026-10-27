"""Scrollbar page regions must not fall back to native hatched fills."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollBar, QWidget

from multicamera.ui.theme import configure_application_style


@pytest.mark.parametrize('orientation', [Qt.Vertical, Qt.Horizontal])
def test_scrollbar_page_regions_have_uniform_dark_background(app, orientation):
    configure_application_style(app)
    parent = QWidget()
    parent.resize(240, 240)
    bar = QScrollBar(orientation, parent)
    bar.setGeometry(10, 10, 8, 200) if orientation == Qt.Vertical else bar.setGeometry(10, 10, 200, 8)
    bar.setRange(0, 1000)
    bar.setPageStep(100)
    bar.setValue(500)
    try:
        parent.show()
        app.processEvents()
        picture = bar.grab().toImage()
        ratio = picture.devicePixelRatio()
        for points in (range(10, 45), range(155, 190)):
            colors = set()
            for offset in points:
                x, y = (4, offset) if orientation == Qt.Vertical else (offset, 4)
                color = picture.pixelColor(round(x * ratio), round(y * ratio))
                colors.add(color.rgb())
                assert max(color.red(), color.green(), color.blue()) < 100
            assert len(colors) == 1
    finally:
        parent.close()
        parent.deleteLater()
