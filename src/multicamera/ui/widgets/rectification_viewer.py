"""Resizable rectification inspection with fit and pixel-level zoom."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout,
)

from ..theme import app_icon


class RectificationViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('校正图检查')
        self.setMinimumSize(360, 260)
        available = self.screen().availableGeometry()
        self.resize(min(1100, available.width() - 32), min(720, available.height() - 32))
        self._pixmap = QPixmap()
        self._scale = 1.
        self._fit = True
        layout = QVBoxLayout(self)
        self.context = QLabel()
        self.context.setWordWrap(True)
        self.context.setTextFormat(Qt.PlainText)
        layout.addWidget(self.context)
        controls = QHBoxLayout()
        for icon, title, callback in (
            ('maximize-2', '适应窗口', self.fit),
            ('zoom-out', '缩小', lambda: self.zoom(.8)),
            ('zoom-in', '放大', lambda: self.zoom(1.25)),
            ('scan-line', '原始像素', self.actual_size),
        ):
            button = QToolButton()
            button.setIcon(app_icon(icon))
            button.setToolTip(title)
            button.setAccessibleName(title)
            button.clicked.connect(callback)
            controls.addWidget(button)
        self.zoom_label = QLabel()
        controls.addWidget(self.zoom_label)
        controls.addStretch()
        close = QToolButton()
        close.setIcon(app_icon('x'))
        close.setToolTip('关闭校正图检查')
        close.setAccessibleName('关闭校正图检查')
        close.clicked.connect(self.close)
        controls.addWidget(close)
        layout.addLayout(controls)
        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignCenter)
        self.image = QLabel()
        self.image.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.scroll.setWidget(self.image)
        layout.addWidget(self.scroll, 1)
        note = QLabel('左目在左，右目在右；参考线不代表匹配已通过。')
        note.setWordWrap(True)
        layout.addWidget(note)

    def set_image(self, image, context):
        self._pixmap = QPixmap.fromImage(image)
        self.context.setText(context)
        self._render()

    def clear_image(self):
        self._pixmap = QPixmap()
        self.image.clear()
        self.image.resize(0, 0)
        self.context.clear()
        self.zoom_label.clear()

    def fit(self):
        self._fit = True
        self._render()

    def actual_size(self):
        self._fit, self._scale = False, 1.
        self._render()

    def zoom(self, factor):
        self._fit = False
        self._scale = min(4., max(.05, self._scale * factor))
        self._render()

    def _render(self):
        if self._pixmap.isNull():
            return
        if self._fit:
            size = self.scroll.viewport().size()
            self._scale = max(.01, min((size.width() - 4) / self._pixmap.width(),
                                      (size.height() - 4) / self._pixmap.height(), 1.))
        size = self._pixmap.size() * self._scale
        self.image.setPixmap(self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.image.resize(size)
        self.zoom_label.setText(f'{self._scale:.0%}')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'scroll') and self._fit:
            self._render()
