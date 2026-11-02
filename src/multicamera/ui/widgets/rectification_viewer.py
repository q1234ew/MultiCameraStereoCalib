"""Resizable rectification inspection with fit and pixel-level zoom."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QToolButton,
    QVBoxLayout,
)

from ..theme import TEXT_DIM, app_icon


class RectificationViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('校正图检查')
        self.setWindowIcon(app_icon('scan-line'))
        self.setMinimumSize(360, 260)
        available = self.screen().availableGeometry()
        self.resize(min(1100, available.width() - 32), min(720, available.height() - 32))
        self._pixmap = QPixmap()
        self._scale = 1.
        self._fit = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)
        title = QLabel('校正图检查')
        title.setObjectName('workbenchSectionTitle')
        layout.addWidget(title)
        self.context_scroll = QScrollArea()
        self.context_scroll.setWidgetResizable(True)
        self.context_scroll.setMaximumHeight(72)
        self.context_scroll.setMinimumHeight(38)
        self.context = QLabel()
        self.context.setWordWrap(True)
        self.context.setTextFormat(Qt.PlainText)
        self.context.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.context.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.context.setAccessibleName('当前校正图来源及状态')
        self.context_scroll.setWidget(self.context)
        layout.addWidget(self.context_scroll)
        controls = QHBoxLayout()
        self._image_buttons = []
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
            button.setFixedSize(32, 32)
            button.clicked.connect(callback)
            controls.addWidget(button)
            self._image_buttons.append(button)
        self._image_buttons[0].setCheckable(True)
        self._image_buttons[3].setCheckable(True)
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
        self.scroll.setAccessibleName('校正图像画布')
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setWordWrap(True)
        self.image.setAccessibleName('左右校正图像')
        self.image.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.scroll.setWidget(self.image)
        layout.addWidget(self.scroll, 1)
        footer = QFrame()
        footer.setObjectName('actionFooter')
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(0, 8, 0, 0)
        self.image_info = QLabel()
        self.image_info.setStyleSheet(f'color:{TEXT_DIM};')
        footer_layout.addWidget(self.image_info)
        note = QLabel('左目在左，右目在右；参考线不代表匹配已通过。')
        note.setWordWrap(True)
        note.setStyleSheet(f'color:{TEXT_DIM};')
        footer_layout.addWidget(note)
        layout.addWidget(footer)
        self.clear_image()

    def set_image(self, image, context):
        self._pixmap = QPixmap.fromImage(image)
        self.context.setText(context)
        self.context.setToolTip(context)
        self.context_scroll.verticalScrollBar().setValue(0)
        self.scroll.setWidgetResizable(False)
        self.image.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._render()

    def clear_image(self):
        self._pixmap = QPixmap()
        self.image.setText('暂无可检查的校正图像')
        self.image.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.scroll.setWidgetResizable(True)
        self.context.clear()
        self.context.setToolTip('')
        self.zoom_label.setText('—')
        self.image_info.setText('暂无图像')
        self._update_controls()

    def _update_controls(self):
        available = not self._pixmap.isNull()
        for button in self._image_buttons:
            button.setEnabled(available)
        self._image_buttons[0].setChecked(available and self._fit)
        self._image_buttons[3].setChecked(available and not self._fit and self._scale == 1.)

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
        self.image_info.setText(
            f'校正拼接图 {self._pixmap.width()} × {self._pixmap.height()} px'
        )
        self._update_controls()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'scroll') and self._fit:
            # Wait for the scroll viewport to finish its own resize before fitting.
            QTimer.singleShot(0, self._render)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._render)
