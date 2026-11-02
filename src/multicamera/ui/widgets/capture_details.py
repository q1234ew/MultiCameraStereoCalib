"""Read-only, bounded inspection of the latest capture acceptance receipt."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QToolButton, QVBoxLayout,
)

from ..theme import TEXT_DIM, app_icon


class CaptureDetailsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        available = self.screen().availableGeometry()
        self.resize(min(720, available.width() - 32), min(480, available.height() - 32))
        self.setWindowIcon(app_icon('list-checks'))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(12)
        self.heading = QLabel()
        self.heading.setObjectName('workbenchSectionTitle')
        self.heading.setTextFormat(Qt.PlainText)
        layout.addWidget(self.heading)
        self.content = QPlainTextEdit()
        self.content.setReadOnly(True)
        self.content.setAccessibleName('完整原图验收记录')
        self.content.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        layout.addWidget(self.content, 1)
        footer = QFrame()
        footer.setObjectName('actionFooter')
        actions = QHBoxLayout(footer)
        actions.setContentsMargins(0, 10, 0, 0)
        self.copy_button = QToolButton()
        self.copy_button.setIcon(app_icon('copy'))
        self.copy_button.setToolTip('复制完整验收记录')
        self.copy_button.setAccessibleName('复制完整验收记录')
        self.copy_button.clicked.connect(self._copy)
        actions.addWidget(self.copy_button)
        self.status = QLabel()
        self.status.setStyleSheet(f'color:{TEXT_DIM};')
        actions.addWidget(self.status, 1)
        self.close_button = QPushButton('关闭')
        self.close_button.setIcon(app_icon('x'))
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        layout.addWidget(footer)

    def set_details(self, title, text):
        self.setWindowTitle(title)
        self.heading.setText(title)
        self.content.setPlainText(text or '暂无原图验收记录')
        self.copy_button.setEnabled(bool(text))
        self.status.setText('验收记录快照' if text else '暂无记录')

    def _copy(self):
        QApplication.clipboard().setText(self.content.toPlainText())
        self.status.setText('已复制完整记录')
