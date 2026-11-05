"""Explicit intent selection before replacing a session workspace."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QPushButton,
                              QScrollArea, QVBoxLayout)

from ..theme import app_icon


class SessionOpenDialog(QDialog):
    def __init__(self, name, summary, differences, *, can_continue, parent=None):
        super().__init__(parent)
        self.mode = None
        self.setWindowTitle('打开历史会话')
        self.resize(640, 460)
        layout = QVBoxLayout(self)
        heading = QLabel(name)
        heading.setWordWrap(True)
        layout.addWidget(heading)
        details = QLabel(summary + '\n\n当前配置差异\n' +
                         ('\n'.join('• ' + item for item in differences) if differences
                          else '未发现配置差异；在线续采仍须核对实物设备及镜头。'))
        details.setWordWrap(True)
        details.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(details)
        layout.addWidget(scroll, 1)
        note = QLabel('离线：恢复照片和结果，不连接相机。\n'
                      '在线：恢复后核对原设备、镜头和分辨率，补齐地址，继续 RGB+IR/辅助采集。\n'
                      '历史文件不删除；已保存照片仍需复核纳入。取消续采配置后保留离线会话。')
        note.setWordWrap(True)
        layout.addWidget(note)
        if not can_continue:
            reason = QLabel('在线续采不可用：需要有效且具有相机绑定的 RGB 双目标定结果。')
            reason.setWordWrap(True)
            layout.addWidget(reason)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Cancel).setText('取消')
        buttons.button(QDialogButtonBox.Cancel).setAutoDefault(False)
        self.offline = QPushButton('离线查看 / 重新标定')
        self.offline.setIcon(app_icon('folder-open'))
        self.online = QPushButton('连接原设备继续采集')
        self.online.setIcon(app_icon('plug'))
        self.online.setEnabled(can_continue)
        self.online.setAutoDefault(False)
        buttons.addButton(self.offline, QDialogButtonBox.ActionRole)
        buttons.addButton(self.online, QDialogButtonBox.ActionRole)
        self.offline.clicked.connect(lambda: self._choose('offline'))
        self.online.clicked.connect(lambda: self._choose('online'))
        buttons.rejected.connect(self.reject)
        self.offline.setDefault(True)
        self.offline.setFocus()
        layout.addWidget(buttons)

    def _choose(self, mode):
        if mode == 'online' and not self.online.isEnabled():
            return
        self.mode = mode
        self.accept()
