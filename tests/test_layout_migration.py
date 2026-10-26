"""Explicit migration must not leave QWidgetItem wrappers at reused addresses."""

import os
import subprocess
import sys

from PySide6.QtWidgets import QVBoxLayout, QWidget, QPushButton

from multicamera.ui.widgets.calib_panel import _detach_from_layout


def test_detach_nested_layout_removes_only_target(app):
    parent = QWidget()
    layout = QVBoxLayout(parent)
    nested = QVBoxLayout()
    first, second = QPushButton('first'), QPushButton('second')
    nested.addWidget(first)
    nested.addWidget(second)
    layout.addLayout(nested)
    assert _detach_from_layout(first)
    assert nested.count() == 1 and nested.itemAt(0).widget() is second
    assert not _detach_from_layout(first)
    assert not _detach_from_layout(QWidget())


def test_reused_native_addresses_never_return_layout_item_as_qobject(tmp_path):
    code = '''
import gc
from PySide6.QtCore import QObject, QCoreApplication, QEvent, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QButtonGroup
from multicamera.ui.widgets.calib_panel import _detach_from_layout
app = QApplication([])
bad = []
class Root(QWidget):
    def childEvent(self, event):
        super().childEvent(event)
        if event.added() and not isinstance(event.child(), QObject):
            bad.append(type(event.child()).__name__)
for cycle in range(250):
    root = Root()
    outer = QVBoxLayout(root)
    inner = QVBoxLayout()
    buttons = [QPushButton(str(index)) for index in range(12)]
    for button in buttons:
        inner.addWidget(button)
    outer.addLayout(inner)
    page = QWidget(root)
    target = QVBoxLayout(page)
    for button in buttons:
        assert _detach_from_layout(button)
        target.addWidget(button)
    for index in range(25):
        timer, action, group = QTimer(root), QAction(root), QButtonGroup(root)
    root.deleteLater()
    del timer, action, group, button, buttons, inner, outer, page, target, root
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()
assert not bad, bad
'''
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', code],
                            capture_output=True, text=True, env=environment, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
