"""Single-pair review window integrated with the existing diagnostic table."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QPushButton, QSplitter,
    QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)
import numpy as np

from ..theme import TEXT_DIM, WARNING, app_icon
from .calibration_diagnostics import CalibrationDiagnosticsDialog


class PairImages(QWidget):
    def __init__(self, preview=None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.labels = []
        self.pixmaps = [None, None]
        for title in ('RGB 左', 'RGB 右'):
            label = QLabel(title)
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(100, 90)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            label.setMaximumHeight(170)
            layout.addWidget(label, 1)
            self.labels.append(label)
        if preview:
            self.set_preview(preview)

    def set_preview(self, preview):
        for index, (label, array, title) in enumerate(zip(self.labels,
                                        (preview.left, preview.right) if preview else (None, None),
                                        ('左原图', '右原图'))):
            self.pixmaps[index] = None
            label.setText(title + '不可用')
            if array is None or not array.size:
                continue
            if array.ndim == 2:
                rgb = np.ascontiguousarray(np.repeat(array[:, :, None], 3, axis=2))
            else:
                rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                           QImage.Format_RGB888).copy()
            self.pixmaps[index] = QPixmap.fromImage(image)
            label.setToolTip(title)
        self._scale_images()

    def _scale_images(self):
        for label, pixmap in zip(self.labels, self.pixmaps):
            if pixmap is not None:
                label.setPixmap(pixmap.scaled(max(1, label.width()), max(1, label.height()),
                                             Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_images()


class CalibrationReviewDialog(QDialog):
    reanalysis_requested = Signal()
    participation_requested = Signal(str)
    retake_requested = Signal(object)
    task_requested = Signal(object)
    preview_requested = Signal(str, object)

    def __init__(self, name, result, previews, parent=None):
        super().__init__(parent)
        self.name, self.rows, self.tasks = name, [], []
        self._previews = previews
        self.setWindowTitle('逐帧误差检查与改进')
        self.setWindowIcon(app_icon('scan-line'))
        self.setMinimumSize(500, 420)
        available = self.screen().availableGeometry()
        self.resize(min(960, available.width() - 32), min(720, available.height() - 32))
        layout = QVBoxLayout(self)
        self.summary = QLabel(f'{name} · 正在分析当前数据…')
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.PlainText)
        layout.addWidget(self.summary)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        page = QWidget()
        page_layout = QHBoxLayout(page)
        splitter = QSplitter()
        page_layout.addWidget(splitter)
        self.list = QListWidget()
        self.list.setAccessibleName('优先复核图像')
        splitter.addWidget(self.list)
        details = QWidget()
        detail_layout = QVBoxLayout(details)
        self.title = QLabel('选择图像查看依据')
        self.title.setTextFormat(Qt.PlainText)
        detail_layout.addWidget(self.title)
        self.images = PairImages()
        detail_layout.addWidget(self.images)
        self.reason = QLabel()
        self.reason.setWordWrap(True)
        self.reason.setTextFormat(Qt.PlainText)
        detail_layout.addWidget(self.reason)
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        self.metrics.setStyleSheet(f'color:{TEXT_DIM};')
        detail_layout.addWidget(self.metrics)
        self.value = QLabel()
        self.value.setWordWrap(True)
        self.value.setTextFormat(Qt.PlainText)
        self.value.setStyleSheet(f'color:{WARNING};')
        detail_layout.addWidget(self.value)
        actions = QHBoxLayout()
        self.retake = QPushButton('按此构图重拍')
        self.retake.setProperty('class', 'primary')
        self.toggle = QPushButton('暂不参与求解')
        actions.addWidget(self.retake)
        actions.addWidget(self.toggle)
        detail_layout.addLayout(actions)
        self.preview = QPushButton('查看原图')
        detail_layout.addWidget(self.preview)
        detail_layout.addStretch()
        splitter.addWidget(details)
        splitter.setSizes([360, 440])
        self.tabs.addTab(page, '图像复核')
        self.diagnostics = CalibrationDiagnosticsDialog({name: result} if result else {}, previews)
        self.diagnostics.setWindowFlags(Qt.Widget)
        self.diagnostics._btn_close.hide()
        self.diagnostics.preview_requested.connect(self.preview_requested)
        self.tabs.addTab(self.diagnostics, '全部误差（上次结果）')
        task_page = QWidget()
        task_layout = QVBoxLayout(task_page)
        self.task_list = QListWidget()
        self.task_list.setAccessibleName('补拍任务清单')
        task_layout.addWidget(self.task_list)
        self.start_task = QPushButton('开始选中补拍')
        self.start_task.setProperty('class', 'primary')
        task_layout.addWidget(self.start_task)
        self.tabs.addTab(task_page, '补拍清单')
        self.notes = QLabel('原图保留；训练误差不是独立精度验证。')
        self.notes.setWordWrap(True)
        layout.addWidget(self.notes)
        refresh = QPushButton('重新分析当前数据')
        refresh.clicked.connect(lambda: self.reanalysis_requested.emit())
        layout.addWidget(refresh, 0, Qt.AlignRight)
        close = QPushButton('关闭')
        close.clicked.connect(self.reject)
        layout.addWidget(close, 0, Qt.AlignRight)
        self.list.currentRowChanged.connect(self._select)
        self.task_list.currentRowChanged.connect(self._actions)
        self.toggle.clicked.connect(lambda: self.participation_requested.emit(self.current()['id']))
        self.retake.clicked.connect(lambda: self.retake_requested.emit(self.current()))
        self.preview.clicked.connect(self._preview)
        self.start_task.clicked.connect(lambda: self.task_requested.emit(
            self.tasks[self.task_list.currentRow()]))
        self.set_busy()

    def current(self):
        index = self.list.currentRow()
        return self.rows[index] if 0 <= index < len(self.rows) else {}

    def set_busy(self, text='正在分析当前数据…'):
        self.summary.setText(f'{self.name} · {text}')
        for button in (self.toggle, self.retake, self.start_task, self.preview):
            button.setEnabled(False)
        self.rows = []
        self.list.clear()
        self.tasks = []
        self.task_list.clear()
        self.title.setText('等待当前数据分析')
        self.images.set_preview(None)
        self.reason.clear()
        self.metrics.clear()
        self.value.clear()

    def set_analysis(self, report, previews, can_retake, mutable):
        selected = self.current().get('id')
        self._previews = previews
        self._can_retake, self._mutable = can_retake, mutable
        self.rows = sorted(report['rows'], key=lambda row: (
            not bool(row['reasons']), 0 if row['id'].isdecimal() else 1,
            int(row['id']) if row['id'].isdecimal() else 0, row['id']))
        self.list.blockSignals(True)
        self.list.clear()
        for row in self.rows:
            state = ('暂不参与' if row['excluded'] else
                     '参与' if row['stereo_available'] else '仅单目观测')
            reason = row['reasons'][0] if row['reasons'] else '基础检查未发现优先复核项'
            self.list.addItem(f"{row['id']} · {state}\n{reason}")
        self.list.blockSignals(False)
        index = next((i for i, row in enumerate(self.rows) if row['id'] == selected), 0)
        self.list.setCurrentRow(index if self.rows else -1)
        self.tasks = report['tasks']
        self.task_list.clear()
        for task in self.tasks:
            self.task_list.addItem(f"{task['title']} · 建议 {task['count']} 对\n{task['reason']}")
        self.task_list.setCurrentRow(0 if self.tasks else -1)
        self.summary.setText(f"{self.name} · 计划参与 {sum(r['participates'] for r in self.rows)} 对"
                             f" · 建议复核 {report['flagged']} 对")
        self.notes.setText(report['notes'] + (' 离线数据可复核；定向补拍需使用匹配的实时组。'
                                           if not can_retake else ''))
        self._select()

    def _select(self, *_):
        row = self.current()
        self.title.setText('图像 ' + row.get('id', '—'))
        preview = row.get('preview')
        self.images.set_preview(preview)
        self.reason.setText('；'.join(row.get('reasons', [])) or '暂无优先复核原因，仍需结合原图判断。')
        def number(value):
            return f'{value:.3f}' if value is not None and np.isfinite(value) else '—'
        self.metrics.setText('上次结果 · 左 / 右内参 RMS：'
                             + number(row.get('left_rms')) + ' / '
                             + number(row.get('right_rms')) + ' px\n校正残差 RMS：'
                             + number(row.get('alignment_rms')) + ' px')
        self.value.setText(row.get('value', ''))
        self.toggle.setText('恢复参与' if row.get('excluded') else '暂不参与求解')
        self._actions()

    def _actions(self, *_):
        row = self.current()
        mutable = getattr(self, '_mutable', False)
        self.toggle.setEnabled(bool(row) and mutable)
        self.retake.setEnabled(bool(row.get('geometry') and all(row['geometry']))
                               and getattr(self, '_can_retake', False) and mutable)
        self.start_task.setEnabled(0 <= self.task_list.currentRow() < len(self.tasks)
                                   and getattr(self, '_can_retake', False) and mutable)
        self.preview.setEnabled(row.get('preview') is not None)

    def _preview(self):
        preview = self.current().get('preview')
        if preview:
            self.preview_requested.emit(preview.key,
                                        {preview.left_id: preview.left, preview.right_id: preview.right})
