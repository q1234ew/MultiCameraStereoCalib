"""Bounded thumbnail rendering over existing, read-only calibration image buffers."""

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal, Slot, QSignalBlocker
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem,
    QStyle, QToolButton, QVBoxLayout, QWidget,
)

from ..theme import BG_DARK, TEXT_DIM, standard_icon, app_icon
from ...calibration.planar import PlanarDetectionResult


@dataclass
class PreviewPair:
    key: str
    left_id: str
    right_id: str
    left: np.ndarray
    right: np.ndarray
    left_detection: object = None
    right_detection: object = None
    detection_origin: str = '预览重检'
    corner_total: int | None = None
    save_token: str | None = None
    save_state: str = ''
    quality_state: str = ''
    quality_reasons: tuple[str, ...] = ()

    @property
    def display_quality_state(self):
        if isinstance(self.left_detection, PlanarDetectionResult):
            return {'可采纳': '验收通过', '待复核': '验收待复核'}.get(
                self.quality_state, self.quality_state)
        return self.quality_state

    def result_text(self):
        def count(detection):
            return (getattr(detection, 'num_corners', getattr(detection, 'num_points', 0))
                    if detection is not None else '—')
        return (f'{self.key} · {self.detection_origin} L={count(self.left_detection)} '
                f'R={count(self.right_detection)} · {self.save_state} · {self.display_quality_state} '
                + '；'.join(self.quality_reasons))


class PairGallery(QWidget):
    pair_selected = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pairs: list[PreviewPair] = []
        self._start = 0
        self._page_size = 4
        self._selected_key = None
        self._icon_cache = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 0, 6, 0)
        outer.setSpacing(4)
        header = QHBoxLayout()
        self._toggle = QToolButton()
        self._toggle.setText("配对图像")
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setAccessibleName("展开或收起配对缩略图")
        self._toggle.toggled.connect(self._set_expanded)
        header.addWidget(self._toggle)
        self._count = QLabel("0 组")
        self._count.setStyleSheet(f"color:{TEXT_DIM};")
        header.addWidget(self._count)
        self._latest_capture = QToolButton()
        self._latest_capture.setIcon(app_icon('camera'))
        self._latest_capture.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._latest_capture.setText('查看最近采图')
        policy = self._latest_capture.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self._latest_capture.setSizePolicy(policy)
        self._latest_capture.setAccessibleName('查看最近采图原图与检测结果')
        self._latest_capture.clicked.connect(self._select_latest_capture)
        self._latest_capture.hide()
        header.addWidget(self._latest_capture)
        header.addStretch()
        outer.addLayout(header)
        self._body = QWidget()
        row = QHBoxLayout(self._body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self._previous = QToolButton()
        self._previous.setIcon(standard_icon(self, QStyle.SP_ArrowLeft))
        self._previous.setToolTip("上一页配对图像")
        self._previous.setAccessibleName("上一页配对图像")
        self._previous.setFixedSize(30, 36)
        self._previous.clicked.connect(lambda: self._page(-1))
        row.addWidget(self._previous)
        self._list = QListWidget()
        self._list.setObjectName("pairThumbnails")
        self._list.setAccessibleName("配对图像缩略图")
        self._list.setViewMode(QListView.IconMode)
        self._list.setFlow(QListView.LeftToRight)
        self._list.setWrapping(False)
        self._list.setMovement(QListView.Static)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setIconSize(QSize(132, 66))
        self._list.setGridSize(QSize(144, 104))
        self._list.setFixedHeight(114)
        self._list.itemClicked.connect(self._select_pair)
        self._list.itemActivated.connect(self._select_pair)
        row.addWidget(self._list, 1)
        self._next = QToolButton()
        self._next.setIcon(standard_icon(self, QStyle.SP_ArrowRight))
        self._next.setToolTip("下一页配对图像")
        self._next.setAccessibleName("下一页配对图像")
        self._next.setFixedSize(30, 36)
        self._next.clicked.connect(lambda: self._page(1))
        row.addWidget(self._next)
        outer.addWidget(self._body)
        self._empty = QLabel('尚无配对图像')
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setFixedHeight(72)
        self._empty.setStyleSheet(f'color:{TEXT_DIM};')
        self._empty.setAccessibleName('图库为空')
        outer.addWidget(self._empty)
        self._refresh_page()
        self._body.hide()
        self._update_empty_state()

    def set_pairs(self, pairs: list[PreviewPair]):
        if len(pairs) == len(self._pairs) and all(
            new.key == old.key and new.left is old.left and new.right is old.right
            and new.save_state == old.save_state and new.detection_origin == old.detection_origin
            and new.quality_state == old.quality_state and new.quality_reasons == old.quality_reasons
            and new.left_detection is old.left_detection and new.right_detection is old.right_detection
            for new, old in zip(pairs, self._pairs)
        ):
            return
        previous_last = self._pairs[-1].key if self._pairs else None
        self._pairs = list(pairs)
        selected_exists = any(pair.key == self._selected_key for pair in self._pairs)
        if not selected_exists:
            self._selected_key = None
        if not self._pairs:
            self._start = 0
            self._selected_key = None
        elif self._pairs[-1].key != previous_last and not selected_exists:
            self._start = max(0, len(self._pairs) - self._page_size)
        else:
            self._start = min(self._start, max(0, len(self._pairs) - self._page_size))
        self._count.setText(f"{len(self._pairs)} 组")
        latest = next((p for p in reversed(self._pairs) if p.detection_origin == '采图原图'), None)
        self._latest_capture.setVisible(latest is not None)
        self._latest_capture.setToolTip(latest.result_text() if latest else '')
        if latest is not None:
            self._latest_capture.setText(f'最近原图 · {latest.display_quality_state or "待验收"}')
        self._body.setVisible(bool(self._pairs) and self._toggle.isChecked())
        self._update_empty_state()
        self._refresh_page()

    def _update_empty_state(self):
        self._empty.setVisible(not self._pairs and self.height() >= 100)

    def clear_selection(self):
        self._selected_key = None
        self._list.clearSelection()

    def show_first_page(self):
        self._start = 0
        self.clear_selection()
        self._refresh_page()

    @Slot(bool)
    def _set_expanded(self, expanded: bool):
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._body.setVisible(expanded and bool(self._pairs))
        self._refresh_page()

    def _page(self, direction: int):
        self._start = max(0, min(self._start + direction * self._page_size,
                                 max(0, len(self._pairs) - self._page_size)))
        self._refresh_page()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_empty_state()
        count = max(1, min(6, (self.width() - 80) // 148))
        if count != self._page_size:
            self._page_size = count
            self._start = min(self._start, max(0, len(self._pairs) - count))
            self._refresh_page()

    def _refresh_page(self):
        visible = set()
        with QSignalBlocker(self._list):
            self._list.clear()
            if self._toggle.isChecked():
                for index in range(self._start, min(len(self._pairs), self._start + self._page_size)):
                    pair = self._pairs[index]
                    signature = (id(pair.left), id(pair.right))
                    cached = self._icon_cache.get(pair.key)
                    icon = cached[1] if cached is not None and cached[0] == signature else QIcon(self._thumbnail(pair))
                    self._icon_cache[pair.key] = (signature, icon)
                    visible.add(pair.key)
                    item = QListWidgetItem(icon, pair.key)
                    item.setData(Qt.UserRole, index)
                    item.setToolTip(f"{pair.left_id} + {pair.right_id}\n{pair.result_text()}")
                    self._list.addItem(item)
                    item.setSelected(pair.key == self._selected_key)
        self._previous.setEnabled(self._start > 0)
        self._next.setEnabled(self._start + self._page_size < len(self._pairs))
        self._icon_cache = {key: value for key, value in self._icon_cache.items() if key in visible}

    @staticmethod
    def _thumbnail(pair: PreviewPair) -> QPixmap:
        canvas = QPixmap(188, 74)
        canvas.fill(QColor(BG_DARK))
        painter = QPainter(canvas)
        for position, frame in enumerate((pair.left, pair.right)):
            height, width = frame.shape[:2]
            scale = min(92 / max(width, 1), 74 / max(height, 1))
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, size[0], size[1], rgb.strides[0], QImage.Format_RGB888)
            painter.drawImage(position * 96 + (92 - size[0]) // 2, (74 - size[1]) // 2, image)
        painter.end()
        return canvas

    @Slot(QListWidgetItem)
    def _select_pair(self, item):
        index = item.data(Qt.UserRole)
        if index is None or not 0 <= index < len(self._pairs):
            return
        pair = self._pairs[index]
        self._selected_key = pair.key
        self.pair_selected.emit(pair.key, {pair.left_id: pair.left, pair.right_id: pair.right})

    def _select_latest_capture(self):
        pair = next((p for p in reversed(self._pairs) if p.detection_origin == '采图原图'), None)
        if pair is not None:
            self._selected_key = pair.key
            index = next(i for i, value in enumerate(self._pairs) if value is pair)
            self._start = max(0, index - self._page_size + 1)
            self._refresh_page()
            self.pair_selected.emit(pair.key, {pair.left_id: pair.left, pair.right_id: pair.right})
