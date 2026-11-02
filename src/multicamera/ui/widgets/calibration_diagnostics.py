"""Read-only per-frame calibration diagnostics and source-image navigation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout,
)

from ...calibration.models import CameraIntrinsics, StereoPairCalibration
from ..theme import PRIMARY, TEXT_DIM, app_icon
from .pair_gallery import PreviewPair


@dataclass
class _FrameRow:
    label: str
    left_error: float | None
    right_error: float | None
    participation: tuple[bool | None, bool | None, bool | None]
    preview: PreviewPair | None = None
    alignment_error: float | None = None


class _ErrorItem(QTableWidgetItem):
    def __init__(self, value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        valid = math.isfinite(number) and number >= 0
        super().__init__(f"{number:.3f}" if valid else "—")
        self.setData(Qt.UserRole, number if valid else math.inf)
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if not valid:
            self.setToolTip("无值或数值无效，不表示零误差")

    def __lt__(self, other):
        return self.data(Qt.UserRole) < other.data(Qt.UserRole)


class _FrameIdItem(QTableWidgetItem):
    def __lt__(self, other):
        def key(value):
            return (0, int(value), value) if value.isdecimal() else (1, 0, value)

        return key(self.text()) < key(other.text())


def _errors_by_id(intr: CameraIntrinsics) -> dict:
    errors = intr.per_frame_errors if intr.per_frame_errors is not None else []
    return {str(frame_id): errors[index] if index < len(errors) else None
            for index, frame_id in enumerate(intr.frame_ids or [])}


def _participates(selection, side: str, frame_id: str, known_ids=()) -> bool | None:
    info = selection.get(side, {})
    if frame_id in {str(value) for value in info.get("used_ids", [])}:
        return True
    if frame_id in {str(value) for value in info.get("excluded_ids", [])}:
        return False
    return True if frame_id in known_ids else None


class CalibrationDiagnosticsDialog(QDialog):
    preview_requested = Signal(str, object)

    def __init__(self, pairs: dict[str, StereoPairCalibration], previews: list[PreviewPair], parent=None):
        super().__init__(parent)
        self._pairs = dict(pairs)
        self._previews = list(previews)
        self.setWindowTitle("逐帧误差检查")
        self.setWindowIcon(app_icon("scan-line"))
        self.setMinimumSize(360, 300)
        available = self.screen().availableGeometry()
        self.resize(min(1000, available.width() - 32), min(640, available.height() - 32))
        content = QVBoxLayout(self)
        content.setContentsMargins(12, 8, 12, 8)
        content.setSpacing(6)
        header = QHBoxLayout()
        title = QLabel("逐帧误差检查")
        title.setObjectName("workbenchSectionTitle")
        header.addWidget(title)
        header.addWidget(QLabel("双目组"))
        self._combo_pair = QComboBox()
        self._combo_pair.setAccessibleName("诊断双目组")
        self._combo_pair.setMinimumContentsLength(4)
        self._combo_pair.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._combo_pair.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._combo_pair.setMinimumWidth(0)
        for name in self._pairs:
            self._combo_pair.addItem(name, name)
            self._combo_pair.setItemData(self._combo_pair.count() - 1, name, Qt.ToolTipRole)
        header.addWidget(self._combo_pair, 1)
        content.addLayout(header)
        summary = QHBoxLayout()
        self._metric_summary = QLabel(
            "训练集重投影误差，不是独立验证。\n— 为缺失或无效值，不表示零误差。")
        self._metric_summary.setWordWrap(True)
        self._metric_summary.setStyleSheet(f"color:{TEXT_DIM};")
        summary.addWidget(self._metric_summary, 1)
        self._btn_details = QToolButton()
        self._btn_details.setCheckable(True)
        self._btn_details.setAccessibleName("指标说明")
        self._btn_details.setToolTip("展开指标说明")
        self._btn_details.setIcon(app_icon("chevron-right"))
        self._btn_details.toggled.connect(self._toggle_details)
        summary.addWidget(self._btn_details)
        content.addLayout(summary)
        self._details_scroll = QScrollArea()
        self._details_scroll.setAccessibleName("详细指标说明")
        self._details_scroll.setWidgetResizable(True)
        self._details_scroll.setFrameShape(QFrame.NoFrame)
        self._details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._notes = QLabel()
        self._notes.setWordWrap(True)
        self._notes.setTextFormat(Qt.PlainText)
        self._notes.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._notes.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._notes.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._notes.setMinimumWidth(0)
        self._notes.setStyleSheet(f"color:{TEXT_DIM};")
        self._details_scroll.setWidget(self._notes)
        self._details_scroll.hide()
        content.addWidget(self._details_scroll)
        self._table = QTableWidget(0, 8)
        self._table.setMinimumSize(0, 64)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._table.setStyleSheet(f'QTableWidget::item:selected {{background:{PRIMARY}; color:white;}}')
        self._table.setAccessibleName("逐帧误差与参与记录")
        self._table.setHorizontalHeaderLabels([
            "帧 ID", "左内参 RMS / px", "右内参 RMS / px", "左参与", "右参与", "双目参与", "原图",
            "校正残差 RMS / px",
        ])
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().hide()
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        content.addWidget(self._table, 1)
        self._empty_state = QLabel(self._table.viewport())
        self._empty_state.setAlignment(Qt.AlignCenter)
        self._empty_state.setWordWrap(True)
        self._empty_state.setTextFormat(Qt.PlainText)
        self._empty_state.setStyleSheet(f"color:{TEXT_DIM}; background:transparent;")
        self._empty_state.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._table.viewport().installEventFilter(self)
        self._footer = QFrame()
        self._footer.setObjectName("actionFooter")
        self._footer.setFixedHeight(1)
        content.addWidget(self._footer)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 2, 0, 0)
        footer.setSpacing(8)
        self._btn_preview = QPushButton("查看原图")
        self._btn_preview.setIcon(app_icon("folder-open"))
        self._btn_preview.setAutoDefault(False)
        self._btn_preview.setToolTip("查看当前帧的原始左右图像")
        self._btn_preview.clicked.connect(self._request_preview)
        footer.addWidget(self._btn_preview)
        self._preview_status = QLabel()
        self._preview_status.setStyleSheet(f"color:{TEXT_DIM}; background:transparent;")
        footer.addWidget(self._preview_status, 1)
        self._btn_close = QPushButton("关闭")
        self._btn_close.setIcon(app_icon("x"))
        self._btn_close.setAutoDefault(False)
        self._btn_close.clicked.connect(self.reject)
        footer.addWidget(self._btn_close)
        content.addLayout(footer)
        self._combo_pair.currentIndexChanged.connect(self._refresh)
        self._table.itemSelectionChanged.connect(self._refresh_preview_button)
        self._table.cellDoubleClicked.connect(self._on_double_clicked)
        self._refresh()

    def _toggle_details(self, expanded):
        self._details_scroll.setVisible(expanded)
        self._btn_details.setIcon(app_icon("chevron-down" if expanded else "chevron-right"))
        self._btn_details.setToolTip("收起指标说明" if expanded else "展开指标说明")

    def resizeEvent(self, event):
        # Bound explanations independently of their length so the table keeps priority.
        self._details_scroll.setFixedHeight(min(80, max(48, self.height() // 6)))
        if not self._btn_details.isChecked():
            # Hidden scroll areas defer child layout until their first show.
            self._notes.resize(max(1, self.width() - 24), self._details_scroll.height())
        super().resizeEvent(event)

    def eventFilter(self, watched, event):
        if watched is self._table.viewport() and event.type() == QEvent.Resize:
            self._empty_state.setGeometry(watched.rect().adjusted(8, 4, -8, -4))
        return super().eventFilter(watched, event)

    def _matching_preview(self, name, pair, frame_id, source_ids):
        candidates = [preview for preview in self._previews
                      if preview.left_id == pair.left_id and preview.right_id == pair.right_id
                      and preview.key.startswith(f"{name} ")
                      and preview.left is not None and preview.right is not None
                      and preview.left.size and preview.right.size]
        exact = [preview for preview in candidates if preview.key == f"{name} {frame_id}"]
        if exact:
            return exact[0] if len(exact) == 1 else None
        # Live source IDs are unpadded, while gallery labels use four digits.
        if frame_id.isdecimal() and any(value != frame_id and value.isdecimal()
                                       and int(value) == int(frame_id) for value in source_ids):
            return None
        numeric = [preview for preview in candidates
                   if frame_id.isdecimal() and preview.key[len(name) + 1:].isdecimal()
                   and int(preview.key[len(name) + 1:]) == int(frame_id)]
        return numeric[0] if len(numeric) == 1 else None

    def _rows(self, name, pair):
        left, right = pair.left_intrinsics, pair.right_intrinsics
        left_errors, right_errors = _errors_by_id(left), _errors_by_id(right)
        selection = pair.calibration_info.get("selection", {})
        alignment = {str(record['frame_id']): record.get('rms_px')
                     for record in pair.calibration_info.get('rectification', {}).get('per_frame', [])}
        ids = dict.fromkeys([*left_errors, *right_errors])
        ids.update(dict.fromkeys(alignment))
        for side in ("left", "right", "stereo"):
            info = selection.get(side, {})
            ids.update(dict.fromkeys(str(value) for key in ("used_ids", "excluded_ids")
                                     for value in info.get(key, [])))
        legacy = left.frame_ids is None or right.frame_ids is None
        rows = []
        for frame_id in sorted(ids, key=lambda value: (0, int(value), value)
                               if value.isdecimal() else (1, 0, value)):
            rows.append(_FrameRow(
                frame_id, left_errors.get(frame_id), right_errors.get(frame_id),
                (_participates(selection, "left", frame_id, left_errors),
                 _participates(selection, "right", frame_id, right_errors),
                 _participates(selection, "stereo", frame_id)),
                None if legacy else self._matching_preview(name, pair, frame_id, ids),
                alignment_error=alignment.get(frame_id),
            ))
        left_legacy = left.per_frame_errors if left.frame_ids is None else None
        right_legacy = right.per_frame_errors if right.frame_ids is None else None
        left_legacy = left_legacy if left_legacy is not None else []
        right_legacy = right_legacy if right_legacy is not None else []
        for index in range(max(len(left_legacy), len(right_legacy))):
            has_left, has_right = index < len(left_legacy), index < len(right_legacy)
            rows.append(_FrameRow(
                f"参与序号 {index + 1}",
                left_legacy[index] if has_left else None,
                right_legacy[index] if has_right else None,
                (True if has_left else None, True if has_right else None, None),
            ))
        return rows

    def _refresh(self, *_):
        self._table.setSortingEnabled(False)
        self._table.clearContents()
        self._table.setRowCount(0)
        name = self._combo_pair.currentData()
        pair = self._pairs.get(name)
        self._combo_pair.setToolTip(name or "暂无双目标定结果")
        self._combo_pair.setEnabled(bool(self._pairs))
        notes = ["训练集重投影误差，不是独立验证；— 表示无值或无效值，不是零误差。"]
        notes.append("左右内参列为各自记录的逐帧误差，单位 px；"
                     "仅 euclidean_rms_px 明确表示欧氏 RMS。")
        rows = []
        if pair is None:
            notes.append("暂无双目标定结果。")
        else:
            left, right = pair.left_intrinsics, pair.right_intrinsics
            if left.frame_ids is None or right.frame_ids is None:
                notes.append("旧 JSON 缺少 frame_ids，仅按参与序号展示缺失 ID 的数据，不映射原图；"
                             "左右同序号不代表同一原图。")
            unknown = [side for side, intr in (("左", left), ("右", right))
                       if not intr.per_frame_error_metric]
            if unknown:
                notes.append(f"{'、'.join(unknown)}侧旧指标口径未记录，不能确认为欧氏 RMS。")
            for side, intr in (("左", left), ("右", right)):
                if intr.per_frame_error_metric and intr.per_frame_error_metric != "euclidean_rms_px":
                    notes.append(f"{side}侧记录指标：{intr.per_frame_error_metric}，非已确认的欧氏 RMS。")
            notes.append("参与状态来自筛选记录及内参帧 ID；未记录表示无法确认。")
            alignment = pair.calibration_info.get('rectification', {})
            if alignment:
                notes.append(f"校正残差列比较选用配对的 {alignment.get('coordinate', '?')} 坐标；受校正像素尺度影响。")
            rows = self._rows(name, pair)
        self._notes.setText("\n".join(notes))
        self._empty_state.setText("暂无双目标定结果" if pair is None else "当前组暂无逐帧记录")
        self._empty_state.setVisible(not rows)
        self._table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            label = _FrameIdItem(row.label)
            label.setData(Qt.UserRole, row)
            self._table.setItem(index, 0, label)
            self._table.setItem(index, 1, _ErrorItem(row.left_error))
            self._table.setItem(index, 2, _ErrorItem(row.right_error))
            for column, used in enumerate(row.participation, 3):
                self._table.setItem(index, column, QTableWidgetItem(
                    "未记录" if used is None else "是" if used else "否",
                ))
            self._table.setItem(index, 6, QTableWidgetItem("可查看" if row.preview else "无原图"))
            self._table.setItem(index, 7, _ErrorItem(row.alignment_error))
        self._table.setSortingEnabled(True)
        self._table.sortItems(0, Qt.AscendingOrder)
        self._table.clearSelection()
        self._table.setCurrentCell(-1, -1)
        self._refresh_preview_button()

    def _current_preview(self):
        if not self._table.selectionModel().hasSelection():
            return None
        item = self._table.item(self._table.currentRow(), 0)
        return item.data(Qt.UserRole).preview if item is not None else None

    def _refresh_preview_button(self):
        preview = self._current_preview()
        self._btn_preview.setEnabled(preview is not None)
        selected = self._table.selectionModel().hasSelection()
        status = "原图可用" if preview else "无匹配原图" if selected else "未选择帧"
        self._preview_status.setText(status)
        self._btn_preview.setToolTip(
            preview.key if preview else "当前帧无可可靠匹配的原图" if selected else "未选择帧"
        )

    def _request_preview(self, *_):
        preview = self._current_preview()
        if preview is not None:
            self.preview_requested.emit(preview.key, {preview.left_id: preview.left, preview.right_id: preview.right})

    def _on_double_clicked(self, row, column):
        self._table.setCurrentCell(row, column)
        self._request_preview()
