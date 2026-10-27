"""Calibration panel — merged stereo workflow (intrinsic+extrinsic in one step)."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPointF, QSignalBlocker, QThread, QEvent
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QImage,
)
from PySide6.QtWidgets import (
    QBoxLayout,
    QComboBox,
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...board.charuco_board import CharucoBoard
from ..threads.model_selection_worker import ModelSelectionWorker
from ...calibration.assessment import (
    CalibrationEvaluator,
    DataAssessment,
    DataAssessor,
    FrameQuality,
    Grade,
)
from ...calibration.intrinsic import IntrinsicCalibrator
from ...calibration.models import (
    AuxCameraCalibration,
    CameraIntrinsics,
    CameraModel,
    MultiCameraRig,
    StereoPairCalibration,
)
from ...calibration.multiview import MultiViewCalibrator
from ...calibration.planar import (
    PatternBoardConfig,
    PlanarDetectionResult,
    PlanarPatternDetector,
)
from ...calibration.stereo import StereoCalibrator
from ...calibration.quality_summary import summarize_quality
from ...io.session import CalibrationSession
from ...perf import perf_timer
from ...streaming.stream_manager import StreamManager
from ..threads.calibration_worker import AuxCalibrationWorker, PairCalibrationWorker
from ..threads.import_worker import AuxImageImportWorker, StereoImageImportWorker
from ..threads.save_worker import FrameSaveWorker
from ..threads.white_balance_worker import WhiteBalanceGuard
from ..theme import (
    ACCENT, BG_CARD, BG_DARK, BG_INPUT,
    BORDER, SUCCESS, TEXT, TEXT_DIM, TEXT_HINT, WARNING, DANGER, standard_icon, app_icon,
)
from ..configuration import ConfigurationImpact
from .capture_guide import (
    CaptureSequence,
    CaptureSequenceWidget,
    build_default_sequence,
    build_multimodal_sequence,
)
from .pair_gallery import PreviewPair

logger = logging.getLogger(__name__)


def _wrap_to_available_width(label):
    policy = label.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Ignored)
    policy.setHeightForWidth(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(policy)


class _WrappingLabel(QLabel):
    """Keep nested scroll layouts from allocating only the unwrapped text height."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWordWrap(True)
        _wrap_to_available_width(self)

    def setText(self, text):
        super().setText(text)
        self._fit_text_height()

    def clear(self):
        self.setText('')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_text_height()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.FontChange, QEvent.StyleChange):
            self._fit_text_height()

    def _fit_text_height(self):
        self.setMinimumHeight(max(0, self.heightForWidth(max(1, self.width()))))


def _detach_from_layout(widget):
    """Explicitly release the old layout item before Qt reparents a widget."""
    parent = widget.parentWidget()
    layout = parent.layout() if parent is not None else None
    if layout is None:
        return False

    def remove(current):
        for index in range(current.count()):
            item = current.itemAt(index)
            if item.widget() is widget:
                current.removeWidget(widget)
                return True
            nested = item.layout()
            if nested is not None and remove(nested):
                return True
        return False

    return remove(layout)


class _ActionPages(QStackedWidget):
    """Only the current operation page determines the fixed footer height."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.currentChanged.connect(self._refresh_size)

    def _refresh_size(self, _):
        self.updateGeometry()

    def sizeHint(self):
        hint = super().sizeHint()
        if self.currentWidget() is not None:
            hint.setHeight(self.currentWidget().sizeHint().height())
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        if self.currentWidget() is not None:
            hint.setHeight(self.currentWidget().minimumSizeHint().height())
        return hint


@dataclass
class _UndoCaptureEntry:
    """One stereo pair's contribution to a single 「采集一帧」 click."""

    pair_name: str
    lid: str
    rid: str
    session_frame_idx: Optional[int]
    save_token: Optional[str]
    pop_intrinsic_l: bool
    pop_intrinsic_r: bool
    pop_stereo: bool
    session: Optional[CalibrationSession] = None


# ═══════════════════════════════════════════════════════════════
#  Step progress timeline
# ═══════════════════════════════════════════════════════════════

class StepTimeline(QWidget):
    """Horizontal step indicator drawn with QPainter."""

    step_clicked = Signal(int)

    def __init__(self, labels: List[str], parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self._labels = labels
        self._count = len(labels)
        self._current = 0
        self._completed: set[int] = set()
        self.setCursor(Qt.PointingHandCursor)

    def set_labels(self, labels: List[str]):
        self._labels = labels
        self._count = len(labels)
        self._completed.clear()
        self._current = 0
        self.update()

    def set_step(self, index: int):
        self._current = max(0, min(index, self._count - 1))
        self.update()

    def mark_completed(self, index: int):
        self._completed.add(index)
        self.update()

    def mark_pending(self, index: int):
        self._completed.discard(index)
        self.update()

    def reset(self):
        self._current = 0
        self._completed.clear()
        self.update()

    def mousePressEvent(self, event):
        if self._count < 2:
            return
        x = event.position().x()
        margin = 32
        usable = self.width() - 2 * margin
        if usable <= 0:
            return
        for i in range(self._count):
            cx = margin + i * usable / max(self._count - 1, 1)
            if abs(x - cx) < 20:
                self.step_clicked.emit(i)
                return

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        if self._count == 0:
            p.end()
            return

        margin = 32
        usable = w - 2 * margin
        cy = 24
        radius = 14
        div = max(self._count - 1, 1)

        for i in range(self._count - 1):
            x1 = margin + i * usable / div
            x2 = margin + (i + 1) * usable / div
            done = i < self._current or i in self._completed
            p.setPen(QPen(QColor(ACCENT if done else BORDER), 2))
            p.drawLine(QPointF(x1 + radius, cy), QPointF(x2 - radius, cy))

        for i in range(self._count):
            cx = margin + i * usable / div
            if i in self._completed:
                g = QLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius)
                g.setColorAt(0, QColor(SUCCESS))
                g.setColorAt(1, QColor(SUCCESS).darker(130))
                p.setBrush(QBrush(g))
                p.setPen(QPen(QColor(SUCCESS).lighter(120), 2))
            elif i == self._current:
                g = QLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius)
                g.setColorAt(0, QColor(ACCENT))
                g.setColorAt(1, QColor(ACCENT).darker(130))
                p.setBrush(QBrush(g))
                p.setPen(QPen(QColor(ACCENT).lighter(130), 2))
            else:
                p.setBrush(QBrush(QColor(BG_INPUT)))
                p.setPen(QPen(QColor(BORDER), 1.5))
            p.drawEllipse(QPointF(cx, cy), radius, radius)

            p.setPen(QPen(QColor("white"), 1))
            font = QFont()
            font.setPixelSize(12)
            font.setBold(True)
            p.setFont(font)
            txt = "✓" if i in self._completed else str(i + 1)
            tw = p.fontMetrics().horizontalAdvance(txt)
            p.drawText(QPointF(cx - tw / 2, cy + 5), txt)

            font.setPixelSize(11)
            font.setBold(i == self._current)
            p.setFont(font)
            if i == self._current:
                p.setPen(QPen(QColor(ACCENT)))
            elif i in self._completed:
                p.setPen(QPen(QColor(SUCCESS)))
            else:
                p.setPen(QPen(QColor(TEXT_DIM)))
            lbl = self._labels[i] if i < len(self._labels) else ""
            tw = p.fontMetrics().horizontalAdvance(lbl)
            p.drawText(QPointF(cx - tw / 2, cy + radius + 16), lbl)
        p.end()


# ═══════════════════════════════════════════════════════════════
#  Collapsible section
# ═══════════════════════════════════════════════════════════════

class CollapsibleSection(QFrame):
    toggled = Signal(bool)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._title_text = title
        self.setStyleSheet(
            f"CollapsibleSection {{ background: {BG_CARD};"
            f" border: 1px solid {BORDER}; border-radius: 8px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = QPushButton(f"▸  {title}")
        self._header.setStyleSheet(
            f"QPushButton {{ text-align: left; padding: 10px 14px;"
            f" background: transparent; border: none; border-radius: 8px;"
            f" font-weight: bold; font-size: 13px; color: {TEXT}; }}"
            f"QPushButton:hover {{ background: {BG_INPUT}; }}"
        )
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.clicked.connect(self.toggle)
        layout.addWidget(self._header)

        self._content = QWidget()
        self._content.setVisible(False)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(14, 0, 14, 12)
        self._content_layout.setSpacing(8)
        layout.addWidget(self._content)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def set_expanded(self, expanded: bool):
        self._expanded = expanded
        self._content.setVisible(expanded)
        arrow = "▾" if expanded else "▸"
        colour = ACCENT if expanded else TEXT
        self._header.setText(f"{arrow}  {self._title_text}")
        self._header.setStyleSheet(
            f"QPushButton {{ text-align: left; padding: 10px 14px;"
            f" background: transparent; border: none; border-radius: 8px;"
            f" font-weight: bold; font-size: 13px; color: {colour}; }}"
            f"QPushButton:hover {{ background: {BG_INPUT}; }}"
        )

    def set_status_hint(self, text: str, color: str = TEXT_DIM):
        arrow = "▾" if self._expanded else "▸"
        self._header.setText(f"{arrow}  {self._title_text}   {text}")

    def toggle(self):
        self.set_expanded(not self._expanded)
        self.toggled.emit(self._expanded)


# ═══════════════════════════════════════════════════════════════
#  Score badge
# ═══════════════════════════════════════════════════════════════

class ScoreBadge(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 8, 2)
        self._lbl = QLabel("--")
        self._lbl.setStyleSheet("font-weight:bold; font-size:12px;")
        lay.addWidget(self._lbl)

    def set_grade(self, grade: Grade, score: float):
        self.show()
        self._lbl.setText(f"{grade.label} ({score:.0f})")
        self.setStyleSheet(f"ScoreBadge{{background:{grade.color};border-radius:4px;}}")
        self._lbl.setStyleSheet("font-weight:bold;font-size:12px;color:white;")

    def clear(self):
        self._lbl.setText("--")
        self.setStyleSheet("")
        self.hide()


# ═══════════════════════════════════════════════════════════════
#  Per-pair status row
# ═══════════════════════════════════════════════════════════════

class _PairRow(QFrame):
    """One stereo pair: [pair_name]  [status]  [标定 button]"""

    def __init__(self, pair_name: str, parent=None):
        super().__init__(parent)
        self.pair_name = pair_name
        self.setStyleSheet(f"background:{BG_INPUT}; border-radius:6px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(8)
        self._lbl_name = _WrappingLabel(pair_name)
        self._lbl_name.setTextFormat(Qt.PlainText)
        self._lbl_name.setWordWrap(True)
        _wrap_to_available_width(self._lbl_name)
        self._lbl_name.setStyleSheet(f"font-size:13px; font-weight:bold; color:{TEXT};")
        top.addWidget(self._lbl_name, 1)
        self.btn = QPushButton("标定")
        self.btn.setProperty("class", "primary")
        self.btn.setMinimumWidth(104)
        self.btn.setEnabled(False)
        top.addWidget(self.btn)
        lay.addLayout(top)

        self._lbl_status = _WrappingLabel("等待采集数据...")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setTextFormat(Qt.PlainText)
        _wrap_to_available_width(self._lbl_status)
        self._lbl_status.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        lay.addWidget(self._lbl_status)

    def set_status(self, text: str, color: str = TEXT_DIM):
        self._lbl_status.setText(text)
        self._lbl_status.setStyleSheet(f"font-size:12px; color:{color};")


# ═══════════════════════════════════════════════════════════════
#  Main panel
# ═══════════════════════════════════════════════════════════════

class CalibrationPanel(QWidget):
    """Merged stereo calibration workflow: capture → calibrate → fuse."""

    AUX_MAX_FRAME_AGE_SEC = 1.0
    AUX_READINESS_INTERVAL_MS = 250

    calibration_finished = Signal(object)
    calibration_reset = Signal()
    pointcloud_ready = Signal(object)
    frame_captured = Signal(str, object)
    sequence_updated = Signal(object)
    images_imported = Signal(dict)  # {camera_id: np.ndarray}
    aux_images_imported = Signal(dict)
    preview_mode_changed = Signal(bool)
    configure_aux_requested = Signal()
    configure_stereo_requested = Signal()
    data_source_changed = Signal()
    offline_imported = Signal()
    workflow_changed = Signal(str, bool)
    preview_pairs_changed = Signal(object)
    export_requested = Signal()
    open_results_requested = Signal()
    log_added = Signal(str)
    capture_started = Signal()
    create_session_requested = Signal()
    model_choice_requested = Signal(object)
    diagnostic_preview_requested = Signal(str, object)

    def __init__(self, stream_manager: StreamManager, board: CharucoBoard, parent=None):
        super().__init__(parent)
        self._sm = stream_manager
        self._board = board
        self._camera_model = CameraModel.PINHOLE
        self._active_step = 0
        self._workflow_phase = "capture"
        self._live_preview_indices: dict[str, list[tuple[int, int, str]]] = {}
        self._result_directory = None
        self._result_files = []
        self._result_errors = []
        self._results_outdated = False
        self._previous_result_directory = None
        self._session: Optional[CalibrationSession] = None

        self._intr_cals: Dict[str, IntrinsicCalibrator] = {}
        self._intrinsics: Dict[str, CameraIntrinsics] = {}
        self._fqs: Dict[str, List[FrameQuality]] = {}

        self._stereo_cals: Dict[str, StereoCalibrator] = {}
        self._pair_calibs: Dict[str, StereoPairCalibration] = {}

        self._rig: Optional[MultiCameraRig] = None
        self._assessor = DataAssessor()
        self._evaluator = CalibrationEvaluator()

        self._capture_sequence = CaptureSequence(build_default_sequence())
        self._aux_capture_sequence = CaptureSequence(build_multimodal_sequence())
        self._sections: List[CollapsibleSection] = []
        self._pair_rows: Dict[str, _PairRow] = {}
        self._undo_stack: List[tuple[List[_UndoCaptureEntry], bool]] = []
        self._pair_capture_press_count: Dict[str, int] = {}
        self._next_session_frame_idx: Optional[int] = None
        self._save_workers: Dict[str, FrameSaveWorker] = {}
        self._save_queue: list[tuple[str, str, np.ndarray, np.ndarray, int]] = []
        self._save_jobs = {}
        self._failed_save_jobs = {}
        self._saved_capture_tokens: set[str] = set()
        self._pending_save_tokens: set[str] = set()
        self._guidance_failed_saves: set[tuple[int, str, str]] = set()
        self._save_sessions: dict[str, CalibrationSession] = {}
        self._canceled_save_tokens: set[str] = set()
        self._calib_worker: Optional[PairCalibrationWorker] = None
        self._model_worker = None
        self._model_report = None
        self._model_report_key = None
        self._model_error = ''
        self._aux_worker: Optional[AuxCalibrationWorker] = None
        self._stereo_import_worker: Optional[StereoImageImportWorker] = None
        self._aux_import_worker: Optional[AuxImageImportWorker] = None
        self._calib_queue: List[str] = []
        self._aux_pairs: list[tuple[str, PlanarDetectionResult, PlanarDetectionResult]] = []
        self._aux_images: list[tuple[np.ndarray, np.ndarray]] = []
        self._aux_intrinsics: Optional[CameraIntrinsics] = None
        self._aux_capture_count = 0
        self._aux_saved_count = 0
        self._aux_imported_count = 0
        self._aux_source_folder = None
        self._aux_observation_board: Optional[PatternBoardConfig] = None
        self._pending_aux_context = None
        self._offline_import_records = []
        self._offline_frame_qualities = {}
        self._offline_lr_swapped = False
        self._offline_source_folder = None
        self._pending_import_folder = None
        self._calib_strategy_key = "standard"
        self._data_filter_key = "all"
        self._unavailable_cameras: set[str] = set()
        self._aux_ui_busy = False
        self._aux_import_busy = False
        self._white_balance_guard = WhiteBalanceGuard(self._sm, self)
        self._wb_resume_auto = False

        self._init_ui()
        self._white_balance_guard.message.connect(self._log)
        self._white_balance_guard.busy_changed.connect(self._on_white_balance_busy)
        self._white_balance_guard.completed.connect(self._on_white_balance_completed)
        self._connect_signals()
        self._update_capture_stats()
        self._activate_step(0)
        self._refresh_disk_hint()
        self._refresh_aux_board_info()
        self._refresh_aux_readiness()
        self._readiness_timer = QTimer(self)
        self._readiness_timer.setInterval(500)
        self._readiness_timer.timeout.connect(self._refresh_capture_readiness)
        self._readiness_timer.start()
        self._refresh_capture_readiness()

    # ── Public setters ────────────────────────────────────────

    def update_board(self, board: CharucoBoard):
        self._board = board
        self._reset_all()

    def set_camera_model(self, model: CameraModel):
        self._camera_model = model
        self._reset_all()
        self._log(f"模型: {'针孔' if model == CameraModel.PINHOLE else '鱼眼'}")

    def set_session(self, session: CalibrationSession, *, preserve_model_assessment: bool = False):
        assessment_name = self._combo_model_pair.currentText()
        rebind_assessment = (
            preserve_model_assessment and self._session is None and self._model_report is not None
            and self._model_report_key == self._model_snapshot_key(assessment_name)
        )
        if session is not self._session:
            if self._failed_save_jobs:
                raise RuntimeError('仍有原图保存失败，请重试保存或撤销对应采集后再切换会话')
            if self._session is not None and self.has_active_background_tasks():
                raise RuntimeError("后台任务尚未结束，不能切换采集会话")
            self._undo_stack.clear()
            self._refresh_undo_btn()
            self._saved_capture_tokens.clear()
        self._white_balance_guard.invalidate()
        self._session = session
        if rebind_assessment:
            self._model_report_key = self._model_snapshot_key(assessment_name)
        self._next_session_frame_idx = session.frame_count
        self._log(f"会话: {session.name}")
        self._refresh_disk_hint()
        self._refresh_model_choice_ui()

    @property
    def rig(self) -> Optional[MultiCameraRig]:
        return self._rig

    @property
    def has_failed_saves(self) -> bool:
        return bool(self._failed_save_jobs)

    @property
    def has_active_operations(self) -> bool:
        return bool(self._auto_timer.isActive() or self._model_worker or self._calib_worker or self._aux_worker or
                    self._stereo_import_worker or self._aux_import_worker or
                    self._save_workers or self._save_queue or self._aux_dataset_busy())

    def has_active_background_tasks(self) -> bool:
        # Result signals can arrive before QThread.run() has actually returned.
        return bool(self.has_active_operations or self._calib_queue or
                    self._pending_save_tokens or
                    any(worker.isRunning() for worker in self.findChildren(QThread)))

    def has_affected_work(self, impact: ConfigurationImpact) -> bool:
        stereo = bool(self._intrinsics or self._pair_calibs or
                      any(cal.num_frames for cal in self._intr_cals.values()) or
                      any(cal.num_frames for cal in self._stereo_cals.values()))
        return ((impact.reset_stereo and stereo) or
                any((name in self._stereo_cals and self._stereo_cals[name].num_frames > 0) or name in self._pair_calibs
                    for name in impact.changed_pairs) or
                any((cid in self._intr_cals and self._intr_cals[cid].num_frames > 0) or cid in self._intrinsics
                    for cid in impact.camera_ids) or
                any(self._pair_capture_press_count.get(name, 0) or
                    getattr(self, "_imported_pair_frames", {}).get(name)
                    for name in impact.changed_pairs) or
                ("offline" in impact.changed_pairs and bool(self._offline_import_records)) or
                (impact.reset_aux and bool(self._aux_images or self._aux_pairs or self._aux_intrinsics or
                                          (self._rig and self._rig.aux_cameras))))

    def apply_configuration(self, board: CharucoBoard, model: CameraModel, impact: ConfigurationImpact):
        if impact.reset_aux:
            self._invalidate_aux_result()
        self._board, self._camera_model = board, model
        if impact.reset_stereo:
            self._reset_all()
        else:
            for name in impact.changed_pairs:
                self._live_preview_indices.pop(name, None)
                self._stereo_cals.pop(name, None)
                self._pair_calibs.pop(name, None)
                self._pair_capture_press_count.pop(name, None)
                if hasattr(self, "_imported_pair_frames"):
                    self._imported_pair_frames.pop(name, None)
            for cid in impact.camera_ids:
                self._intr_cals.pop(cid, None)
                self._intrinsics.pop(cid, None)
                self._fqs.pop(cid, None)
            self._undo_stack = [([e for e in entries if e.pair_name not in impact.changed_pairs], advanced)
                                for entries, advanced in self._undo_stack]
            self._undo_stack = [(entries, advanced) for entries, advanced in self._undo_stack if entries]
            if "offline" in impact.changed_pairs:
                self._offline_import_records.clear()
                self._btn_swap_offline.setVisible(False)
                for attr in ("_imported_frames", "_import_last_pair", "_import_view_idx"):
                    if hasattr(self, attr):
                        delattr(self, attr)
                self._btn_import_prev.setVisible(False)
                self._btn_import_next.setVisible(False)
                self._lbl_import_nav.setVisible(False)
            if impact.reset_aux:
                self._aux_pairs.clear()
                self._aux_images.clear()
                self._aux_intrinsics = None
                self._aux_capture_count = 0
                self._aux_saved_count = 0
                self._aux_imported_count = 0
                self._aux_source_folder = None
                self._aux_observation_board = None
                self._aux_capture_sequence = CaptureSequence(build_multimodal_sequence())
                self._aux_seq_widget.set_sequence(self._aux_capture_sequence)
                self._aux_cap_progress.setValue(0)
                self._lbl_aux_status.setText("辅助配置已更新，等待重新采集或导入")
                self._sections[2].set_status_hint("")
                self._timeline.mark_pending(2)
                if self._rig is not None:
                    self._rig.aux_cameras.clear()
            if impact.changed_pairs or impact.reset_aux:
                self._rig = self._rebuild_rig_from_pairs() if self._pair_calibs else None
            if impact.changed_pairs:
                self._quality_report.clear()
                self._clear_rect_preview()
                self._timeline.mark_pending(1)
                self._timeline.mark_pending(3)
                self._btn_multi.setEnabled(bool(self._pair_calibs))
                self._btn_cloud.setEnabled(bool(self._pair_calibs))
            self._refresh_undo_btn()
        self.refresh_pairs()
        if impact.changed_pairs or impact.reset_aux or impact.reset_stereo:
            self.set_result_storage(None)
        if impact.changed_pairs or impact.reset_stereo:
            self._capture_sequence.reset()
            self._seq_widget.refresh()
            self._data_score.clear()
            self._lbl_tips.clear()
            self._sections[0].set_status_hint("")
            self._update_capture_stats()
        self._refresh_aux_readiness()

    @property
    def workflow_phase(self) -> str:
        return self._workflow_phase

    @property
    def result_directory(self):
        return self._result_directory

    @property
    def offline_selection(self):
        selected = self._selected_pairs()
        return bool(selected) and all(self._sm.is_virtual_pair(name) for name in selected)

    def show_workflow(self, phase: str, multimodal: bool = False):
        if phase not in {"capture", "calibration", "results", "pointcloud"}:
            raise ValueError(f"Unknown workflow phase: {phase}")
        step = 3 if phase == "pointcloud" else (2 if multimodal else (0 if phase == "capture" else 1))
        self._activate_step(step, phase=phase)

    def preview_pairs(self) -> list[PreviewPair]:
        if self._active_step == 2:
            left, right = self._rgb_left_camera_id() or "RGB_L", self._aux_camera_id()
            return [PreviewPair(f"融合 {index + 1:04d}", left, right, rgb, aux)
                    for index, (rgb, aux) in enumerate(self._aux_images)]
        pairs = []
        for name, cfg in self._selected_pairs().items():
            imported = getattr(self, "_imported_pair_frames", {}).get(name)
            if imported is not None:
                labels = [str(record[0]) for record in self._offline_import_records] if name == 'offline' else []
                pairs.extend(PreviewPair(f"{name} {labels[i] if i < len(labels) else f'{i + 1:04d}'}",
                                        cfg.left.camera_id, cfg.right.camera_id, left, right)
                             for i, (left, right) in enumerate(imported))
                continue
            left_cal, right_cal = self._intr_cals.get(cfg.left.camera_id), self._intr_cals.get(cfg.right.camera_id)
            if left_cal is None or right_cal is None:
                continue
            for left, right, frame_id in self._live_preview_indices.get(name, []):
                if (left < len(left_cal.stored_images) and right < len(right_cal.stored_images) and
                        left_cal.stored_images[left] is not None and right_cal.stored_images[right] is not None):
                    pairs.append(PreviewPair(f"{name} {int(frame_id):04d}" if frame_id.isdecimal() else f"{name} {frame_id}",
                                             cfg.left.camera_id, cfg.right.camera_id,
                                             left_cal.stored_images[left], right_cal.stored_images[right]))
        return pairs

    def _refresh_preview_pairs(self):
        self.preview_pairs_changed.emit(self.preview_pairs())

    def set_result_storage(self, directory, files=None, errors=None):
        self._result_directory = directory
        self._result_files = list(files or [])
        self._result_errors = list(errors or [])
        self._refresh_result_summary()

    def set_result_model(self, rig):
        self._rig = rig
        if rig is not None:
            self._results_outdated = False
            self._previous_result_directory = None
        if rig is None:
            self.set_result_storage(None)
        self._refresh_result_summary()

    def _refresh_result_summary(self):
        rig = self._rig
        if rig is None:
            self._lbl_result_summary.setText("采集数据已变化，请重新标定" if self._results_outdated else "尚无标定结果")
            self._lbl_result_decision.setText("尚无可评估结果")
            self._lbl_result_decision.setStyleSheet(f'color:{TEXT_DIM}; font-size:15px; font-weight:bold;')
            self._lbl_result_risks.clear()
        else:
            decision = summarize_quality(rig)
            color = DANGER if decision.level == 'poor' else WARNING if decision.level == 'fair' else SUCCESS
            self._lbl_result_decision.setText(decision.title)
            self._lbl_result_decision.setStyleSheet(f'color:{color}; font-size:15px; font-weight:bold;')
            self._lbl_result_risks.setText('\n'.join(decision.risks[:3]) or '请核对校正图与实际使用场景')
            self._lbl_result_risks.setToolTip('\n'.join(decision.risks))
            lines = [f"参考相机：{rig.reference_camera}", f"双目 {len(rig.pairs)} 组 · 辅助 {len(rig.aux_cameras)} 路", ""]
            for name, pair in rig.pairs.items():
                lines.extend([name, f"左右内参 RMS  {pair.left_intrinsics.rms_error:.3f} / "
                              f"{pair.right_intrinsics.rms_error:.3f} px",
                              f"双目 RMS  {pair.stereo.rms_error:.3f} px", ""])
                for key, title in (('left', '左内参'), ('right', '右内参'), ('stereo', '双目')):
                    selected = pair.calibration_info.get('selection', {}).get(key, {})
                    if not selected:
                        continue
                    used, total = selected.get('used_count'), selected.get('input_count')
                    lines.append(f"{title}参与 {used if used is not None else '未知'} / "
                                 f"{total if total is not None else '未知'} 帧")
                    excluded = selected.get('excluded_ids', [])
                    if excluded:
                        lines.append('未参与帧：' + '、'.join(map(str, excluded)))
            for aux in rig.aux_cameras.values():
                lines.extend([aux.camera_id, f"融合 RMS  {aux.rms_error:.3f} px · {aux.frame_count} 组", ""])
            lines.extend(['', *decision.details, '', *decision.risks])
            self._lbl_result_summary.setText("\n".join(lines).rstrip())
        if self._result_errors:
            status = (f"已保存 {len(self._result_files)} 个文件 · 部分格式未生成\n"
                      if self._result_directory is not None else "保存失败\n")
            if self._result_directory is not None:
                status += f'结果目录：{Path(self._result_directory).resolve()}\n\n'
            status += "\n".join(self._result_errors)
        elif self._result_directory is not None:
            directory = Path(self._result_directory)
            status = f"文件已保存：{len(self._result_files)} 个\n{directory.resolve()}"
        else:
            status = "文件待生成或另存" if rig is not None else ""
            if self._results_outdated:
                status = '旧结果已过期，未代表当前采集数据'
                if self._previous_result_directory is not None:
                    status += f'\n历史文件保留：{Path(self._previous_result_directory).resolve()}'
        self._lbl_result_storage.setText(status)
        storage_tip = str(self._result_directory or '')
        if self._result_files:
            storage_tip += '\n本次文件：\n' + '\n'.join(Path(path).name for path in self._result_files)
        self._lbl_result_storage.setToolTip(storage_tip)
        self._btn_export_results.setEnabled(rig is not None)
        self._btn_open_results.setEnabled(self._result_directory is not None)
        self._btn_copy_results.setEnabled(self._result_directory is not None)
        self._btn_view_results.setEnabled(rig is not None)

    # ── UI construction ───────────────────────────────────────

    def _init_ui(self):
        scroll = QScrollArea()
        self._inspector_scroll = scroll
        scroll.setWidgetResizable(True)

        container = QWidget()
        ml = QVBoxLayout(container)
        ml.setContentsMargins(16, 16, 16, 16)
        ml.setSpacing(14)
        self._lbl_task_title = QLabel("当前任务")
        self._lbl_task_title.setStyleSheet(f"font-size:16px; font-weight:bold; color:{TEXT};")
        task_heading = QHBoxLayout()
        task_heading.addWidget(self._lbl_task_title)
        task_heading.addStretch()
        self._lbl_task_phase = QLabel()
        self._lbl_task_phase.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        task_heading.addWidget(self._lbl_task_phase)
        ml.addLayout(task_heading)

        # Timeline: stereo capture/calibration, optional multimodal mono, then point cloud.
        self._timeline = StepTimeline(["共视采图", "双目标定", "多模态", "点云融合"])
        ml.addWidget(self._timeline)
        self._timeline.hide()

        # ── 0. Capture ────────────────────────────────
        s0 = CollapsibleSection("1  同步共视采图（双目标定数据）")
        l0 = s0.content_layout

        self._lbl_stereo_task = QLabel(
            "业界面常见要求：每步为「同一时刻左右目」一对图像；标定板须在双目共视内、整板清晰；"
            "本任务序列为多距×横向平拍 → 中距倾角 → 边角大倾角（与 OpenCV / 工业双目流程一致）。"
        )
        self._lbl_stereo_task.setWordWrap(True)
        self._lbl_stereo_task.setStyleSheet(
            f"font-size:11px; color:{TEXT_HINT}; padding: 0 0 6px 0;"
        )
        l0.addWidget(self._lbl_stereo_task)
        self._lbl_stereo_task.hide()

        # pair selector
        pair_row = QHBoxLayout()
        pair_row.addWidget(QLabel("采集目标"))
        self._combo_pair = QComboBox()
        self._combo_pair.addItem("全部相机组")
        self._combo_pair.setMinimumWidth(120)
        pair_row.addWidget(self._combo_pair, 1)
        l0.addLayout(pair_row)
        self._lbl_rgb_board = QLabel()
        self._lbl_rgb_board.setWordWrap(True)
        self._lbl_rgb_board.setStyleSheet(f"color:{TEXT_DIM};")
        l0.addWidget(self._lbl_rgb_board)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("最少配对数"))
        self._spin_min = QSpinBox()
        self._spin_min.setRange(5, 200)
        self._spin_min.setValue(20)
        self._spin_min.setFixedWidth(70)
        min_row.addWidget(self._spin_min)
        min_row.addStretch()
        l0.addLayout(min_row)

        btn_row = QHBoxLayout()
        self._btn_capture = QPushButton("采集一帧")
        self._btn_capture.setProperty("class", "primary")
        btn_row.addWidget(self._btn_capture)
        self._btn_auto = QCheckBox("自动采集")
        self._btn_auto.setToolTip("每 2 秒检查一次；双路检测清晰、运动稳定且接近建议区域时采集")
        btn_row.addWidget(self._btn_auto)
        self._btn_import = QPushButton("导入图片")
        self._btn_import.setIcon(app_icon('folder-input'))
        self._btn_import.setToolTip(
            "支持递归导入：*_left/*_right、left_*/right_*、*_L/*_R、left/right 子目录"
        )
        btn_row.addWidget(self._btn_import)
        self._btn_swap_offline = QPushButton("交换离线左右图")
        self._btn_swap_offline.setIcon(app_icon('arrow-left-right'))
        self._btn_swap_offline.setToolTip(
            "交换离线图像角色并清空该组旧标定结果；不修改原文件，需重新标定并导出"
        )
        self._btn_swap_offline.setVisible(False)
        btn_row.addWidget(self._btn_swap_offline)
        self._btn_undo = QToolButton()
        self._btn_undo.setIcon(app_icon('undo-2'))
        self._btn_undo.setFixedSize(36, 36)
        self._btn_undo.setAccessibleName("撤销上一帧")
        self._btn_undo.setEnabled(False)
        self._btn_undo.setToolTip(
            "撤销最近一次成功的同步采集（标定缓冲与会话目录中的 PNG）"
        )
        btn_row.addWidget(self._btn_undo)
        l0.addLayout(btn_row)

        self._lbl_disk_hint = QLabel()
        self._lbl_disk_hint.setWordWrap(True)
        self._lbl_disk_hint.setStyleSheet(
            f"font-size:11px; color:{TEXT_HINT}; padding: 2px 0;"
        )
        l0.addWidget(self._lbl_disk_hint)

        # import navigation (hidden until images loaded)
        import_nav = QHBoxLayout()
        import_nav.setSpacing(4)
        self._btn_import_prev = QPushButton("◀")
        self._btn_import_prev.setFixedWidth(36)
        self._btn_import_prev.setFixedHeight(28)
        self._btn_import_prev.setCursor(Qt.PointingHandCursor)
        self._btn_import_prev.setStyleSheet(
            f"QPushButton {{ font-size:16px; color:{ACCENT}; background:{BG_INPUT};"
            f" border:1px solid {BORDER}; border-radius:4px; padding:0; }}"
            f"QPushButton:hover {{ background:{BG_CARD}; border-color:{ACCENT}; }}"
            f"QPushButton:disabled {{ color:{TEXT_HINT}; }}"
        )
        self._btn_import_prev.setVisible(False)
        self._btn_import_prev.clicked.connect(self._on_import_prev)
        import_nav.addWidget(self._btn_import_prev)
        self._lbl_import_nav = QLabel("")
        self._lbl_import_nav.setAlignment(Qt.AlignCenter)
        self._lbl_import_nav.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        self._lbl_import_nav.setVisible(False)
        import_nav.addWidget(self._lbl_import_nav, 1)
        self._btn_import_next = QPushButton("▶")
        self._btn_import_next.setFixedWidth(36)
        self._btn_import_next.setFixedHeight(28)
        self._btn_import_next.setCursor(Qt.PointingHandCursor)
        self._btn_import_next.setStyleSheet(
            f"QPushButton {{ font-size:16px; color:{ACCENT}; background:{BG_INPUT};"
            f" border:1px solid {BORDER}; border-radius:4px; padding:0; }}"
            f"QPushButton:hover {{ background:{BG_CARD}; border-color:{ACCENT}; }}"
            f"QPushButton:disabled {{ color:{TEXT_HINT}; }}"
        )
        self._btn_import_next.setVisible(False)
        self._btn_import_next.clicked.connect(self._on_import_next)
        import_nav.addWidget(self._btn_import_next)
        l0.addLayout(import_nav)

        self._cap_progress = QProgressBar()
        self._cap_progress.setTextVisible(False)
        self._cap_progress.setFixedHeight(8)

        self._seq_widget = CaptureSequenceWidget()
        self._seq_widget.set_compact_mode(True)
        self._seq_widget.set_sequence(self._capture_sequence)
        l0.addWidget(self._seq_widget)

        self._lbl_capture_count = QLabel()
        self._lbl_capture_count.setStyleSheet(f'color:{TEXT_DIM}; font-size:12px;')
        l0.addWidget(self._lbl_capture_count)
        l0.addWidget(self._cap_progress)

        self._lbl_assess = QLabel("尚无质量评估")
        self._lbl_assess.setWordWrap(True)
        self._lbl_assess.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        l0.addWidget(self._lbl_assess)
        self._data_score = ScoreBadge()
        self._data_score.clear()
        l0.addWidget(self._data_score)
        self._lbl_tips = QLabel("")
        self._lbl_tips.setWordWrap(True)
        self._lbl_tips.setStyleSheet(f"font-size:11px; color:{WARNING};")
        l0.addWidget(self._lbl_tips)

        ml.addWidget(s0)
        self._sections.append(s0)

        # ── 1. Stereo calibration (intrinsic + extrinsic) ─────
        s1 = CollapsibleSection("2  双目标定")
        l1 = s1.content_layout

        self._lbl_calib_hint = _WrappingLabel(
            "每组双目一键完成：左右相机内参 + 立体外参"
        )
        self._lbl_calib_hint.setWordWrap(True)
        self._lbl_calib_hint.setStyleSheet(f"font-size:12px; color:{TEXT_HINT};")
        l1.addWidget(self._lbl_calib_hint)
        self._btn_configure_stereo = QToolButton()
        self._btn_configure_stereo.setIcon(app_icon('settings'))
        self._btn_configure_stereo.setText('模型与标定板')
        self._btn_configure_stereo.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_configure_stereo.setAccessibleName('配置双目相机模型与标定板')
        self._btn_configure_stereo.clicked.connect(self.configure_stereo_requested)
        l1.addWidget(self._btn_configure_stereo)

        model_row = QHBoxLayout()
        self._combo_model_pair = QComboBox()
        self._combo_model_pair.setAccessibleName('模型评估对象')
        self._combo_model_pair.currentTextChanged.connect(self._refresh_model_choice_ui)
        model_row.addWidget(self._combo_model_pair, 1)
        self._btn_assess_model = QToolButton()
        self._btn_assess_model.setText('自动评估模型')
        self._btn_assess_model.setIcon(app_icon('scan-line'))
        self._btn_assess_model.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_assess_model.clicked.connect(self._start_model_assessment)
        model_row.addWidget(self._btn_assess_model)
        self._btn_cancel_model = QToolButton()
        self._btn_cancel_model.setText('取消评估')
        self._btn_cancel_model.setIcon(app_icon('x'))
        self._btn_cancel_model.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_cancel_model.clicked.connect(self._cancel_model_assessment)
        self._btn_cancel_model.hide()
        model_row.addWidget(self._btn_cancel_model)
        l1.addLayout(model_row)
        self._lbl_model_choice = _WrappingLabel('至少 9 对多姿态观测后可评估针孔/鱼眼模型')
        self._lbl_model_choice.setWordWrap(True)
        self._lbl_model_choice.setTextFormat(Qt.PlainText)
        l1.addWidget(self._lbl_model_choice)
        self._btn_apply_model = QToolButton()
        self._btn_apply_model.setText('应用推荐模型')
        self._btn_apply_model.setIcon(app_icon('check'))
        self._btn_apply_model.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_apply_model.clicked.connect(self._request_model_choice)
        self._btn_apply_model.hide()
        l1.addWidget(self._btn_apply_model)

        strategy_row = QHBoxLayout()
        strategy_row.addWidget(QLabel("优化策略"))
        self._combo_calib_strategy = QComboBox()
        self._combo_calib_strategy.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._combo_calib_strategy.setMinimumContentsLength(12)
        self._combo_calib_strategy.addItem("标准：固定内参求双目外参", "standard")
        self._combo_calib_strategy.addItem("稳健：固定切向畸变/高阶畸变", "robust")
        self._combo_calib_strategy.addItem("联合精修：允许双目阶段微调内参", "joint_refine")
        self._combo_calib_strategy.setToolTip(
            "切换策略会清空已有标定结果，但保留采集/导入的图像检测数据"
        )
        strategy_row.addWidget(self._combo_calib_strategy, 1)
        l1.addLayout(strategy_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("数据筛选"))
        self._combo_data_filter = QComboBox()
        self._combo_data_filter.addItem("全部使用", "all")
        self._combo_data_filter.addItem("自动质量筛选", "quality")
        self._combo_data_filter.addItem("角点丰富优先", "corners")
        self._combo_data_filter.addItem("鱼眼兼容：跨帧共同角点", "fisheye_common")
        self._combo_data_filter.setToolTip(
            "评分筛选保留约 80% 且不少于最少配对数，不保证姿态最优；不删除原图。\n"
            "鱼眼兼容模式须明确选择：保留全部有效帧，双目阶段只使用跨帧共同 ID 的角点。"
        )
        filter_row.addWidget(self._combo_data_filter, 1)
        l1.addLayout(filter_row)

        # pair rows will be added here dynamically
        self._calib_container = QVBoxLayout()
        self._calib_container.setSpacing(6)
        l1.addLayout(self._calib_container)

        # batch button
        self._btn_calib_all = QPushButton("一键标定全部")
        self._btn_calib_all.setProperty("class", "primary")
        self._btn_calib_all.setEnabled(False)
        l1.addWidget(self._btn_calib_all)

        # rectification preview
        self._lbl_rect_context = QLabel()
        self._lbl_rect_context.setWordWrap(True)
        self._lbl_rect_context.hide()
        l1.addWidget(self._lbl_rect_context)
        self._lbl_rect_preview = QLabel()
        self._lbl_rect_preview.setAlignment(Qt.AlignCenter)
        self._lbl_rect_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._lbl_rect_preview.setMinimumHeight(50)
        self._lbl_rect_preview.setStyleSheet(f"background:{BG_DARK};border-radius:4px;")
        self._lbl_rect_preview.hide()
        l1.addWidget(self._lbl_rect_preview)

        self._quality_report = QTextEdit()
        self._quality_report.setReadOnly(True)
        self._btn_frame_diagnostics = QPushButton("逐帧误差检查")
        self._btn_frame_diagnostics.setIcon(app_icon('scan-line'))
        self._btn_frame_diagnostics.setEnabled(False)
        self._btn_frame_diagnostics.clicked.connect(self._show_frame_diagnostics)
        l1.addWidget(self._btn_frame_diagnostics)
        self._quality_report.setMaximumHeight(150)
        self._quality_report.setPlaceholderText("标定完成后显示质量评估、风险提示和优化建议")
        l1.addWidget(self._quality_report)

        ml.addWidget(s1)
        self._sections.append(s1)

        # ── 2. Multimodal auxiliary mono calibration ─────────
        s2 = CollapsibleSection("3  RGB 左目 + 红外/辅助标定")
        l2 = s2.content_layout

        self._lbl_aux_info = QLabel(
            "导入 rgb_left_XXXX.* 与 aux_XXXX.* 配对图像，先完成 RGB 双目标定后可求辅助单目内参和 AUX→RGB_L 外参。"
        )
        self._lbl_aux_info.setWordWrap(True)
        self._lbl_aux_info.setStyleSheet(f"font-size:12px; color:{TEXT_HINT};")
        l2.addWidget(self._lbl_aux_info)
        self._lbl_aux_info.hide()

        self._lbl_aux_board = QLabel()
        self._lbl_aux_board.setWordWrap(True)
        self._lbl_aux_board.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        l2.addWidget(self._lbl_aux_board)
        self._lbl_aux_board.hide()

        self._lbl_aux_sources = QLabel()
        self._lbl_aux_sources.setWordWrap(True)
        self._lbl_aux_sources.setStyleSheet(f"font-size:12px; color:{TEXT};")
        l2.addWidget(self._lbl_aux_sources)
        self._lbl_aux_sources.hide()

        self._lbl_aux_readiness = QLabel()
        self._lbl_aux_readiness.setWordWrap(True)
        self._lbl_aux_readiness.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        l2.addWidget(self._lbl_aux_readiness)
        self._lbl_aux_readiness.hide()

        self._aux_task_values = {}
        self._aux_task_labels = {}
        task_grid = QGridLayout()
        task_grid.setHorizontalSpacing(12)
        task_grid.setVerticalSpacing(10)
        task_grid.setColumnStretch(1, 1)
        for row, (key, title) in enumerate((('sensor', '传感器组合'), ('intrinsics', 'RGB 左目内参'),
                                          ('capture', '配对采集'), ('calibration', '融合标定'))):
            self._aux_task_values[key] = self._inspector_value(task_grid, row, title)
            self._aux_task_labels[key] = task_grid.itemAtPosition(row, 0).widget()
        self._btn_rgb_prerequisite = QToolButton()
        self._btn_rgb_prerequisite.setIcon(standard_icon(self, QStyle.SP_ArrowRight))
        self._btn_rgb_prerequisite.setFixedSize(28, 28)
        self._btn_rgb_prerequisite.setToolTip("前往 RGB 双目标定")
        self._btn_rgb_prerequisite.setAccessibleName("前往 RGB 双目标定")
        self._btn_rgb_prerequisite.clicked.connect(lambda: self.show_workflow('calibration'))
        task_grid.addWidget(self._btn_rgb_prerequisite, 1, 2)
        l2.addLayout(task_grid)

        self._aux_board_section = QWidget()
        board_layout = QVBoxLayout(self._aux_board_section)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setSpacing(6)
        board_header = QHBoxLayout()
        self._btn_board_details = QToolButton()
        self._btn_board_details.setText("标定板")
        self._btn_board_details.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_board_details.setArrowType(Qt.RightArrow)
        self._btn_board_details.setCheckable(True)
        self._btn_board_details.setToolTip("展开 / 收起标定板详细参数")
        self._btn_board_details.toggled.connect(self._set_board_details)
        board_header.addWidget(self._btn_board_details)
        board_header.addStretch()

        self._btn_configure_aux = QToolButton()
        self._btn_configure_aux.setIcon(app_icon('settings'))
        self._btn_configure_aux.setFixedSize(30, 30)
        self._btn_configure_aux.setToolTip('配置相机与标定板')
        self._btn_configure_aux.setAccessibleName("配置相机与标定板")
        self._btn_configure_aux.clicked.connect(lambda: self.configure_aux_requested.emit())
        board_header.addWidget(self._btn_configure_aux)
        board_layout.addLayout(board_header)
        self._lbl_board_compact = QLabel()
        self._lbl_board_compact.setWordWrap(True)
        self._lbl_board_compact.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px;")
        board_layout.addWidget(self._lbl_board_compact)
        self._aux_board_details = QWidget()
        board_grid = QGridLayout(self._aux_board_details)
        board_grid.setContentsMargins(0, 0, 0, 0)
        board_grid.setHorizontalSpacing(12)
        board_grid.setVerticalSpacing(10)
        board_grid.setColumnStretch(1, 1)
        self._aux_board_values = {}
        for row, (key, title) in enumerate((('type', '类型'), ('geometry', '规格'),
                                          ('spacing', '方格 / 点间距'), ('points', '检测点'))):
            self._aux_board_values[key] = self._inspector_value(board_grid, row, title)
        board_layout.addWidget(self._aux_board_details)
        self._aux_board_details.hide()
        l2.addWidget(self._aux_board_section)

        self._aux_seq_widget = CaptureSequenceWidget()
        self._aux_seq_widget.set_compact_mode(True)
        self._aux_seq_widget.set_sequence(self._aux_capture_sequence)
        l2.addWidget(self._aux_seq_widget)

        self._aux_cap_progress = QProgressBar()
        self._aux_cap_progress.setFormat("%v / %m")
        self._aux_cap_progress.setMaximum(self._aux_capture_sequence.total)
        self._aux_cap_progress.setValue(0)
        self._aux_progress_header = QWidget()
        progress_header = QHBoxLayout(self._aux_progress_header)
        progress_header.setContentsMargins(0, 0, 0, 0)
        progress_header.addWidget(QLabel("有效配对"))
        progress_header.addStretch()
        self._lbl_aux_pair_count = QLabel("0 / 13")
        progress_header.addWidget(self._lbl_aux_pair_count)
        l2.addWidget(self._aux_progress_header)
        self._aux_cap_progress.setTextVisible(False)
        self._aux_cap_progress.setFixedHeight(8)
        l2.addWidget(self._aux_cap_progress)

        aux_btn_row = QHBoxLayout()
        self._btn_capture_aux = QPushButton("采集多模态一帧")
        self._btn_capture_aux.setProperty("class", "primary")
        aux_btn_row.addWidget(self._btn_capture_aux)
        self._btn_import_aux = QPushButton("导入多模态图片")
        aux_btn_row.addWidget(self._btn_import_aux)
        self._btn_calib_aux = QPushButton("标定辅助单目")
        self._btn_calib_aux.setEnabled(False)
        l2.addLayout(aux_btn_row)
        l2.addWidget(self._btn_calib_aux)

        self._lbl_aux_status = QLabel("等待多模态配对图像")
        self._lbl_aux_status.setWordWrap(True)
        self._lbl_aux_status.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        l2.addWidget(self._lbl_aux_status)

        ml.addWidget(s2)
        self._sections.append(s2)

        # ── 3. Fusion / multi-view ────────────────────
        s3 = CollapsibleSection("4  点云融合")
        l3 = s3.content_layout

        self._lbl_fusion_info = QLabel("全部双目标定完成后可生成融合点云")
        self._lbl_fusion_info.setWordWrap(True)
        self._lbl_fusion_info.setStyleSheet(f"font-size:12px; color:{TEXT_HINT};")
        l3.addWidget(self._lbl_fusion_info)

        self._btn_multi = QPushButton("联合优化")
        self._btn_multi.setProperty("class", "primary")
        self._btn_multi.setEnabled(False)
        l3.addWidget(self._btn_multi)

        self._lbl_multi = QLabel("")
        self._lbl_multi.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        l3.addWidget(self._lbl_multi)

        self._btn_cloud = QPushButton("生成融合点云")
        self._btn_cloud.setProperty("class", "primary")
        self._btn_cloud.setEnabled(False)
        l3.addWidget(self._btn_cloud)

        ml.addWidget(s3)
        self._sections.append(s3)

        # ── Log ──────────────────────────────────────
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(100)
        self._log_text.document().setMaximumBlockCount(1000)
        self._log_text.setPlaceholderText("操作日志...")
        ml.addWidget(self._log_text)

        self._result_body = QWidget()
        results = QVBoxLayout(self._result_body)
        results.setContentsMargins(0, 0, 0, 0)
        results.setAlignment(Qt.AlignTop)
        self._lbl_result_decision = _WrappingLabel("尚无可评估结果")
        self._lbl_result_decision.setWordWrap(True)
        self._lbl_result_decision.setTextFormat(Qt.PlainText)
        results.addWidget(self._lbl_result_decision)
        self._lbl_result_risks = _WrappingLabel()
        self._lbl_result_risks.setWordWrap(True)
        self._lbl_result_risks.setTextFormat(Qt.PlainText)
        self._lbl_result_risks.setStyleSheet(f'color:{TEXT_DIM}; font-size:12px;')
        results.addWidget(self._lbl_result_risks)
        self._btn_result_details = QToolButton()
        self._btn_result_details.setText('质量指标详情')
        self._btn_result_details.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_result_details.setCheckable(True)
        self._btn_result_details.setArrowType(Qt.RightArrow)
        self._btn_result_details.toggled.connect(self._toggle_result_details)
        results.addWidget(self._btn_result_details)
        self._lbl_result_summary = _WrappingLabel("尚无标定结果")
        self._lbl_result_summary.setTextFormat(Qt.PlainText)
        self._lbl_result_summary.setWordWrap(True)
        self._lbl_result_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        results.addWidget(self._lbl_result_summary)
        self._lbl_result_summary.hide()
        self._lbl_result_storage = _WrappingLabel()
        self._lbl_result_storage.setTextFormat(Qt.PlainText)
        self._lbl_result_storage.setWordWrap(True)
        self._lbl_result_storage.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._lbl_result_storage.setStyleSheet(f"color:{TEXT_DIM};")
        results.addWidget(self._lbl_result_storage)
        ml.addWidget(self._result_body)

        for section in self._sections:
            section._header.hide()
            section.setStyleSheet("CollapsibleSection { border:none; background:transparent; }")
            section.content_layout.setContentsMargins(0, 0, 0, 0)
            section.content_layout.setAlignment(Qt.AlignTop)

        ml.addStretch()
        for label in container.findChildren(QLabel):
            if label.wordWrap():
                _wrap_to_available_width(label)
        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll, 1)
        self._build_action_footer(outer)

    def take_log_widget(self) -> QTextEdit:
        self._log_text.parentWidget().layout().removeWidget(self._log_text)
        return self._log_text

    @staticmethod
    def _inspector_value(grid, row, title):
        label = QLabel(title)
        label.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px;")
        value = QLabel("—")
        value.setTextFormat(Qt.PlainText)
        value.setWordWrap(True)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        value.setMinimumHeight(28)
        value.setStyleSheet(f"color:{TEXT}; font-size:12px;")
        grid.addWidget(label, row, 0, Qt.AlignTop)
        grid.addWidget(value, row, 1)
        return value

    def _build_action_footer(self, outer):
        # Implicit layout migration can leave stale QWidgetItem wrappers in PySide.
        for button in (self._btn_capture, self._btn_import, self._btn_auto, self._btn_undo,
                       self._btn_swap_offline, self._btn_calib_all, self._btn_capture_aux,
                       self._btn_import_aux, self._btn_calib_aux, self._btn_multi, self._btn_cloud):
            _detach_from_layout(button)
        footer = QFrame()
        self._action_footer = footer
        footer.setObjectName("actionFooter")
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        self._footer_info = QWidget()
        self._footer_info.setMinimumWidth(210)
        self._footer_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        info = QVBoxLayout(self._footer_info)
        info.setContentsMargins(0, 0, 0, 0)
        info.setSpacing(6)
        info.setAlignment(Qt.AlignTop)
        layout.addWidget(self._footer_info)
        self._lbl_raw_save = QLabel()
        self._lbl_raw_save.setWordWrap(False)
        self._lbl_raw_save.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self._lbl_raw_save.setMinimumHeight(self._lbl_raw_save.fontMetrics().height())
        self._lbl_raw_save.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        info.addWidget(self._lbl_raw_save)
        self._lbl_capture_readiness = QLabel()
        self._lbl_capture_readiness.setWordWrap(True)
        self._lbl_capture_readiness.setTextFormat(Qt.PlainText)
        self._lbl_capture_readiness.setStyleSheet(f'color:{TEXT_DIM}; background:transparent; font-size:12px;')
        info.addWidget(self._lbl_capture_readiness)
        self._btn_retry_save = QToolButton()
        self._btn_retry_save.setText('重试保存原图')
        self._btn_retry_save.setIcon(app_icon('rotate-ccw'))
        self._btn_retry_save.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_retry_save.setToolTip('使用保留的原图重试，不重复检测或增加采集记录')
        self._btn_retry_save.clicked.connect(self._retry_failed_saves)
        self._btn_retry_save.hide()
        info.addWidget(self._btn_retry_save)
        self._wb_progress = QWidget()
        progress_row = QHBoxLayout(self._wb_progress)
        progress_row.setContentsMargins(0, 0, 0, 0)
        self._lbl_wb_progress = QLabel()
        self._lbl_wb_progress.setWordWrap(True)
        self._lbl_wb_progress.setStyleSheet(f"color:{ACCENT}; background:transparent; font-size:12px;")
        progress_row.addWidget(self._lbl_wb_progress, 1)
        self._btn_cancel_wb = QToolButton()
        self._btn_cancel_wb.setText("取消准备")
        self._btn_cancel_wb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_cancel_wb.setIcon(app_icon('x'))
        self._btn_cancel_wb.setToolTip("取消白平衡准备及本次待执行的采图")
        self._btn_cancel_wb.clicked.connect(self._cancel_white_balance)
        progress_row.addWidget(self._btn_cancel_wb)
        info.addWidget(self._wb_progress)
        self._wb_progress.hide()
        self._wb_progress_timer = QTimer(self)
        self._wb_progress_timer.setInterval(250)
        self._wb_progress_timer.timeout.connect(self._refresh_white_balance_progress)
        self._check_wb = QCheckBox("白平衡")
        self._check_wb.setStyleSheet(f"background:transparent; color:{TEXT_DIM}; font-size:12px;")
        self._check_wb.setChecked(True)
        self._check_wb.setToolTip("采图前准备两路 RGB，策略由右侧选项决定；不回写历史色温。无控制接口的相机可手动关闭。")
        self._check_wb.toggled.connect(self._on_white_balance_policy_changed)
        wb_policy_row = QHBoxLayout()
        wb_policy_row.setSpacing(6)
        wb_policy_row.addWidget(self._check_wb)
        self._combo_wb_mode = QComboBox()
        self._combo_wb_mode.addItem('自动稳定（不锁定）', 'automatic')
        self._combo_wb_mode.addItem('手动冻结验证（可能变色）', 'locked')
        self._combo_wb_mode.setAccessibleName('采图白平衡策略')
        self._combo_wb_mode.setToolTip('默认保持自动白平衡并检查稳定，不保证参数固定或颜色准确。手动冻结仅供已验证设备使用，可能触发变色。')
        self._combo_wb_mode.currentIndexChanged.connect(self._on_white_balance_mode_changed)
        wb_policy_row.addWidget(self._combo_wb_mode, 1)
        info.addLayout(wb_policy_row)
        self._footer_pages = _ActionPages()
        layout.addWidget(self._footer_pages)

        def page():
            widget = QWidget()
            box = QVBoxLayout(widget)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(6)
            box.setAlignment(Qt.AlignTop)
            self._footer_pages.addWidget(widget)
            return box

        capture = page()
        self._capture_footer_layout = capture
        self._btn_capture.setText("采集配对图像")
        self._btn_capture.setIcon(app_icon('camera', '#ffffff'))
        self._btn_capture.setMinimumHeight(40)
        capture.addWidget(self._btn_capture)
        row = QHBoxLayout()
        row.addWidget(self._btn_import, 1)
        row.addWidget(self._btn_auto)
        self._capture_import_row = row
        capture.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self._btn_undo)
        self._btn_skip_stereo = QToolButton()
        self._btn_skip_stereo.setIcon(app_icon('skip-forward'))
        self._btn_skip_stereo.setToolTip('跳过当前建议，不新增采图记录')
        self._btn_skip_stereo.setAccessibleName('跳过当前双目采图建议')
        self._btn_skip_stereo.clicked.connect(self._seq_widget._on_skip)
        row.addWidget(self._btn_skip_stereo)
        row.addWidget(self._btn_swap_offline)
        row.addStretch()
        self._btn_to_calibration = QPushButton("前往标定")
        self._btn_to_calibration.setIcon(app_icon('chevron-right'))
        self._btn_to_calibration.clicked.connect(lambda: self.show_workflow("calibration"))
        row.addWidget(self._btn_to_calibration)
        self._capture_nav_row = row
        capture.addLayout(row)

        calibration = page()
        calibration.addWidget(self._btn_calib_all)
        self._btn_view_results = QPushButton("查看结果")
        self._btn_view_results.clicked.connect(lambda: self.show_workflow("results", self._active_step == 2))
        calibration.addWidget(self._btn_view_results)

        auxiliary = page()
        self._btn_capture_aux.setText("采集配对图像")
        self._btn_capture_aux.setIcon(app_icon('camera', '#ffffff'))
        self._btn_capture_aux.setMinimumHeight(40)
        auxiliary.addWidget(self._btn_capture_aux)
        row = QHBoxLayout()
        self._btn_import_aux.setText("导入图片")
        self._btn_import_aux.setIcon(app_icon('folder-input'))
        row.addWidget(self._btn_import_aux, 1)
        to_aux_calibration = QPushButton("前往标定")
        to_aux_calibration.clicked.connect(lambda: self.show_workflow("calibration", True))
        row.addWidget(to_aux_calibration, 1)
        auxiliary.addLayout(row)

        aux_calibration = page()
        self._btn_calib_aux.setText("开始融合标定")
        self._btn_calib_aux.setProperty("class", "primary")
        self._btn_calib_aux.setMinimumHeight(40)
        aux_calibration.addWidget(self._btn_calib_aux)
        aux_results = QPushButton("查看结果")
        aux_results.clicked.connect(lambda: self.show_workflow("results", True))
        aux_calibration.addWidget(aux_results)

        result_page = page()
        self._btn_export_results = QPushButton("导出标定结果")
        self._btn_export_results.setIcon(app_icon('download', '#ffffff'))
        self._btn_export_results.setProperty("class", "primary")
        self._btn_export_results.clicked.connect(self.export_requested)
        result_page.addWidget(self._btn_export_results)
        self._btn_open_results = QPushButton("打开结果目录")
        self._btn_open_results.setIcon(app_icon('folder-open'))
        self._btn_open_results.clicked.connect(self.open_results_requested)
        result_row = QHBoxLayout()
        result_row.addWidget(self._btn_open_results, 1)
        self._btn_copy_results = QToolButton()
        self._btn_copy_results.setIcon(app_icon('copy'))
        self._btn_copy_results.setFixedSize(32, 32)
        self._btn_copy_results.setToolTip('复制完整结果路径')
        self._btn_copy_results.setAccessibleName('复制完整结果路径')
        self._btn_copy_results.clicked.connect(self._copy_result_directory)
        result_row.addWidget(self._btn_copy_results)
        result_page.addLayout(result_row)

        cloud = page()
        self._btn_multi.setProperty("class", "secondary")
        cloud.addWidget(self._btn_multi)
        cloud.addWidget(self._btn_cloud)
        outer.addWidget(footer)
        self._refresh_result_summary()
        self._refresh_capture_source_ui()

    def set_capture_footer_compact(self, compact):
        if compact != getattr(self, '_footer_compact', False):
            self._footer_compact = compact
            if compact:
                self._capture_footer_layout.removeWidget(self._btn_capture)
                self._capture_import_row.insertWidget(0, self._btn_capture, 1)
            else:
                self._capture_import_row.removeWidget(self._btn_capture)
                self._capture_footer_layout.insertWidget(0, self._btn_capture)
        self._action_footer.layout().setDirection(
            QBoxLayout.LeftToRight if compact else QBoxLayout.TopToBottom,
        )
        self._footer_info.setMaximumWidth(280 if compact else 16777215)
        self._action_footer.layout().setStretch(0, 1 if compact else 0)
        self._action_footer.layout().setStretch(1, 2 if compact else 0)

    def _set_board_details(self, expanded):
        self._aux_board_details.setVisible(expanded)
        self._btn_board_details.setArrowType(Qt.NoArrow)
        self._btn_board_details.setIcon(app_icon('chevron-down' if expanded else 'chevron-right'))

    def _toggle_result_details(self, expanded):
        self._lbl_result_summary.setVisible(expanded)
        self._btn_result_details.setArrowType(Qt.NoArrow)
        self._btn_result_details.setIcon(app_icon('chevron-down' if expanded else 'chevron-right'))

    def _copy_result_directory(self):
        if self._result_directory is not None:
            QApplication.clipboard().setText(str(Path(self._result_directory).resolve()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_aux_seq_widget') and len(self._sections) > 2:
            self._apply_inspector_density()

    def _apply_inspector_density(self):
        capture = self._workflow_phase == 'capture'
        compact = capture and self.height() < 720
        key = (compact, capture, self._active_step)
        if getattr(self, '_inspector_density_key', None) == key:
            return
        self._inspector_density_key = key
        for guide in (self._seq_widget, self._aux_seq_widget):
            guide.set_diagram_height(200 if compact else 260)
        outer = self._inspector_scroll.widget().layout()
        margin = 8 if compact else 16
        outer.setContentsMargins(margin, margin, margin, margin)
        outer.setSpacing(4 if compact else 14)
        self._sections[0].content_layout.setSpacing(4 if compact else 8)
        self._lbl_rgb_board.setVisible(not compact)
        self._lbl_disk_hint.setVisible(not compact)
        self._combo_pair.setToolTip(self._lbl_rgb_board.text())
        for name in ('sensor', 'calibration'):
            self._aux_task_labels[name].setVisible(not compact)
            self._aux_task_values[name].setVisible(not compact)
        layout = self._sections[2].content_layout
        layout.setSpacing(4 if compact else 8)
        layout.removeWidget(self._aux_board_section)
        if compact:
            layout.addWidget(self._aux_board_section)
        else:
            layout.insertWidget(layout.indexOf(self._aux_seq_widget), self._aux_board_section)
        self._lbl_aux_status.setVisible(not compact)

    def _apply_workflow_layout(self):
        result = self._workflow_phase == "results"
        for index, section in enumerate(self._sections):
            section.setVisible(not result and index == self._active_step)
            section._header.hide()
        self._result_body.setVisible(result)
        config = self._board.config
        self._lbl_rgb_board.setText(f"ChArUco · {config.cols} × {config.rows} 格子\n"
                                    f"方格 {config.square_length * 1000:g} mm · 标记 {config.marker_length * 1000:g} mm")
        auxiliary = self._active_step == 2
        capture = self._workflow_phase == "capture"
        self._aux_seq_widget.setVisible(auxiliary and capture)
        self._aux_cap_progress.setVisible(auxiliary and capture)
        self._aux_progress_header.setVisible(auxiliary and capture)
        if not auxiliary:
            self._lbl_aux_status.hide()
        phase_names = {"capture": "采集", "calibration": "标定", "results": "结果", "pointcloud": "点云工具"}
        self._lbl_task_phase.setText(f"{'融合' if auxiliary else '双目'}{phase_names[self._workflow_phase]}")
        if getattr(self, '_board_capture_phase', None) != capture:
            self._board_capture_phase = capture
            self._btn_board_details.setChecked(not capture)
        self._apply_inspector_density()
        index = (4 if result else 5 if self._active_step == 3 else
                 (2 if capture else 3) if auxiliary else (0 if capture else 1))
        self._footer_pages.setCurrentIndex(index)
        self._lbl_raw_save.setVisible(capture)
        self._check_wb.setVisible(capture)
        self._refresh_result_summary()
        self._refresh_capture_source_ui()

    def _connect_signals(self):
        self._spin_min.valueChanged.connect(self._on_min_pairs_changed)
        self._btn_capture.clicked.connect(self._on_capture)
        self._btn_undo.clicked.connect(self._on_undo_capture)
        self._btn_auto.toggled.connect(self._on_auto_toggle)
        self._btn_import.clicked.connect(self._on_import_images)
        self._btn_swap_offline.clicked.connect(self._on_swap_offline_images)
        self._combo_calib_strategy.currentIndexChanged.connect(
            self._on_calib_strategy_changed
        )
        self._combo_data_filter.currentIndexChanged.connect(
            self._on_data_filter_changed
        )
        self._btn_capture_aux.clicked.connect(self._on_capture_aux)
        self._btn_import_aux.clicked.connect(self._on_import_aux_images)
        self._btn_calib_aux.clicked.connect(self._on_calibrate_aux)
        self._btn_calib_all.clicked.connect(self._on_calibrate_all)
        self._btn_multi.clicked.connect(self._on_calibrate_multiview)
        self._btn_cloud.clicked.connect(self._on_generate_cloud)

        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(2000)
        self._auto_timer.timeout.connect(self._on_auto_capture)
        self._combo_pair.currentTextChanged.connect(self._on_capture_target_changed)
        self._aux_readiness_timer = QTimer(self)
        self._aux_readiness_timer.setInterval(self.AUX_READINESS_INTERVAL_MS)
        self._aux_readiness_timer.timeout.connect(self._refresh_aux_readiness)
        self._aux_readiness_timer.start()
        self._sm.camera_connected.connect(self._on_aux_camera_connected)
        self._sm.camera_disconnected.connect(self._on_aux_camera_disconnected)

        self._timeline.step_clicked.connect(self._activate_step)
        self._seq_widget.user_navigated.connect(self._on_sequence_user_nav)
        self._aux_seq_widget.user_navigated.connect(self._on_aux_sequence_user_nav)
        for i, sec in enumerate(self._sections):
            idx = i
            sec.toggled.connect(lambda _, ii=idx: self._on_section_toggled(ii))

    # ── Navigation ────────────────────────────────────────────

    def _activate_step(self, step: int, *, phase: str | None = None):
        previous_step = self._active_step
        if phase is not None:
            self._workflow_phase = phase
        elif step in (0, 1, 3):
            self._workflow_phase = {0: "capture", 1: "calibration", 3: "pointcloud"}[step]
        elif previous_step != 2:
            self._workflow_phase = "capture"
        if step != 0 and self._auto_timer.isActive():
            self._auto_timer.stop()
            self._btn_auto.setChecked(False)
            self._btn_auto.setText("自动采集")
        self._active_step = step
        self._timeline.set_step(step)
        for i, sec in enumerate(self._sections):
            sec.set_expanded(i == step)
        sequence = self._aux_capture_sequence if step == 2 else self._capture_sequence
        self.sequence_updated.emit(sequence)
        self.preview_mode_changed.emit(step == 2)
        self._apply_workflow_layout()
        self.workflow_changed.emit(self._workflow_phase, step == 2)
        self._refresh_preview_pairs()

    def _on_section_toggled(self, index: int):
        self._activate_step(index)

    @Slot()
    def _on_sequence_user_nav(self):
        """User changed active guide step (grid / prev-next / skip)."""
        self._activate_step(0)

    @Slot()
    def _on_aux_sequence_user_nav(self):
        """User changed active multimodal guide step."""
        self._activate_step(2)

    # ── Pair selector ────────────────────────────────────────

    def refresh_pairs(self):
        """Sync the pair combo and calibration rows with current StreamManager state."""
        cur = self._combo_pair.currentText()
        with QSignalBlocker(self._combo_pair):
            self._combo_pair.clear()
            self._combo_pair.addItem("全部相机组")
            for name in self._sm.stereo_pairs:
                self._combo_pair.addItem(name)
            idx = self._combo_pair.findText(cur)
            if idx >= 0:
                self._combo_pair.setCurrentIndex(idx)
        self._rebuild_pair_rows()
        self._refresh_aux_board_info()
        self._refresh_aux_readiness()
        self.preview_mode_changed.emit(self._active_step == 2)
        self._update_capture_stats()
        self._refresh_preview_pairs()
        self._refresh_capture_source_ui()

    @Slot(str)
    def _on_capture_target_changed(self, text):
        self._update_capture_stats()
        self._refresh_preview_pairs()
        self._refresh_capture_source_ui()
        self.data_source_changed.emit()

    def _selected_pairs(self) -> Dict:
        all_pairs = self._sm.stereo_pairs
        sel = self._combo_pair.currentText()
        if sel and sel != "全部相机组" and sel in all_pairs:
            return {sel: all_pairs[sel]}
        return dict(all_pairs)

    def _on_min_pairs_changed(self, value):
        for calibrator in (*self._intr_cals.values(), *self._stereo_cals.values()):
            calibrator.min_frames = value
        self._update_capture_stats()
        self._sync_pair_states()

    # ── Dynamic pair rows ─────────────────────────────────────

    def _rebuild_pair_rows(self):
        if hasattr(self, '_combo_model_pair'):
            names = list(self._sm.stereo_pairs)
            if names != [self._combo_model_pair.itemText(i) for i in range(self._combo_model_pair.count())]:
                previous = self._combo_model_pair.currentText()
                with QSignalBlocker(self._combo_model_pair):
                    self._combo_model_pair.clear()
                    self._combo_model_pair.addItems(names)
                    if previous in names:
                        self._combo_model_pair.setCurrentText(previous)
        for name in list(self._pair_rows):
            if name not in self._sm.stereo_pairs:
                row = self._pair_rows.pop(name)
                self._calib_container.removeWidget(row)
                row.hide()
                row.deleteLater()
        for index, name in enumerate(self._sm.stereo_pairs):
            row = self._pair_rows.get(name)
            if row is None:
                row = _PairRow(name, self)
                row.btn.clicked.connect(lambda _, n=name: self._calibrate_pair(n))
                self._pair_rows[name] = row
            if self._calib_container.indexOf(row) != index:
                self._calib_container.insertWidget(index, row)

        self._sync_pair_states()

    def _pair_data_ready(self, pair_name):
        cfg = self._sm.stereo_pairs.get(pair_name)
        if cfg is None:
            return False
        calibrators = [self._stereo_cals.get(pair_name), self._intr_cals.get(cfg.left.camera_id),
                       self._intr_cals.get(cfg.right.camera_id)]
        return all(cal is not None and cal.ready for cal in calibrators)

    def _sync_pair_states(self):
        self._refresh_model_choice_ui()
        any_ready = False
        all_calibrated = True

        for name, row in self._pair_rows.items():
            cal = self._stereo_cals.get(name)
            done = name in self._pair_calibs
            ready = self._pair_data_ready(name)

            if done:
                pc = self._pair_calibs[name]
                rms_i_l = pc.left_intrinsics.rms_error
                rms_i_r = pc.right_intrinsics.rms_error
                rms_s = pc.stereo.rms_error
                bl_mm = np.linalg.norm(pc.stereo.T) * 1000.0
                grades = [self._evaluator.evaluate_intrinsic(pc.left_intrinsics),
                          self._evaluator.evaluate_intrinsic(pc.right_intrinsics),
                          self._evaluator.evaluate_stereo(pc)]
                worst = min(grades, key=lambda quality: quality.score).grade
                color = DANGER if worst == Grade.POOR else WARNING if worst == Grade.FAIR else SUCCESS
                row.set_status(
                    f"已计算 · 内参 RMS\n左 {rms_i_l:.3f} px · 右 {rms_i_r:.3f} px\n"
                    f"双目 RMS {rms_s:.3f} px\n基线 {bl_mm:.1f} mm",
                    color,
                )
                row.btn.setEnabled(ready)
                row.btn.setText("重新标定")
                any_ready = any_ready or ready
            elif ready:
                row.set_status(f"{cal.num_frames} 对已就绪，可开始标定", ACCENT)
                row.btn.setEnabled(True)
                row.btn.setText("标定")
                any_ready = True
                all_calibrated = False
            elif cal is not None:
                pc = self._pair_capture_press_count.get(name, 0)
                cfg = self._sm.stereo_pairs[name]
                left = self._intr_cals.get(cfg.left.camera_id)
                right = self._intr_cals.get(cfg.right.camera_id)
                row.set_status(
                    f"数据记录 {pc} 对 · 有效配对 {cal.num_frames}/{cal.min_frames} · "
                    f"左内参 {left.num_frames if left else 0} / 右内参 {right.num_frames if right else 0}",
                    TEXT_DIM,
                )
                row.btn.setEnabled(False)
                row.btn.setText("标定")
                all_calibrated = False
            else:
                row.set_status("等待采集数据...", TEXT_HINT)
                row.btn.setEnabled(False)
                row.btn.setText("标定")
                all_calibrated = False

        self._btn_calib_all.setEnabled(any_ready)
        self._btn_frame_diagnostics.setEnabled(bool(self._pair_calibs))
        self._btn_calib_all.setText('开始标定' if len(self._pair_rows) == 1 else '标定全部就绪组')
        if (self._calib_worker is not None or self._aux_dataset_busy() or self._stereo_import_worker is not None
                or self._pending_save_tokens or self._failed_save_jobs):
            self._btn_calib_all.setEnabled(False)
            for row in self._pair_rows.values():
                row.btn.setEnabled(False)
                row.btn.setToolTip('请先等待原图保存完成，或重试/撤销保存失败的采集'
                                   if self._pending_save_tokens or self._failed_save_jobs else '后台任务进行中')
        else:
            for name, row in self._pair_rows.items():
                row.btn.setToolTip('使用当前数据重新计算并生成标定文件' if self._pair_data_ready(name)
                                   else '左右内参与有效配对均须达到最少配对数')

        n_pairs = len(self._sm.stereo_pairs)
        n_done = len(self._pair_calibs)

        if all_calibrated and n_pairs > 0:
            self._timeline.mark_completed(1)
            self._sections[1].set_status_hint("✓ 全部完成", SUCCESS)

            if n_pairs >= 2:
                self._btn_multi.setEnabled(True)
                self._btn_cloud.setEnabled(False)
                self._lbl_fusion_info.setText(
                    f"已标定 {n_done} 组双目，可进行联合优化后生成点云"
                )
            else:
                self._btn_multi.setEnabled(False)
                self._btn_cloud.setEnabled(True)
                self._lbl_fusion_info.setText("单组双目标定完成，可直接生成点云")
        else:
            self._btn_multi.setEnabled(False)
            if n_pairs >= 2:
                self._lbl_fusion_info.setText(
                    f"已标定 {n_done}/{n_pairs} 组，全部完成后可融合"
                )
            elif n_pairs == 0:
                self._lbl_fusion_info.setText("请先配置相机组")

    # ── Capture ───────────────────────────────────────────────

    def _refresh_disk_hint(self):
        caption = "创建会话并采集" if self._session is None else "采集配对图像"
        if self._pending_save_tokens:
            caption = f'正在保存 {len(self._pending_save_tokens)} 组'
        if not self._white_balance_guard.busy:
            self._btn_capture.setText(caption)
            self._btn_capture_aux.setText(caption)
        if self._session is None:
            self._lbl_disk_hint.setText(
                "原图保存：未启用（无采集会话）"
            )
            self._lbl_disk_hint.setStyleSheet(f"font-size:12px; color:{WARNING};")
            self._lbl_raw_save.setText("无采集会话 · 原图未落盘")
            self._lbl_raw_save.setStyleSheet(f"font-size:12px; color:{WARNING}; background:transparent;")
            self._lbl_raw_save.setToolTip("当前仅保留内存数据，原始图片尚未保存到会话目录")
        else:
            base = self._session.resolved_images_base()
            self._lbl_disk_hint.setText(
                f"原图保存：{self._session.name}"
            )
            self._lbl_disk_hint.setToolTip(str(base))
            self._lbl_disk_hint.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
            self._lbl_raw_save.setText("原图保存已启用")
            self._lbl_raw_save.setToolTip(str(base))
            self._lbl_raw_save.setStyleSheet(f"font-size:12px; color:{TEXT_DIM}; background:transparent;")
            if self._active_step != 2:
                pending, failed = len(self._pending_save_tokens), len(self._failed_save_jobs)
                self._lbl_raw_save.setText(
                    f'本次已落盘 {len(self._saved_capture_tokens)} 组 · 保存中 {pending} · 失败 {failed}'
                )
                if failed:
                    self._lbl_raw_save.setStyleSheet(f'color:{DANGER}; background:transparent; font-size:12px;')
        self._btn_retry_save.setVisible(bool(self._failed_save_jobs) and self._active_step != 2)
        self._btn_retry_save.setEnabled(not self._pending_save_tokens and not self._white_balance_guard.busy)
        if self.offline_selection and self._active_step != 2:
            self._lbl_raw_save.setText('离线原图保留在源目录')
            self._lbl_raw_save.setToolTip(str(self._offline_source_folder or '导入目录；原文件不修改'))
            self._lbl_raw_save.setStyleSheet(f"font-size:12px; color:{TEXT_DIM}; background:transparent;")
        if self._active_step == 2 and (self._aux_images or self._aux_imported_count):
            parts = []
            paths = []
            if self._aux_imported_count:
                parts.append(f'离线 {self._aux_imported_count} 组保留在源目录')
                paths.append(str(self._aux_source_folder or '导入源目录；未复制到会话'))
            if self._aux_saved_count:
                parts.append(f'实时原图已落盘 {self._aux_saved_count} 组')
                if self._session is not None:
                    paths.append(str(self._session.session_dir / self._session.MULTIMODAL_DIR))
            self._lbl_raw_save.setText(' · '.join(parts) or '原图仅在内存中，未确认落盘')
            self._lbl_raw_save.setToolTip('\n'.join(paths))

    def _refresh_capture_readiness(self):
        visible = self._workflow_phase == 'capture' and self._active_step == 0 and not self.offline_selection
        self._lbl_capture_readiness.setVisible(visible)
        if not visible:
            return
        pairs = self._selected_pairs()
        statuses = [(name, *self._sm.pair_capture_status(name)) for name in pairs]
        text = '；'.join(f'{name}: {status}' for name, _, status in statuses) or '请先配置左右相机'
        self._lbl_capture_readiness.setText(
            text if len(statuses) <= 1 else f'{statuses[0][0]}: {statuses[0][2]} · 共 {len(statuses)} 组'
        )
        board = self._board.config
        self._lbl_capture_readiness.setToolTip(
            f'{text}\nChArUco {board.cols}×{board.rows} 格，方格 {board.square_length * 1000:g} mm\n'
            + ('白平衡：' + self._combo_wb_mode.currentText() if self._check_wb.isChecked() else '白平衡管理已关闭')
            + '\n保存目录：' + (str(self._session.resolved_images_base()) if self._session else '采集时创建会话')
        )
        ready = bool(statuses) and all(state == 'ready' for _, state, _ in statuses)
        self._lbl_capture_readiness.setStyleSheet(
            f'color:{SUCCESS if ready else WARNING}; background:transparent; font-size:12px;'
        )
        self._btn_capture.setToolTip(text + '\n角点和清晰度在采图时按原始图像检查')

    def _refresh_capture_source_ui(self):
        offline = self.offline_selection and self._active_step != 2
        capture = self._workflow_phase == 'capture'
        busy = getattr(self, '_ui_operation_busy', False)
        for control in (self._btn_capture, self._btn_auto, self._btn_undo, self._btn_skip_stereo):
            control.setVisible(not offline)
        self._check_wb.setVisible(capture and not offline)
        self._combo_wb_mode.setVisible(capture and not offline and self._check_wb.isChecked())
        if offline and self._btn_auto.isChecked():
            self._btn_auto.setChecked(False)
        self._seq_widget.setVisible(not offline)
        self._btn_capture.setEnabled(not busy and not offline)
        self._btn_auto.setEnabled(not busy and not offline)
        if offline != getattr(self, '_offline_footer_mode', False):
            self._offline_footer_mode = offline
            if offline:
                self._capture_nav_row.removeWidget(self._btn_to_calibration)
                self._capture_import_row.insertWidget(0, self._btn_to_calibration, 1)
            else:
                self._capture_import_row.removeWidget(self._btn_to_calibration)
                self._capture_nav_row.addWidget(self._btn_to_calibration)
        new_class = 'primary' if offline else None
        if self._btn_to_calibration.property('class') != new_class:
            self._btn_to_calibration.setProperty('class', new_class)
            self._btn_to_calibration.style().unpolish(self._btn_to_calibration)
            self._btn_to_calibration.style().polish(self._btn_to_calibration)
        self._btn_to_calibration.setEnabled(not busy)
        model = '针孔' if self._camera_model == CameraModel.PINHOLE else '鱼眼'
        board = self._board.config
        self._lbl_calib_hint.setText(
            f"{model}模型 · ChArUco {board.cols}×{board.rows} 格\n"
            f"方格 {board.square_length * 1000:g} mm · 标记 {board.marker_length * 1000:g} mm"
        )
        with QSignalBlocker(self._combo_calib_strategy):
            self._combo_calib_strategy.setItemText(
                1, '稳健：固定鱼眼 k3/k4' if self._camera_model == CameraModel.FISHEYE
                else '稳健：固定切向/高阶畸变',
            )
        fish_index = self._combo_data_filter.findData('fisheye_common')
        self._combo_data_filter.model().item(fish_index).setEnabled(self._camera_model == CameraModel.FISHEYE)
        if self._camera_model != CameraModel.FISHEYE and self._combo_data_filter.currentData() == 'fisheye_common':
            with QSignalBlocker(self._combo_data_filter):
                self._combo_data_filter.setCurrentIndex(0)
            self._data_filter_key = 'all'
        self._btn_configure_stereo.setEnabled(not busy)
        self._refresh_disk_hint()
        self._refresh_capture_readiness()

    def _refresh_undo_btn(self):
        self._btn_undo.setEnabled(bool(self._undo_stack) and self._calib_worker is None)

    def _start_frame_save(
        self,
        pair_name: str,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[Optional[int], Optional[str]]:
        if self._session is None:
            return None, None
        if self._next_session_frame_idx is None:
            self._next_session_frame_idx = self._session.frame_count
        frame_idx = self._next_session_frame_idx
        self._next_session_frame_idx += 1
        token = uuid.uuid4().hex
        self._pending_save_tokens.add(token)
        self._save_sessions[token] = self._session
        job = (token, pair_name, left.copy(), right.copy(), frame_idx)
        self._save_jobs[token] = job
        self._save_queue.append(job)
        self._start_next_frame_save()
        self._refresh_disk_hint()
        return frame_idx, token

    def _start_next_frame_save(self):
        if self._save_workers or not self._save_queue or self._session is None:
            return
        token, pair_name, left, right, frame_idx = self._save_queue.pop(0)
        if token in self._canceled_save_tokens:
            self._pending_save_tokens.discard(token)
            self._canceled_save_tokens.discard(token)
            self._save_sessions.pop(token, None)
            self._save_jobs.pop(token, None)
            self._start_next_frame_save()
            self._refresh_disk_hint()
            return
        worker = FrameSaveWorker(
            token,
            self._save_sessions[token],
            pair_name,
            left,
            right,
            frame_idx,
            parent=self,
        )
        worker.saved.connect(self._on_frame_save_finished)
        worker.failed.connect(self._on_frame_save_failed)
        worker.finished.connect(worker.deleteLater)
        self._save_workers[token] = worker
        worker.start()

    @Slot(str, str, int)
    def _on_frame_save_finished(self, token: str, pair_name: str, frame_idx: int):
        self._save_workers.pop(token, None)
        self._pending_save_tokens.discard(token)
        session = self._save_sessions.pop(token, None)
        self._save_jobs.pop(token, None)
        if token in self._canceled_save_tokens:
            self._canceled_save_tokens.discard(token)
            if session is not None:
                try:
                    session.delete_saved_pair_frame(pair_name, frame_idx)
                except Exception as exc:
                    logger.exception("Undo saved frame failed: pair=%s frame=%s", pair_name, frame_idx)
                    self._log(f"{pair_name}: 撤销图片失败（帧 {frame_idx}）— {exc}")
        elif session is not None and session is self._session:
            self._saved_capture_tokens.add(token)
            self._guidance_failed_saves.discard((id(session), pair_name, f'{frame_idx:04d}'))
            self._log(f'{pair_name}: 原图已落盘（帧 {frame_idx}）')
        self._start_next_frame_save()
        self._refresh_disk_hint()
        self._sync_pair_states()
        self._update_capture_stats()

    @Slot(str, str, int, str)
    def _on_frame_save_failed(self, token: str, pair_name: str, frame_idx: int, reason: str):
        self._save_workers.pop(token, None)
        self._pending_save_tokens.discard(token)
        session = self._save_sessions.pop(token, None)
        job = self._save_jobs.pop(token, None)
        if token in self._canceled_save_tokens:
            self._canceled_save_tokens.discard(token)
            self._start_next_frame_save()
            self._refresh_disk_hint()
            self._sync_pair_states()
            self._update_capture_stats()
            return
        if job is not None and session is self._session:
            self._failed_save_jobs[token] = (session, job)
        self._log(f"{pair_name}: 保存图片失败（帧 {frame_idx}）— {reason}")
        self._guidance_failed_saves.add((id(session), pair_name, f"{frame_idx:04d}"))
        self._start_next_frame_save()
        self._btn_auto.setChecked(False)
        self._refresh_disk_hint()
        self._sync_pair_states()
        self._update_capture_stats()

    def _retry_failed_saves(self):
        if self.has_active_background_tasks():
            self._log('请等待当前任务结束后重试保存')
            return
        for token, (session, job) in list(self._failed_save_jobs.items()):
            if session is not self._session:
                continue
            self._pending_save_tokens.add(token)
            self._save_sessions[token] = session
            self._save_jobs[token] = job
            self._save_queue.append(job)
            del self._failed_save_jobs[token]
        self._start_next_frame_save()
        self._refresh_disk_hint()
        self._sync_pair_states()

    def _undo_apply_entry(self, e: _UndoCaptureEntry) -> None:
        self._failed_save_jobs.pop(e.save_token, None)
        self._saved_capture_tokens.discard(e.save_token)
        if e.session_frame_idx is not None:
            self._guidance_failed_saves.discard((id(e.session), e.pair_name, f'{e.session_frame_idx:04d}'))
        if e.pop_stereo and e.pop_intrinsic_l and e.pop_intrinsic_r and self._live_preview_indices.get(e.pair_name):
            self._live_preview_indices[e.pair_name].pop()
        if e.pop_stereo and e.pair_name in self._stereo_cals:
            self._stereo_cals[e.pair_name].pop_last_frame()
        if e.pop_intrinsic_r and e.rid in self._intr_cals:
            self._intr_cals[e.rid].pop_last_frame()
            if self._fqs.get(e.rid):
                self._fqs[e.rid].pop()
        if e.pop_intrinsic_l and e.lid in self._intr_cals:
            self._intr_cals[e.lid].pop_last_frame()
            if self._fqs.get(e.lid):
                self._fqs[e.lid].pop()
        if e.save_token is not None and e.save_token in self._pending_save_tokens:
            self._canceled_save_tokens.add(e.save_token)
        elif e.session_frame_idx is not None and e.session is not None:
            try:
                e.session.delete_saved_pair_frame(e.pair_name, e.session_frame_idx)
            except Exception as exc:
                logger.exception("Delete undone capture failed: pair=%s frame=%s", e.pair_name, e.session_frame_idx)
                self._log(f"{e.pair_name}: 内存已撤销，但磁盘图片删除失败（帧 {e.session_frame_idx}）：{exc}")

    @Slot()
    def _on_undo_capture(self):
        if self._calib_worker is not None or self._stereo_import_worker is not None:
            self._log("标定任务正在运行，请稍后再撤销")
            return
        if not self._undo_stack:
            return
        batch, advanced_guide = self._undo_stack.pop()
        for ent in reversed(batch):
            self._undo_apply_entry(ent)
            self._invalidate_pair_result(ent.pair_name)
            pn = ent.pair_name
            self._pair_capture_press_count[pn] = max(
                0, self._pair_capture_press_count.get(pn, 0) - 1
            )
        if advanced_guide:
            self._capture_sequence.rewind_last_advance()
            self._seq_widget.refresh()
            self.sequence_updated.emit(self._capture_sequence)
        self._refresh_undo_btn()
        self._update_capture_stats()
        self._sync_pair_states()
        self._log("已撤销上一帧采集")
        self._refresh_preview_pairs()
        self._refresh_disk_hint()

    @Slot()
    def _on_capture(self):
        with perf_timer("capture click", threshold_ms=100.0):
            self._capture_impl()

    def _capture_impl(self):
        automatic = getattr(self, '_auto_capture_request', False)
        if self._pending_save_tokens or self._failed_save_jobs:
            self._log('请等待原图保存完成；保存失败请先重试或撤销对应采集')
            return
        if self._aux_dataset_busy():
            self._log("辅助任务进行中，请稍候再修改双目数据")
            return
        if self._calib_worker is not None or self._aux_worker is not None:
            self._log("标定任务正在运行，请稍后再采集")
            return
        pairs = self._selected_pairs()
        if not pairs:
            self._log("未配置相机组，请先通过 ⚙ 添加相机")
            return

        available = any(self._sm.get_sync_pair(name) is not None or
                        (not automatic and self._sm.get_latest_pair_relaxed(name) is not None)
                        for name in pairs)
        if not available:
            self._log('采集未开始：' + '；'.join(
                f'{name}: {self._sm.pair_capture_status(name)[1]}' for name in pairs
            ))
            self._refresh_capture_readiness()
            return
        if not self._ensure_capture_session():
            return
        resume_capture = self._on_auto_capture if automatic else self._on_capture
        if not self._prepare_white_balance(pairs, resume_capture):
            return

        self.capture_started.emit()

        captured = 0
        batch_entries: List[_UndoCaptureEntry] = []
        prepared = []

        for name, pair_cfg in pairs.items():
            pair_frames = self._sm.get_sync_pair(name)
            relaxed_note = ""
            if pair_frames is None:
                if automatic:
                    self._log(f'{name}: 未满足软件时间配对，本次不自动采集（非硬件同步）')
                    continue
                pair_frames = self._sm.get_latest_pair_relaxed(name)
                if pair_frames is None:
                    continue
                tol_ms = self._sm.sync_tolerance_seconds * 1000.0
                relaxed_note = (
                    f"{name}: 左右帧未满足严格同步（≤{tol_ms:.0f}ms），"
                    "已改用各自最新帧；进度仍会累计。"
                )
            left, right = pair_frames
            lid = pair_cfg.left.camera_id
            rid = pair_cfg.right.camera_id
            min_f = self._spin_min.value()

            if name not in self._stereo_cals:
                self._intr_cals[lid] = IntrinsicCalibrator(
                    self._board, self._camera_model, min_f
                )
                self._intr_cals[rid] = IntrinsicCalibrator(
                    self._board, self._camera_model, min_f
                )
                self._stereo_cals[name] = StereoCalibrator(
                    self._board, self._camera_model, min_f
                )
                self._fqs[lid] = []
                self._fqs[rid] = []
                self._rebuild_pair_rows()

            det_l = det_r = None
            if automatic:
                validator = getattr(self, '_capture_validator', None)
                if validator is None:
                    self._log(f'{name}: 自动采集缺少原图验收，已跳过')
                    continue
                try:
                    det_l = self._intr_cals[lid].detector.detect(left)
                    det_r = self._intr_cals[rid].detector.detect(right)
                    accepted, reason = validator(name, left, right, det_l, det_r)
                except Exception as exc:
                    logger.exception('Automatic raw capture validation failed: pair=%s', name)
                    self._log(f'{name}: 自动采集原图验收异常 — {exc}；未保存或推进计数')
                    continue
                if not accepted:
                    self._log(f'{name}: 自动采集原图未通过 — {reason}；未保存或推进计数')
                    continue
            prepared.append((name, lid, rid, left, right, det_l, det_r, relaxed_note))

        # Validate the whole batch before queued saves or new observations change readiness.
        for name, lid, rid, left, right, det_l, det_r, relaxed_note in prepared:
            n_ib_l = self._intr_cals[lid].num_frames
            n_ib_r = self._intr_cals[rid].num_frames
            n_sb = self._stereo_cals[name].num_frames
            source_id = (f"{self._next_session_frame_idx:04d}" if self._next_session_frame_idx is not None
                         else str(self._pair_capture_press_count.get(name, 0) + 1))
            if automatic:
                self._intr_cals[lid].add_detection(det_l, image=left, frame_id=source_id)
                self._intr_cals[rid].add_detection(det_r, image=right, frame_id=source_id)
            else:
                det_l = self._intr_cals[lid].add_frame(left, frame_id=source_id)
                det_r = self._intr_cals[rid].add_frame(right, frame_id=source_id)
            similar_l = self._intr_cals[lid].find_similar_detection(
                det_l, exclude_last=self._intr_cals[lid].num_frames > n_ib_l,
            )
            similar_r = self._intr_cals[rid].find_similar_detection(
                det_r, exclude_last=self._intr_cals[rid].num_frames > n_ib_r,
            )
            if similar_l is not None and similar_r is not None:
                self._log(
                    f"{name}: 重复构图提醒 — 左图近似样本 {similar_l + 1}，"
                    f"右图近似样本 {similar_r + 1}；本次仍保留，建议改变位置、大小或倾角"
                )
            self._store_stereo_detection(name, det_l, det_r, source=source_id)
            if (self._stereo_cals[name].num_frames > n_sb and
                    self._intr_cals[lid].num_frames > n_ib_l and self._intr_cals[rid].num_frames > n_ib_r):
                self._live_preview_indices.setdefault(name, []).append((
                    len(self._intr_cals[lid].stored_images) - 1, len(self._intr_cals[rid].stored_images) - 1, source_id,
                ))
            if (
                self._intr_cals[lid].num_frames > n_ib_l
                or self._intr_cals[rid].num_frames > n_ib_r
                or self._stereo_cals[name].num_frames > n_sb
            ):
                self._invalidate_pair_result(name)

            fi = len(self._fqs[lid])
            fq_l = self._assessor.assess_frame(left, det_l.num_corners, fi)
            fq_r = self._assessor.assess_frame(right, det_r.num_corners, fi)
            self._fqs[lid].append(fq_l)
            self._fqs[rid].append(fq_r)

            sess_idx = None
            save_token = None
            if self._session:
                sess_idx, save_token = self._start_frame_save(name, left, right)

            if relaxed_note:
                self._log(relaxed_note)

            self._pair_capture_press_count[name] = (
                self._pair_capture_press_count.get(name, 0) + 1
            )

            captured += 1
            self.frame_captured.emit(lid, det_l)
            self.frame_captured.emit(rid, det_r)

            sc = self._stereo_cals[name]
            ql = "OK" if fq_l.usable else "⚠"
            qr = "OK" if fq_r.usable else "⚠"
            self._log(
                f"{name} [{source_id}]: L={det_l.num_corners}[{ql}] "
                f"R={det_r.num_corners}[{qr}] 帧{sc.num_frames}"
            )

            batch_entries.append(
                _UndoCaptureEntry(
                    pair_name=name,
                    lid=lid,
                    rid=rid,
                    session_frame_idx=sess_idx,
                    save_token=save_token,
                    pop_intrinsic_l=self._intr_cals[lid].num_frames > n_ib_l,
                    pop_intrinsic_r=self._intr_cals[rid].num_frames > n_ib_r,
                    pop_stereo=self._stereo_cals[name].num_frames > n_sb,
                    session=self._session,
                )
            )

        if captured == 0:
            self._log(
                "本次未采集：无可用配对或原图验收未通过，详见上述原因。"
            )
            return

        advanced_guide = False
        if not self._capture_sequence.finished:
            self._capture_sequence.advance()
            advanced_guide = True
            self._seq_widget.refresh()
            self.sequence_updated.emit(self._capture_sequence)

        self._undo_stack.append((batch_entries, advanced_guide))
        self._refresh_undo_btn()

        self._update_capture_stats()
        self._sync_pair_states()
        self._refresh_preview_pairs()

    @Slot()
    def _on_import_images(self):
        """Load pre-captured image pairs from a folder (NNNN_left.png / NNNN_right.png)."""
        if self._failed_save_jobs:
            self._log('仍有原图保存失败，请先重试保存或撤销对应采集后再导入')
            return
        if self._aux_dataset_busy():
            self._log("辅助任务进行中，请稍候再导入双目数据")
            return
        if self._stereo_import_worker is not None:
            self._log("图片导入正在进行，请稍候")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if not folder:
            return

        import os

        self._log(f"正在后台导入图片: {os.path.basename(folder)}")
        self._pending_import_folder = Path(folder).resolve()
        self._set_import_busy(True)
        worker = StereoImageImportWorker(folder, self._board, parent=self)
        worker.log.connect(self._log)
        worker.finished_ok.connect(self._on_import_images_ready)
        worker.failed.connect(self._on_import_images_failed)
        worker.finished.connect(worker.deleteLater)
        self._stereo_import_worker = worker
        self._refresh_aux_readiness()
        worker.start()

    @Slot(object)
    def _on_import_images_ready(self, img_pairs):
        self._offline_frame_qualities = dict(getattr(self._stereo_import_worker, 'frame_qualities', {}))
        self._stereo_import_worker = None
        self._set_import_busy(False)
        self._log(f"导入 {len(img_pairs)} 对图片")

        self._offline_import_records = list(img_pairs)
        self._offline_source_folder = self._pending_import_folder
        self._pending_import_folder = None
        self._offline_lr_swapped = False
        self._import_view_idx = 0
        self._load_offline_records(swapped=False, log_frames=True)
        self.offline_imported.emit()

    def _load_offline_records(self, swapped: bool, log_frames: bool = False):
        img_pairs = list(self._offline_import_records)
        if not img_pairs:
            self._log("没有可交换的离线图片，请先导入图片")
            return

        pair_name = "offline"
        lid, rid = "offline_L", "offline_R"
        min_f = max(self._spin_min.minimum(), min(self._spin_min.value(), len(img_pairs)))
        self._spin_min.setValue(min_f)

        self._reset_offline_pair(pair_name, lid, rid, min_f)
        self._combo_pair.setCurrentText(pair_name)

        for idx, img_l, img_r, det_l, det_r in img_pairs:
            if swapped:
                img_l, img_r = img_r, img_l
                det_l, det_r = det_r, det_l
            self._intr_cals[lid].add_detection(det_l, image=img_l, frame_id=str(idx))
            self._intr_cals[rid].add_detection(det_r, image=img_r, frame_id=str(idx))
            self._store_stereo_detection(pair_name, det_l, det_r, source=str(idx))
            for camera_id in (lid, rid):
                reason = self._intr_cals[camera_id].last_rejection_reason
                if reason:
                    self._log(f"[{idx}] {camera_id}: 未用于内参标定 — {reason}")

            fi = len(self._fqs[lid])
            cached_quality = self._offline_frame_qualities.get(str(idx))
            if cached_quality is not None:
                fq_l, fq_r = cached_quality[::-1] if swapped else cached_quality
            else:
                fq_l = self._assessor.assess_frame(img_l, det_l.num_corners, fi)
                fq_r = self._assessor.assess_frame(img_r, det_r.num_corners, fi)
            self._fqs[lid].append(fq_l)
            self._fqs[rid].append(fq_r)

            if log_frames:
                self._log(
                    f"  [{idx}] L={det_l.num_corners} R={det_r.num_corners} "
                    f"帧{self._stereo_cals[pair_name].num_frames}"
                )

        displayed_pairs = [
            (ir, il) if swapped else (il, ir)
            for _, il, ir, _, _ in img_pairs
        ]
        self._import_last_pair = displayed_pairs[-1]
        self._imported_frames = [(lid, rid, il, ir) for il, ir in displayed_pairs]
        self._imported_pair_frames = {pair_name: displayed_pairs}
        self._import_view_idx = min(
            getattr(self, "_import_view_idx", 0),
            len(self._imported_frames) - 1,
        )

        self._seq_widget.refresh()
        self.sequence_updated.emit(self._capture_sequence)
        self._update_capture_stats()
        self._sync_pair_states()
        self._timeline.mark_completed(0)
        usable = self._stereo_cals[pair_name].num_frames
        self._sections[0].set_status_hint(
            f"已加载 {len(img_pairs)} 对 · 有效配对 {usable} 对",
            SUCCESS if self._pair_data_ready(pair_name) else WARNING,
        )
        direction = "（已交换左右图）" if swapped else ""
        self._log(f"导入完成{direction}，{len(img_pairs)} 对已加载，{self._stereo_cals[pair_name].num_frames} 对可用于标定")

        self._pair_capture_press_count[pair_name] = len(img_pairs)

        self._show_imported_frame(self._import_view_idx)
        self._btn_import_prev.setVisible(True)
        self._btn_import_next.setVisible(True)
        self._lbl_import_nav.setVisible(True)
        self._btn_swap_offline.setVisible(True)
        self._btn_swap_offline.setEnabled(self._calib_worker is None)
        self._btn_swap_offline.setText(
            "恢复离线左右图" if swapped else "交换离线左右图"
        )
        self._update_import_nav_label()
        self._btn_import_prev.hide()
        self._btn_import_next.hide()
        self._lbl_import_nav.hide()
        self._refresh_preview_pairs()
        self._refresh_capture_source_ui()
        self.data_source_changed.emit()

    @Slot()
    def _on_swap_offline_images(self):
        if not self._offline_import_records:
            self._log("请先导入离线左右图")
            return
        if self._calib_worker is not None:
            self._log("标定任务正在运行，请稍后再交换离线左右图")
            return
        self._offline_lr_swapped = not self._offline_lr_swapped
        self._load_offline_records(swapped=self._offline_lr_swapped)
        if self._offline_lr_swapped:
            self._log("离线左右图已交换：offline_L 使用原右图，offline_R 使用原左图")
        else:
            self._log("离线左右图已恢复：offline_L 使用原左图，offline_R 使用原右图")

    @Slot(str)
    def _on_import_images_failed(self, message: str):
        self._stereo_import_worker = None
        self._pending_import_folder = None
        self._set_import_busy(False)
        self._log('离线导入失败：' + message.splitlines()[0])

    def _reset_offline_pair(self, pair_name: str, lid: str, rid: str, min_f: int):
        self._live_preview_indices.pop(pair_name, None)
        from ...streaming.stream_manager import CameraConfig, StereoPairConfig

        if pair_name in self._sm.stereo_pairs:
            self._sm.remove_stereo_pair(pair_name)
        for cid in (lid, rid):
            self._intr_cals.pop(cid, None)
            self._intrinsics.pop(cid, None)
            self._fqs.pop(cid, None)
        self._stereo_cals.pop(pair_name, None)
        self._pair_calibs.pop(pair_name, None)
        self._pair_capture_press_count.pop(pair_name, None)
        self._rig = None
        self.calibration_reset.emit()

        virt_pair = StereoPairConfig(
            name=pair_name,
            left=CameraConfig(camera_id=lid, url="", role="left", group=pair_name),
            right=CameraConfig(camera_id=rid, url="", role="right", group=pair_name),
        )
        self._sm.add_virtual_stereo_pair(virt_pair)

        self._intr_cals[lid] = IntrinsicCalibrator(self._board, self._camera_model, min_f)
        self._intr_cals[rid] = IntrinsicCalibrator(self._board, self._camera_model, min_f)
        self._stereo_cals[pair_name] = StereoCalibrator(self._board, self._camera_model, min_f)
        self._fqs[lid] = []
        self._fqs[rid] = []
        self._rebuild_pair_rows()
        self.refresh_pairs()

    def _show_imported_frame(self, idx: int):
        if not hasattr(self, "_imported_frames") or not self._imported_frames:
            return
        idx = max(0, min(idx, len(self._imported_frames) - 1))
        self._import_view_idx = idx
        lid, rid, img_l, img_r = self._imported_frames[idx]
        self.images_imported.emit({lid: img_l, rid: img_r})
        self._update_import_nav_label()

    def select_offline_preview(self, key: str):
        for index, record in enumerate(self._offline_import_records):
            if key == f'offline {record[0]}':
                self._show_imported_frame(index)
                break

    def _update_import_nav_label(self):
        if not hasattr(self, "_imported_frames"):
            return
        n = len(self._imported_frames)
        i = self._import_view_idx
        self._lbl_import_nav.setText(f"第 {i + 1} / {n} 帧")
        self._btn_import_prev.setEnabled(i > 0)
        self._btn_import_next.setEnabled(i < n - 1)

    @Slot()
    def _on_import_prev(self):
        if hasattr(self, "_import_view_idx"):
            self._show_imported_frame(self._import_view_idx - 1)

    @Slot()
    def _on_import_next(self):
        if hasattr(self, "_import_view_idx"):
            self._show_imported_frame(self._import_view_idx + 1)

    @Slot(bool)
    def _on_auto_toggle(self, checked: bool):
        if checked:
            if not self._ensure_capture_session():
                return
            self._auto_timer.start()
            self._btn_auto.setText("自动采集")
        else:
            self._auto_timer.stop()
            self._btn_auto.setText("自动采集")

    def _on_auto_capture(self):
        if self._active_step != 0 or self._workflow_phase != 'capture':
            return
        readiness = getattr(self, '_capture_readiness', None)
        if readiness is not None and readiness():
            self._auto_capture_request = True
            try:
                self._on_capture()
            finally:
                self._auto_capture_request = False

    def _update_capture_stats(self):
        min_f = self._spin_min.value()
        sel = self._selected_pairs()
        counts_press = [
            self._pair_capture_press_count.get(n, 0)
            for n in sel if n in self._stereo_cals
        ]
        mc = min(counts_press) if counts_press else 0
        self._cap_progress.setMaximum(min_f)
        stereo_ns = [self._stereo_cals[n].num_frames if n in self._stereo_cals else 0 for n in sel]
        mn_st = min(stereo_ns) if stereo_ns else 0
        prefix = '待落盘数据 · 有效配对' if self._pending_save_tokens or self._failed_save_jobs else '有效配对'
        self._lbl_capture_count.setText(f'{prefix} {mn_st} / {min_f}')
        self._cap_progress.setValue(min(mn_st, min_f))
        self._cap_progress.setVisible(mn_st > 0)
        self._cap_progress.setToolTip(f'采集记录 {mc} · 有效配对 {mn_st} · 目标 {min_f}')

        assessments: List[DataAssessment] = []
        for cid, cal in self._intr_cals.items():
            fqs = self._fqs.get(cid, [])
            if not fqs:
                continue
            assessments.append(
                self._assessor.assess_dataset(
                    cid, fqs, cal.compute_coverage(), cal.get_corner_distribution()
                )
            )
        if assessments:
            ac = np.mean([a.coverage for a in assessments])
            ai = np.mean([a.image_coverage or 0. for a in assessments])
            au = np.mean([a.distribution_uniformity for a in assessments])
            tu = sum(a.usable_frames for a in assessments)
            tf = sum(a.total_frames for a in assessments)
            self._lbl_assess.setText(
                f"板角点完整度 {ac:.0%} | 视场网格 {ai:.0%} | 均匀度 {au:.0%} | 可用 {tu}/{tf}"
            )
            worst = min(assessments, key=lambda a: a.score)
            self._data_score.set_grade(worst.grade, worst.score)
            tips = worst.suggestions[:2]
            self._lbl_tips.setText("  ".join(tips) if tips else "")
        else:
            self._lbl_assess.setText('尚无质量评估' if not mn_st else '已有配对图像，质量评估待生成')
            self._data_score.clear()
            self._lbl_tips.clear()
        self._lbl_tips.setVisible(bool(self._lbl_tips.text()))

        any_ready = any(self._pair_data_ready(name) for name in self._selected_pairs())
        if any_ready:
            self._sections[0].set_status_hint("✓ 可标定", SUCCESS)

    def _calib_strategy_label(self) -> str:
        return self._combo_calib_strategy.currentText().split("：", 1)[0]

    def _store_stereo_detection(self, pair_name, left, right, source=""):
        try:
            return self._stereo_cals[pair_name].add_detection_pair(left, right, frame_id=source or None)
        except (ValueError, TypeError) as exc:
            self._log(f"{pair_name} {source}: 未用于双目标定 — {exc}")
            return False

    def _data_filter_label(self) -> str:
        return self._combo_data_filter.currentText()

    def _calib_strategy_flags(self) -> tuple[int, int]:
        key = self._combo_calib_strategy.currentData() or "standard"
        if self._camera_model == CameraModel.FISHEYE:
            intrinsic_flags = cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4 if key == "robust" else 0
            stereo_flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS if key == "joint_refine"
                            else cv2.fisheye.CALIB_FIX_INTRINSIC)
            return intrinsic_flags, stereo_flags
        if key == "robust":
            intrinsic_flags = (
                cv2.CALIB_ZERO_TANGENT_DIST
                | cv2.CALIB_FIX_K3
                | cv2.CALIB_FIX_K4
                | cv2.CALIB_FIX_K5
                | cv2.CALIB_FIX_K6
            )
            return intrinsic_flags, cv2.CALIB_FIX_INTRINSIC
        if key == "joint_refine":
            return 0, cv2.CALIB_USE_INTRINSIC_GUESS
        return 0, cv2.CALIB_FIX_INTRINSIC

    def _update_calibration_quality_report(self, pair_name: str, q_l, q_r, q_s):
        def fmt_quality(title: str, q) -> list[str]:
            lines = [
                f"{title}: {q.grade.label}  分数 {q.score:.0f}  RMS {q.rms_error:.4f}px"
            ]
            for k, v in q.details.items():
                if k in {"模型", "RMS 重投影误差"}:
                    continue
                lines.append(f"  - {k}: {v}")
            if q.suggestions:
                lines.append("  - 建议: " + "；".join(q.suggestions[:3]))
            return lines

        lines = [
            f"标定质量评估 · {pair_name}",
            f"策略: {self._calib_strategy_label()}",
            f"数据筛选: {self._data_filter_label()}",
            "",
        ]
        lines.extend(fmt_quality("左目内参", q_l))
        lines.append("")
        lines.extend(fmt_quality("右目内参", q_r))
        lines.append("")
        lines.extend(fmt_quality("双目外参", q_s))

        worst = min((q_l, q_r, q_s), key=lambda q: q.score)
        lines.append("")
        lines.append(f"总体: {worst.grade.label}（按最低分项评定）")
        if worst.grade in {Grade.FAIR, Grade.POOR}:
            lines.append("优化策略: 先核对镜头模型、板尺寸和左右配对，再检查异常帧及边角/倾角覆盖；约束策略并不等同鲁棒剔除。")
        elif (self._combo_calib_strategy.currentData() or "standard") == "joint_refine":
            lines.append("优化策略: 联合精修适合最终对比，若结果不稳定可回到标准策略。")
        else:
            lines.append("结果复核: 训练集误差不代表实际精度，请检查校正残差并验证独立图像；第三方使用前须确认镜头模型。")

        self._quality_report.setPlainText("\n".join(lines))

    def _show_frame_diagnostics(self):
        from .calibration_diagnostics import CalibrationDiagnosticsDialog

        if not self._pair_calibs:
            return
        current = getattr(self, '_diagnostic_dialog', None)
        if current is not None:
            current.raise_()
            current.activateWindow()
            return
        dialog = CalibrationDiagnosticsDialog(self._pair_calibs, self.preview_pairs(), self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.preview_requested.connect(self.diagnostic_preview_requested)
        self.calibration_reset.connect(dialog.reject)
        dialog.finished.connect(lambda _: setattr(self, '_diagnostic_dialog', None))
        self._diagnostic_dialog = dialog
        dialog.show()

    # ── Stereo calibration (per pair: intrinsic + extrinsic) ──

    def _calibrate_pair(self, pair_name: str, continue_queue: bool = False):
        """Run full calibration for one stereo pair: intrinsic L+R then stereo."""
        if self._model_worker is not None:
            self._log('模型评估中，请等待完成后再标定')
            return
        if self._pending_save_tokens or self._failed_save_jobs:
            self._log('原图尚未全部保存，完成保存或撤销失败采集后再标定')
            self._calib_queue.clear()
            return
        if self._aux_dataset_busy():
            self._log("辅助任务进行中，请稍候再执行双目标定")
            return
        if self._stereo_import_worker is not None:
            self._log("双目导入进行中，请稍候再标定")
            return
        if self._calib_worker is not None:
            if continue_queue:
                self._calib_queue.append(pair_name)
            else:
                self._log("已有标定任务正在运行，请稍候")
            return

        pair_cfg = self._sm.stereo_pairs.get(pair_name)
        cal = self._stereo_cals.get(pair_name)
        if not pair_cfg or not cal or not self._pair_data_ready(pair_name):
            self._log(f"{pair_name}: 数据不足，左右内参与有效配对均须达到最少配对数")
            if continue_queue:
                self._start_next_calibration()
            return

        self._activate_step(1)
        lid = pair_cfg.left.camera_id
        rid = pair_cfg.right.camera_id
        intrinsic_flags, stereo_flags = self._calib_strategy_flags()
        left_cal, right_cal, stereo_cal = self._intr_cals[lid], self._intr_cals[rid], cal

        self._invalidate_pair_result(pair_name)
        self._set_calibration_busy(True)
        worker = PairCalibrationWorker(
            pair_name=pair_name,
            left_id=lid,
            right_id=rid,
            left_calibrator=left_cal,
            right_calibrator=right_cal,
            stereo_calibrator=stereo_cal,
            intrinsic_flags=intrinsic_flags,
            stereo_flags=stereo_flags,
            strategy_name=self._calib_strategy_label(),
            data_filter_key=self._combo_data_filter.currentData() or "all",
            parent=self,
        )
        worker.log.connect(self._log)
        worker.finished_ok.connect(self._on_pair_calibration_finished)
        worker.failed.connect(self._on_pair_calibration_failed)
        worker.finished.connect(worker.deleteLater)
        self._calib_worker = worker
        self._refresh_aux_readiness()
        worker.start()

    @Slot()
    def _on_calibrate_all(self):
        """Batch calibrate all ready pairs."""
        if self._model_worker is not None:
            self._log('模型评估中，请等待完成后再标定')
            return
        if self._pending_save_tokens or self._failed_save_jobs:
            self._log('原图尚未全部保存，完成保存或撤销失败采集后再标定')
            return
        if self._aux_dataset_busy():
            self._log("辅助任务进行中，请稍候再执行双目标定")
            return
        if self._stereo_import_worker is not None:
            self._log("双目导入进行中，请稍候再标定")
            return
        if self._calib_worker is not None:
            self._log("已有标定任务正在运行，请稍候")
            return
        self._calib_queue = []
        for name in list(self._sm.stereo_pairs.keys()):
            cal = self._stereo_cals.get(name)
            if cal and self._pair_data_ready(name):
                self._calib_queue.append(name)
        self._start_next_calibration()

    def _start_next_calibration(self):
        if self._calib_worker is not None:
            return
        if not self._calib_queue:
            self._set_calibration_busy(False)
            self._sync_pair_states()
            return
        name = self._calib_queue.pop(0)
        self._calibrate_pair(name, continue_queue=True)

    @Slot(int)
    def _on_calib_strategy_changed(self, _index: int = 0):
        key = self._combo_calib_strategy.currentData() or "standard"
        if key == self._calib_strategy_key:
            return
        self._calib_strategy_key = key
        self._clear_calibration_results()
        self._log(f"已切换标定优化策略：{self._calib_strategy_label()}")

    @Slot(int)
    def _on_data_filter_changed(self, _index: int = 0):
        key = self._combo_data_filter.currentData() or "all"
        if key == self._data_filter_key:
            return
        self._data_filter_key = key
        self._clear_calibration_results()
        self._log(f"已切换数据筛选策略：{self._data_filter_label()}")

    def _clear_calibration_results(self):
        if self._pair_calibs or self._rig is not None or self._result_directory is not None:
            self._mark_results_outdated()
        self._clear_rect_preview()
        self._intrinsics.clear()
        self._pair_calibs.clear()
        self._rig = None
        self._aux_intrinsics = None
        self._btn_cloud.setEnabled(False)
        self._timeline.mark_pending(1)
        self._timeline.mark_pending(2)
        self._sections[1].set_status_hint("")
        self._quality_report.clear()
        self.calibration_reset.emit()
        self._sync_pair_states()

    def _mark_results_outdated(self):
        self._results_outdated = True
        self._previous_result_directory = self._result_directory or self._previous_result_directory

    def _invalidate_pair_result(self, pair_name: str):
        """Keep collected observations, but invalidate results derived from older data."""
        pair = self._pair_calibs.pop(pair_name, None)
        if pair is None:
            return
        self._mark_results_outdated()
        self._intrinsics.pop(pair.left_id, None)
        self._intrinsics.pop(pair.right_id, None)
        self._rig = None
        self._aux_intrinsics = None
        self._quality_report.clear()
        self._clear_rect_preview()
        self._btn_multi.setEnabled(False)
        self._btn_cloud.setEnabled(False)
        self._timeline.mark_pending(1)
        self._timeline.mark_pending(2)
        self._timeline.mark_pending(3)
        self._sections[1].set_status_hint("数据已更新，待重新标定", WARNING)
        self._sections[2].set_status_hint("")
        self._sections[3].set_status_hint("")
        self.calibration_reset.emit()
        self._refresh_result_summary()
        self._log(f"{pair_name}: 样本已更新，旧标定结果已失效，可直接重新标定")

    @Slot(str, str, str, object, object, object)
    def _on_pair_calibration_finished(
        self,
        pair_name: str,
        lid: str,
        rid: str,
        intr_l,
        intr_r,
        pair_calib,
    ):
        self._calib_worker = None
        self._intrinsics[lid] = intr_l
        self._intrinsics[rid] = intr_r
        self._pair_calibs[pair_name] = pair_calib
        if self._model_report is not None and self._model_report_key == self._model_snapshot_key(pair_name):
            pair_calib.calibration_info['model_assessment'] = self._model_report

        self._sync_pair_states()
        self._publish_rig_for_export()
        try:
            q_l = self._evaluator.evaluate_intrinsic(intr_l)
            q_r = self._evaluator.evaluate_intrinsic(intr_r)
            q_s = self._evaluator.evaluate_stereo(pair_calib)
            bl_mm = np.linalg.norm(pair_calib.stereo.T) * 1000.0
            self._log(f"  {lid}: RMS={intr_l.rms_error:.4f}px [{q_l.grade.label}]")
            self._log(f"  {rid}: RMS={intr_r.rms_error:.4f}px [{q_r.grade.label}]")
            self._log(
                f"  ✓ {pair_name}: 外参 RMS={pair_calib.stereo.rms_error:.4f}px "
                f"基线={bl_mm:.1f}mm [{q_s.grade.label}]"
            )
            self._update_calibration_quality_report(pair_name, q_l, q_r, q_s)
        except Exception as exc:
            logger.exception("Calibration quality display failed: pair=%s", pair_name)
            self._log(f"{pair_name}: 质量评估显示失败（标定结果已生成）：{exc}")
        try:
            self._clear_rect_preview()
            self._show_rect_preview(pair_name, pair_calib)
        except Exception as exc:
            logger.exception("Rectification preview failed: pair=%s", pair_name)
            self._log(f"{pair_name}: 校正预览失败（标定结果已生成）：{exc}")
        self._start_next_calibration()

    @Slot(str, str)
    def _on_pair_calibration_failed(self, pair_name: str, message: str):
        self._calib_worker = None
        self._log(f"  {pair_name}: 标定失败 — {message.splitlines()[0]}")
        self._sync_pair_states()
        self._start_next_calibration()

    def _set_calibration_busy(self, busy: bool):
        self._ui_operation_busy = busy
        dialog = getattr(self, '_diagnostic_dialog', None)
        if busy and dialog is not None:
            dialog.reject()
        if busy and self._btn_auto.isChecked():
            self._btn_auto.setChecked(False)
        self._btn_capture.setEnabled(not busy)
        self._btn_import.setEnabled(not busy)
        self._btn_swap_offline.setEnabled(
            (not busy) and bool(self._offline_import_records)
        )
        self._btn_undo.setEnabled((not busy) and bool(self._undo_stack))
        self._btn_auto.setEnabled(not busy)
        self._combo_pair.setEnabled(not busy)
        self._spin_min.setEnabled(not busy)
        self._combo_calib_strategy.setEnabled(not busy)
        self._combo_data_filter.setEnabled(not busy)
        self._btn_frame_diagnostics.setEnabled(not busy and bool(self._pair_calibs))
        self._btn_calib_all.setEnabled(False if busy else self._btn_calib_all.isEnabled())
        for row in self._pair_rows.values():
            row.btn.setEnabled(False if busy else row.btn.isEnabled())
        self._refresh_aux_readiness()
        self._refresh_capture_source_ui()
        self._refresh_model_choice_ui()

    def _model_snapshot_key(self, name):
        pair = self._sm.stereo_pairs.get(name)
        if pair is None:
            return None
        cals = [self._intr_cals.get(cid) for cid in (pair.left.camera_id, pair.right.camera_id)]
        stereo = self._stereo_cals.get(name)
        return (name, id(self._board), id(self._session),
                tuple((id(cal), cal.data_revision if cal else None) for cal in cals),
                id(stereo), tuple(stereo.frame_ids) if stereo else ())

    def _refresh_model_choice_ui(self, *_):
        if not hasattr(self, '_btn_assess_model'):
            return
        busy = self.has_active_background_tasks() or self._white_balance_guard.busy or self.has_failed_saves
        self._btn_assess_model.setEnabled(not busy and bool(self._combo_model_pair.currentText()))
        self._combo_model_pair.setEnabled(not busy)
        valid = (self._model_report is not None and self._model_report_key ==
                 self._model_snapshot_key(self._combo_model_pair.currentText()))
        model = self._model_report.get('recommended_model') if valid else None
        needs_apply = model is not None and (
            model != self._camera_model.value or
            (model == 'fisheye' and self._combo_data_filter.currentData() != 'fisheye_common'))
        self._btn_apply_model.setVisible(needs_apply)
        self._btn_apply_model.setEnabled(not busy and valid)
        self._btn_cancel_model.setVisible(self._model_worker is not None)
        self._btn_cancel_model.setEnabled(self._model_worker is not None and
                                         not self._model_worker.isInterruptionRequested())
        if self._model_worker is not None:
            return
        if valid:
            values = self._model_report['candidates']
            def score(key):
                value = values[key]['rms_px']
                return f'{value:.3f} px' if value is not None else '评估失败'
            title = {'pinhole': '推荐针孔', 'fisheye': '推荐鱼眼'}.get(model, '暂不改变模型')
            self._lbl_model_choice.setText(
                f'{title} · 留出右图预测：针孔 {score("pinhole")} / 鱼眼 {score("fisheye")}\n'
                + self._model_report['reason'])
            self._lbl_model_choice.setToolTip(str(self._model_report))
        elif self._model_report is not None:
            self._lbl_model_choice.setText('数据或评估对象已变化，请重新评估模型')
        else:
            self._lbl_model_choice.setText(self._model_error or '至少 9 对多姿态观测后可评估针孔/鱼眼模型')

    def _start_model_assessment(self):
        if self.has_active_background_tasks() or self._white_balance_guard.busy or self.has_failed_saves:
            self._log('请等待当前任务完成并处理保存失败后再评估模型')
            return
        name = self._combo_model_pair.currentText()
        pair = self._sm.stereo_pairs.get(name)
        stereo = self._stereo_cals.get(name)
        if pair is None or stereo is None:
            self._log('请先采集或导入双目观测')
            return
        left, right = (self._intr_cals.get(cid) for cid in (pair.left.camera_id, pair.right.camera_id))
        if left is None or right is None:
            self._log('左右内参观测不足，无法评估模型')
            return
        self._model_report_key = self._model_snapshot_key(name)
        self._model_report, self._model_error = None, ''
        worker = ModelSelectionWorker(
            left.copy_with_indices(list(range(left.num_frames)), keep_images=False),
            right.copy_with_indices(list(range(right.num_frames)), keep_images=False),
            stereo.copy_with_indices(list(range(stereo.num_frames))), self)
        self._model_worker = worker
        worker.progress.connect(self._lbl_model_choice.setText)
        worker.result_ready.connect(self._on_model_assessed)
        worker.failed.connect(self._on_model_assessment_failed)
        worker.finished.connect(self._on_model_assessment_finished)
        worker.finished.connect(worker.deleteLater)
        self._set_calibration_busy(True)
        self._sync_pair_states()
        self._lbl_model_choice.setText('正在比较针孔和鱼眼模型，请稍候')
        worker.start()

    def _on_model_assessed(self, report):
        if (self._model_worker is not None and not self._model_worker.isInterruptionRequested() and
                self._model_report_key == self._model_snapshot_key(self._combo_model_pair.currentText())):
            self._model_report = report
            self._log('模型评估完成：' + report['reason'])

    def _cancel_model_assessment(self):
        if self._model_worker is not None:
            self._model_worker.requestInterruption()
            self._btn_cancel_model.setEnabled(False)
            self._lbl_model_choice.setText('正在取消模型评估，请等待当前计算安全结束')

    def _on_model_assessment_failed(self, message):
        self._model_error = '模型评估未完成：' + message
        self._log(self._model_error)

    def _on_model_assessment_finished(self):
        if self._model_worker is not None and self._model_worker.isInterruptionRequested():
            self._model_report = None
            self._model_error = '模型评估已取消，当前模型未改变'
        self._model_worker = None
        self._set_calibration_busy(False)
        self._sync_pair_states()
        self._refresh_model_choice_ui()
        QTimer.singleShot(0, self._refresh_model_choice_ui)

    def _request_model_choice(self):
        if (self._model_report is None or self.has_active_background_tasks() or
                self._model_report_key != self._model_snapshot_key(self._combo_model_pair.currentText())):
            return
        model = self._model_report.get('recommended_model')
        if model is not None:
            self.model_choice_requested.emit(CameraModel(model))

    def apply_recommended_model(self, model):
        """Change projection interpretation, retaining model-independent raw observations."""
        if self._pair_calibs or self._rig is not None:
            self._mark_results_outdated()
        self._camera_model = CameraModel(model)
        for cal in (*self._intr_cals.values(), *self._stereo_cals.values()):
            cal.model = self._camera_model
        self._clear_calibration_results()
        self._invalidate_aux_result()
        policy = 'fisheye_common' if self._camera_model == CameraModel.FISHEYE else 'all'
        self._combo_data_filter.setCurrentIndex(self._combo_data_filter.findData(policy))
        self._refresh_capture_source_ui()
        self._refresh_model_choice_ui()
        self.data_source_changed.emit()
        self._log(f'已应用推荐模型 {self._camera_model.value}；原始观测保留，请重新标定')

    def _clear_rect_preview(self):
        self._lbl_rect_preview.clear()
        self._lbl_rect_preview.hide()
        self._lbl_rect_context.clear()
        self._lbl_rect_context.hide()

    def _show_rect_preview(self, pair_name: str, pair: StereoPairCalibration):
        self._clear_rect_preview()
        sync = self._sm.get_sync_pair(pair_name)
        if sync is not None:
            left, right = sync
            source = '实时配对'
        else:
            frames = getattr(self, '_imported_pair_frames', {}).get(pair_name, [])
            if not frames:
                return
            left, right = frames[-1]
            source = '该组最后导入配对'
        for image, intr in ((left, pair.left_intrinsics), (right, pair.right_intrinsics)):
            if image.shape[1::-1] != tuple(intr.image_size):
                raise ValueError('预览原图尺寸与标定尺寸不一致，已隐藏校正预览')
        preview = self._evaluator.generate_rectification_preview(
            left, right, pair.stereo, num_lines=15
        )
        h, w = preview.shape[:2]
        tw = max(self._inspector_scroll.viewport().width() - 64, 100)
        sc = min(tw / max(w, 1), 160.0 / max(h, 1))
        nw, nh = int(w * sc), int(h * sc)
        if nw > 0 and nh > 0:
            small = cv2.resize(preview, (nw, nh))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, nw, nh, nw * 3, QImage.Format_RGB888)
            self._lbl_rect_preview.setPixmap(QPixmap.fromImage(qimg))
            model = '鱼眼' if pair.stereo.model == CameraModel.FISHEYE else '针孔'
            self._lbl_rect_context.setText(f'校正预览 · {pair_name} · {model}模型 · {source}')
            self._lbl_rect_context.setToolTip('左侧为左目、右侧为右目；参考线不代表已验证匹配。此图不随原图缩略图选择变化。')
            self._lbl_rect_context.show()
            self._lbl_rect_preview.show()

    # ── Multimodal auxiliary mono calibration ────────────────

    @Slot()
    def _on_capture_aux(self):
        self._activate_step(2, phase="capture")
        if self._aux_operation_busy():
            self._log("后台任务进行中，请稍候再采集")
            return
        frames, problem = self._aux_live_frames()
        if frames is None:
            self._log(problem)
            self._refresh_aux_readiness()
            return
        aux = self._sm.auxiliary_camera
        if aux is None:
            self._log("请先在配置窗口启用并选择融合辅助单目相机")
            return
        rgb_left_id = self._rgb_left_camera_id()
        if not rgb_left_id:
            self._log("请先配置 RGB 左相机")
            return

        if not self._ensure_capture_session():
            return
        rgb_pairs = {name: pair for name, pair in self._sm.stereo_pairs.items()
                     if pair.left.camera_id == rgb_left_id}
        if not self._prepare_white_balance(rgb_pairs, self._on_capture_aux):
            return

        frames, problem = self._aux_live_frames()
        if frames is None:
            self._log(problem)
            return
        rgb_frame, aux_frame, _ = frames
        self.capture_started.emit()

        board_cfg = self._aux_board_config()
        if self._aux_pairs and self._aux_observation_board != board_cfg:
            self._log("辅助板参数已变更，请重新导入图片或通过配置重置后重新采集")
            return
        detector = PlanarPatternDetector(board_cfg)
        rgb_det = detector.detect(rgb_frame)
        aux_det = detector.detect(aux_frame)
        capture_index = self._aux_capture_count
        idx = f"live_{capture_index:04d}"
        existing_ids = {key for key, _, _ in self._aux_pairs}
        while idx in existing_ids:
            capture_index += 1
            idx = f"live_{capture_index:04d}"
        if self._session is not None:
            try:
                self._session.save_multimodal_pair(rgb_frame, aux_frame)
            except Exception as exc:
                self._log(f"{idx}: 多模态图片保存失败，未增加采集记录，请重试采集 — {str(exc).splitlines()[0]}")
                return
            self._aux_saved_count += 1
        self._aux_images.append((rgb_frame.copy(), aux_frame.copy()))

        if rgb_det.valid and aux_det.valid:
            self._invalidate_aux_result()
            self._aux_observation_board = board_cfg
            self._aux_pairs.append((idx, rgb_det, aux_det))
            self.frame_captured.emit(rgb_left_id, rgb_det)
            self.frame_captured.emit(aux.camera_id, aux_det)
            self._log(f"{idx}: RGB_L={rgb_det.num_points} AUX={aux_det.num_points}")
            if not self._aux_capture_sequence.finished:
                self._aux_capture_sequence.advance()
                self._aux_seq_widget.refresh()
                self.sequence_updated.emit(self._aux_capture_sequence)
        else:
            self._log(f"{idx}: 平面板检测失败 RGB_L={rgb_det.valid} AUX={aux_det.valid}")

        self._aux_capture_count = capture_index + 1
        self._refresh_aux_capture_state()

    @Slot()
    def _on_import_aux_images(self):
        if self._failed_save_jobs:
            self._log('仍有双目原图保存失败，请先重试保存或撤销对应采集后再导入')
            return
        self._activate_step(2, phase="capture")
        if self._aux_operation_busy():
            self._log("后台任务进行中，请稍候再导入多模态图片")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择多模态图片目录")
        if not folder:
            return

        board_cfg = self._aux_board_config()
        self._log(f"正在后台导入多模态图片，板参数: {board_cfg.to_dict()}")
        self._set_aux_import_busy(True)
        worker = AuxImageImportWorker(folder, board_cfg, parent=self)
        worker.finished_ok.connect(self._on_import_aux_images_ready)
        worker.failed.connect(self._on_import_aux_images_failed)
        worker.finished.connect(worker.deleteLater)
        self._aux_import_worker = worker
        worker.start()

    @Slot(object)
    def _on_import_aux_images_ready(self, img_pairs):
        worker = self._aux_import_worker
        self._aux_import_worker = None
        self._set_aux_import_busy(False)
        board_cfg = worker.board_config if worker is not None else self._aux_board_config()
        if board_cfg != self._aux_board_config():
            self._log("辅助板参数在导入期间已变更，已忽略旧检测结果，请重新导入")
            return
        self._aux_observation_board = board_cfg
        self._invalidate_aux_result()
        self._aux_pairs.clear()
        self._aux_images.clear()
        self._aux_capture_count = 0
        self._aux_saved_count = 0
        self._aux_imported_count = len(img_pairs)
        self._aux_source_folder = getattr(worker, 'folder', None)
        self._aux_capture_sequence.reset()

        for key, rgb, aux, rgb_det, aux_det in img_pairs:
            self._aux_images.append((rgb, aux))
            if rgb_det.valid and aux_det.valid:
                self._aux_pairs.append((key, rgb_det, aux_det))
                self._log(f"  [{key}] RGB_L={rgb_det.num_points} AUX={aux_det.num_points}")
            else:
                self._log(f"  [{key}] 检测失败 RGB_L={rgb_det.valid} AUX={aux_det.valid}")

        valid = len(self._aux_pairs)
        # Imported board detections do not identify which guided poses were achieved.
        self._aux_seq_widget.refresh()
        self.sequence_updated.emit(self._aux_capture_sequence)
        self._refresh_aux_capture_state()
        if self._aux_images:
            rgb, aux = self._aux_images[-1]
            aux_id = self._aux_camera_id()
            rgb_left_id = self._rgb_left_camera_id() or "RGB_L"
            self.aux_images_imported.emit({rgb_left_id: rgb, aux_id: aux})
        self._log(f"多模态导入完成：有效 {valid}/{len(img_pairs)} 组")
        self._log('离线原图保留在源目录，未复制到会话；导入不自动确认引导姿态')

    @Slot(str)
    def _on_import_aux_images_failed(self, message: str):
        self._aux_import_worker = None
        self._set_aux_import_busy(False)
        self._log(message.splitlines()[0])

    @Slot()
    def _on_calibrate_aux(self):
        self._activate_step(2, phase="calibration")
        if self._aux_operation_busy():
            self._log("后台任务进行中，请稍候再标定")
            return
        problem = self._aux_calibration_problem()
        if problem:
            self._log(problem)
            self._refresh_aux_readiness()
            return
        rgb_left_id = self._rgb_left_camera_id()
        board_cfg = self._aux_board_config()
        self._log(f"多模态标定板参数: {board_cfg.to_dict()}")
        aux_id = self._aux_camera_id()
        modality = self._aux_modality()

        min_frames = min(max(3, self._spin_min.value()), len(self._aux_pairs))
        self._set_aux_busy(True)
        self._pending_aux_context = (aux_id, modality, board_cfg, rgb_left_id)
        worker = AuxCalibrationWorker(
            aux_id=aux_id,
            board_config=board_cfg,
            camera_model=self._camera_model,
            detections=list(self._aux_pairs),
            rgb_left_intrinsics=self._intrinsics[rgb_left_id],
            min_intrinsic_frames=min_frames,
            parent=self,
        )
        worker.log.connect(self._log)
        worker.finished_ok.connect(self._on_aux_calibration_finished)
        worker.failed.connect(self._on_aux_calibration_failed)
        worker.finished.connect(worker.deleteLater)
        self._aux_worker = worker
        worker.start()

    @Slot(object, object)
    def _on_aux_calibration_finished(self, aux_intr, extr):
        self._aux_worker = None
        aux_id, modality, board_cfg, rgb_left_id = self._pending_aux_context
        if (board_cfg != self._aux_board_config() or aux_id != self._aux_camera_id() or
                modality != self._aux_modality() or rgb_left_id != self._rgb_left_camera_id()):
            self._set_aux_busy(False)
            self._log("标定期间辅助配置已变更，未应用旧结果，请重新导入/采集后标定")
            return
        self._aux_intrinsics = aux_intr
        aux_calib = AuxCameraCalibration(
            camera_id=aux_id,
            modality=modality,
            intrinsics=aux_intr,
            T_rgb_left_aux=extr.T_rgb_left_aux,
            rms_error=extr.rms_error,
            board_type=board_cfg.pattern_type.value,
            frame_count=extr.frame_count,
            per_frame_errors=extr.per_frame_errors,
            board_config=board_cfg.to_dict(),
        )
        rig = self._ensure_rig()
        rig.reference_camera = rgb_left_id
        rig.aux_cameras[aux_id] = aux_calib
        self._lbl_aux_status.setText(
            f"✓ {aux_id} ({modality}) RMS={extr.rms_error:.3f}px，参与帧 {extr.frame_count}"
        )
        self._sections[2].set_status_hint("✓ 完成", SUCCESS)
        self._timeline.mark_completed(2)
        self._log(
            f"✓ {aux_id}: T_rgb_left_aux 已生成，RMS={extr.rms_error:.4f}px，帧数={extr.frame_count}"
        )
        if self._session is not None:
            try:
                self._session.record_multimodal_result(
                    aux_camera_id=aux_id,
                    modality=modality,
                    board_type=board_cfg.pattern_type.value,
                    valid_pairs=extr.frame_count,
                    rms_error=extr.rms_error,
                    board_config=board_cfg.to_dict(),
                )
            except Exception as exc:
                logger.exception("Save multimodal result metadata failed: camera=%s", aux_id)
                self._log(f"{aux_id}: 标定已完成，但会话记录保存失败，结果仍可导出：{exc}")
        self._set_aux_busy(False)
        self.calibration_finished.emit(rig)

    @Slot(str)
    def _on_aux_calibration_failed(self, message: str):
        self._aux_worker = None
        self._set_aux_busy(False)
        aux_id = self._aux_camera_id()
        self._log(f"{aux_id}: 多模态标定失败 — {message.splitlines()[0]}")

    def _set_aux_busy(self, busy: bool):
        self._aux_ui_busy = busy
        self._set_calibration_busy(busy)
        if not busy:
            self._sync_pair_states()
        self._refresh_aux_readiness()

    def _set_import_busy(self, busy: bool):
        self._set_calibration_busy(busy)
        if not busy:
            self._sync_pair_states()
        self._btn_import.setEnabled(not busy)
        self._btn_swap_offline.setEnabled(
            (not busy) and bool(self._offline_import_records)
        )
        self._btn_capture.setEnabled(not busy)
        self._btn_auto.setEnabled(not busy)
        self._btn_undo.setEnabled((not busy) and bool(self._undo_stack))
        self._btn_import.setText('正在导入…' if busy else '导入图片')
        self._refresh_capture_source_ui()

    def _set_aux_import_busy(self, busy: bool):
        self._aux_import_busy = busy
        self._set_calibration_busy(busy)
        if not busy:
            self._sync_pair_states()
        self._refresh_aux_readiness()

    def _refresh_aux_capture_state(self):
        valid = len(self._aux_pairs)
        total = self._aux_capture_sequence.total
        self._aux_cap_progress.setMaximum(total)
        self._aux_cap_progress.setValue(min(valid, total))
        self._aux_cap_progress.setFormat(f'有效配对 {valid} 组 · 建议 {total} 组')
        self._refresh_aux_readiness()
        self._lbl_aux_status.setText(f"最少 3 组 · 建议 {total} 组")
        self._refresh_preview_pairs()
        self._refresh_disk_hint()

    @Slot(str)
    def _on_aux_camera_connected(self, camera_id: str):
        self._unavailable_cameras.discard(camera_id)
        self._refresh_aux_readiness()

    @Slot(str, str)
    def _on_aux_camera_disconnected(self, camera_id: str, reason: str):
        self._unavailable_cameras.add(camera_id)
        self._refresh_aux_readiness()

    def _aux_live_frames(self):
        aux, rgb_id = self._sm.auxiliary_camera, self._rgb_left_camera_id()
        if aux is None or not aux.url or rgb_id is None:
            return None, "采图未就绪：请配置 RGB 左目和辅助实时视频流"
        entries = [self._sm.get_latest_frame(cid) if cid not in self._unavailable_cameras else None
                   for cid in (rgb_id, aux.camera_id)]
        now = time.time()
        states = []
        for name, entry in zip(("RGB 左目", "红外/辅助"), entries):
            fresh = entry is not None and np.isfinite(entry[1]) and 0 <= now - entry[1] <= self.AUX_MAX_FRAME_AGE_SEC
            states.append(f"{name} {'✓' if fresh else '无新帧'}")
        if any(entry is None or not np.isfinite(entry[1]) or
               not 0 <= now - entry[1] <= self.AUX_MAX_FRAME_AGE_SEC for entry in entries):
            return None, "采图未就绪：" + "；".join(states) + f"（帧需在 {self.AUX_MAX_FRAME_AGE_SEC:g} 秒内）"
        dt = abs(entries[0][1] - entries[1][1])
        if dt > self._sm.sync_tolerance_seconds:
            return None, (f"采图未就绪：两路时间差 {dt * 1000:.0f} ms，"
                          f"超过 {self._sm.sync_tolerance_seconds * 1000:.0f} ms")
        return (entries[0][0], entries[1][0], dt), (f"采图就绪：{'；'.join(states)}；"
                                                  f"软件配对时间差 {dt * 1000:.0f} ms（非硬件同步）")

    def _aux_calibration_problem(self) -> str:
        rgb_id = self._rgb_left_camera_id()
        if not rgb_id or rgb_id not in self._intrinsics:
            return "标定未就绪：请先完成 RGB 左目所在双目标定"
        if self._aux_observation_board is not None and self._aux_observation_board != self._aux_board_config():
            return "标定未就绪：辅助板参数与已检测图像不一致，请重新导入或采集"
        if len(self._aux_pairs) < 3:
            return f"标定未就绪：有效配对 {len(self._aux_pairs)} 组，至少需要 3 组"
        if any(rgb.image_size != self._intrinsics[rgb_id].image_size for _, rgb, _ in self._aux_pairs):
            return "标定未就绪：配对 RGB 图像分辨率与左目内参不一致，请重新标定或采集"
        if len({aux.image_size for _, _, aux in self._aux_pairs}) > 1:
            return "标定未就绪：辅助图像分辨率不一致，请使用同一分辨率重新采集或导入"
        return ""

    def _invalidate_aux_result(self):
        had_result = bool(self._aux_intrinsics is not None or
                          (self._rig is not None and self._rig.aux_cameras))
        if had_result:
            self._mark_results_outdated()
        self._aux_intrinsics = None
        if had_result:
            # MainWindow shares this rig; remove stale AUX without discarding valid RGB results.
            if self._rig is not None:
                self._rig.aux_cameras.clear()
            self._lbl_aux_status.setText('辅助数据已更新，旧融合结果已失效，请重新标定')
        self._timeline.mark_pending(2)
        self._sections[2].set_status_hint("")
        if had_result:
            self.set_result_storage(None)

    def _aux_dataset_busy(self) -> bool:
        return bool(self._model_worker or self._aux_ui_busy or self._aux_import_busy or self._aux_worker or self._aux_import_worker or
                    self._white_balance_guard.busy)

    def _ensure_capture_session(self):
        if self._session is None:
            self.create_session_requested.emit()
        if self._session is None:
            self._btn_auto.setChecked(False)
            self._log("采图已取消：未创建可保存原图的会话")
            return False
        return True

    def _prepare_white_balance(self, pairs, resume):
        if not self._check_wb.isChecked():
            return True
        urls = []
        for pair in pairs.values():
            if pair.left.stream_type == 'rgb' and any(
                self._sm.is_camera_running(cam.camera_id) for cam in (pair.left, pair.right)
            ):
                urls.extend(cam.url for cam in (pair.left, pair.right) if cam.url)
        context = (self._active_step, self._workflow_phase, id(self._session))

        def resume_current():
            if context == (self._active_step, self._workflow_phase, id(self._session)):
                resume()
        self._wb_resume_auto = self._btn_auto.isChecked()
        if self._combo_wb_mode.currentData() == 'automatic':
            return self._white_balance_guard.ensure(urls, resume_current, lock=False)
        return self._white_balance_guard.ensure(urls, resume_current)

    def _on_white_balance_mode_changed(self, *_):
        self._white_balance_guard.invalidate()
        automatic = self._combo_wb_mode.currentData() == 'automatic'
        self._log('白平衡策略：自动稳定（不锁定），每次采图重新检查，不保证参数固定' if automatic else
                  '白平衡策略：手动冻结验证（可能变色）；关闭自动不代表设备能保持当前颜色增益')

    def _on_white_balance_policy_changed(self, enabled):
        self._white_balance_guard.invalidate()
        if not enabled:
            self._log("采图前白平衡管理已关闭；后续采图不检查颜色稳定或锁定状态")
        self._refresh_capture_source_ui()

    def _on_white_balance_busy(self, busy):
        self._wb_progress.setVisible(busy)
        if busy:
            self._wb_progress_timer.start()
            self._refresh_white_balance_progress()
            self._btn_capture.setText('白平衡准备中…')
            self._btn_capture_aux.setText('白平衡准备中…')
        else:
            self._wb_progress_timer.stop()
            self._refresh_disk_hint()
        self._check_wb.setEnabled(not busy)
        self._combo_wb_mode.setEnabled(not busy)
        self._set_calibration_busy(busy)
        if not busy:
            self._sync_pair_states()

    def _refresh_white_balance_progress(self):
        guard = self._white_balance_guard
        elapsed = max(0, time.monotonic() - guard.started_at)
        self._lbl_wb_progress.setText(f'{guard.stage} · 已等待 {elapsed:.0f} 秒')
        self._btn_cancel_wb.setEnabled(not guard.cancel_requested)

    def _cancel_white_balance(self):
        self._white_balance_guard.cancel()
        self._refresh_white_balance_progress()

    def _on_white_balance_completed(self, success):
        if success and self._wb_resume_auto and self._active_step == 0 and self._workflow_phase == 'capture':
            self._btn_auto.setChecked(True)
        self._wb_resume_auto = False

    def _aux_operation_busy(self) -> bool:
        return bool(self._aux_dataset_busy() or self._calib_worker or self._stereo_import_worker)

    @Slot()
    def _refresh_aux_readiness(self):
        frames, capture_status = self._aux_live_frames()
        problem = self._aux_calibration_problem()
        busy = self._aux_operation_busy()
        if busy:
            calibration_status = "后台任务进行中，请稍候"
        else:
            calibration_status = problem or f"标定就绪：RGB 左目内参 ✓；有效配对 {len(self._aux_pairs)} 组"
        self._lbl_aux_readiness.setText(f"{capture_status}\n{calibration_status}")
        aux = self._sm.auxiliary_camera
        intrinsics_ready = self._rgb_left_camera_id() in self._intrinsics
        summary = {
            'sensor': ('RGB + 声像仪 RGB' if aux and aux.modality == 'acoustic_rgb' else 'RGB + 红外', TEXT),
            'intrinsics': ('已就绪' if intrinsics_ready else '未完成', SUCCESS if intrinsics_ready else WARNING),
            'capture': ('白平衡准备中' if self._white_balance_guard.busy else '处理中' if busy else '已就绪' if frames is not None else '未就绪',
                        SUCCESS if frames is not None and not busy else WARNING),
            'calibration': ('等待白平衡' if self._white_balance_guard.busy else '处理中' if busy else '已就绪' if not problem else '需先标定 RGB' if not intrinsics_ready else '数据待就绪',
                            SUCCESS if not problem and not busy else WARNING),
        }
        for key, (text, color) in summary.items():
            value = self._aux_task_values[key]
            if value.text() != text:
                value.setText(text)
                value.setStyleSheet(f'color:{color}; font-size:12px; background:transparent;')
        self._aux_task_values['capture'].setToolTip(capture_status)
        self._aux_task_values['calibration'].setToolTip(calibration_status)
        self._btn_rgb_prerequisite.setVisible(not intrinsics_ready)
        self._lbl_aux_pair_count.setText(f"{len(self._aux_pairs)} / {self._aux_capture_sequence.total}")
        self._lbl_aux_pair_count.setToolTip('有效平面板配对数 / 建议采图组数；不代表姿态覆盖或标定精度')
        self._btn_capture_aux.setEnabled(not busy and frames is not None)
        self._btn_capture_aux.setToolTip(capture_status)
        self._btn_calib_aux.setEnabled(not busy and not problem)
        self._btn_import_aux.setEnabled(not busy)
        self._btn_calib_aux.setToolTip(calibration_status)
        if busy:
            hint = "处理中"
        elif problem:
            hint = "待就绪" if self._sm.auxiliary_camera is not None or self._aux_pairs else "未配置"
        elif self._aux_intrinsics is not None and self._rig is not None and self._aux_camera_id() in self._rig.aux_cameras:
            hint = "✓ 完成"
        else:
            hint = "✓ 可标定"
        self._sections[2].set_status_hint(hint)

    def _aux_board_config(self) -> PatternBoardConfig:
        aux = self._sm.auxiliary_camera
        return aux.pattern_board_config() if aux is not None else PatternBoardConfig.ir_chessboard()

    def _refresh_aux_board_info(self):
        board = self._aux_board_config()
        pattern = "棋盘格" if board.pattern_type.value == "chessboard" else "圆点阵"
        geometry = (f"{board.cols + 1} 列 × {board.rows + 1} 行格子" if pattern == "棋盘格"
                    else f"{board.cols} 列 × {board.rows} 行圆心")
        self._lbl_aux_board.setText(f"{pattern} · {geometry}\n{board.square_size * 1000:g} mm · "
                                    f"检测点 {board.cols} × {board.rows}（{board.cols * board.rows} 点）")
        dimension = f'{board.cols + 1}×{board.rows + 1} 格子' if pattern == '棋盘格' else f'{board.cols}×{board.rows} 圆心'
        self._lbl_board_compact.setText(f'{pattern} · {dimension} · {board.square_size * 1000:g} mm')
        self._lbl_board_compact.setToolTip(self._lbl_aux_board.text())
        for key, value in (('type', pattern), ('geometry', geometry),
                           ('spacing', f'{board.square_size * 1000:g} mm'),
                           ('points', f'{board.cols} × {board.rows}（{board.cols * board.rows} 点）')):
            self._aux_board_values[key].setText(value)
            self._aux_board_values[key].setToolTip(value)
        aux = self._sm.auxiliary_camera
        name = "声像仪 RGB" if aux is not None and aux.modality == "acoustic_rgb" else "红外（中间）"
        self._lbl_aux_sources.setText(
            f"RGB 左目：{self._rgb_left_camera_id() or '未配置'}\n"
            f"{name}：{aux.camera_id if aux is not None else '未配置'}"
        )
        self._lbl_aux_sources.setToolTip(
            f"辅助视频流：{aux.url if aux is not None and aux.url else '未配置实时地址'}"
        )
        self._btn_configure_aux.setText("配置")
        self._btn_configure_aux.setToolTip("相机与标定板配置\n" + self._lbl_aux_sources.text())

    def _aux_camera_id(self) -> str:
        aux = self._sm.auxiliary_camera
        return aux.camera_id if aux is not None else "aux_cam"

    def _aux_modality(self) -> str:
        aux = self._sm.auxiliary_camera
        return aux.modality if aux is not None else "ir"

    def _rgb_left_camera_id(self) -> Optional[str]:
        for pair in self._sm.stereo_pairs.values():
            if pair.left.stream_type == "rgb":
                return pair.left.camera_id
        for name, pair in self._pair_calibs.items():
            cfg = self._sm.stereo_pairs.get(name)
            if cfg is None or cfg.left.stream_type == "rgb":
                return pair.left_id
        return None

    def _ensure_rig(self) -> MultiCameraRig:
        if self._rig is not None:
            return self._rig
        return self._rebuild_rig_from_pairs()

    def _rebuild_rig_from_pairs(self) -> MultiCameraRig:
        previous_aux = dict(self._rig.aux_cameras) if self._rig is not None else {}
        reference = self._rgb_left_camera_id()
        available = {cid for pair in self._pair_calibs.values() for cid in (pair.left_id, pair.right_id)}
        mv = MultiViewCalibrator(reference_camera=reference if reference in available else "")
        for pc in self._pair_calibs.values():
            mv.add_pair_calibration(pc)
        self._rig = mv.calibrate()
        self._rig.aux_cameras.update(previous_aux)
        return self._rig

    def _publish_rig_for_export(self):
        if not self._pair_calibs:
            return
        try:
            rig = self._rebuild_rig_from_pairs()
        except Exception as exc:
            self._log(f"生成导出结果失败: {str(exc).splitlines()[0]}")
            return
        self._btn_multi.setEnabled(True)
        self._btn_cloud.setEnabled(True)
        self._lbl_multi.setText(
            f"✓ {len(rig.extrinsics)} 相机, 参考: {rig.reference_camera}"
        )
        self._log(f"已生成标定结果：{len(rig.pairs)} 组，可导出")
        self.calibration_finished.emit(rig)

    # ── Multi-view ────────────────────────────────────────────

    @Slot()
    def _on_calibrate_multiview(self):
        self._activate_step(3)
        self._log("联合优化...")

        try:
            self._rebuild_rig_from_pairs()
            self._lbl_multi.setText(
                f"✓ {len(self._rig.extrinsics)} 相机, "
                f"参考: {self._rig.reference_camera}"
            )
            self._log(f"联合完成: 参考 {self._rig.reference_camera}")
            self._btn_cloud.setEnabled(True)
            self.calibration_finished.emit(self._rig)
        except Exception as e:
            self._log(f"联合失败: {e}")
            self._lbl_multi.setText("失败")

    # ── Point Cloud ───────────────────────────────────────────

    @Slot()
    def _on_generate_cloud(self):
        if self._rig is None and self._pair_calibs:
            try:
                self._rebuild_rig_from_pairs()
                self.calibration_finished.emit(self._rig)
            except Exception as e:
                self._log(f"构建失败: {e}")
                return

        if self._rig is None:
            return

        self._log("生成点云...")
        from ...fusion.fusion import MultiViewFusion
        from ...fusion.pointcloud import depth_to_pointcloud
        from ...fusion.stereo_matching import StereoMatcher

        matcher = StereoMatcher()
        clouds = {}
        for pn, pair in self._rig.pairs.items():
            sr = pair.stereo
            if sr.Q is None or sr.map1_left is None:
                continue
            sync = self._image_pair_for_cloud(pn)
            if sync is None:
                continue
            left, right = sync
            rl, rr = matcher.rectify(left, right, sr)
            disp = matcher.compute_disparity(rl, rr)
            pts3 = matcher.compute_depth(disp, sr.Q)
            pts, cols = depth_to_pointcloud(pts3, rl)
            clouds[pn] = (pts, cols)
            self._log(f"  {pn}: {len(pts)} 点")

        if not clouds:
            self._log("无有效数据")
            return
        try:
            fused = MultiViewFusion().fuse(clouds, self._rig)
            self._log(f"融合完成: {len(fused.points)} 点")
            self._timeline.mark_completed(3)
            self._sections[3].set_status_hint("✓ 完成", SUCCESS)
            self.pointcloud_ready.emit(fused)
        except Exception as e:
            self._log(f"融合失败: {e}")

    # ── Helpers ───────────────────────────────────────────────

    def _image_pair_for_cloud(self, pair_name: str):
        sync = self._sm.get_sync_pair(pair_name)
        if sync is not None:
            return sync
        if not hasattr(self, "_imported_pair_frames"):
            return None
        frames = self._imported_pair_frames.get(pair_name)
        if not frames:
            return None
        idx = getattr(self, "_import_view_idx", 0)
        return frames[max(0, min(idx, len(frames) - 1))]

    def _reset_all(self):
        self._clear_rect_preview()
        self._model_report = None
        self._model_report_key = None
        self._model_error = ''
        self._failed_save_jobs.clear()
        self._saved_capture_tokens.clear()
        self._results_outdated = False
        self._previous_result_directory = None
        self._live_preview_indices.clear()
        self._intr_cals.clear()
        self._intrinsics.clear()
        self._stereo_cals.clear()
        self._pair_calibs.clear()
        self._fqs.clear()
        self._rig = None
        self._aux_pairs.clear()
        self._aux_images.clear()
        self._aux_intrinsics = None
        self._aux_capture_count = 0
        self._aux_saved_count = 0
        self._aux_imported_count = 0
        self._aux_source_folder = None
        self._aux_observation_board = None
        self._offline_import_records = []
        self._offline_frame_qualities = {}
        self._offline_lr_swapped = False
        self._offline_source_folder = None
        self._pending_import_folder = None
        self._btn_swap_offline.setVisible(False)
        self._btn_swap_offline.setText("交换离线左右图")
        self._quality_report.clear()
        self._aux_capture_sequence = CaptureSequence(build_multimodal_sequence())
        self._aux_seq_widget.set_sequence(self._aux_capture_sequence)
        self._aux_cap_progress.setMaximum(self._aux_capture_sequence.total)
        self._aux_cap_progress.setValue(0)
        self._btn_calib_aux.setEnabled(False)
        self._lbl_aux_status.setText("等待多模态配对图像")
        self._data_score.clear()
        self._capture_sequence.reset()
        self._seq_widget.refresh()
        self._timeline.reset()
        self._undo_stack.clear()
        self._refresh_undo_btn()
        self._pair_capture_press_count.clear()
        self._canceled_save_tokens.update(self._pending_save_tokens)
        self._calib_queue.clear()
        for attr in (
            "_import_last_pair",
            "_imported_frames",
            "_imported_pair_frames",
            "_import_view_idx",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        for sec in self._sections:
            sec.set_status_hint("")
        self._rebuild_pair_rows()
        self._activate_step(0)
        self._update_capture_stats()
        self.calibration_reset.emit()

    def _log(self, msg: str):
        logger.info(msg)
        self._log_text.append(msg)
        self.log_added.emit(msg)

    def append_log(self, msg: str):
        self._log(msg)
