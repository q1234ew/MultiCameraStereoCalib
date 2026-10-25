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
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPointF, QSignalBlocker
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
        self._lbl_name = QLabel(pair_name)
        self._lbl_name.setStyleSheet(f"font-size:13px; font-weight:bold; color:{TEXT};")
        top.addWidget(self._lbl_name)
        top.addStretch()
        self.btn = QPushButton("标定")
        self.btn.setProperty("class", "primary")
        self.btn.setFixedWidth(72)
        self.btn.setEnabled(False)
        top.addWidget(self.btn)
        lay.addLayout(top)

        self._lbl_status = QLabel("等待采集数据...")
        self._lbl_status.setWordWrap(True)
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
    workflow_changed = Signal(str, bool)
    preview_pairs_changed = Signal(object)
    export_requested = Signal()
    open_results_requested = Signal()
    log_added = Signal(str)
    capture_started = Signal()
    create_session_requested = Signal()

    def __init__(self, stream_manager: StreamManager, board: CharucoBoard, parent=None):
        super().__init__(parent)
        self._sm = stream_manager
        self._board = board
        self._camera_model = CameraModel.PINHOLE
        self._active_step = 0
        self._workflow_phase = "capture"
        self._live_preview_indices: dict[str, list[tuple[int, int]]] = {}
        self._result_directory = None
        self._result_files = []
        self._result_errors = []
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
        self._pending_save_tokens: set[str] = set()
        self._canceled_save_tokens: set[str] = set()
        self._calib_worker: Optional[PairCalibrationWorker] = None
        self._aux_worker: Optional[AuxCalibrationWorker] = None
        self._stereo_import_worker: Optional[StereoImageImportWorker] = None
        self._aux_import_worker: Optional[AuxImageImportWorker] = None
        self._calib_queue: List[str] = []
        self._aux_pairs: list[tuple[str, PlanarDetectionResult, PlanarDetectionResult]] = []
        self._aux_images: list[tuple[np.ndarray, np.ndarray]] = []
        self._aux_intrinsics: Optional[CameraIntrinsics] = None
        self._aux_capture_count = 0
        self._aux_observation_board: Optional[PatternBoardConfig] = None
        self._pending_aux_context = None
        self._offline_import_records = []
        self._offline_lr_swapped = False
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

    # ── Public setters ────────────────────────────────────────

    def update_board(self, board: CharucoBoard):
        self._board = board
        self._reset_all()

    def set_camera_model(self, model: CameraModel):
        self._camera_model = model
        self._reset_all()
        self._log(f"模型: {'针孔' if model == CameraModel.PINHOLE else '鱼眼'}")

    def set_session(self, session: CalibrationSession):
        self._white_balance_guard.invalidate()
        self._session = session
        self._next_session_frame_idx = session.frame_count
        self._log(f"会话: {session.name}")
        self._refresh_disk_hint()

    @property
    def rig(self) -> Optional[MultiCameraRig]:
        return self._rig

    @property
    def has_active_operations(self) -> bool:
        return bool(self._auto_timer.isActive() or self._calib_worker or self._aux_worker or
                    self._stereo_import_worker or self._aux_import_worker or
                    self._save_workers or self._save_queue or self._aux_dataset_busy())

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
                self._lbl_rect_preview.clear()
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
                pairs.extend(PreviewPair(f"{name} {i + 1:04d}", cfg.left.camera_id, cfg.right.camera_id, left, right)
                             for i, (left, right) in enumerate(imported))
                continue
            left_cal, right_cal = self._intr_cals.get(cfg.left.camera_id), self._intr_cals.get(cfg.right.camera_id)
            if left_cal is None or right_cal is None:
                continue
            for index, (left, right) in enumerate(self._live_preview_indices.get(name, [])):
                if left < len(left_cal.stored_images) and right < len(right_cal.stored_images):
                    pairs.append(PreviewPair(f"{name} {index + 1:04d}", cfg.left.camera_id, cfg.right.camera_id,
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
        if rig is None:
            self.set_result_storage(None)
        self._refresh_result_summary()

    def _refresh_result_summary(self):
        rig = self._rig
        if rig is None:
            self._lbl_result_summary.setText("尚无标定结果")
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
            for aux in rig.aux_cameras.values():
                lines.extend([aux.camera_id, f"融合 RMS  {aux.rms_error:.3f} px · {aux.frame_count} 组", ""])
            lines.extend(['', *decision.details, '', *decision.risks])
            self._lbl_result_summary.setText("\n".join(lines).rstrip())
        if self._result_errors:
            status = "保存异常\n" + "\n".join(self._result_errors)
        elif self._result_directory is not None:
            directory = Path(self._result_directory)
            status = f"文件已保存：{len(self._result_files)} 个\n{directory.parent.name}/{directory.name}"
        else:
            status = "文件待生成或另存" if rig is not None else ""
        self._lbl_result_storage.setText(status)
        self._lbl_result_storage.setToolTip(str(self._result_directory or ''))
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
        min_row.addWidget(QLabel("最少帧数"))
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
        self._btn_swap_offline.setToolTip(
            "导入离线图片后可交换左右图像源；交换后需重新标定并导出"
        )
        self._btn_swap_offline.setVisible(False)
        btn_row.addWidget(self._btn_swap_offline)
        self._btn_undo = QToolButton()
        self._btn_undo.setIcon(app_icon('undo-2'))
        self._btn_undo.setFixedSize(32, 32)
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

        self._lbl_calib_hint = QLabel(
            "每组双目一键完成：左右相机内参 + 立体外参"
        )
        self._lbl_calib_hint.setWordWrap(True)
        self._lbl_calib_hint.setStyleSheet(f"font-size:12px; color:{TEXT_HINT};")
        l1.addWidget(self._lbl_calib_hint)

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
        self._combo_data_filter.setToolTip(
            "筛选只作用于本次标定计算，不删除已采集/导入的原始数据"
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
        self._lbl_rect_preview = QLabel()
        self._lbl_rect_preview.setAlignment(Qt.AlignCenter)
        self._lbl_rect_preview.setMinimumHeight(50)
        self._lbl_rect_preview.setStyleSheet(f"background:{BG_DARK};border-radius:4px;")
        l1.addWidget(self._lbl_rect_preview)

        self._quality_report = QTextEdit()
        self._quality_report.setReadOnly(True)
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
        self._lbl_result_decision = QLabel("尚无可评估结果")
        self._lbl_result_decision.setWordWrap(True)
        self._lbl_result_decision.setTextFormat(Qt.PlainText)
        results.addWidget(self._lbl_result_decision)
        self._lbl_result_risks = QLabel()
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
        self._lbl_result_summary = QLabel("尚无标定结果")
        self._lbl_result_summary.setTextFormat(Qt.PlainText)
        self._lbl_result_summary.setWordWrap(True)
        self._lbl_result_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        results.addWidget(self._lbl_result_summary)
        self._lbl_result_summary.hide()
        self._lbl_result_storage = QLabel()
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
        self._lbl_calib_hint.hide()

        ml.addStretch()
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
        footer = QFrame()
        self._action_footer = footer
        footer.setObjectName("actionFooter")
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        self._lbl_raw_save = QLabel()
        self._lbl_raw_save.setWordWrap(True)
        self._lbl_raw_save.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        layout.addWidget(self._lbl_raw_save)
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
        layout.addWidget(self._wb_progress)
        self._wb_progress.hide()
        self._wb_progress_timer = QTimer(self)
        self._wb_progress_timer.setInterval(250)
        self._wb_progress_timer.timeout.connect(self._refresh_white_balance_progress)
        self._check_wb = QCheckBox("采图前稳定并锁定 RGB 白平衡")
        self._check_wb.setStyleSheet(f"background:transparent; color:{TEXT_DIM}; font-size:12px;")
        self._check_wb.setChecked(True)
        self._check_wb.setToolTip("首次实时采图前准备两路 RGB；不回写历史色温。无控制接口的相机可手动关闭此项。")
        self._check_wb.toggled.connect(self._on_white_balance_policy_changed)
        layout.addWidget(self._check_wb)
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
        self._btn_capture.setText("采集配对图像")
        self._btn_capture.setIcon(app_icon('camera', '#ffffff'))
        self._btn_capture.setMinimumHeight(40)
        capture.addWidget(self._btn_capture)
        row = QHBoxLayout()
        row.addWidget(self._btn_import, 1)
        row.addWidget(self._btn_auto)
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
        self._btn_to_calibration.clicked.connect(lambda: self.show_workflow("calibration"))
        row.addWidget(self._btn_to_calibration)
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

    def _connect_signals(self):
        self._spin_min.valueChanged.connect(self._update_capture_stats)
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

    @Slot(str)
    def _on_capture_target_changed(self, text):
        self._update_capture_stats()
        self._refresh_preview_pairs()

    def _selected_pairs(self) -> Dict:
        all_pairs = self._sm.stereo_pairs
        sel = self._combo_pair.currentText()
        if sel and sel != "全部相机组" and sel in all_pairs:
            return {sel: all_pairs[sel]}
        return dict(all_pairs)

    # ── Dynamic pair rows ─────────────────────────────────────

    def _rebuild_pair_rows(self):
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

    def _sync_pair_states(self):
        any_ready = False
        all_calibrated = True

        for name, row in self._pair_rows.items():
            cal = self._stereo_cals.get(name)
            done = name in self._pair_calibs

            if done:
                pc = self._pair_calibs[name]
                rms_i_l = pc.left_intrinsics.rms_error
                rms_i_r = pc.right_intrinsics.rms_error
                rms_s = pc.stereo.rms_error
                bl_mm = np.linalg.norm(pc.stereo.T) * 1000.0
                row.set_status(
                    f"✓  内参 L={rms_i_l:.3f}px R={rms_i_r:.3f}px  |  "
                    f"外参 RMS={rms_s:.3f}px 基线={bl_mm:.1f}mm",
                    SUCCESS,
                )
                row.btn.setEnabled(cal is not None and cal.ready)
                row.btn.setText("重新标定")
                any_ready = any_ready or (cal is not None and cal.ready)
            elif cal is not None and cal.ready:
                row.set_status(f"{cal.num_frames} 帧已就绪，可开始标定", ACCENT)
                row.btn.setEnabled(True)
                row.btn.setText("标定")
                any_ready = True
                all_calibrated = False
            elif cal is not None:
                pc = self._pair_capture_press_count.get(name, 0)
                row.set_status(
                    f"按下采集 {pc} 次 · 立体可用 {cal.num_frames} 帧（需 {self._spin_min.value()} 帧）",
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
        if self._calib_worker is not None or self._aux_dataset_busy() or self._stereo_import_worker is not None:
            self._btn_calib_all.setEnabled(False)
            for row in self._pair_rows.values():
                row.btn.setEnabled(False)

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
        self._save_queue.append((token, pair_name, left.copy(), right.copy(), frame_idx))
        self._start_next_frame_save()
        return frame_idx, token

    def _start_next_frame_save(self):
        if self._save_workers or not self._save_queue or self._session is None:
            return
        token, pair_name, left, right, frame_idx = self._save_queue.pop(0)
        if token in self._canceled_save_tokens:
            self._pending_save_tokens.discard(token)
            self._canceled_save_tokens.discard(token)
            self._start_next_frame_save()
            return
        worker = FrameSaveWorker(
            token,
            self._session,
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
        if token in self._canceled_save_tokens:
            self._canceled_save_tokens.discard(token)
            if self._session is not None:
                self._session.delete_saved_pair_frame(pair_name, frame_idx)
        self._start_next_frame_save()

    @Slot(str, str, int, str)
    def _on_frame_save_failed(self, token: str, pair_name: str, frame_idx: int, reason: str):
        self._save_workers.pop(token, None)
        self._pending_save_tokens.discard(token)
        if token in self._canceled_save_tokens:
            self._canceled_save_tokens.discard(token)
            self._start_next_frame_save()
            return
        self._log(f"{pair_name}: 保存图片失败（帧 {frame_idx}）— {reason}")
        self._start_next_frame_save()

    def _undo_apply_entry(self, e: _UndoCaptureEntry) -> None:
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
        elif e.session_frame_idx is not None and self._session:
            self._session.delete_saved_pair_frame(e.pair_name, e.session_frame_idx)

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

    @Slot()
    def _on_capture(self):
        with perf_timer("capture click", threshold_ms=100.0):
            self._capture_impl()

    def _capture_impl(self):
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

        if not self._ensure_capture_session():
            return
        resume_capture = self._on_auto_capture if self._auto_timer.isActive() else self._on_capture
        if not self._prepare_white_balance(pairs, resume_capture):
            return

        self.capture_started.emit()

        captured = 0
        batch_entries: List[_UndoCaptureEntry] = []

        for name, pair_cfg in pairs.items():
            pair_frames = self._sm.get_sync_pair(name)
            relaxed_note = ""
            if pair_frames is None:
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

            n_ib_l = self._intr_cals[lid].num_frames
            n_ib_r = self._intr_cals[rid].num_frames
            n_sb = self._stereo_cals[name].num_frames

            det_l = self._intr_cals[lid].add_frame(left)
            det_r = self._intr_cals[rid].add_frame(right)
            self._stereo_cals[name].add_detection_pair(det_l, det_r)
            if (self._stereo_cals[name].num_frames > n_sb and
                    self._intr_cals[lid].num_frames > n_ib_l and self._intr_cals[rid].num_frames > n_ib_r):
                self._live_preview_indices.setdefault(name, []).append((
                    len(self._intr_cals[lid].stored_images) - 1, len(self._intr_cals[rid].stored_images) - 1,
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
                f"{name}: L={det_l.num_corners}[{ql}] "
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
                )
            )

        if captured == 0:
            self._log(
                "未取到左右图像（请先连接相机并确认两路均有预览）。"
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
        self._stereo_import_worker = None
        self._set_import_busy(False)
        self._log(f"导入 {len(img_pairs)} 对图片")

        self._offline_import_records = list(img_pairs)
        self._offline_lr_swapped = False
        self._load_offline_records(swapped=False, log_frames=True)

    def _load_offline_records(self, swapped: bool, log_frames: bool = False):
        img_pairs = list(self._offline_import_records)
        if not img_pairs:
            self._log("没有可交换的离线图片，请先导入图片")
            return

        pair_name = "offline"
        lid, rid = "offline_L", "offline_R"
        min_f = min(self._spin_min.value(), len(img_pairs))
        self._spin_min.setValue(min_f)

        self._reset_offline_pair(pair_name, lid, rid, min_f)
        self._capture_sequence.reset()

        for idx, img_l, img_r, det_l, det_r in img_pairs:
            if swapped:
                img_l, img_r = img_r, img_l
                det_l, det_r = det_r, det_l
            self._intr_cals[lid].add_detection(det_l, image=img_l)
            self._intr_cals[rid].add_detection(det_r, image=img_r)
            self._stereo_cals[pair_name].add_detection_pair(det_l, det_r)

            fi = len(self._fqs[lid])
            fq_l = self._assessor.assess_frame(img_l, det_l.num_corners, fi)
            fq_r = self._assessor.assess_frame(img_r, det_r.num_corners, fi)
            self._fqs[lid].append(fq_l)
            self._fqs[rid].append(fq_r)

            if not self._capture_sequence.finished:
                self._capture_sequence.advance()

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
        self._sections[0].set_status_hint(f"✓ {len(img_pairs)} 帧已导入", SUCCESS)
        direction = "（已交换左右图）" if swapped else ""
        self._log(f"导入完成{direction}，{self._stereo_cals[pair_name].num_frames} 帧可用")

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
        self._set_import_busy(False)
        self._log(message.splitlines()[0])

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
            self._on_capture()

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
        self._lbl_capture_count.setText(f'有效配对 {mn_st} / {min_f}')
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
            au = np.mean([a.distribution_uniformity for a in assessments])
            tu = sum(a.usable_frames for a in assessments)
            tf = sum(a.total_frames for a in assessments)
            self._lbl_assess.setText(
                f"覆盖 {ac:.0%} | 均匀 {au:.0%} | 可用 {tu}/{tf}"
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

        any_ready = any(c.ready for c in self._stereo_cals.values())
        if any_ready:
            self._sections[0].set_status_hint("✓ 可标定", SUCCESS)

    def _calib_strategy_label(self) -> str:
        return self._combo_calib_strategy.currentText().split("：", 1)[0]

    def _data_filter_label(self) -> str:
        return self._combo_data_filter.currentText()

    def _calib_strategy_flags(self) -> tuple[int, int]:
        key = self._combo_calib_strategy.currentData() or "standard"
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

    def _filtered_calibrators_for_pair(self, pair_name: str, lid: str, rid: str):
        filter_key = self._combo_data_filter.currentData() or "all"
        left_cal = self._intr_cals[lid]
        right_cal = self._intr_cals[rid]
        stereo_cal = self._stereo_cals[pair_name]
        if filter_key == "all":
            return left_cal, right_cal, stereo_cal

        if filter_key == "quality":
            left_idx = self._quality_filter_indices(left_cal)
            right_idx = self._quality_filter_indices(right_cal)
            stereo_idx = self._top_count_indices(
                stereo_cal.frame_common_corner_counts,
                stereo_cal.min_frames,
                keep_ratio=0.8,
            )
        else:
            left_idx = self._top_count_indices(
                left_cal.frame_corner_counts,
                left_cal.min_frames,
                keep_ratio=0.8,
            )
            right_idx = self._top_count_indices(
                right_cal.frame_corner_counts,
                right_cal.min_frames,
                keep_ratio=0.8,
            )
            stereo_idx = self._top_count_indices(
                stereo_cal.frame_common_corner_counts,
                stereo_cal.min_frames,
                keep_ratio=0.8,
            )

        f_left = left_cal.copy_with_indices(left_idx)
        f_right = right_cal.copy_with_indices(right_idx)
        f_stereo = stereo_cal.copy_with_indices(stereo_idx)

        if not (f_left.ready and f_right.ready and f_stereo.ready):
            self._log("数据筛选后帧数不足，已回退为全部使用")
            return left_cal, right_cal, stereo_cal

        self._log(
            f"数据筛选[{self._data_filter_label()}]: "
            f"L {f_left.num_frames}/{left_cal.num_frames}, "
            f"R {f_right.num_frames}/{right_cal.num_frames}, "
            f"Stereo {f_stereo.num_frames}/{stereo_cal.num_frames}"
        )
        return f_left, f_right, f_stereo

    def _quality_filter_indices(self, cal):
        scores = []
        images = cal.stored_images
        counts = cal.frame_corner_counts
        max_corners = max(counts) if counts else 1
        for idx, count in enumerate(counts):
            corner_score = count / max(max_corners, 1)
            image_score = 0.6
            if idx < len(images):
                img = images[idx]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(gray.mean())
                sharp_score = min(sharpness / 120.0, 1.0)
                bright_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
                image_score = 0.65 * sharp_score + 0.35 * bright_score
            scores.append(0.55 * corner_score + 0.45 * image_score)
        return self._top_score_indices(scores, cal.min_frames, keep_ratio=0.8)

    @staticmethod
    def _top_count_indices(counts: list[int], min_frames: int, keep_ratio: float) -> list[int]:
        return CalibrationPanel._top_score_indices(
            [float(c) for c in counts],
            min_frames,
            keep_ratio,
        )

    @staticmethod
    def _top_score_indices(scores: list[float], min_frames: int, keep_ratio: float) -> list[int]:
        n = len(scores)
        if n == 0:
            return []
        keep = max(min_frames, int(np.ceil(n * keep_ratio)))
        keep = min(n, keep)
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        return sorted(ranked[:keep])

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
            lines.append("优化策略: 优先补采边角/倾角样本，剔除高误差帧后重跑；必要时尝试稳健策略。")
        elif (self._combo_calib_strategy.currentData() or "standard") == "joint_refine":
            lines.append("优化策略: 联合精修适合最终对比，若结果不稳定可回到标准策略。")
        else:
            lines.append("优化策略: 当前结果可用；如需第三方使用，请导出 OpenCV 标准 Stereo YAML。")

        self._quality_report.setPlainText("\n".join(lines))

    # ── Stereo calibration (per pair: intrinsic + extrinsic) ──

    def _calibrate_pair(self, pair_name: str, continue_queue: bool = False):
        """Run full calibration for one stereo pair: intrinsic L+R then stereo."""
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
        if not pair_cfg or not cal or not cal.ready:
            self._log(f"{pair_name}: 数据不足")
            if continue_queue:
                self._start_next_calibration()
            return

        self._activate_step(1)
        lid = pair_cfg.left.camera_id
        rid = pair_cfg.right.camera_id
        intrinsic_flags, stereo_flags = self._calib_strategy_flags()
        left_cal, right_cal, stereo_cal = self._filtered_calibrators_for_pair(
            pair_name, lid, rid
        )

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
            if cal and cal.ready:
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

    def _invalidate_pair_result(self, pair_name: str):
        """Keep collected observations, but invalidate results derived from older data."""
        pair = self._pair_calibs.pop(pair_name, None)
        if pair is None:
            return
        self._intrinsics.pop(pair.left_id, None)
        self._intrinsics.pop(pair.right_id, None)
        self._rig = None
        self._aux_intrinsics = None
        self._quality_report.clear()
        self._lbl_rect_preview.clear()
        self._btn_multi.setEnabled(False)
        self._btn_cloud.setEnabled(False)
        self._timeline.mark_pending(1)
        self._timeline.mark_pending(2)
        self._timeline.mark_pending(3)
        self._sections[1].set_status_hint("数据已更新，待重新标定", WARNING)
        self._sections[2].set_status_hint("")
        self._sections[3].set_status_hint("")
        self.calibration_reset.emit()
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
        self._combo_calib_strategy.setEnabled(not busy)
        self._combo_data_filter.setEnabled(not busy)
        self._btn_calib_all.setEnabled(False if busy else self._btn_calib_all.isEnabled())
        for row in self._pair_rows.values():
            row.btn.setEnabled(False if busy else row.btn.isEnabled())
        self._refresh_aux_readiness()

    def _show_rect_preview(self, pair_name: str, pair: StereoPairCalibration):
        sync = self._sm.get_sync_pair(pair_name)
        if sync is not None:
            left, right = sync
        elif hasattr(self, "_import_last_pair"):
            left, right = self._import_last_pair
        else:
            return
        preview = self._evaluator.generate_rectification_preview(
            left, right, pair.stereo, num_lines=15
        )
        h, w = preview.shape[:2]
        tw = max(self._lbl_rect_preview.width(), 200)
        sc = min(tw / max(w, 1), 160.0 / max(h, 1))
        nw, nh = int(w * sc), int(h * sc)
        if nw > 0 and nh > 0:
            small = cv2.resize(preview, (nw, nh))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, nw, nh, nw * 3, QImage.Format_RGB888)
            self._lbl_rect_preview.setPixmap(QPixmap.fromImage(qimg))

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

        rgb_frame, aux_frame, _ = frames
        self.capture_started.emit()

        board_cfg = self._aux_board_config()
        if self._aux_pairs and self._aux_observation_board != board_cfg:
            self._log("辅助板参数已变更，请重新导入图片或通过配置重置后重新采集")
            return
        detector = PlanarPatternDetector(board_cfg)
        rgb_det = detector.detect(rgb_frame)
        aux_det = detector.detect(aux_frame)
        idx = f"live_{self._aux_capture_count:04d}"
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

        if self._session is not None:
            try:
                self._session.save_multimodal_pair(rgb_frame, aux_frame)
            except Exception as exc:
                self._log(f"{idx}: 多模态图片保存失败 — {str(exc).splitlines()[0]}")

        self._aux_capture_count += 1
        self._refresh_aux_capture_state()

    @Slot()
    def _on_import_aux_images(self):
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

        for key, rgb, aux, rgb_det, aux_det in img_pairs:
            self._aux_images.append((rgb, aux))
            if rgb_det.valid and aux_det.valid:
                self._aux_pairs.append((key, rgb_det, aux_det))
                self._log(f"  [{key}] RGB_L={rgb_det.num_points} AUX={aux_det.num_points}")
            else:
                self._log(f"  [{key}] 检测失败 RGB_L={rgb_det.valid} AUX={aux_det.valid}")

        valid = len(self._aux_pairs)
        for _ in range(min(valid, self._aux_capture_sequence.total)):
            if self._aux_capture_sequence.finished:
                break
            self._aux_capture_sequence.advance()
        self._aux_seq_widget.refresh()
        self._refresh_aux_capture_state()
        if self._aux_images:
            rgb, aux = self._aux_images[-1]
            aux_id = self._aux_camera_id()
            rgb_left_id = self._rgb_left_camera_id() or "RGB_L"
            self.aux_images_imported.emit({rgb_left_id: rgb, aux_id: aux})
        self._log(f"多模态导入完成：有效 {valid}/{len(img_pairs)} 组")

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
            self._session.record_multimodal_result(
                aux_camera_id=aux_id,
                modality=modality,
                board_type=board_cfg.pattern_type.value,
                valid_pairs=extr.frame_count,
                rms_error=extr.rms_error,
                board_config=board_cfg.to_dict(),
            )
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
        self._refresh_aux_readiness()
        self._lbl_aux_status.setText(f"最少 3 组 · 建议 {total} 组")
        self._refresh_preview_pairs()

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
        return (entries[0][0], entries[1][0], dt), f"采图就绪：{'；'.join(states)}；时间差 {dt * 1000:.0f} ms"

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
        self._aux_intrinsics = None
        if self._rig is not None:
            self._rig.aux_cameras.pop(self._aux_camera_id(), None)
        self._timeline.mark_pending(2)
        self._sections[2].set_status_hint("")
        self.set_result_storage(None)

    def _aux_dataset_busy(self) -> bool:
        return bool(self._aux_ui_busy or self._aux_import_busy or self._aux_worker or self._aux_import_worker or
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
        return self._white_balance_guard.ensure(urls, resume_current)

    def _on_white_balance_policy_changed(self, enabled):
        self._white_balance_guard.invalidate()
        if not enabled:
            self._log("采图前白平衡管理已关闭；后续采图不检查颜色稳定或锁定状态")

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
        self._aux_observation_board = None
        self._offline_import_records = []
        self._offline_lr_swapped = False
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
