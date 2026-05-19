"""Calibration panel — merged stereo workflow (intrinsic+extrinsic in one step)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPointF
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
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
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
from ...io.session import CalibrationSession
from ...perf import perf_timer
from ...streaming.stream_manager import StreamManager
from ..threads.calibration_worker import AuxCalibrationWorker, PairCalibrationWorker
from ..threads.import_worker import AuxImageImportWorker, StereoImageImportWorker
from ..threads.save_worker import FrameSaveWorker
from ..theme import (
    ACCENT, BG_CARD, BG_DARK, BG_INPUT,
    BORDER, SUCCESS, TEXT, TEXT_DIM, TEXT_HINT, WARNING,
)
from .capture_guide import (
    CaptureSequence,
    CaptureSequenceWidget,
    build_default_sequence,
    build_multimodal_sequence,
)

logger = logging.getLogger(__name__)


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
        self._lbl.setText(f"{grade.label} ({score:.0f})")
        self.setStyleSheet(f"ScoreBadge{{background:{grade.color};border-radius:4px;}}")
        self._lbl.setStyleSheet("font-weight:bold;font-size:12px;color:white;")

    def clear(self):
        self._lbl.setText("--")
        self.setStyleSheet("")


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

    calibration_finished = Signal(object)
    pointcloud_ready = Signal(object)
    frame_captured = Signal(str, object)
    sequence_updated = Signal(object)
    images_imported = Signal(dict)  # {camera_id: np.ndarray}

    def __init__(self, stream_manager: StreamManager, board: CharucoBoard, parent=None):
        super().__init__(parent)
        self._sm = stream_manager
        self._board = board
        self._camera_model = CameraModel.PINHOLE
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
        self._pending_aux_context = None

        self._init_ui()
        self._connect_signals()
        self._activate_step(0)
        self._refresh_disk_hint()

    # ── Public setters ────────────────────────────────────────

    def update_board(self, board: CharucoBoard):
        self._board = board
        self._reset_all()

    def set_camera_model(self, model: CameraModel):
        self._camera_model = model
        self._reset_all()
        self._log(f"模型: {'针孔' if model == CameraModel.PINHOLE else '鱼眼'}")

    def set_session(self, session: CalibrationSession):
        self._session = session
        self._next_session_frame_idx = session.frame_count
        self._log(f"会话: {session.name}")
        self._refresh_disk_hint()

    # ── UI construction ───────────────────────────────────────

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        ml = QVBoxLayout(container)
        ml.setContentsMargins(8, 8, 8, 8)
        ml.setSpacing(8)

        # Timeline: stereo capture/calibration, optional multimodal mono, then point cloud.
        self._timeline = StepTimeline(["共视采图", "双目标定", "多模态", "点云融合"])
        ml.addWidget(self._timeline)

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

        # pair selector
        pair_row = QHBoxLayout()
        pair_row.addWidget(QLabel("采集目标"))
        self._combo_pair = QComboBox()
        self._combo_pair.addItem("全部相机组")
        self._combo_pair.setMinimumWidth(120)
        pair_row.addWidget(self._combo_pair, 1)
        l0.addLayout(pair_row)

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
        self._btn_auto = QPushButton("自动采集")
        self._btn_auto.setCheckable(True)
        btn_row.addWidget(self._btn_auto)
        self._btn_import = QPushButton("导入图片")
        btn_row.addWidget(self._btn_import)
        self._btn_undo = QPushButton("撤销上一帧")
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
        self._cap_progress.setFormat("%v / %m")
        l0.addWidget(self._cap_progress)

        self._seq_widget = CaptureSequenceWidget()
        self._seq_widget.set_sequence(self._capture_sequence)
        l0.addWidget(self._seq_widget)

        self._lbl_assess = QLabel("覆盖 -- | 均匀 -- | 可用 --")
        self._lbl_assess.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        l0.addWidget(self._lbl_assess)
        self._data_score = ScoreBadge()
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

        ml.addWidget(s1)
        self._sections.append(s1)

        # ── 2. Multimodal auxiliary mono calibration ─────────
        s2 = CollapsibleSection("3  多模态单目标定")
        l2 = s2.content_layout

        self._lbl_aux_info = QLabel(
            "导入 rgb_left_XXXX.* 与 aux_XXXX.* 配对图像，先完成 RGB 双目标定后可求辅助单目内参和 AUX→RGB_L 外参。"
        )
        self._lbl_aux_info.setWordWrap(True)
        self._lbl_aux_info.setStyleSheet(f"font-size:12px; color:{TEXT_HINT};")
        l2.addWidget(self._lbl_aux_info)

        self._aux_seq_widget = CaptureSequenceWidget()
        self._aux_seq_widget.set_sequence(self._aux_capture_sequence)
        l2.addWidget(self._aux_seq_widget)

        self._aux_cap_progress = QProgressBar()
        self._aux_cap_progress.setFormat("%v / %m")
        self._aux_cap_progress.setMaximum(self._aux_capture_sequence.total)
        self._aux_cap_progress.setValue(0)
        l2.addWidget(self._aux_cap_progress)

        aux_btn_row = QHBoxLayout()
        self._btn_capture_aux = QPushButton("采集多模态一帧")
        self._btn_capture_aux.setProperty("class", "primary")
        aux_btn_row.addWidget(self._btn_capture_aux)
        self._btn_import_aux = QPushButton("导入多模态图片")
        aux_btn_row.addWidget(self._btn_import_aux)
        self._btn_calib_aux = QPushButton("标定辅助单目")
        self._btn_calib_aux.setProperty("class", "primary")
        self._btn_calib_aux.setEnabled(False)
        aux_btn_row.addWidget(self._btn_calib_aux)
        l2.addLayout(aux_btn_row)

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
        self._log_text.setPlaceholderText("操作日志...")
        ml.addWidget(self._log_text)

        ml.addStretch()
        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _connect_signals(self):
        self._btn_capture.clicked.connect(self._on_capture)
        self._btn_undo.clicked.connect(self._on_undo_capture)
        self._btn_auto.toggled.connect(self._on_auto_toggle)
        self._btn_import.clicked.connect(self._on_import_images)
        self._btn_capture_aux.clicked.connect(self._on_capture_aux)
        self._btn_import_aux.clicked.connect(self._on_import_aux_images)
        self._btn_calib_aux.clicked.connect(self._on_calibrate_aux)
        self._btn_calib_all.clicked.connect(self._on_calibrate_all)
        self._btn_multi.clicked.connect(self._on_calibrate_multiview)
        self._btn_cloud.clicked.connect(self._on_generate_cloud)

        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(2000)
        self._auto_timer.timeout.connect(self._on_capture)

        self._timeline.step_clicked.connect(self._activate_step)
        self._seq_widget.user_navigated.connect(self._on_sequence_user_nav)
        self._aux_seq_widget.user_navigated.connect(self._on_aux_sequence_user_nav)
        for i, sec in enumerate(self._sections):
            idx = i
            sec.toggled.connect(lambda _, ii=idx: self._on_section_toggled(ii))

    # ── Navigation ────────────────────────────────────────────

    def _activate_step(self, step: int):
        self._timeline.set_step(step)
        for i, sec in enumerate(self._sections):
            sec.set_expanded(i == step)

    def _on_section_toggled(self, index: int):
        self._timeline.set_step(index)
        for i, sec in enumerate(self._sections):
            if i != index:
                sec.set_expanded(False)

    @Slot()
    def _on_sequence_user_nav(self):
        """User changed active guide step (grid / prev-next / skip)."""
        self.sequence_updated.emit(self._capture_sequence)

    @Slot()
    def _on_aux_sequence_user_nav(self):
        """User changed active multimodal guide step."""
        self.sequence_updated.emit(self._aux_capture_sequence)

    # ── Pair selector ────────────────────────────────────────

    def refresh_pairs(self):
        """Sync the pair combo and calibration rows with current StreamManager state."""
        cur = self._combo_pair.currentText()
        self._combo_pair.clear()
        self._combo_pair.addItem("全部相机组")
        for name in self._sm.stereo_pairs:
            self._combo_pair.addItem(name)
        idx = self._combo_pair.findText(cur)
        if idx >= 0:
            self._combo_pair.setCurrentIndex(idx)
        self._rebuild_pair_rows()

    def _selected_pairs(self) -> Dict:
        all_pairs = self._sm.stereo_pairs
        sel = self._combo_pair.currentText()
        if sel and sel != "全部相机组" and sel in all_pairs:
            return {sel: all_pairs[sel]}
        return dict(all_pairs)

    # ── Dynamic pair rows ─────────────────────────────────────

    def _rebuild_pair_rows(self):
        for r in self._pair_rows.values():
            r.setParent(None)
            r.deleteLater()
        self._pair_rows.clear()

        for name in self._sm.stereo_pairs:
            row = _PairRow(name, self)
            row.btn.clicked.connect(lambda _, n=name: self._calibrate_pair(n))
            self._calib_container.addWidget(row)
            self._pair_rows[name] = row

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
                row.btn.setEnabled(False)
                row.btn.setText("完成")
            elif cal is not None and cal.ready:
                row.set_status(f"{cal.num_frames} 帧已就绪，可开始标定", ACCENT)
                row.btn.setEnabled(True)
                any_ready = True
                all_calibrated = False
            elif cal is not None:
                pc = self._pair_capture_press_count.get(name, 0)
                row.set_status(
                    f"按下采集 {pc} 次 · 立体可用 {cal.num_frames} 帧（需 {self._spin_min.value()} 帧）",
                    TEXT_DIM,
                )
                row.btn.setEnabled(False)
                all_calibrated = False
            else:
                row.set_status("等待采集数据...", TEXT_HINT)
                row.btn.setEnabled(False)
                all_calibrated = False

        self._btn_calib_all.setEnabled(any_ready)

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
        if self._session is None:
            self._lbl_disk_hint.setText(
                "磁盘保存：请先「文件 → 新建标定会话」，采集图像才会写入会话目录 "
                "（路径形如 sessions/<会话名>/images/<相机组名>/left_XXXX.png）。"
            )
        else:
            base = self._session.resolved_images_base()
            self._lbl_disk_hint.setText(
                f"磁盘保存：PNG 写入目录「{base}」（按相机组分子文件夹）。"
            )

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
        if not self._undo_stack:
            return
        batch, advanced_guide = self._undo_stack.pop()
        for ent in reversed(batch):
            self._undo_apply_entry(ent)
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

    @Slot()
    def _on_capture(self):
        with perf_timer("capture click", threshold_ms=100.0):
            self._capture_impl()

    def _capture_impl(self):
        pairs = self._selected_pairs()
        if not pairs:
            self._log("未配置相机组，请先通过 ⚙ 添加相机")
            return

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

    @Slot()
    def _on_import_images(self):
        """Load pre-captured image pairs from a folder (NNNN_left.png / NNNN_right.png)."""
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
        worker.finished_ok.connect(self._on_import_images_ready)
        worker.failed.connect(self._on_import_images_failed)
        worker.finished.connect(worker.deleteLater)
        self._stereo_import_worker = worker
        worker.start()

    @Slot(object)
    def _on_import_images_ready(self, img_pairs):
        self._stereo_import_worker = None
        self._set_import_busy(False)
        self._log(f"导入 {len(img_pairs)} 对图片")

        pair_name = "offline"
        lid, rid = "offline_L", "offline_R"
        min_f = min(self._spin_min.value(), len(img_pairs))
        self._spin_min.setValue(min_f)

        if pair_name not in self._stereo_cals:
            from ...streaming.stream_manager import CameraConfig, StereoPairConfig
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

        for idx, img_l, img_r, det_l, det_r in img_pairs:
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

            self._log(
                f"  [{idx}] L={det_l.num_corners} R={det_r.num_corners} "
                f"帧{self._stereo_cals[pair_name].num_frames}"
            )

        self._import_last_pair = (img_pairs[-1][1], img_pairs[-1][2])
        self._imported_frames = [(lid, rid, il, ir) for (_, il, ir, _, _) in img_pairs]
        self._imported_pair_frames = {pair_name: [(il, ir) for (_, il, ir, _, _) in img_pairs]}
        self._import_view_idx = 0

        self._seq_widget.refresh()
        self.sequence_updated.emit(self._capture_sequence)
        self._update_capture_stats()
        self._sync_pair_states()
        self._timeline.mark_completed(0)
        self._sections[0].set_status_hint(f"✓ {len(img_pairs)} 帧已导入", SUCCESS)
        self._log(f"导入完成，{self._stereo_cals[pair_name].num_frames} 帧可用")

        self._pair_capture_press_count[pair_name] = len(img_pairs)

        self._show_imported_frame(0)
        self._btn_import_prev.setVisible(True)
        self._btn_import_next.setVisible(True)
        self._lbl_import_nav.setVisible(True)
        self._update_import_nav_label()

    @Slot(str)
    def _on_import_images_failed(self, message: str):
        self._stereo_import_worker = None
        self._set_import_busy(False)
        self._log(message.splitlines()[0])

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
            self._btn_auto.setText("⏹ 停止")
        else:
            self._auto_timer.stop()
            self._btn_auto.setText("自动采集")

    def _update_capture_stats(self):
        min_f = self._spin_min.value()
        sel = self._selected_pairs()
        counts_press = [
            self._pair_capture_press_count.get(n, 0)
            for n in sel if n in self._stereo_cals
        ]
        mc = min(counts_press) if counts_press else 0
        self._cap_progress.setMaximum(min_f)
        self._cap_progress.setValue(min(mc, min_f))

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
            stereo_ns = [
                self._stereo_cals[n].num_frames
                for n in sel if n in self._stereo_cals
            ]
            mn_st = min(stereo_ns) if stereo_ns else 0
            self._lbl_assess.setText(
                f"采集 {mc}/{min_f} | 立体可用 {mn_st}/{min_f} | "
                f"覆盖 {ac:.0%} | 均匀 {au:.0%} | 可用 {tu}/{tf}"
            )
            worst = min(assessments, key=lambda a: a.score)
            self._data_score.set_grade(worst.grade, worst.score)
            tips = worst.suggestions[:2]
            self._lbl_tips.setText("  ".join(tips) if tips else "")
        else:
            stereo_ns = [
                self._stereo_cals[n].num_frames
                for n in sel if n in self._stereo_cals
            ]
            mn_st = min(stereo_ns) if stereo_ns else 0
            self._lbl_assess.setText(
                f"采集 {mc}/{min_f} | 立体可用 {mn_st}/{min_f} | 覆盖 -- | 均匀 -- | 可用 --"
            )

        any_ready = any(c.ready for c in self._stereo_cals.values())
        if any_ready:
            self._sections[0].set_status_hint("✓ 可标定", SUCCESS)

    # ── Stereo calibration (per pair: intrinsic + extrinsic) ──

    def _calibrate_pair(self, pair_name: str, continue_queue: bool = False):
        """Run full calibration for one stereo pair: intrinsic L+R then stereo."""
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

        self._set_calibration_busy(True)
        worker = PairCalibrationWorker(
            pair_name=pair_name,
            left_id=lid,
            right_id=rid,
            left_calibrator=self._intr_cals[lid],
            right_calibrator=self._intr_cals[rid],
            stereo_calibrator=cal,
            left_intrinsics=self._intrinsics.get(lid),
            right_intrinsics=self._intrinsics.get(rid),
            parent=self,
        )
        worker.log.connect(self._log)
        worker.finished_ok.connect(self._on_pair_calibration_finished)
        worker.failed.connect(self._on_pair_calibration_failed)
        worker.finished.connect(worker.deleteLater)
        self._calib_worker = worker
        worker.start()

    @Slot()
    def _on_calibrate_all(self):
        """Batch calibrate all ready pairs."""
        if self._calib_worker is not None:
            self._log("已有标定任务正在运行，请稍候")
            return
        self._calib_queue = []
        for name in list(self._sm.stereo_pairs.keys()):
            if name in self._pair_calibs:
                continue
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
        self._show_rect_preview(pair_name, pair_calib)
        self._sync_pair_states()
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
        self._btn_undo.setEnabled((not busy) and bool(self._undo_stack))
        self._btn_auto.setEnabled(not busy)
        self._btn_calib_all.setEnabled(False if busy else self._btn_calib_all.isEnabled())
        for row in self._pair_rows.values():
            row.btn.setEnabled(False if busy else row.btn.isEnabled())

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
        aux = self._sm.auxiliary_camera
        if aux is None:
            self._log("请先在配置窗口启用并选择融合辅助单目相机")
            return
        rgb_left_id = self._rgb_left_camera_id()
        if not rgb_left_id:
            self._log("请先配置 RGB 左相机")
            return

        rgb_entry = self._sm.get_latest_frame(rgb_left_id)
        aux_entry = self._sm.get_latest_frame(aux.camera_id)
        if rgb_entry is None or aux_entry is None:
            self._log("未取到 RGB_L/AUX 最新帧，请先连接相机并确认两路有预览")
            return

        rgb_frame, rgb_ts = rgb_entry
        aux_frame, aux_ts = aux_entry
        dt_ms = abs(rgb_ts - aux_ts) * 1000.0
        if dt_ms > self._sm.sync_tolerance_seconds * 1000.0:
            self._log(f"RGB_L/AUX 时间差 {dt_ms:.0f}ms，已使用最新帧作为软同步采集")

        board_cfg = self._aux_board_config()
        detector = PlanarPatternDetector(board_cfg)
        rgb_det = detector.detect(rgb_frame)
        aux_det = detector.detect(aux_frame)
        idx = f"live_{self._aux_capture_count:04d}"
        self._aux_images.append((rgb_frame.copy(), aux_frame.copy()))

        if rgb_det.valid and aux_det.valid:
            self._aux_pairs.append((idx, rgb_det, aux_det))
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
        self.images_imported.emit({rgb_left_id: rgb_frame, aux.camera_id: aux_frame})

    @Slot()
    def _on_import_aux_images(self):
        if self._aux_import_worker is not None:
            self._log("多模态图片导入正在进行，请稍候")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择多模态图片目录")
        if not folder:
            return

        board_cfg = self._aux_board_config()
        self._log("正在后台导入多模态图片")
        self._set_aux_import_busy(True)
        worker = AuxImageImportWorker(folder, board_cfg, parent=self)
        worker.finished_ok.connect(self._on_import_aux_images_ready)
        worker.failed.connect(self._on_import_aux_images_failed)
        worker.finished.connect(worker.deleteLater)
        self._aux_import_worker = worker
        worker.start()

    @Slot(object)
    def _on_import_aux_images_ready(self, img_pairs):
        self._aux_import_worker = None
        self._set_aux_import_busy(False)
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
            self.images_imported.emit({"RGB_L": rgb, aux_id: aux})
        self._log(f"多模态导入完成：有效 {valid}/{len(img_pairs)} 组")

    @Slot(str)
    def _on_import_aux_images_failed(self, message: str):
        self._aux_import_worker = None
        self._set_aux_import_busy(False)
        self._log(message.splitlines()[0])

    @Slot()
    def _on_calibrate_aux(self):
        if len(self._aux_pairs) < 3:
            self._log("多模态配对观测不足，至少需要 3 组有效图像")
            return
        rgb_left_id = self._rgb_left_camera_id()
        if not rgb_left_id or rgb_left_id not in self._intrinsics:
            self._log("请先完成 RGB_L 所在双目标定，再进行辅助单目标定")
            return

        board_cfg = self._aux_board_config()
        aux_id = self._aux_camera_id()
        modality = self._aux_modality()

        min_frames = min(max(3, self._spin_min.value()), len(self._aux_pairs))
        if self._aux_worker is not None:
            self._log("已有多模态标定任务正在运行，请稍候")
            return

        self._activate_step(2)
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
        self._btn_capture_aux.setEnabled(not busy)
        self._btn_import_aux.setEnabled(not busy)
        self._btn_calib_aux.setEnabled((not busy) and len(self._aux_pairs) >= 3)

    def _set_import_busy(self, busy: bool):
        self._btn_import.setEnabled(not busy)
        self._btn_capture.setEnabled(not busy)
        self._btn_auto.setEnabled(not busy)
        self._btn_undo.setEnabled((not busy) and bool(self._undo_stack))

    def _set_aux_import_busy(self, busy: bool):
        self._btn_capture_aux.setEnabled(not busy)
        self._btn_import_aux.setEnabled(not busy)
        self._btn_calib_aux.setEnabled((not busy) and len(self._aux_pairs) >= 3)

    def _refresh_aux_capture_state(self):
        valid = len(self._aux_pairs)
        total = self._aux_capture_sequence.total
        self._aux_cap_progress.setMaximum(total)
        self._aux_cap_progress.setValue(min(valid, total))
        self._btn_calib_aux.setEnabled(valid >= 3 and self._aux_worker is None)
        self._lbl_aux_status.setText(
            f"多模态观测 {valid}/{total} 组有效，至少 3 组可标定，建议按引导采满"
        )
        if valid >= total:
            self._sections[2].set_status_hint("✓ 采图完成", SUCCESS)
        elif valid >= 3:
            self._sections[2].set_status_hint("✓ 可标定", SUCCESS)

    def _aux_board_config(self) -> PatternBoardConfig:
        aux = self._sm.auxiliary_camera
        pattern = aux.board_pattern if aux is not None else "chessboard"
        cfg = self._board.config
        return PatternBoardConfig(
            cols=max(2, cfg.cols - 1),
            rows=max(2, cfg.rows - 1),
            square_size=cfg.square_length,
            pattern_type=pattern,
        )

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
        for pair in self._sm.stereo_pairs.values():
            return pair.left.camera_id
        for pair in self._pair_calibs.values():
            return pair.left_id
        return None

    def _ensure_rig(self) -> MultiCameraRig:
        if self._rig is not None:
            return self._rig
        mv = MultiViewCalibrator()
        for pc in self._pair_calibs.values():
            mv.add_pair_calibration(pc)
        self._rig = mv.calibrate()
        return self._rig

    # ── Multi-view ────────────────────────────────────────────

    @Slot()
    def _on_calibrate_multiview(self):
        self._activate_step(3)
        self._log("联合优化...")

        try:
            previous_aux = dict(self._rig.aux_cameras) if self._rig is not None else {}
            mv = MultiViewCalibrator()
            for pc in self._pair_calibs.values():
                mv.add_pair_calibration(pc)
            self._rig = mv.calibrate()
            self._rig.aux_cameras.update(previous_aux)
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
            mv = MultiViewCalibrator()
            for pc in self._pair_calibs.values():
                mv.add_pair_calibration(pc)
            try:
                self._rig = mv.calibrate()
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
        self.sequence_updated.emit(self._capture_sequence)

    def _log(self, msg: str):
        logger.info(msg)
        self._log_text.append(msg)
