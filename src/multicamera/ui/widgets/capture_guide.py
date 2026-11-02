"""Visual capture sequence guide — position, angle, and ordering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QAction,
    QPainter,
    QPen,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..opencv_unicode_text import TextPainter
from .board_motion import BoardTilt, TILT_LABELS, tilt_point
from ..theme import (BG_DARK, BG_MID, BORDER, TEXT, TEXT_DIM, TEXT_HINT, ACCENT, PRIMARY,
                     SUCCESS, WARNING, standard_icon, app_icon)


# ═══════════════════════════════════════════════════════════════
#  Data model
# ═══════════════════════════════════════════════════════════════


# Image position is independent of physical board distance.
STEREO_ZONE_LABELS = {
    (0, 0): "画面左上",
    (0, 1): "画面上方",
    (0, 2): "画面右上",
    (1, 0): "画面左侧",
    (1, 1): "画面中央",
    (1, 2): "画面右侧",
    (2, 0): "画面左下",
    (2, 1): "画面下方",
    (2, 2): "画面右下",
}


class CaptureStatus(Enum):
    PENDING = auto()
    DONE = auto()
    SKIPPED = auto()


@dataclass
class CaptureTarget:
    """One planned capture: a normalised position + desired tilt."""
    zone_row: int          # 0-2  (3×3 grid)
    zone_col: int          # 0-2
    tilt: BoardTilt
    status: CaptureStatus = CaptureStatus.PENDING
    scale: str = "medium"
    target_center: tuple[float, float] | None = None
    target_area: float | None = None  # Candidate reference only; distance band remains the gate.

    @property
    def norm_cx(self) -> float:
        """Normalised x-centre in [0,1]."""
        return self.target_center[0] if self.target_center is not None else (self.zone_col + 0.5) / 3.0

    @property
    def norm_cy(self) -> float:
        return self.target_center[1] if self.target_center is not None else (self.zone_row + 0.5) / 3.0

    @property
    def label(self) -> str:
        zone = STEREO_ZONE_LABELS.get((self.zone_row, self.zone_col), "?")
        return f"{zone}，{TILT_LABELS[self.tilt]}"


# Bootstrap plan: centre, cross, corners, then varied tilts.
_STEREO_NINE_FRONTAL: Tuple[Tuple[int, int], ...] = (
    (1, 1),
    (1, 0),
    (1, 2),
    (0, 1),
    (2, 1),
    (0, 0),
    (0, 2),
    (2, 0),
    (2, 2),
)


def build_default_sequence() -> List[CaptureTarget]:
    """Heuristic bootstrap plan, not measured pose coverage or optimal design."""
    seq: List[CaptureTarget] = []

    for r, c in _STEREO_NINE_FRONTAL:
        seq.append(CaptureTarget(r, c, BoardTilt.FRONT))

    for tilt in (
        BoardTilt.TILT_LEFT,
        BoardTilt.TILT_RIGHT,
        BoardTilt.TILT_UP,
        BoardTilt.TILT_DOWN,
    ):
        seq.append(CaptureTarget(1, 1, tilt))

    seq.append(CaptureTarget(0, 0, BoardTilt.TILT_RIGHT))
    seq.append(CaptureTarget(0, 2, BoardTilt.TILT_LEFT))
    seq.append(CaptureTarget(2, 0, BoardTilt.TILT_RIGHT))
    seq.append(CaptureTarget(2, 2, BoardTilt.TILT_LEFT))
    seq.append(CaptureTarget(0, 1, BoardTilt.TILT_DOWN))

    return seq


def build_multimodal_sequence() -> List[CaptureTarget]:
    """Independent planar plan: central distances/tilts, then grouped image heights."""
    from ...calibration.aux_guidance import initial_plan

    tilts = {'front': BoardTilt.FRONT, 'left': BoardTilt.TILT_LEFT,
             'right': BoardTilt.TILT_RIGHT, 'up': BoardTilt.TILT_UP, 'down': BoardTilt.TILT_DOWN}
    return [CaptureTarget(t.row, t.col, tilts[t.tilt], scale=t.scale) for t in initial_plan()]


# ═══════════════════════════════════════════════════════════════
#  Capture sequence state
# ═══════════════════════════════════════════════════════════════


class CaptureSequence:
    """Holds the ordered list of targets and tracks progress.

    The active step for overlays is ``_index`` (see ``current_index``), independent
    of per-step DONE/SKIPPED flags, so the user can jump to any step to align
    the live guide.
    """

    def __init__(self, targets: Optional[List[CaptureTarget]] = None):
        self.targets = targets or build_default_sequence()
        self._index = 0

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def current(self) -> Optional[CaptureTarget]:
        n = len(self.targets)
        if 0 <= self._index < n:
            return self.targets[self._index]
        return None

    @property
    def total(self) -> int:
        return len(self.targets)

    @property
    def done_count(self) -> int:
        return sum(1 for t in self.targets if t.status == CaptureStatus.DONE)

    @property
    def finished(self) -> bool:
        n = len(self.targets)
        if n == 0:
            return True
        return self._index >= n

    def jump_to(self, step_index: int) -> None:
        """Focus a step (0..N-1) for the guide. Does not change DONE/SKIPPED."""
        if not self.targets:
            return
        n = len(self.targets)
        self._index = max(0, min(int(step_index), n - 1))

    def go_to_next(self) -> None:
        """Move focus forward; at last step moves to the \"all done\" state (index==N)."""
        n = len(self.targets)
        if n == 0:
            return
        if self._index < n:
            self._index += 1

    def go_to_prev(self) -> None:
        """Move focus back; from \"all done\" (index==N) returns to last step."""
        n = len(self.targets)
        if n == 0:
            return
        if self._index > 0:
            self._index -= 1

    def advance(self):
        n = len(self.targets)
        if self._index < n:
            self.targets[self._index].status = CaptureStatus.DONE
        self._index += 1

    def rewind_last_advance(self) -> bool:
        """撤销最近一次 ``advance()``：恢复上一引导步为未完成并回退一步索引。"""
        if self._index <= 0:
            return False
        self._index -= 1
        if self._index < len(self.targets):
            self.targets[self._index].status = CaptureStatus.PENDING
        return True

    def skip_current(self) -> None:
        n = len(self.targets)
        if self._index >= n:
            return
        t = self.targets[self._index]
        if t.status == CaptureStatus.PENDING:
            t.status = CaptureStatus.SKIPPED
        self._index += 1

    def reset(self):
        self._index = 0
        for t in self.targets:
            t.status = CaptureStatus.PENDING


# ═══════════════════════════════════════════════════════════════
#  Camera-overlay rendering (OpenCV, drawn onto each frame)
# ═══════════════════════════════════════════════════════════════

_ZONE_BORDER_PAD = 0.03  # 3% padding from edges

# 倾角在示意中的口语说明（相对镜头/画面）
TILT_POSE_HINT: dict[BoardTilt, str] = {**TILT_LABELS, BoardTilt.FRONT: "正对镜头，平行像面"}


def board_corners_f(
    cx: float, cy: float, half_w: float, half_h: float, tilt: BoardTilt
) -> np.ndarray:
    """Lens-view projection of the same motion used by the animation."""
    points = np.array([tilt_point(x, y, 1., tilt) for x, y in
                       ((-1.45, -1), (1.45, -1), (1.45, 1), (-1.45, 1))])
    projected = points[:, :2] / points[:, 2:3]
    # Fit the slot without reversing perspective: the nearer edge stays larger.
    projected /= np.maximum(np.max(np.abs(projected), axis=0), 1e-9)
    return projected * (half_w, half_h) + (cx, cy)


def paint_board_pose_schematic(
    p: QPainter, rect: QRect, tilt: BoardTilt, *, with_reference: bool = True
) -> None:
    """右侧「标定板摆放」示意：板面 + 光轴 + 相机，全部在 rect 内。"""
    p.save()
    m = 6
    inner = rect.adjusted(m, m, -m, -m)
    if inner.width() < 40 or inner.height() < 50:
        p.restore()
        return

    ix, iy = float(inner.x()), float(inner.y())
    iw, ih = float(inner.width()), float(inner.height())
    cx = ix + iw / 2.0

    # vertical layout: hint(16%) → board(50%) → axis gap → camera+label(18%)
    hint_h = ih * 0.14
    cam_zone_h = ih * 0.18
    board_zone_top = iy + hint_h
    board_zone_bot = iy + ih - cam_zone_h
    board_zone_h = board_zone_bot - board_zone_top

    # pose hint text
    hint = TILT_POSE_HINT.get(tilt, "")
    if hint:
        fh = QFont()
        fh.setPixelSize(10)
        p.setFont(fh)
        p.setPen(QColor("#b0c8e0"))
        p.drawText(
            QRectF(ix, iy, iw, hint_h),
            int(Qt.AlignHCenter | Qt.AlignVCenter) | int(Qt.TextWordWrap), hint,
        )

    # optical axis (dashed)
    p.setPen(QPen(QColor(120, 180, 255, 130), 1, Qt.PenStyle.DashLine))
    p.drawLine(int(cx), int(board_zone_bot + 2), int(cx), int(board_zone_top + 4))

    # camera body
    cam_y = min(iy + ih - cam_zone_h * 0.65, iy + ih - 24)
    cam_w = min(44.0, iw * 0.4)
    p.setPen(QPen(QColor("#6ecff6"), 1.5))
    p.setBrush(QBrush(QColor("#1a3a52")))
    p.drawRoundedRect(int(cx - cam_w / 2), int(cam_y), int(cam_w), 10, 3, 3)
    p.setBrush(QBrush(QColor("#0d2137")))
    p.drawEllipse(int(cx - 4), int(cam_y + 2), 8, 7)
    f_cam = QFont()
    f_cam.setPixelSize(8)
    p.setFont(f_cam)
    p.setPen(QColor("#5ca8c8"))
    p.drawText(
        QRect(int(cx - 28), int(cam_y + 12), 56, 10),
        Qt.AlignCenter, "相机视向 ↑",
    )

    # board in the middle zone
    sc = min(iw, board_zone_h) * 0.28
    bcx = cx
    bcy = board_zone_top + board_zone_h * 0.45
    hw, hh = sc * 1.2, sc * 0.95
    if with_reference and tilt != BoardTilt.FRONT:
        ref = board_corners_f(bcx, bcy, hw * 0.85, hh * 0.85, BoardTilt.FRONT)
        poly_ref = QPolygonF([QPointF(x, y) for x, y in ref])
        p.setPen(QPen(QColor(100, 100, 110, 90), 1.5, Qt.PenStyle.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(poly_ref)
    corners = board_corners_f(bcx, bcy, hw, hh, tilt)
    poly = QPolygonF([QPointF(x, y) for x, y in corners])
    p.setPen(QPen(QColor(0, 255, 200), 2.5))
    p.setBrush(QBrush(QColor(0, 220, 180, 65)))
    p.drawPolygon(poly)
    p.setPen(QPen(QColor(255, 230, 120), 1))
    for t_val in (0.33, 0.66):
        a = corners[0] * (1 - t_val) + corners[1] * t_val
        b = corners[3] * (1 - t_val) + corners[2] * t_val
        p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
    a = corners[0] * 0.5 + corners[3] * 0.5
    b = corners[1] * 0.5 + corners[2] * 0.5
    p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
    p.restore()


def _draw_opencv_pose_inset(vis: np.ndarray, tilt: BoardTilt, *, compact: bool = False) -> None:
    """画面右下小窗：大比例板面 + 与参考姿态对比，避免遮挡顶步号与角点 OSD。"""
    h, w = vis.shape[:2]
    if compact or w < 480 or h < 320:
        return
    ih = min(150, max(100, h // 5))
    iw = min(210, max(150, w // 4))
    x0 = w - iw - 8
    y0 = h - ih - 44
    if y0 < 0 or x0 < 0:
        return
    roi = vis[y0 : y0 + ih, x0 : x0 + iw]
    overlay = roi.copy()
    overlay[:] = (22, 32, 48)
    icx, icy = float(iw) * 0.5, float(ih) * 0.42
    sc = min(iw, ih) * 0.2
    hw, hh = sc * 1.1, sc * 0.88
    if tilt != BoardTilt.FRONT:
        ref = board_corners_f(icx, icy, hw * 0.9, hh * 0.9, BoardTilt.FRONT).astype(
            np.int32
        )
        cv2.polylines(overlay, [ref], True, (90, 90, 100), 1, cv2.LINE_AA)
    pts = board_corners_f(icx, icy, hw, hh, tilt).astype(np.int32)
    cv2.fillPoly(overlay, [pts], (35, 95, 85))
    cv2.polylines(overlay, [pts], True, (0, 255, 200), 2, cv2.LINE_AA)
    for t in (0.33, 0.66):
        a = (pts[0] * (1 - t) + pts[1] * t).astype(int)
        b = (pts[3] * (1 - t) + pts[2] * t).astype(int)
        cv2.line(overlay, tuple(a), tuple(b), (255, 200, 80), 1, cv2.LINE_AA)
    t = 0.5
    a = (pts[0] * (1 - t) + pts[3] * t).astype(int)
    b = (pts[1] * (1 - t) + pts[2] * t).astype(int)
    cv2.line(overlay, tuple(a), tuple(b), (255, 200, 80), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.88, roi, 0.12, 0, roi)
    with TextPainter(vis) as tp:
        tp.text(x0 + 4, y0 + 16, "摆放示意", scale=0.4, color=(0, 255, 200))
        tp.text(x0 + iw // 2, y0 + ih - 8, TILT_LABELS[tilt], scale=0.38,
                color=(0, 255, 200), anchor_center=True)


def draw_target_overlay(
    frame: np.ndarray,
    target: CaptureTarget,
    step_num: int,
    total_steps: int,
    *,
    compact: bool = False,
) -> np.ndarray:
    """Draw the target zone rectangle + tilt arrow + step label on frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # zone rectangle
    pad_x = int(w * _ZONE_BORDER_PAD)
    pad_y = int(h * _ZONE_BORDER_PAD)
    zone_w = (w - 2 * pad_x) // 3
    zone_h = (h - 2 * pad_y) // 3
    x1 = pad_x + target.zone_col * zone_w
    y1 = pad_y + target.zone_row * zone_h
    x2 = x1 + zone_w
    y2 = y1 + zone_h

    overlay = vis.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 200), -1)
    cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 200), 2)

    # centre of zone
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    if target.target_center is not None:
        cx, cy = int(target.norm_cx * w), int(target.norm_cy * h)

    _draw_tilt_indicator(vis, cx, cy, zone_w, zone_h, target.tilt)
    _draw_opencv_pose_inset(vis, target.tilt, compact=compact)

    if compact:
        return vis

    step_text = f"[{step_num}/{total_steps}]"
    with TextPainter(vis) as tp:
        tp.text(w - 8, 26, step_text, scale=0.55,
                color=(0, 255, 200), anchor_right=True)
        tp.text(cx, y2 - 8, target.label, scale=0.45,
                color=(255, 255, 255), anchor_center=True)

    return vis


def _draw_tilt_indicator(
    img: np.ndarray,
    cx: int, cy: int,
    zone_w: int, zone_h: int,
    tilt: BoardTilt,
):
    """画取景区中心的大比例板面示意（与摆放小窗同几何，略放大）。"""
    half_w = max(zone_w // 4, 10)
    half_h = max(zone_h // 4, 8)
    ptsf = board_corners_f(float(cx), float(cy), float(half_w), float(half_h), tilt)
    pts = ptsf.astype(np.int32)
    if tilt != BoardTilt.FRONT:
        ref = board_corners_f(
            float(cx), float(cy), float(half_w * 0.85), float(half_h * 0.85), BoardTilt.FRONT
        ).astype(np.int32)
        cv2.polylines(img, [ref], True, (100, 100, 120), 1, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (255, 255, 0), 2, cv2.LINE_AA)

    # draw a small grid inside to represent chessboard pattern
    _draw_board_grid(img, pts, 3, 2)


def _draw_board_grid(img: np.ndarray, quad: np.ndarray, cols: int, rows: int):
    """Draw internal grid lines inside a quadrilateral."""
    color = (200, 200, 100)
    for i in range(1, cols):
        t = i / cols
        p_top = (quad[0] * (1 - t) + quad[1] * t).astype(int)
        p_bot = (quad[3] * (1 - t) + quad[2] * t).astype(int)
        cv2.line(img, tuple(p_top), tuple(p_bot), color, 1, cv2.LINE_AA)
    for j in range(1, rows):
        t = j / rows
        p_left = (quad[0] * (1 - t) + quad[3] * t).astype(int)
        p_right = (quad[1] * (1 - t) + quad[2] * t).astype(int)
        cv2.line(img, tuple(p_left), tuple(p_right), color, 1, cv2.LINE_AA)


# _put_text → draw_text_baseline_bgr (supports CJK; cv2.putText does not)


# ═══════════════════════════════════════════════════════════════
#  Qt Widget — visual sequence panel (sidebar)
# ═══════════════════════════════════════════════════════════════

_ROW_LABELS = ("上", "中", "下")
_COL_LABELS = ("左", "中", "右")


def paint_board_tilt_symbol(
    p: QPainter,
    cx: float,
    cy: float,
    r: float,
    tilt: BoardTilt,
    *,
    line_scale: float = 1.0,
) -> None:
    """Draw a perspective board icon showing tilt direction."""
    lw = max(1.5, 1.8 * line_scale)
    hw, hh = r * 0.9, r * 0.65
    corners = board_corners_f(cx, cy, hw, hh, tilt)
    poly = QPolygonF([QPointF(x, y) for x, y in corners])
    fill_alpha = int(60 * line_scale)
    p.setPen(QPen(QColor(255, 220, 100), lw))
    p.setBrush(QBrush(QColor(255, 220, 100, min(fill_alpha, 120))))
    p.drawPolygon(poly)
    # cross-hair lines on the board surface
    p.setPen(QPen(QColor(255, 240, 180, int(100 * min(line_scale, 1.5))), max(1.0, lw * 0.5)))
    mid_t = corners[0] * 0.5 + corners[1] * 0.5
    mid_b = corners[3] * 0.5 + corners[2] * 0.5
    p.drawLine(int(mid_t[0]), int(mid_t[1]), int(mid_b[0]), int(mid_b[1]))
    mid_l = corners[0] * 0.5 + corners[3] * 0.5
    mid_r = corners[1] * 0.5 + corners[2] * 0.5
    p.drawLine(int(mid_l[0]), int(mid_l[1]), int(mid_r[0]), int(mid_r[1]))


# (title, slice start, slice end exclusive) — matches build_default_sequence()
class _StepDots(QWidget):
    """Compact dot indicator bar: one dot per step, clickable."""

    dot_clicked = Signal(int)
    _DOT_R = 5
    _DOT_GAP = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._targets: List[CaptureTarget] = []
        self._active_index = 0
        self.setFixedHeight(self._DOT_R * 2 + 8)
        self.setCursor(Qt.PointingHandCursor)

    def set_data(self, targets: List[CaptureTarget], active: int) -> None:
        self._targets = targets
        self._active_index = active
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._targets:
            x = event.position().x()
            total_w = len(self._targets) * (self._DOT_R * 2 + self._DOT_GAP) - self._DOT_GAP
            x0 = (self.width() - total_w) / 2.0
            idx = int((x - x0 + self._DOT_GAP / 2) / (self._DOT_R * 2 + self._DOT_GAP))
            idx = max(0, min(idx, len(self._targets) - 1))
            self.dot_clicked.emit(idx)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        if not self._targets:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        n = len(self._targets)
        r = self._DOT_R
        gap = self._DOT_GAP
        total_w = n * (r * 2 + gap) - gap
        x0 = (self.width() - total_w) / 2.0
        cy = self.height() / 2.0
        for i, t in enumerate(self._targets):
            cx = x0 + i * (r * 2 + gap) + r
            is_cur = i == self._active_index
            if t.status == CaptureStatus.DONE:
                p.setBrush(QBrush(QColor("#52b788")))
                p.setPen(Qt.NoPen)
            elif t.status == CaptureStatus.SKIPPED:
                p.setBrush(QBrush(QColor("#4a4a5a")))
                p.setPen(Qt.NoPen)
            elif is_cur:
                p.setBrush(QBrush(QColor("#00b4d8")))
                p.setPen(QPen(QColor("#90e0ef"), 1.5))
            else:
                p.setBrush(QBrush(QColor("#2a3a4a")))
                p.setPen(Qt.NoPen)
            draw_r = r + 2 if is_cur else r
            p.drawEllipse(QPointF(cx, cy), draw_r, draw_r)
        p.end()


class _LargeGuideDiagram(QWidget):
    """Prominent FOV + zone + tilt illustration (mature calib apps style)."""

    zone_selected = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target: Optional[CaptureTarget] = None
        self._step_index = 0
        self._total = 0
        self._finished = False
        self._targets = []
        self.completion_message = None
        self.show_distance = False
        self.recommendations_only = False
        self.setMouseTracking(True)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.setStyleSheet(f"background:{BG_DARK};")

    def set_guide(
        self,
        target: Optional[CaptureTarget],
        step_index: int,
        total: int,
        finished: bool,
    ) -> None:
        self._target = target
        self._step_index = step_index
        self._total = max(total, 1)
        self._finished = finished
        self.update()

    @property
    def completion_title(self):
        if self.completion_message:
            return self.completion_message[0]
        done = sum(t.status == CaptureStatus.DONE for t in self._targets)
        return '采图计划已完成' if done == self._total else '引导已浏览完'

    def grid_rect(self):
        pose_width = max(100., (self.width() - 8) * .38)
        width = max(80., self.width() - pose_width - 42)
        available = max(1., self.height() - 56)
        height = min(width, available - 4)
        return QRectF(30., 46. + (available - height) / 2., width, height)

    def zone_at(self, point):
        rect = self.grid_rect()
        if self._target is None or not rect.contains(point):
            return None
        return min(2, int((point.y() - rect.y()) / (rect.height() / 3))), min(
            2, int((point.x() - rect.x()) / (rect.width() / 3)))

    def mousePressEvent(self, event):
        zone = self.zone_at(event.position())
        if event.button() == Qt.LeftButton and zone is not None:
            self.zone_selected.emit(*zone)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        zone = self.zone_at(event.position())
        self.setCursor(Qt.PointingHandCursor if zone else Qt.ArrowCursor)
        if zone is not None:
            targets = [t for t in self._targets if (t.zone_row, t.zone_col) == zone]
            done = sum(t.status == CaptureStatus.DONE for t in targets)
            self.setToolTip(
                f'{STEREO_ZONE_LABELS[zone]} · 参考动作 {len(targets)} 项\n浏览不增加采集记录'
                if self.recommendations_only else
                f'{STEREO_ZONE_LABELS[zone]} · 已采 {done}/{len(targets)}\n'
                '选择该区域待采姿态；距离为计划提示，并非实时测距')
        else:
            self.setToolTip(self.accessibleName() + '\n记录状态不代表实际姿态已自动验证')
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(BG_DARK))

        if self._finished and self._target is None:
            p.setPen(QColor("#52b788"))
            f_done = QFont()
            f_done.setPixelSize(18)
            f_done.setBold(True)
            p.setFont(f_done)
            done = sum(t.status == CaptureStatus.DONE for t in self._targets)
            skipped = sum(t.status == CaptureStatus.SKIPPED for t in self._targets)
            p.drawText(QRect(0, h // 2 - 30, w, 30), Qt.AlignCenter, self.completion_title)
            f_sub = QFont()
            f_sub.setPixelSize(11)
            p.setFont(f_sub)
            p.setPen(QColor(TEXT_DIM))
            p.drawText(
                QRect(0, h // 2 + 4, w, 36), Qt.AlignCenter,
                self.completion_message[1] if self.completion_message else
                f"已采 {done} · 跳过 {skipped} · 可继续补拍",
            )
            return

        if self._target is None:
            p.setPen(QColor(TEXT_HINT))
            p.drawText(QRect(0, h // 2 - 10, w, 24), Qt.AlignCenter, "无引导数据")
            return

        t = self._target
        step_text = f"{self._step_index + 1}/{self._total}"
        label_text = t.label
        if self.show_distance:
            label_text = {'small': '远', 'medium': '中', 'large': '近'}[t.scale] + ' · ' + label_text
        status, status_color = {
            CaptureStatus.PENDING: ('待采', ACCENT), CaptureStatus.DONE: ('已采', SUCCESS),
            CaptureStatus.SKIPPED: ('已跳过', WARNING),
        }[t.status]
        if self.recommendations_only and t.status == CaptureStatus.PENDING:
            status = '参考'

        # ── header: badge | label | hint ──
        hdr_h = 28
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(BG_MID)))
        p.drawRoundedRect(0, 0, w, hdr_h, 8, 8)

        from PySide6.QtGui import QFontMetrics

        # step badge — auto width
        f_badge = QFont()
        f_badge.setPixelSize(11)
        f_badge.setBold(True)
        p.setFont(f_badge)
        badge_w = QFontMetrics(f_badge).horizontalAdvance(step_text) + 14
        p.setBrush(QBrush(QColor(PRIMARY)))
        p.drawRoundedRect(6, 4, badge_w, hdr_h - 8, 4, 4)
        p.setPen(QColor("#ffffff"))
        p.drawText(QRect(6, 4, badge_w, hdr_h - 8), Qt.AlignCenter, step_text)

        # label — measured position
        f_lbl = QFont()
        f_lbl.setPixelSize(13)
        f_lbl.setBold(True)
        p.setFont(f_lbl)
        p.setPen(QColor(TEXT))
        lbl_x = 6 + badge_w + 8
        status_width = 46
        label_text = QFontMetrics(f_lbl).elidedText(label_text, Qt.ElideRight, max(1, w - lbl_x - status_width - 8))
        lbl_w = QFontMetrics(f_lbl).horizontalAdvance(label_text)
        p.drawText(QRect(lbl_x, 0, lbl_w + 4, hdr_h), Qt.AlignVCenter, label_text)

        # Keep record status visible even when the title is elided.
        f_hint = QFont()
        f_hint.setPixelSize(10)
        p.setFont(f_hint)
        p.setPen(QColor(status_color))
        p.drawText(QRect(w - status_width - 8, 0, status_width, hdr_h),
                   Qt.AlignRight | Qt.AlignVCenter, status)

        # ── content area: left = FOV grid, right = board pose ──
        top_y = hdr_h + 18
        avail_h = float(h - top_y - 10)
        avail_w = float(w - 8)

        # right pose panel: 38% width
        po_w = max(100.0, avail_w * 0.38)
        pose_x = int(w - po_w - 4)
        pose_rect = QRect(pose_x, top_y, int(po_w), int(avail_h))
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawLine(pose_x - 2, top_y, pose_x - 2, h - 10)
        p.setBrush(Qt.NoBrush)
        paint_board_pose_schematic(p, pose_rect, t.tilt, with_reference=True)

        # left FOV grid
        grid = self.grid_rect()
        gx0, gy0, gw, gh = grid.x(), grid.y(), grid.width(), grid.height()
        cw, ch = gw / 3.0, gh / 3.0

        # row/col labels
        f_ax = QFont()
        f_ax.setPixelSize(10)
        p.setFont(f_ax)
        p.setPen(QColor(TEXT_HINT))
        for j, name in enumerate(_COL_LABELS):
            cx = gx0 + j * cw + cw / 2
            p.drawText(
                QRect(int(cx - 16), int(gy0 - 14), 32, 14),
                Qt.AlignHCenter | Qt.AlignBottom, name,
            )
        for i, name in enumerate(_ROW_LABELS):
            cy = gy0 + i * ch + ch / 2
            p.drawText(
                QRect(2, int(cy - 8), 26, 16),
                Qt.AlignRight | Qt.AlignVCenter, name,
            )

        # grid frame
        p.setPen(QPen(QColor(BORDER), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(int(gx0) - 1, int(gy0) - 1, int(gw) + 2, int(gh) + 2, 4, 4)
        p.setPen(QPen(QColor(255, 255, 255, 45), 1))
        for k in range(1, 3):
            x = int(gx0 + k * cw)
            p.drawLine(x, int(gy0), x, int(gy0 + gh))
            y = int(gy0 + k * ch)
            p.drawLine(int(gx0), y, int(gx0 + gw), y)

        # Counts represent recorded guide steps, not verified pose coverage.
        for row in range(3):
            for col in range(3):
                done = sum(target.status == CaptureStatus.DONE for target in self._targets
                           if (target.zone_row, target.zone_col) == (row, col))
                if done:
                    p.setPen(QColor(SUCCESS))
                    p.drawText(QRect(int(gx0 + col * cw + 4), int(gy0 + row * ch + 2),
                                     int(cw - 8), 14), Qt.AlignRight, str(done))

        # target cell
        zx = int(gx0 + t.zone_col * cw)
        zy = int(gy0 + t.zone_row * ch)
        highlight = QColor(PRIMARY)
        highlight.setAlpha(100)
        p.setBrush(QBrush(highlight))
        p.setPen(QPen(QColor(ACCENT), 2.5))
        p.drawRect(zx + 1, zy + 1, int(cw) - 2, int(ch) - 2)
        zcx = gx0 + t.norm_cx * gw
        zcy = gy0 + t.norm_cy * gh
        paint_board_tilt_symbol(
            p, zcx, zcy, float(min(cw, ch) * (
                min(.55, .34 * np.sqrt(t.target_area / .07)) if t.target_area is not None else
                {'small': .23, 'medium': .34, 'large': .46}[t.scale]
                if self.show_distance else .34)), t.tilt, line_scale=2.0,
        )

        count = len(self._targets)
        if count:
            segment = (w - 2 * (count - 1)) / count
            for index, target in enumerate(self._targets):
                color = SUCCESS if target.status == CaptureStatus.DONE else WARNING if target.status == CaptureStatus.SKIPPED else BORDER
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(ACCENT if index == self._step_index else color))
                p.drawRoundedRect(QRectF(index * (segment + 2), h - 4, segment, 3), 1, 1)


class _ExpandedGuideDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.setWindowTitle('采图引导')
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(760, 520)
        self.setMinimumSize(620, 400)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        self.canvas = _LargeGuideDiagram()
        self.canvas.zone_selected.connect(owner._on_zone_selected)
        layout.addWidget(self.canvas, 1)
        navigation = QHBoxLayout()
        self.buttons = []
        for name, text, callback in (('chevron-left', '上一姿态', owner._on_prev),
                                     ('chevron-right', '下一姿态', owner._on_next),
                                     ('scan-line', '下个待采', owner._on_next_pending)):
            button = QPushButton(text)
            button.setIcon(app_icon(name))
            button.setMinimumHeight(36)
            button.clicked.connect(callback)
            navigation.addWidget(button)
            self.buttons.append(button)
        layout.addLayout(navigation)


class CaptureSequenceWidget(QFrame):
    """Visual capture sequence: current step summary + phased mini-map."""

    step_changed = Signal(int)
    """Reserved for programmatic step notifications."""
    user_navigated = Signal()
    """Emitted when the user changes the active guide step (click, buttons, keys)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet(f"CaptureSequenceWidget {{background:transparent; border-top:1px solid {BORDER};}}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(8)

        # ── header: title + progress ──
        header = QHBoxLayout()
        header.setSpacing(6)
        self._title_lbl = QLabel("采图引导")
        self._title_lbl.setStyleSheet(
            f"font-weight:bold; font-size:13px; color:{TEXT};"
        )
        header.addWidget(self._title_lbl)
        header.addStretch()
        self._progress_lbl = QLabel("0 / 0")
        self._progress_lbl.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        header.addWidget(self._progress_lbl)
        self._detail_dialog = None
        self._btn_expand = QToolButton()
        self._btn_expand.setIcon(app_icon('maximize-2'))
        self._btn_expand.setFixedSize(24, 24)
        self._btn_expand.setToolTip('展开采图引导')
        self._btn_expand.setAccessibleName('展开采图引导')
        self._btn_expand.clicked.connect(self._open_expanded_guide)
        header.addWidget(self._btn_expand)
        self._step_menu = QMenu(self)
        self._step_actions = []
        self._btn_steps = QToolButton()
        self._btn_steps.setIcon(app_icon('list-checks'))
        self._btn_steps.setFixedSize(24, 20)
        self._btn_steps.setStyleSheet('padding:0; border:0;')
        self._btn_steps.setToolTip('姿态清单：查看记录状态并跳转')
        self._btn_steps.setAccessibleName('选择采图姿态')
        self._btn_steps.setPopupMode(QToolButton.InstantPopup)
        self._btn_steps.setMenu(self._step_menu)
        header.addWidget(self._btn_steps)
        outer.addLayout(header)

        # ── primary: large diagram (contains step#, zone, pose, caption) ──
        self._large_guide = _LargeGuideDiagram()
        self._large_guide.zone_selected.connect(self._on_zone_selected)
        outer.addWidget(self._large_guide)

        # ── compact nav: ← / → / skip ──
        nav = QHBoxLayout()
        nav.setSpacing(4)
        self._btn_prev = QPushButton()
        self._btn_prev.setIcon(standard_icon(self, QStyle.SP_ArrowLeft))
        self._btn_prev.setToolTip("上一个采图姿态")
        self._btn_prev.setAccessibleName("上一个采图姿态")
        self._btn_next = QPushButton()
        self._btn_next.setIcon(standard_icon(self, QStyle.SP_ArrowRight))
        self._btn_next.setToolTip("查看下一姿态，不计为已采")
        self._btn_next.setAccessibleName("下一个采图姿态")
        self._btn_skip = QPushButton("跳过")
        self._btn_skip.setIcon(app_icon('skip-forward'))
        self._btn_skip.setToolTip('将当前姿态标记为跳过，不新增采图记录')
        for b in (self._btn_prev, self._btn_next, self._btn_skip):
            b.setFixedHeight(32)
            b.setStyleSheet("padding:2px 4px;")
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_skip.setMaximumWidth(76)
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next.clicked.connect(self._on_next)
        self._btn_skip.clicked.connect(self._on_skip)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._btn_next)
        nav.addWidget(self._btn_skip)
        outer.addLayout(nav)

        # ── step dot indicator ──
        self._dots = _StepDots()
        self._dots.dot_clicked.connect(self._on_dot_clicked)
        outer.addWidget(self._dots)

        self._sequence: Optional[CaptureSequence] = None
        self.recommendations_only = False

        self.setFocusPolicy(Qt.StrongFocus)
        sc_l = QShortcut(QKeySequence(Qt.Key_Left), self)
        sc_l.setContext(Qt.WidgetWithChildrenShortcut)
        sc_l.activated.connect(self._on_prev)
        sc_r = QShortcut(QKeySequence(Qt.Key_Right), self)
        sc_r.setContext(Qt.WidgetWithChildrenShortcut)
        sc_r.activated.connect(self._on_next)

    def set_compact_mode(self, compact: bool):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed if compact else QSizePolicy.Preferred)
        self.layout().setAlignment(Qt.AlignTop)
        self._large_guide.setMinimumHeight(200 if compact else 260)
        self._large_guide.setMaximumHeight(204 if compact else 16777215)
        self.layout().setContentsMargins(0 if compact else 8, 6,
                                        0 if compact else 8, 6)
        self._progress_lbl.setVisible(not compact)
        self._dots.setVisible(not compact)

    def set_diagram_height(self, height):
        self._large_guide.setMinimumHeight(height)
        self._large_guide.setMaximumHeight(height + 4)

    def set_completion_message(self, title, subtitle):
        self._large_guide.completion_message = (title, subtitle) if title else None
        if self._detail_dialog is not None:
            self._detail_dialog.canvas.completion_message = self._large_guide.completion_message

    def _open_expanded_guide(self):
        if self._detail_dialog is None:
            self._detail_dialog = _ExpandedGuideDialog(self)
            self._detail_dialog.finished.connect(self._on_detail_closed)
        self._update_display()
        self._detail_dialog.show()
        self._detail_dialog.raise_()
        self._detail_dialog.activateWindow()

    def _on_detail_closed(self):
        self._detail_dialog = None

    def hideEvent(self, event):
        if self._detail_dialog is not None:
            self._detail_dialog.close()
        super().hideEvent(event)

    def set_sequence(self, seq: CaptureSequence):
        self._sequence = seq
        self._step_menu.clear()
        self._step_actions = []
        for index in range(seq.total):
            action = QAction(self._step_menu)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, i=index: self._on_dot_clicked(i))
            self._step_menu.addAction(action)
            self._step_actions.append(action)
        self._step_menu.addSeparator()
        self._pending_action = self._step_menu.addAction('下个待采姿态')
        self._pending_action.setIcon(app_icon('scan-line'))
        self._pending_action.triggered.connect(self._on_next_pending)
        self._update_display()

    def _on_next_pending(self):
        seq = self._sequence
        if seq is None:
            return
        for offset in range(1, seq.total + 1):
            index = (seq.current_index + offset) % seq.total
            if seq.targets[index].status == CaptureStatus.PENDING:
                self._on_dot_clicked(index)
                return

    def _on_zone_selected(self, row, col):
        if self._sequence is None:
            return
        candidates = [i for i, target in enumerate(self._sequence.targets)
                      if (target.zone_row, target.zone_col) == (row, col)]
        if candidates:
            index = next((i for i in candidates if self._sequence.targets[i].status == CaptureStatus.PENDING),
                         candidates[0])
            self._on_dot_clicked(index)

    def refresh(self):
        if self._sequence is None:
            return
        self._update_display()

    def _on_dot_clicked(self, index: int) -> None:
        if self._sequence is None:
            return
        self.setFocus()
        self._sequence.jump_to(index)
        self._update_display()
        self.user_navigated.emit()

    def _on_prev(self) -> None:
        if self._sequence is None:
            return
        self._sequence.go_to_prev()
        self._update_display()
        self.user_navigated.emit()

    def _on_next(self) -> None:
        if self._sequence is None:
            return
        self._sequence.go_to_next()
        self._update_display()
        self.user_navigated.emit()

    def _on_skip(self) -> None:
        if self._sequence is None:
            return
        self._sequence.skip_current()
        self._update_display()
        self.user_navigated.emit()

    def _update_display(self):
        if self._sequence is None:
            return
        done = self._sequence.done_count
        total = self._sequence.total
        self._progress_lbl.setText(f"{done}/{total}")
        skipped = sum(t.status == CaptureStatus.SKIPPED for t in self._sequence.targets)
        self._title_lbl.setText(f'引导步骤 · 已记录 {done}/{total}')
        self._title_lbl.setToolTip('引导步骤记录独立于图像配对数量；导入图片不自动证明姿态覆盖。')
        summary = f'已采 {done} · 跳过 {skipped} · 待采 {total - done - skipped}'
        self._btn_steps.setToolTip(summary + '\n姿态清单：查看记录状态并跳转')
        self._large_guide._targets = self._sequence.targets
        self._large_guide.recommendations_only = self.recommendations_only
        if self.recommendations_only:
            idx = self._sequence.current_index
            self._title_lbl.setText('参考动作 · 非采集进度')
            self._title_lbl.setToolTip('图中编号是参考动作序号；原图已保存、验收通过和引导采用分别统计。')
            self._progress_lbl.setText(f'参考动作 {min(idx + 1, total)}/{total}')
            self._btn_steps.setToolTip('参考动作清单；浏览或跳过参考动作不会增加采集记录')
            self._btn_steps.setAccessibleName('浏览参考动作清单')
            self._btn_expand.setToolTip('展开参考动作示意')
            self._btn_expand.setAccessibleName('展开参考动作示意')
            self._pending_action.setText('下个参考动作')
            self._btn_prev.setToolTip('浏览上个参考动作，不增加采集记录')
            self._btn_prev.setAccessibleName('上个参考动作')
            self._btn_next.setToolTip('浏览下个参考动作，不增加采集记录')
            self._btn_next.setAccessibleName('下个参考动作')
            self._btn_skip.setToolTip('跳过当前参考动作，不增加采集记录')
        names = {CaptureStatus.DONE: ('已采', 'check', SUCCESS),
                 CaptureStatus.SKIPPED: ('跳过', 'circle-minus', WARNING),
                 CaptureStatus.PENDING: ('待采', 'scan-line', TEXT_DIM)}
        for index, (target, action) in enumerate(zip(self._sequence.targets, self._step_actions)):
            label, icon, color = names[target.status]
            if self.recommendations_only and target.status == CaptureStatus.PENDING:
                label = '参考'
            elif self.recommendations_only and target.status == CaptureStatus.SKIPPED:
                label = '已跳过参考'
            current = ' · 当前' if index == self._sequence.current_index else ''
            action.setText(f'{index + 1:02d}  {target.label}  · {label}{current}')
            action.setIcon(app_icon(icon, color))
            action.setChecked(index == self._sequence.current_index)
        self._pending_action.setEnabled(any(t.status == CaptureStatus.PENDING for t in self._sequence.targets))

        idx = self._sequence.current_index
        self._dots.set_data(self._sequence.targets, idx)

        cur = self._sequence.current
        accessible = (f'{idx + 1}/{total}，{cur.label}；{summary}' if cur is not None else
                      '；'.join(self._large_guide.completion_message)
                      if self._large_guide.completion_message else f'引导已浏览完；{summary}')
        if self.recommendations_only and cur is not None:
            accessible = f'参考动作 {idx + 1}/{total}，{cur.label}；浏览不增加采集记录'
        self._large_guide.setAccessibleName(accessible)
        self._large_guide.setToolTip(accessible + '\n记录状态不代表姿态覆盖已自动验证')
        if cur is not None:
            self._large_guide.set_guide(cur, idx, total, False)
        elif self._sequence.finished:
            self._large_guide.set_guide(None, 0, total, True)
        else:
            self._large_guide.set_guide(None, 0, total, False)

        n = total
        self._btn_prev.setEnabled(n > 0 and idx > 0)
        self._btn_next.setEnabled(n > 0 and idx < n)
        self._btn_skip.setEnabled(n > 0 and idx < n)
        if self._detail_dialog is not None:
            detail = self._detail_dialog
            detail.canvas.completion_message = self._large_guide.completion_message
            detail.canvas.show_distance = self._large_guide.show_distance
            detail.canvas.recommendations_only = self.recommendations_only
            detail.canvas._targets = self._sequence.targets
            detail.canvas.set_guide(cur, idx, total, self._sequence.finished)
            detail.canvas.setAccessibleName(accessible)
            detail.canvas.setToolTip(self._large_guide.toolTip())
            detail.buttons[0].setEnabled(self._btn_prev.isEnabled())
            detail.buttons[1].setEnabled(self._btn_next.isEnabled())
            detail.buttons[2].setEnabled(self._pending_action.isEnabled())
