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
    QPainter,
    QPen,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..opencv_unicode_text import TextPainter


# ═══════════════════════════════════════════════════════════════
#  Data model
# ═══════════════════════════════════════════════════════════════


class BoardTilt(Enum):
    FRONT = auto()
    TILT_LEFT = auto()
    TILT_RIGHT = auto()
    TILT_UP = auto()
    TILT_DOWN = auto()


TILT_LABELS = {
    BoardTilt.FRONT: "正面",
    BoardTilt.TILT_LEFT: "左倾",
    BoardTilt.TILT_RIGHT: "右倾",
    BoardTilt.TILT_UP: "上倾",
    BoardTilt.TILT_DOWN: "下倾",
}

# 与视场分块一致：行≈成像距离感（上远下近），列=横向，符合 OpenCV / 工业双目标定常见叙述
STEREO_ZONE_LABELS = {
    (0, 0): "远距-左",
    (0, 1): "远距-中",
    (0, 2): "远距-右",
    (1, 0): "中距-左",
    (1, 1): "中距-中",
    (1, 2): "中距-右",
    (2, 0): "近距-左",
    (2, 1): "近距-中",
    (2, 2): "近距-右",
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

    @property
    def norm_cx(self) -> float:
        """Normalised x-centre in [0,1]."""
        return (self.zone_col + 0.5) / 3.0

    @property
    def norm_cy(self) -> float:
        return (self.zone_row + 0.5) / 3.0

    @property
    def label(self) -> str:
        zone = STEREO_ZONE_LABELS.get((self.zone_row, self.zone_col), "?")
        return f"{zone}，{TILT_LABELS[self.tilt]}"


# 中距-中优先（共视大、易同步），再十字扩展，四角收尾 — 与 ZED/ROS/OC 文档里「先中后外」习惯一致
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
    """业界常见立体标定 ChArUco 采图序：多距×横向共视 + 中距倾角 + 边距大倾角（共 18 步）。"""
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
    """RGB_L + AUX 平面板采图序：覆盖视场 + 倾角，用于跨模态单目标定。"""
    seq: List[CaptureTarget] = []
    for r, c in (
        (1, 1),
        (1, 0),
        (1, 2),
        (0, 1),
        (2, 1),
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
    ):
        seq.append(CaptureTarget(r, c, BoardTilt.FRONT))

    for tilt in (
        BoardTilt.TILT_LEFT,
        BoardTilt.TILT_RIGHT,
        BoardTilt.TILT_UP,
        BoardTilt.TILT_DOWN,
    ):
        seq.append(CaptureTarget(1, 1, tilt))
    return seq


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
TILT_POSE_HINT: dict[BoardTilt, str] = {
    BoardTilt.FRONT: "正对镜头，平行像面",
    BoardTilt.TILT_LEFT: "向左侧倾",
    BoardTilt.TILT_RIGHT: "向右侧倾",
    BoardTilt.TILT_UP: "向上侧倾",
    BoardTilt.TILT_DOWN: "向下侧倾",
}


def board_corners_f(
    cx: float, cy: float, half_w: float, half_h: float, tilt: BoardTilt
) -> np.ndarray:
    """标定板四角 (4x2)，与旧版 OpenCV 示意几何一致，便于多尺度复用。"""
    if tilt == BoardTilt.FRONT:
        return np.array(
            [
                [cx - half_w, cy - half_h],
                [cx + half_w, cy - half_h],
                [cx + half_w, cy + half_h],
                [cx - half_w, cy + half_h],
            ],
            dtype=np.float64,
        )
    if tilt == BoardTilt.TILT_LEFT:
        return np.array(
            [
                [cx - half_w + half_w / 3, cy - half_h + half_h / 4],
                [cx + half_w, cy - half_h],
                [cx + half_w, cy + half_h],
                [cx - half_w + half_w / 3, cy + half_h - half_h / 4],
            ],
            dtype=np.float64,
        )
    if tilt == BoardTilt.TILT_RIGHT:
        return np.array(
            [
                [cx - half_w, cy - half_h],
                [cx + half_w - half_w / 3, cy - half_h + half_h / 4],
                [cx + half_w - half_w / 3, cy + half_h - half_h / 4],
                [cx - half_w, cy + half_h],
            ],
            dtype=np.float64,
        )
    if tilt == BoardTilt.TILT_UP:
        return np.array(
            [
                [cx - half_w + half_w / 4, cy - half_h + half_h / 3],
                [cx + half_w - half_w / 4, cy - half_h + half_h / 3],
                [cx + half_w, cy + half_h],
                [cx - half_w, cy + half_h],
            ],
            dtype=np.float64,
        )
    if tilt == BoardTilt.TILT_DOWN:
        return np.array(
            [
                [cx - half_w, cy - half_h],
                [cx + half_w, cy - half_h],
                [cx + half_w - half_w / 4, cy + half_h - half_h / 3],
                [cx - half_w + half_w / 4, cy + half_h - half_h / 3],
            ],
            dtype=np.float64,
        )
    return board_corners_f(cx, cy, half_w, half_h, BoardTilt.FRONT)


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
    cam_y = iy + ih - cam_zone_h * 0.65
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


def _draw_opencv_pose_inset(vis: np.ndarray, tilt: BoardTilt) -> None:
    """画面右下小窗：大比例板面 + 与参考姿态对比，避免遮挡顶步号与角点 OSD。"""
    h, w = vis.shape[:2]
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

    _draw_tilt_indicator(vis, cx, cy, zone_w, zone_h, target.tilt)
    _draw_opencv_pose_inset(vis, target.tilt)

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

_ROW_LABELS = ("远", "中", "近")
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target: Optional[CaptureTarget] = None
        self._step_index = 0
        self._total = 0
        self._finished = False
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.setStyleSheet("background: #0a1628; border-radius: 8px;")

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

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0a1525"))

        if self._finished and self._target is None:
            p.setPen(QColor("#52b788"))
            f_done = QFont()
            f_done.setPixelSize(18)
            f_done.setBold(True)
            p.setFont(f_done)
            p.drawText(QRect(0, h // 2 - 30, w, 30), Qt.AlignCenter, "采图已完成")
            f_sub = QFont()
            f_sub.setPixelSize(11)
            p.setFont(f_sub)
            p.setPen(QColor("#8da6c4"))
            p.drawText(
                QRect(0, h // 2 + 4, w, 36), Qt.AlignCenter,
                "可补拍或进入「双目标定」  ·  点 ← 回到任一步",
            )
            return

        if self._target is None:
            p.setPen(QColor("#657786"))
            p.drawText(QRect(0, h // 2 - 10, w, 24), Qt.AlignCenter, "无引导数据")
            return

        t = self._target
        step_text = f"{self._step_index + 1}/{self._total}"
        label_text = t.label
        if t.tilt == BoardTilt.FRONT:
            hint_text = "共视整板 · 板面平行像面 · 无运动糊"
        else:
            hint_text = "共视 · 侧倾 20°–40° · 角点可检"

        # ── header: badge | label | hint ──
        hdr_h = 28
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(20, 40, 65, 200)))
        p.drawRoundedRect(0, 0, w, hdr_h, 8, 8)

        from PySide6.QtGui import QFontMetrics

        # step badge — auto width
        f_badge = QFont()
        f_badge.setPixelSize(11)
        f_badge.setBold(True)
        p.setFont(f_badge)
        badge_w = QFontMetrics(f_badge).horizontalAdvance(step_text) + 14
        p.setBrush(QBrush(QColor("#0077b6")))
        p.drawRoundedRect(6, 4, badge_w, hdr_h - 8, 4, 4)
        p.setPen(QColor("#ffffff"))
        p.drawText(QRect(6, 4, badge_w, hdr_h - 8), Qt.AlignCenter, step_text)

        # label — measured position
        f_lbl = QFont()
        f_lbl.setPixelSize(13)
        f_lbl.setBold(True)
        p.setFont(f_lbl)
        p.setPen(QColor("#e0f0ff"))
        lbl_x = 6 + badge_w + 8
        lbl_w = QFontMetrics(f_lbl).horizontalAdvance(label_text)
        p.drawText(QRect(lbl_x, 0, lbl_w + 4, hdr_h), Qt.AlignVCenter, label_text)

        # hint — fill remaining space
        f_hint = QFont()
        f_hint.setPixelSize(10)
        p.setFont(f_hint)
        p.setPen(QColor("#6a90b0"))
        hint_x = lbl_x + lbl_w + 10
        p.drawText(QRect(hint_x, 0, w - hint_x - 4, hdr_h), Qt.AlignVCenter, hint_text)

        # ── content area: left = FOV grid, right = board pose ──
        top_y = hdr_h + 4
        avail_h = float(h - top_y - 4)
        avail_w = float(w - 8)

        # right pose panel: 38% width
        po_w = max(140.0, avail_w * 0.38)
        pose_x = int(w - po_w - 4)
        pose_rect = QRect(pose_x, top_y, int(po_w), int(avail_h))
        p.setPen(QPen(QColor("#263a50"), 1))
        p.setBrush(QBrush(QColor(8, 18, 32, 180)))
        p.drawRoundedRect(pose_rect, 6, 6)
        p.setBrush(Qt.NoBrush)
        paint_board_pose_schematic(p, pose_rect, t.tilt, with_reference=True)

        # left FOV grid
        grid_left = 30.0
        grid_right = float(pose_x - 8)
        gw = max(80.0, grid_right - grid_left)
        gh = min(gw, avail_h - 4)
        gx0 = grid_left
        gy0 = float(top_y) + (avail_h - gh) / 2.0
        cw, ch = gw / 3.0, gh / 3.0

        # row/col labels
        f_ax = QFont()
        f_ax.setPixelSize(10)
        p.setFont(f_ax)
        p.setPen(QColor("#5c7a9a"))
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
        p.setPen(QPen(QColor("#3d5a7a"), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(int(gx0) - 1, int(gy0) - 1, int(gw) + 2, int(gh) + 2, 4, 4)
        p.setPen(QPen(QColor(255, 255, 255, 45), 1))
        for k in range(1, 3):
            x = int(gx0 + k * cw)
            p.drawLine(x, int(gy0), x, int(gy0 + gh))
            y = int(gy0 + k * ch)
            p.drawLine(int(gx0), y, int(gx0 + gw), y)

        # target cell
        zx = int(gx0 + t.zone_col * cw)
        zy = int(gy0 + t.zone_row * ch)
        p.setBrush(QBrush(QColor(0, 200, 255, 90)))
        p.setPen(QPen(QColor(0, 255, 220), 2.5))
        p.drawRect(zx + 1, zy + 1, int(cw) - 2, int(ch) - 2)
        zcx = zx + int(cw) / 2.0
        zcy = zy + int(ch) / 2.0
        paint_board_tilt_symbol(
            p, zcx, zcy, float(min(cw, ch) * 0.34), t.tilt, line_scale=2.0,
        )


class CaptureSequenceWidget(QFrame):
    """Visual capture sequence: current step summary + phased mini-map."""

    step_changed = Signal(int)
    """Reserved for programmatic step notifications."""
    user_navigated = Signal()
    """Emitted when the user changes the active guide step (click, buttons, keys)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet(
            "CaptureSequenceWidget { background: #0d1b2a; border-radius: 8px; "
            "border: 1px solid #1b263b; }"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        # ── header: title + progress ──
        header = QHBoxLayout()
        header.setSpacing(6)
        self._title_lbl = QLabel("采图引导")
        self._title_lbl.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #e0e1dd;"
        )
        header.addWidget(self._title_lbl)
        header.addStretch()
        self._progress_lbl = QLabel("0 / 0")
        self._progress_lbl.setStyleSheet("font-size: 12px; color: #adb5bd;")
        header.addWidget(self._progress_lbl)
        outer.addLayout(header)

        # ── primary: large diagram (contains step#, zone, pose, caption) ──
        self._large_guide = _LargeGuideDiagram()
        outer.addWidget(self._large_guide)

        # ── compact nav: ← / → / skip ──
        nav = QHBoxLayout()
        nav.setSpacing(4)
        self._btn_prev = QPushButton("←")
        self._btn_next = QPushButton("→")
        self._btn_skip = QPushButton("跳过")
        for b in (self._btn_prev, self._btn_next, self._btn_skip):
            b.setFixedHeight(26)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_skip.setMaximumWidth(60)
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

        self.setFocusPolicy(Qt.StrongFocus)
        sc_l = QShortcut(QKeySequence(Qt.Key_Left), self)
        sc_l.setContext(Qt.WidgetWithChildrenShortcut)
        sc_l.activated.connect(self._on_prev)
        sc_r = QShortcut(QKeySequence(Qt.Key_Right), self)
        sc_r.setContext(Qt.WidgetWithChildrenShortcut)
        sc_r.activated.connect(self._on_next)

    def set_sequence(self, seq: CaptureSequence):
        self._sequence = seq
        self._update_display()

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

        idx = self._sequence.current_index
        self._dots.set_data(self._sequence.targets, idx)

        cur = self._sequence.current
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
