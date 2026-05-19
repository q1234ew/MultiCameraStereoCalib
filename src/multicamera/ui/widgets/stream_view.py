"""Multi-stream video preview with calibration board visual guidance."""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal, Slot, QRectF
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...board.charuco_board import CharucoBoard
from ...board.detector import CharucoDetector, DetectionResult
from ...perf import perf_timer
from ...streaming.stream_manager import StreamManager
from ..opencv_unicode_text import TextPainter
from ..theme import ACCENT, BG_CARD, BG_DARK, BORDER, TEXT_DIM, TEXT_HINT
from .capture_guide import CaptureSequence, CaptureTarget, draw_target_overlay


# ═══════════════════════════════════════════════════════════════
#  Coverage tracker — accumulates corner positions per camera
# ═══════════════════════════════════════════════════════════════

GRID_ROWS, GRID_COLS = 5, 5


class CoverageTracker:
    """Tracks corner spatial distribution over collected frames."""

    def __init__(self, image_size: Tuple[int, int]):
        self.w, self.h = image_size
        self.grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
        self.total_frames = 0

    def update(self, result: DetectionResult):
        if not result.valid:
            return
        self.total_frames += 1
        pts = result.charuco_corners.reshape(-1, 2)
        for px, py in pts:
            ci = min(int(px / self.w * GRID_COLS), GRID_COLS - 1)
            ri = min(int(py / self.h * GRID_ROWS), GRID_ROWS - 1)
            self.grid[ri, ci] += 1

    @property
    def coverage(self) -> float:
        return float((self.grid > 0).sum()) / self.grid.size

    @property
    def uniformity(self) -> float:
        if self.grid.sum() == 0:
            return 0.0
        normed = self.grid.astype(np.float64) / self.grid.sum()
        ideal = 1.0 / normed.size
        return max(0.0, 1.0 - np.std(normed) / ideal)


# ═══════════════════════════════════════════════════════════════
#  OSD overlay renderer
# ═══════════════════════════════════════════════════════════════

def draw_guide_overlay(
    frame: np.ndarray,
    result: DetectionResult,
    tracker: CoverageTracker,
    board_total_corners: int,
    capture_target: Optional[CaptureTarget] = None,
    step_num: int = 0,
    total_steps: int = 0,
) -> np.ndarray:
    """Draw coverage heatmap, target zone, status indicators and OSD."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # 1) Coverage heatmap overlay (semi-transparent)
    if tracker.total_frames > 0:
        heatmap = _render_heatmap(tracker.grid, w, h)
        vis = cv2.addWeighted(vis, 0.85, heatmap, 0.15, 0)

    # 2) Target zone overlay from capture sequence
    if capture_target is not None:
        vis = draw_target_overlay(vis, capture_target, step_num, total_steps)

    # 3) Detection status border
    if result.valid:
        ratio = result.num_corners / max(board_total_corners, 1)
        if ratio > 0.5:
            border_color = (0, 200, 0)
        else:
            border_color = (0, 200, 255)
    else:
        border_color = (0, 0, 200)

    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), border_color, 3)

    # 4) Top-left OSD + 5) Bottom guidance — batch in one conversion
    n = result.num_corners
    cov = tracker.coverage
    uni = tracker.uniformity
    osd = f"角点: {n}/{board_total_corners}  覆盖: {cov:.0%}  均匀: {uni:.0%}  帧: {tracker.total_frames}"

    if capture_target is not None:
        guide = f"双目共视目标位姿: {capture_target.label}"
    elif tracker.total_frames == 0:
        guide = "双目共视内放置标定板，与右栏任务序一致后采集"
    else:
        guide = f"覆盖率 {cov:.0%} | 均匀度 {uni:.0%}"

    with TextPainter(vis) as tp:
        tp.text(6, 22, osd, scale=0.45, color=(255, 255, 255), bg=(0, 0, 0))
        tp.text(6, h - 12, guide, scale=0.50, color=(0, 255, 200), bg=(0, 0, 0))

    return vis


def _render_heatmap(grid: np.ndarray, w: int, h: int) -> np.ndarray:
    norm = grid.astype(np.float32)
    if norm.max() > 0:
        norm = norm / norm.max()
    norm_u8 = (norm * 255).astype(np.uint8)
    small_color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    return cv2.resize(small_color, (w, h), interpolation=cv2.INTER_NEAREST)


def _scale_detection_result(
    result: DetectionResult,
    scale: float,
    image_size: tuple[int, int],
) -> DetectionResult:
    corners = None
    if result.charuco_corners is not None:
        corners = result.charuco_corners.copy()
        corners *= scale

    marker_corners = []
    for marker in result.marker_corners:
        scaled = marker.copy()
        scaled *= scale
        marker_corners.append(scaled)

    return DetectionResult(
        charuco_corners=corners,
        charuco_ids=result.charuco_ids,
        marker_corners=marker_corners,
        marker_ids=result.marker_ids,
        image_size=image_size,
    )



# ═══════════════════════════════════════════════════════════════
#  Camera tile widget
# ═══════════════════════════════════════════════════════════════


class _PreviewLabel(QLabel):
    """预览标签：不显式依赖 pixmap 尺寸作为 minimumSizeHint。

    默认 QLabel 会把 pixmap 尺寸当成最小尺寸，并排布局时左右列会因上一帧缩放结果不一致而宽度失衡。
    """

    def minimumSizeHint(self) -> QSize:
        return QSize(160, 120)

    def sizeHint(self) -> QSize:
        return QSize(320, 240)


class CameraView(QFrame):
    """Single camera preview tile with modern styling."""

    clicked = Signal(str)

    def __init__(self, camera_id: str, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._expanded = False
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("单击放大该路预览；再次单击或按 Esc 恢复双目并排")
        self.setStyleSheet(
            f"CameraView {{ background: {BG_CARD}; border: 1px solid {BORDER};"
            f" border-radius: 8px; }}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # top: camera id + status dot
        header = QLabel(f"● {camera_id}")
        header.setAlignment(Qt.AlignLeft)
        header.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {TEXT_DIM};"
            f" padding: 2px 4px; background: transparent;"
        )
        header.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._header = header
        layout.addWidget(header)

        self._last_frame: Optional[np.ndarray] = None
        self._last_label_size: Optional[tuple[int, int]] = None
        self._last_pixmap: Optional[QPixmap] = None

        self._image_label = _PreviewLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setMinimumSize(160, 120)
        self._image_label.setStyleSheet(
            f"background: {BG_DARK}; border-radius: 4px;"
        )
        self._image_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self._image_label)

        self._info = QLabel("等待连接...")
        self._info.setAlignment(Qt.AlignCenter)
        self._info.setStyleSheet(
            f"font-size: 11px; color: {TEXT_HINT}; padding: 2px;"
            f" background: transparent;"
        )
        self._info.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self._info)

    def set_expanded_mode(self, expanded: bool):
        """标题提示：放大模式下再次点击可恢复并排。"""
        self._expanded = expanded
        prefix = "● "
        suffix = "  ·  点击恢复并排预览" if expanded else ""
        self._header.setText(f"{prefix}{self.camera_id}{suffix}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.camera_id)
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_frame is not None:
            self._render_cached_frame()

    def update_frame(self, frame: np.ndarray):
        self._last_frame = frame
        self._last_label_size = None
        self._render_cached_frame()

    def _render_cached_frame(self):
        if self._last_frame is None:
            return
        frame = self._last_frame
        lw = max(1, self._image_label.width())
        lh = max(1, self._image_label.height())
        label_size = (lw, lh)
        if self._last_pixmap is not None and self._last_label_size == label_size:
            self._image_label.setPixmap(self._last_pixmap)
            return
        with perf_timer(f"preview render {self.camera_id}", threshold_ms=30.0):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            scaled = QPixmap.fromImage(qimg).scaled(
                QSize(lw, lh),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self._last_pixmap = scaled
        self._last_label_size = label_size
        self._image_label.setPixmap(scaled)

    def refresh_preview_if_cached(self) -> None:
        """布局完成后用缓存帧重绘（tile 尺寸已更新）。"""
        self._render_cached_frame()

    def update_info(self, text: str):
        self._info.setText(text)

    def set_connected(self, connected: bool):
        if connected:
            self._header.setStyleSheet(
                f"font-weight: bold; font-size: 12px; color: {ACCENT};"
                f" padding: 2px 4px; background: transparent;"
            )
            self.setStyleSheet(
                f"CameraView {{ background: {BG_CARD}; border: 1px solid {ACCENT};"
                f" border-radius: 8px; }}"
            )
        else:
            self._header.setStyleSheet(
                f"font-weight: bold; font-size: 12px; color: {TEXT_HINT};"
                f" padding: 2px 4px; background: transparent;"
            )
            self.setStyleSheet(
                f"CameraView {{ background: {BG_CARD}; border: 1px solid {BORDER};"
                f" border-radius: 8px; }}"
            )


# ═══════════════════════════════════════════════════════════════
#  Multi-stream view with guide overlay
# ═══════════════════════════════════════════════════════════════

class _EmptyState(QWidget):
    """Welcoming placeholder when no cameras are connected."""

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(BG_DARK))

        # camera icon placeholder
        icon_y = h // 2 - 60
        p.setPen(QPen(QColor(BORDER), 2))
        p.setBrush(QColor(BG_CARD))
        p.drawRoundedRect(w // 2 - 40, icon_y, 80, 56, 8, 8)
        p.drawEllipse(w // 2 - 10, icon_y + 14, 20, 20)

        # text
        p.setPen(QColor(TEXT_DIM))
        font = QFont()
        font.setPixelSize(16)
        p.setFont(font)
        p.drawText(QRectF(0, icon_y + 70, w, 30), Qt.AlignCenter,
                   "尚未连接相机")

        font.setPixelSize(13)
        p.setFont(font)
        p.setPen(QColor(TEXT_HINT))
        p.drawText(QRectF(0, icon_y + 100, w, 50), Qt.AlignCenter,
                   "请通过 ⚙ 相机配置 添加双目相机组\n然后点击 ▶ 连接相机")
        p.end()


class MultiStreamView(QWidget):
    """Dynamic grid of camera views with calibration board guide overlay."""

    UI_FPS_CAP = 15
    DETECT_FPS_CAP = 2.0
    DETECT_MAX_WIDTH = 960

    def __init__(
        self,
        stream_manager: StreamManager,
        board: CharucoBoard,
        parent=None,
    ):
        super().__init__(parent)
        self._stream_manager = stream_manager
        self._board = board
        self._detector = CharucoDetector(board)
        self._show_guide = True

        self._views: Dict[str, CameraView] = {}
        self._fps: Dict[str, float] = {}
        self._trackers: Dict[str, CoverageTracker] = {}
        self._capture_sequence: Optional[CaptureSequence] = None

        self._last_render_ts: Dict[str, float] = {}
        self._last_detect_ts: Dict[str, float] = {}
        self._last_detection: Dict[str, DetectionResult] = {}
        self._min_interval = 1.0 / self.UI_FPS_CAP
        self._detect_min_interval = 1.0 / self.DETECT_FPS_CAP
        self._expanded_id: Optional[str] = None

        self._layout = QGridLayout(self)
        self._layout.setSpacing(6)
        self._layout.setContentsMargins(6, 6, 6, 6)

        self._empty = _EmptyState()
        self._layout.addWidget(self._empty, 0, 0)

        self._stream_manager.frame_received.connect(self._on_frame)
        self._stream_manager.camera_connected.connect(self._on_connected)
        self._stream_manager.camera_disconnected.connect(self._on_disconnected)
        self._stream_manager.fps_updated.connect(self._on_fps)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def update_board(self, board: CharucoBoard):
        self._board = board
        self._detector = CharucoDetector(board)
        self._trackers.clear()
        self._last_detection.clear()
        self._last_detect_ts.clear()

    def set_capture_sequence(self, seq: Optional[CaptureSequence]):
        self._capture_sequence = seq

    def set_show_guide(self, show: bool):
        self._show_guide = show

    def set_show_detections(self, show: bool):
        self._show_guide = show

    def show_static_images(self, images: Dict[str, np.ndarray]):
        """Display a set of static images (e.g. from offline import)."""
        for camera_id, frame in images.items():
            view = self._get_or_create_view(camera_id)
            view.set_connected(True)
            h, w = frame.shape[:2]

            if camera_id not in self._trackers:
                self._trackers[camera_id] = CoverageTracker((w, h))

            result = self._detect_for_preview(frame)
            if self._show_guide:
                display = self._detector.draw_detected(frame, result)
                display = draw_guide_overlay(
                    display, result,
                    self._trackers[camera_id],
                    self._board.num_corners,
                )
            else:
                display = frame

            view.update_frame(display)
            view.update_info(f"{w}x{h} | 静态")

    def reset_trackers(self):
        self._trackers.clear()

    def notify_capture(self, camera_id: str, result: DetectionResult):
        tracker = self._trackers.get(camera_id)
        if tracker is not None:
            tracker.update(result)

    @Slot(str, np.ndarray, float)
    def _on_frame(self, camera_id: str, frame: np.ndarray, ts: float):
        now = time.monotonic()
        last = self._last_render_ts.get(camera_id, 0.0)
        if now - last < self._min_interval:
            return
        self._last_render_ts[camera_id] = now

        view = self._get_or_create_view(camera_id)
        h, w = frame.shape[:2]

        if camera_id not in self._trackers:
            self._trackers[camera_id] = CoverageTracker((w, h))

        last_detect = self._last_detect_ts.get(camera_id, 0.0)
        if now - last_detect >= self._detect_min_interval or camera_id not in self._last_detection:
            self._last_detect_ts[camera_id] = now
            result = self._detect_for_preview(frame)
            self._last_detection[camera_id] = result
        else:
            result = self._last_detection[camera_id]

        if self._show_guide:
            cur_target = None
            step_num = 0
            total_steps = 0
            if self._capture_sequence is not None and not self._capture_sequence.finished:
                cur_target = self._capture_sequence.current
                step_num = self._capture_sequence.current_index + 1
                total_steps = self._capture_sequence.total

            display = self._detector.draw_detected(frame, result)
            display = draw_guide_overlay(
                display, result,
                self._trackers[camera_id],
                self._board.num_corners,
                capture_target=cur_target,
                step_num=step_num,
                total_steps=total_steps,
            )
        else:
            display = frame

        view.update_frame(display)

        fps = self._fps.get(camera_id, 0)
        view.update_info(f"{w}x{h} | {fps:.1f} fps")

    def _detect_for_preview(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        scale = 1.0
        detect_frame = frame
        if w > self.DETECT_MAX_WIDTH:
            scale = self.DETECT_MAX_WIDTH / float(w)
            detect_frame = cv2.resize(
                frame,
                (self.DETECT_MAX_WIDTH, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        with perf_timer("preview charuco detect", threshold_ms=40.0):
            result = self._detector.detect(detect_frame)
        if scale == 1.0:
            return result
        return _scale_detection_result(result, 1.0 / scale, (w, h))

    @Slot(str)
    def _on_connected(self, camera_id: str):
        view = self._get_or_create_view(camera_id)
        view.set_connected(True)

    @Slot(str, str)
    def _on_disconnected(self, camera_id: str, reason: str):
        view = self._views.get(camera_id)
        if view:
            view.set_connected(False)
            view.update_info(f"断开: {reason}")

    @Slot(str, float)
    def _on_fps(self, camera_id: str, fps: float):
        self._fps[camera_id] = fps

    @Slot(str)
    def _on_camera_tile_clicked(self, camera_id: str):
        if self._expanded_id == camera_id:
            self._expanded_id = None
        else:
            self._expanded_id = camera_id
        self._relayout()
        self.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._expanded_id:
            self._expanded_id = None
            self._relayout()
            event.accept()
            return
        super().keyPressEvent(event)

    def _get_or_create_view(self, camera_id: str) -> CameraView:
        if camera_id not in self._views:
            view = CameraView(camera_id, self)
            view.clicked.connect(self._on_camera_tile_clicked)
            self._views[camera_id] = view
            self._relayout()
        return self._views[camera_id]

    def _relayout(self):
        for i in reversed(range(self._layout.count())):
            item = self._layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        n = len(self._views)
        if n == 0:
            self._expanded_id = None
            self._empty.setParent(self)
            self._layout.addWidget(self._empty, 0, 0)
            return

        self._empty.setParent(None)

        if self._expanded_id is not None and self._expanded_id not in self._views:
            self._expanded_id = None

        if self._expanded_id is not None:
            view = self._views[self._expanded_id]
            self._layout.addWidget(view, 0, 0)
            self._layout.setRowStretch(0, 1)
            self._layout.setColumnStretch(0, 1)
            for cid, v in self._views.items():
                v.set_expanded_mode(cid == self._expanded_id)
            return

        for v in self._views.values():
            v.set_expanded_mode(False)

        self._layout.setRowStretch(0, 0)
        self._layout.setColumnStretch(0, 0)

        cols = max(1, math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        for i, (cid, view) in enumerate(sorted(self._views.items())):
            r, c = divmod(i, cols)
            self._layout.addWidget(view, r, c)

        for c in range(cols):
            self._layout.setColumnStretch(c, 1)
        for r in range(rows):
            self._layout.setRowStretch(r, 1)

        QTimer.singleShot(0, self._refresh_tile_pixmaps_after_layout)

    def _refresh_tile_pixmaps_after_layout(self) -> None:
        for v in self._views.values():
            v.refresh_preview_if_cached()
