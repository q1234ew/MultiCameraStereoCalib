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
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...board.charuco_board import CharucoBoard
from ...board.detector import CharucoDetector, DetectionResult
from ...calibration.planar import PlanarDetectionResult, PlanarPatternDetector, PatternBoardConfig
from ...perf import perf_timer
from ...streaming.stream_manager import StreamManager
from ..opencv_unicode_text import TextPainter
from ..theme import BG_CARD, BG_DARK, BORDER, TEXT, TEXT_DIM, TEXT_HINT, SUCCESS, DANGER, standard_icon
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

    def update(self, result: DetectionResult | PlanarDetectionResult):
        if not result.valid:
            return
        self.total_frames += 1
        points = (result.image_points if isinstance(result, PlanarDetectionResult)
                  else result.charuco_corners)
        pts = points.reshape(-1, 2)
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
    compact_text: bool = False,
    guidance_label: str = "双目共视",
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
        vis = draw_target_overlay(
            vis,
            capture_target,
            step_num,
            total_steps,
            compact=compact_text,
        )

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

    if compact_text:
        return vis

    # 4) Top-left OSD + 5) Bottom guidance — batch in one conversion
    n = result.num_corners
    cov = tracker.coverage
    uni = tracker.uniformity
    osd = f"角点: {n}/{board_total_corners}  覆盖: {cov:.0%}  均匀: {uni:.0%}  帧: {tracker.total_frames}"

    if capture_target is not None:
        guide = f"{guidance_label}目标位姿: {capture_target.label}"
    elif tracker.total_frames == 0:
        guide = f"{guidance_label}内放置标定板，与右栏任务序一致后采集"
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


class _HeaderLabel(QLabel):
    def __init__(self, text=""):
        self._full_text = text
        super().__init__(text)
        self.setTextFormat(Qt.PlainText)

    def setText(self, text):
        self._full_text = text
        self.setToolTip(text)
        super().setText(self.fontMetrics().elidedText(text, Qt.ElideRight, max(1, self.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setText(self._full_text)


class CameraView(QFrame):
    """Single camera preview tile with modern styling."""

    clicked = Signal(str)
    close_requested = Signal(str)

    def __init__(self, camera_id: str, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.corner_count = None
        self._display_name = camera_id
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
        header = _HeaderLabel(camera_id)
        header.setAlignment(Qt.AlignLeft)
        header.setStyleSheet(
            f"font-weight: bold; font-size: 13px; color: {TEXT_DIM};"
            f" padding: 2px 4px; background: transparent;"
        )
        header.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._header = header
        header.setMinimumWidth(0)
        header.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self._status_dot = QLabel("●")
        self._status_dot.setFixedWidth(14)
        self._status_dot.setStyleSheet(f"color:{TEXT_HINT}; background:transparent;")
        header_row.addWidget(self._status_dot)
        header_row.addWidget(header, 1)
        self._close_button = QToolButton(self)
        self._close_button.setIcon(standard_icon(self, QStyle.SP_TitleBarCloseButton))
        self._close_button.setFixedSize(28, 28)
        self._close_button.setIconSize(QSize(18, 18))
        self._close_button.setStyleSheet("padding:0; border:1px solid transparent;")
        self._close_button.setAutoRaise(True)
        self._close_button.setToolTip("关闭此路预览（保持相机连接）")
        self._close_button.setAccessibleName("关闭相机预览")
        self._close_button.clicked.connect(self._request_close)
        header_row.addWidget(self._close_button)
        layout.addLayout(header_row)

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
        self._info_color = TEXT_DIM
        self._info.setTextFormat(Qt.PlainText)
        self._info.setMinimumHeight(24)
        self._info.setWordWrap(True)
        self._info.setAlignment(Qt.AlignCenter)
        self._info.setStyleSheet(
            f"font-size: 12px; color: {TEXT_DIM}; padding: 2px;"
            f" background: transparent;"
        )
        self._info.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self._info)

    def set_expanded_mode(self, expanded: bool):
        """标题提示：放大模式下再次点击可恢复并排。"""
        self._expanded = expanded
        suffix = " · 已放大" if expanded else ""
        self._header.setText(f"{self._display_name}{suffix}")

    @Slot()
    def _request_close(self):
        self.close_requested.emit(self.camera_id)

    def set_display_name(self, text: str):
        if self._display_name == text:
            return
        self._display_name = text
        self.set_expanded_mode(self._expanded)

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

    def clear_frame(self, info: str = "等待连接..."):
        """Drop cached image data so stale camera frames cannot remain visible."""
        self._last_frame = None
        self.corner_count = None
        self._last_label_size = None
        self._last_pixmap = None
        self._image_label.clear()
        self.update_info(info)

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
        color = DANGER if "无法连接" in text or "失败" in text else TEXT_DIM
        if color != self._info_color:
            self._info_color = color
            self._info.setStyleSheet(f"font-size:12px; color:{color}; padding:2px; background:transparent;")

    def set_connected(self, connected: bool):
        if connected:
            self._close_button.setToolTip("关闭此路预览（保持相机连接）")
            self._status_dot.setStyleSheet(f"color:{SUCCESS}; background:transparent;")
            self._header.setStyleSheet(
                f"font-weight: bold; font-size: 13px; color: {TEXT};"
                f" padding: 2px 4px; background: transparent;"
            )
            self.setStyleSheet(
                f"CameraView {{ background: {BG_CARD}; border: 1px solid {BORDER};"
                f" border-radius: 8px; }}"
            )
        else:
            self._status_dot.setStyleSheet(f"color:{TEXT_HINT}; background:transparent;")
            self._header.setStyleSheet(
                f"font-weight: bold; font-size: 13px; color: {TEXT_DIM};"
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

    previews_closed = False

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
                   "相机预览已关闭" if self.previews_closed else "尚未连接相机")

        font.setPixelSize(13)
        p.setFont(font)
        p.setPen(QColor(TEXT_HINT))
        p.drawText(QRectF(0, icon_y + 100, w, 50), Qt.AlignCenter,
                   "" if self.previews_closed else "请通过 ⚙ 相机配置 添加双目相机组\n然后点击 ▶ 连接相机")
        p.end()


class MultiStreamView(QWidget):
    """Dynamic grid of camera views with calibration board guide overlay."""

    preview_visibility_changed = Signal()
    presentation_changed = Signal(str)
    comparison_visible_changed = Signal(bool)
    review_changed = Signal(str, bool)
    preview_detected = Signal(str, object, object)

    UI_FPS_CAP = 15
    DETECT_FPS_CAP = 2.0
    DETECT_MAX_WIDTH = 960
    MAX_PREVIEW_FRAME_AGE_SEC = 0.35

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
        self._hidden_cameras: set[str] = set()
        self._fps: Dict[str, float] = {}
        self._trackers: Dict[str, CoverageTracker] = {}
        self._capture_sequence: Optional[CaptureSequence] = None

        self._last_render_ts: Dict[str, float] = {}
        self._last_detect_ts: Dict[str, float] = {}
        self._last_detection: Dict[str, DetectionResult] = {}
        self._min_interval = 1.0 / self.UI_FPS_CAP
        self._detect_min_interval = 1.0 / self.DETECT_FPS_CAP
        self._expanded_id: Optional[str] = None
        self._multimodal_mode = False
        self._rgb_left_id: Optional[str] = None
        self._planar_board = PatternBoardConfig.ir_chessboard()
        self._planar_detector = PlanarPatternDetector(self._planar_board)
        self._connection_state: Dict[str, tuple[bool, str]] = {}
        self._static_images: Dict[bool, Dict[str, np.ndarray]] = {}
        self._mode_trackers: Dict[bool, Dict[str, CoverageTracker]] = {}
        self._presentation = "split"
        self._comparison_layout = None
        self._review_images: Optional[Dict[str, np.ndarray]] = None
        self._review_key = ""
        self._known_sizes = {}

        self._layout = QGridLayout(self)
        self._layout.setSpacing(6)
        self._layout.setContentsMargins(6, 6, 6, 6)

        self._empty = _EmptyState(self)
        self._layout.addWidget(self._empty, 0, 0)
        self._layout_refresh_timer = QTimer(self)
        self._layout_refresh_timer.setSingleShot(True)
        self._layout_refresh_timer.timeout.connect(self._refresh_tile_pixmaps_after_layout)

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
        self._mode_trackers.clear()

    def set_capture_sequence(self, seq: Optional[CaptureSequence]):
        self._capture_sequence = seq

    def _multimodal_ids(self) -> tuple[str, str]:
        aux = self._stream_manager.auxiliary_camera
        return self._rgb_left_id or "RGB_L", aux.camera_id if aux is not None else "aux_cam"

    def _accept_camera(self, camera_id: str) -> bool:
        if camera_id in self._hidden_cameras:
            return False
        if self._multimodal_mode:
            return camera_id in self._multimodal_ids()
        pair_ids = {cid for pair in self._stream_manager.stereo_pairs.values()
                    for cid in (pair.left.camera_id, pair.right.camera_id)}
        aux = self._stream_manager.auxiliary_camera
        return camera_id in pair_ids if pair_ids else aux is None or camera_id != aux.camera_id

    def set_comparison_host(self, layout):
        self._comparison_layout = layout
        self._relayout()

    @property
    def can_focus_aux(self) -> bool:
        return (self._multimodal_mode and self._stream_manager.auxiliary_camera is not None and
                all(cid in self._visible_camera_ids() for cid in self._multimodal_ids()))

    @property
    def presentation_mode(self) -> str:
        ids = self._visible_camera_ids()
        focus = (self._presentation == "aux_focus" and self._multimodal_mode and self._comparison_layout is not None
                 and all(cid in ids for cid in self._multimodal_ids()) and self._expanded_id is None)
        return "aux_focus" if focus else "split"

    @property
    def review_key(self) -> str:
        return self._review_key

    def review_matches(self, images) -> bool:
        return (self._review_images is not None and set(images) == set(self._review_images) and
                all(self._review_images[cid] is frame for cid, frame in images.items()))

    def set_presentation(self, mode: str):
        if mode not in {"split", "aux_focus"}:
            raise ValueError(f"Unknown preview presentation: {mode}")
        self._presentation = mode
        self._expanded_id = None
        self._relayout()

    def _visible_camera_ids(self):
        stereo_order = list(dict.fromkeys(
            camera.camera_id for pair in self._stream_manager.stereo_pairs.values()
            for camera in (pair.left, pair.right)
        ))
        stereo_order.extend(cid for cid in self._views if cid not in stereo_order)
        candidates = (tuple(cid for cid in stereo_order if cid in self._review_images)
                      if self._review_images is not None else
                      self._multimodal_ids() if self._multimodal_mode else tuple(stereo_order))
        return [cid for cid in candidates if cid in self._views and self._accept_camera(cid)]

    def displayed_corner_counts(self):
        return {cid: view.corner_count for cid, view in self._views.items()}

    def show_review_images(self, key: str, images: Dict[str, np.ndarray]):
        self._expanded_id = None
        self._review_images = {cid: frame for cid, frame in images.items() if self._accept_camera(cid)}
        self._review_key = key
        for cid, frame in self._review_images.items():
            view = self._get_or_create_view(cid)
            view.set_display_name(self._camera_label(cid))
            result = self._detect_for_preview(frame)
            view.corner_count = result.num_corners
            view.update_frame(self._draw_detection(frame, result) if self._show_guide else frame)
            h, w = frame.shape[:2]
            view.update_info(f"{w}x{h} | 回看 {key} | 预览角点 {result.num_corners}/{self._preview_corner_count}")
        self._relayout()
        self.review_changed.emit(key, True)

    def _cancel_review(self):
        self._review_images = None
        self._review_key = ""
        self.review_changed.emit("", False)

    def end_review(self):
        if self._review_images is None:
            return
        self._cancel_review()
        self._last_render_ts.clear()
        self._last_detect_ts.clear()
        self.set_multimodal_mode(self._multimodal_mode, self._rgb_left_id, force_refresh=True)

    def close_preview(self, camera_id: str):
        if camera_id not in self._views:
            return
        self._hidden_cameras.add(camera_id)
        self._remove_view(camera_id, preserve_tracker=True)
        self.preview_visibility_changed.emit()

    @property
    def has_closed_previews(self) -> bool:
        return bool(self._hidden_cameras)

    def restore_closed_previews(self):
        self._hidden_cameras.clear()
        self.set_multimodal_mode(self._multimodal_mode, self._rgb_left_id, force_refresh=True)
        self.preview_visibility_changed.emit()

    def set_multimodal_mode(
        self, enabled: bool, rgb_left_id: Optional[str] = None, *, force_refresh: bool = False,
    ):
        was_multimodal = self._multimodal_mode
        aux = self._stream_manager.auxiliary_camera
        board = aux.pattern_board_config() if aux is not None else PatternBoardConfig.ir_chessboard()
        changed = (self._multimodal_mode != enabled or self._rgb_left_id != rgb_left_id or
                   self._planar_board != board)
        self._multimodal_mode = enabled
        self._rgb_left_id = rgb_left_id
        self._planar_board = board
        self._planar_detector = PlanarPatternDetector(board)
        if not enabled and not was_multimodal and not force_refresh:
            return
        if changed:
            self._cancel_review()
            if was_multimodal != enabled:
                self._mode_trackers[was_multimodal] = self._trackers
                self._trackers = self._mode_trackers.pop(enabled, {})
            else:
                self._trackers.clear()
            self._last_detection.clear()
            self._last_detect_ts.clear()
            self._last_render_ts.clear()
            self._expanded_id = None
        if enabled:
            ids = self._multimodal_ids()
        else:
            ids = tuple(cid for cid in self._stream_manager.configured_camera_urls() if self._accept_camera(cid))
        ids = tuple(cid for cid in ids if cid not in self._hidden_cameras)
        for cid in list(self._views):
            if cid not in ids:
                self._remove_view(cid, relayout=False)
        for cid in ids:
            view = self._get_or_create_view(cid)
            view.set_display_name(self._camera_label(cid))
            if enabled and cid == self._multimodal_ids()[1] and (aux is None or not aux.url):
                view.set_connected(False)
                view.clear_frame("未配置红外/辅助视频流，可配置相机或导入离线图片")
                continue
            if enabled and cid == self._multimodal_ids()[0] and rgb_left_id is None:
                view.set_connected(False)
                view.clear_frame("未配置 RGB 左相机，可配置相机或导入离线图片")
                continue
            connected, reason = self._connection_state.get(cid, (False, "尚未连接"))
            view.set_connected(connected)
            if changed or force_refresh or view._last_frame is None:
                view.clear_frame("已连接，等待视频帧..." if connected else reason)
            entry = self._stream_manager.get_latest_frame(cid)
            if entry is not None and connected:
                self._on_frame(cid, *entry)
        if (changed or force_refresh) and enabled in self._static_images and not any(
            self._connection_state.get(cid, (False, ""))[0] for cid in ids
        ):
            self._show_static_images(self._static_images[enabled])
        self._relayout()

    def _camera_label(self, camera_id: str) -> str:
        if not self._multimodal_mode:
            roles = {role for pair in self._stream_manager.stereo_pairs.values()
                     for role, camera in (("左目", pair.left), ("右目", pair.right)) if camera.camera_id == camera_id}
            return f"{next(iter(roles))} · {camera_id}" if len(roles) == 1 else camera_id
        if camera_id == self._multimodal_ids()[0]:
            return f"RGB 左目 · {camera_id}"
        aux = self._stream_manager.auxiliary_camera
        name = "声像仪 RGB" if aux is not None and aux.modality == "acoustic_rgb" else "红外（中间）"
        return f"{name} · {camera_id}"

    @property
    def _preview_corner_count(self) -> int:
        if self._multimodal_mode:
            return self._planar_board.cols * self._planar_board.rows
        return self._board.num_corners

    def _draw_detection(self, frame: np.ndarray, result: DetectionResult, *, include_target=True) -> np.ndarray:
        if not self._multimodal_mode:
            vis = frame.copy()
            if result.valid:
                points = result.charuco_corners.reshape(-1, 2)
                hull = cv2.convexHull(points.astype(np.float32)).astype(np.int32)
                cv2.polylines(vis, [hull], True, (91, 182, 231), 2, cv2.LINE_AA)
                for point in points:
                    cv2.circle(vis, tuple(point.astype(int)), 2, (91, 182, 231), -1)
            target = self._capture_sequence.current if self._capture_sequence else None
            if include_target and target is not None and self._review_images is None:
                h, w = frame.shape[:2]
                x = int(target.norm_cx * w)
                y = int(target.norm_cy * h)
                extent = {'small': .09, 'medium': .14, 'large': .20}.get(target.scale, .14)
                rx, ry = int(w * extent), int(h * extent)
                # A suggested image region, not a calibrated 3D target projection.
                for start, end in (
                    ((x - rx, y - ry), (x + rx, y - ry)),
                    ((x + rx, y - ry), (x + rx, y + ry)),
                    ((x + rx, y + ry), (x - rx, y + ry)),
                    ((x - rx, y + ry), (x - rx, y - ry)),
                ):
                    a, b = np.asarray(start), np.asarray(end)
                    length = float(np.linalg.norm(b - a))
                    for offset in range(0, int(length), 18):
                        p = a + (b - a) * (offset / length)
                        q = a + (b - a) * (min(offset + 10, length) / length)
                        cv2.line(vis, tuple(p.astype(int)), tuple(q.astype(int)), (255, 210, 60), 2)
            return vis
        vis = frame.copy()
        if result.valid:
            cv2.drawChessboardCorners(
                vis, (self._planar_board.cols, self._planar_board.rows),
                result.charuco_corners, True,
            )
        return vis

    def set_show_guide(self, show: bool):
        self._show_guide = show

    def set_show_detections(self, show: bool):
        self._show_guide = show

    def show_static_images(self, images: Dict[str, np.ndarray]):
        """Keep stereo offline images separate from the multimodal preview."""
        self._static_images[False] = dict(images)
        if not self._multimodal_mode:
            self._show_static_images(images)

    def show_multimodal_images(self, images: Dict[str, np.ndarray]):
        self._static_images[True] = dict(images)
        if self._multimodal_mode:
            self._show_static_images(images)

    def _show_static_images(self, images: Dict[str, np.ndarray]):
        visible_ids = set(images.keys()) - self._hidden_cameras
        visible_ids = {cid for cid in visible_ids if self._accept_camera(cid)}
        if self._multimodal_mode:
            visible_ids &= set(self._multimodal_ids())
        for camera_id in list(self._views.keys()):
            if camera_id not in visible_ids:
                self._remove_view(camera_id, relayout=False)

        for camera_id, frame in images.items():
            if camera_id not in visible_ids:
                continue
            view = self._get_or_create_view(camera_id)
            view.set_display_name(self._camera_label(camera_id) + ' · 离线')
            view.set_connected(False)
            view._status_dot.setToolTip('离线图像，不代表相机已连接')
            view._close_button.setToolTip('关闭这路离线图像预览')
            h, w = frame.shape[:2]

            if camera_id not in self._trackers:
                self._trackers[camera_id] = CoverageTracker((w, h))

            result = self._detect_for_preview(frame)
            view.corner_count = result.num_corners
            if self._show_guide:
                display = self._draw_detection(frame, result, include_target=False)
            else:
                display = frame

            view.update_frame(display)
            suffix = f" | 预览角点 {result.num_corners}/{self._preview_corner_count}"
            view.update_info(f"{w}x{h} | 离线图像{suffix}")
        if self._multimodal_mode:
            # Keep both roles visible even when only one offline image is available.
            for cid in self._multimodal_ids():
                if cid in self._hidden_cameras:
                    continue
                view = self._get_or_create_view(cid)
                view.set_display_name(self._camera_label(cid))
                if cid not in visible_ids:
                    view.clear_frame("缺少该路配对图片")
        self._relayout()

    def show_connecting(self, camera_urls: Dict[str, str], *, preserve_trackers: bool = False):
        """Create placeholder tiles for configured cameras before frames arrive."""
        self._cancel_review()
        visible_ids = {cid for cid in camera_urls if self._accept_camera(cid)}
        self._fps.clear()
        if not preserve_trackers:
            self._trackers.clear()
        self._last_render_ts.clear()
        self._last_detect_ts.clear()
        self._last_detection.clear()
        self._expanded_id = None
        self._known_sizes.clear()
        self._static_images.clear()
        if not preserve_trackers:
            self._mode_trackers.clear()
        self._connection_state = {cid: (False, "连接中...") for cid in camera_urls}
        if self._multimodal_mode:
            visible_ids &= set(self._multimodal_ids())

        for camera_id in list(self._views.keys()):
            if camera_id not in visible_ids:
                self._remove_view(camera_id, relayout=False)

        for camera_id, url in camera_urls.items():
            if camera_id not in visible_ids:
                continue
            view = self._get_or_create_view(camera_id)
            view.set_display_name(self._camera_label(camera_id))
            view.set_connected(False)
            view.clear_frame("连接中...")
            view.setToolTip(
                f"相机: {camera_id}\n视频流: {url}\n状态: 正在连接，等待首帧"
            )
        self._relayout()
        if self._multimodal_mode:
            self.set_multimodal_mode(True, self._rgb_left_id)

    def clear_all(self):
        """Remove all preview tiles and cached detections."""
        self._cancel_review()
        self._hidden_cameras.clear()
        for camera_id in list(self._views.keys()):
            self._remove_view(camera_id, relayout=False)
        self._fps.clear()
        self._trackers.clear()
        self._last_render_ts.clear()
        self._last_detect_ts.clear()
        self._last_detection.clear()
        self._expanded_id = None
        self._connection_state.clear()
        self._static_images.clear()
        self._mode_trackers.clear()
        self._known_sizes.clear()
        self._relayout()
        self.preview_visibility_changed.emit()

    def clear_camera_frames(self, camera_ids: list[str], info: str = "等待新画面..."):
        """Clear cached preview images for specific cameras without removing tiles."""
        for camera_id in camera_ids:
            view = self._views.get(camera_id)
            if view is not None:
                view.clear_frame(info)
                view.set_connected(True)
            self._fps.pop(camera_id, None)
            self._last_render_ts.pop(camera_id, None)
            self._last_detect_ts.pop(camera_id, None)
            self._last_detection.pop(camera_id, None)

    def reset_trackers(self):
        self._trackers.clear()
        self._mode_trackers.clear()

    def notify_capture(self, camera_id: str, result: DetectionResult | PlanarDetectionResult):
        tracker = self._trackers.get(camera_id)
        if tracker is None or (tracker.w, tracker.h) != result.image_size:
            tracker = CoverageTracker(result.image_size)
            self._trackers[camera_id] = tracker
        tracker.update(result)

    @Slot(str, np.ndarray, float)
    def _on_frame(self, camera_id: str, frame: np.ndarray, ts: float):
        if self._review_images is not None or not self._accept_camera(camera_id):
            return
        now = time.monotonic()
        frame_age = time.time() - ts
        if frame_age > self.MAX_PREVIEW_FRAME_AGE_SEC:
            return
        last = self._last_render_ts.get(camera_id, 0.0)
        if now - last < self._min_interval:
            return
        self._last_render_ts[camera_id] = now

        view = self._get_or_create_view(camera_id)
        view.set_display_name(self._camera_label(camera_id))
        h, w = frame.shape[:2]

        if camera_id not in self._trackers:
            self._trackers[camera_id] = CoverageTracker((w, h))

        last_detect = self._last_detect_ts.get(camera_id, 0.0)
        if now - last_detect >= self._detect_min_interval or camera_id not in self._last_detection:
            self._last_detect_ts[camera_id] = now
            result = self._detect_for_preview(frame)
            self._last_detection[camera_id] = result
            self.preview_detected.emit(camera_id, result, frame)
        else:
            result = self._last_detection[camera_id]

        display = self._draw_detection(frame, result) if self._show_guide else frame
        view.corner_count = result.num_corners

        view.update_frame(display)
        if self._known_sizes.get(camera_id) != (w, h):
            self._known_sizes[camera_id] = (w, h)
            self._relayout()

        fps = self._fps.get(camera_id, 0)
        suffix = f" | 预览角点 {result.num_corners}/{self._preview_corner_count}"
        view.update_info(f"{w}x{h} | {fps:.1f} fps{suffix}")

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
        with perf_timer("preview board detect", threshold_ms=40.0):
            if self._multimodal_mode:
                planar = self._planar_detector.detect(detect_frame)
                # Preview-only adapter: these sequential IDs are never calibration observations.
                result = DetectionResult(
                    planar.image_points.reshape(-1, 1, 2) if planar.valid else None,
                    np.arange(planar.num_points, dtype=np.int32).reshape(-1, 1) if planar.valid else None,
                    [], None, planar.image_size,
                )
            else:
                result = self._detector.detect(detect_frame)
        if scale == 1.0:
            return result
        return _scale_detection_result(result, 1.0 / scale, (w, h))

    @staticmethod
    def _use_compact_overlay(frame: np.ndarray) -> bool:
        """Low-resolution sensors should not receive large in-frame text overlays."""
        h, w = frame.shape[:2]
        return w < 480 or h < 320

    @Slot(str)
    def _on_connected(self, camera_id: str):
        self._connection_state[camera_id] = (True, "已连接，等待视频帧...")
        if self._review_images is not None:
            return
        if not self._accept_camera(camera_id):
            return
        view = self._get_or_create_view(camera_id)
        view.set_display_name(self._camera_label(camera_id))
        view.set_connected(True)
        view.clear_frame("已连接，等待视频帧...")
        view.setToolTip(f"相机: {camera_id}\n状态: 已连接")

    @Slot(str, str)
    def _on_disconnected(self, camera_id: str, reason: str):
        prefix = "无法连接" if reason not in {"已断开", "已移除", "重新连接"} else "断开"
        self._connection_state[camera_id] = (False, f"{prefix}: {reason}")
        self._last_detection.pop(camera_id, None)
        self._known_sizes.pop(camera_id, None)
        self._last_detect_ts.pop(camera_id, None)
        view = self._views.get(camera_id)
        if view:
            view.set_connected(False)
            if self._review_images is not None and camera_id in self._review_images:
                view.update_info(f"回看 {self._review_key} | 实时{prefix}: {reason}")
                return
            view.clear_frame(f"{prefix}: {reason}")
            view.setToolTip(f"相机: {camera_id}\n状态: {prefix}\n原因: {reason}")

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
            view.close_requested.connect(self.close_preview)
            self._views[camera_id] = view
            self._relayout()
        return self._views[camera_id]

    def _remove_view(self, camera_id: str, relayout: bool = True, *, preserve_tracker: bool = False):
        self._known_sizes.pop(camera_id, None)
        view = self._views.pop(camera_id, None)
        if view is not None:
            view.clear_frame()
            self._layout.removeWidget(view)
            if self._comparison_layout is not None:
                self._comparison_layout.removeWidget(view)
            view.hide()
            view.deleteLater()
        self._fps.pop(camera_id, None)
        if not preserve_tracker:
            self._trackers.pop(camera_id, None)
        self._last_render_ts.pop(camera_id, None)
        self._last_detect_ts.pop(camera_id, None)
        self._last_detection.pop(camera_id, None)
        if self._expanded_id == camera_id:
            self._expanded_id = None
        if relayout:
            self._relayout()

    def _relayout(self):
        for row in range(self._layout.rowCount()):
            self._layout.setRowStretch(row, 0)
        for col in range(self._layout.columnCount()):
            self._layout.setColumnStretch(col, 0)
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item and item.widget():
                item.widget().hide()
        if self._comparison_layout is not None:
            while self._comparison_layout.count():
                item = self._comparison_layout.takeAt(0)
                if item and item.widget():
                    item.widget().hide()

        ids = self._visible_camera_ids()
        if self._expanded_id not in ids:
            self._expanded_id = None
        n = len(ids)
        mode = self.presentation_mode
        self.comparison_visible_changed.emit(mode == "aux_focus")
        self.presentation_changed.emit(mode)
        if n == 0:
            self._expanded_id = None
            self._empty.previews_closed = bool(self._hidden_cameras)
            self._empty.update()
            self._layout.addWidget(self._empty, 0, 0)
            self._empty.show()
            return

        self._empty.hide()

        if mode == "aux_focus":
            rgb_id, aux_id = self._multimodal_ids()
            self._layout.addWidget(self._views[aux_id], 0, 0)
            self._comparison_layout.addWidget(self._views[rgb_id])
            for cid in (rgb_id, aux_id):
                self._views[cid].set_expanded_mode(False)
                self._views[cid].show()
            self._layout.setRowStretch(0, 1)
            self._layout.setColumnStretch(0, 1)
            self._layout_refresh_timer.start(0)
            return

        if self._expanded_id is not None and self._expanded_id not in self._views:
            self._expanded_id = None

        if self._expanded_id is not None:
            view = self._views[self._expanded_id]
            self._layout.addWidget(view, 0, 0)
            view.show()
            self._layout.setRowStretch(0, 1)
            self._layout.setColumnStretch(0, 1)
            for cid, v in self._views.items():
                v.set_expanded_mode(cid == self._expanded_id)
            self._layout_refresh_timer.start(0)
            return

        for v in self._views.values():
            v.set_expanded_mode(False)

        self._layout.setRowStretch(0, 0)
        self._layout.setColumnStretch(0, 0)

        cols = max(1, math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        order = ids
        for i, cid in enumerate(order):
            if cid not in self._views:
                continue
            view = self._views[cid]
            r, c = divmod(i, cols)
            self._layout.addWidget(view, r, c)
            view.show()

        for c in range(cols):
            self._layout.setColumnStretch(c, 1)
        if self._multimodal_mode and n == 2:
            for column, cid in enumerate(ids):
                frame = self._views[cid]._last_frame
                ratio = frame.shape[1] / frame.shape[0] if frame is not None else (16 / 9 if column == 0 else 4 / 3)
                self._layout.setColumnStretch(column, max(1, round(ratio * 100)))
        for r in range(rows):
            self._layout.setRowStretch(r, 1)

        self._layout_refresh_timer.start(0)

    def _refresh_tile_pixmaps_after_layout(self) -> None:
        for v in self._views.values():
            v.refresh_preview_if_cached()
