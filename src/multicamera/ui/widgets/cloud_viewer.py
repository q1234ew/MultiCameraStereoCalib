"""3D point cloud viewer with QPainter fallback for empty state."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QPoint, QPointF, Slot, QRectF
from PySide6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
    QImage,
)
from PySide6.QtWidgets import QWidget


class PointCloudViewer(QWidget):
    """Software-rendered 3D point cloud viewer.

    Uses QPainter for robust cross-platform rendering without
    OpenGL context/shader compatibility issues on macOS.
    Switches to optimised QImage blitting for large point clouds.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

        self._points: Optional[np.ndarray] = None  # (N,3) float32
        self._colors: Optional[np.ndarray] = None  # (N,3) float32 [0,1]
        self._num_points = 0

        self._rot_x = 25.0
        self._rot_y = -40.0
        self._distance = 5.0
        self._center = np.zeros(3, dtype=np.float32)
        self._point_size = 2

        self._last_pos = QPoint()

        # Axis colours
        self._axis_colors = [
            QColor(220, 60, 60),   # X - red
            QColor(60, 180, 60),   # Y - green
            QColor(60, 100, 220),  # Z - blue
        ]

    # ── Public API ────────────────────────────────────────────

    @Slot(object)
    def set_pointcloud(self, pcd):
        """Accept an Open3D PointCloud or (points, colors) tuple."""
        if hasattr(pcd, "points"):
            self._points = np.asarray(pcd.points, dtype=np.float32)
            if hasattr(pcd, "has_colors") and pcd.has_colors():
                self._colors = np.asarray(pcd.colors, dtype=np.float32)
            else:
                self._colors = np.full_like(self._points, 0.7)
        elif isinstance(pcd, tuple) and len(pcd) == 2:
            self._points = np.asarray(pcd[0], dtype=np.float32)
            self._colors = (
                np.asarray(pcd[1], dtype=np.float32)
                if pcd[1] is not None
                else None
            )
        else:
            return

        if self._colors is None:
            self._colors = np.full_like(self._points, 0.7)

        self._num_points = len(self._points)
        self._auto_zoom()
        self.update()

    # ── Projection helpers ────────────────────────────────────

    def _build_view_matrix(self) -> np.ndarray:
        """Build a 4x4 view matrix from orbit parameters."""
        rx = np.radians(self._rot_x)
        ry = np.radians(self._rot_y)

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)

        rot_x = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx],
        ], dtype=np.float32)

        rot_y = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy],
        ], dtype=np.float32)

        R = rot_x @ rot_y

        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R
        t = -R @ self._center + np.array([0, 0, -self._distance], dtype=np.float32)
        view[:3, 3] = t
        return view

    def _build_proj_matrix(self, w: int, h: int) -> np.ndarray:
        """Simple perspective projection matrix."""
        fov = 45.0
        aspect = w / max(h, 1)
        near, far = 0.01, 1000.0
        f = 1.0 / np.tan(np.radians(fov) / 2.0)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2 * far * near / (near - far)
        proj[3, 2] = -1
        return proj

    def _project(
        self, points: np.ndarray, w: int, h: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project (N,3) points to (N,2) screen coords + depths.

        Returns (screen_xy, depths, visible_mask).
        """
        view = self._build_view_matrix()
        proj = self._build_proj_matrix(w, h)
        mvp = proj @ view

        N = len(points)
        homog = np.ones((N, 4), dtype=np.float32)
        homog[:, :3] = points

        clip = (mvp @ homog.T).T  # (N,4)
        w_clip = clip[:, 3]

        visible = w_clip > 0.001
        ndc = np.zeros((N, 3), dtype=np.float32)
        ndc[visible] = clip[visible, :3] / w_clip[visible, np.newaxis]

        screen = np.zeros((N, 2), dtype=np.float32)
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * w
        screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h

        return screen, ndc[:, 2], visible

    def _auto_zoom(self):
        if self._points is None or len(self._points) == 0:
            return
        self._center = self._points.mean(axis=0)
        extent = np.linalg.norm(
            self._points.max(axis=0) - self._points.min(axis=0)
        )
        self._distance = max(extent * 1.5, 0.5)

    # ── Painting ──────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(20, 20, 30))

        self._draw_axes(painter, w, h)

        if self._num_points == 0:
            self._draw_empty_hint(painter, w, h)
        else:
            self._draw_points(painter, w, h)

        painter.end()

    def _draw_empty_hint(self, painter: QPainter, w: int, h: int):
        painter.setPen(QPen(QColor(120, 120, 150)))
        font = QFont()
        font.setPointSize(14)
        painter.setFont(font)
        painter.drawText(
            QRectF(0, 0, w, h),
            Qt.AlignCenter,
            "完成标定后此处显示 3D 点云\n\n"
            "鼠标左键: 旋转  |  滚轮: 缩放  |  中键: 平移",
        )

    def _draw_axes(self, painter: QPainter, w: int, h: int):
        """Draw a small XYZ axis indicator in the bottom-left corner."""
        origin = np.array([[0, 0, 0]], dtype=np.float32)
        axis_len = self._distance * 0.08
        tips = np.array([
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len],
        ], dtype=np.float32) + self._center

        all_pts = np.vstack([origin + self._center, tips])  # (4,3)
        screen, _, vis = self._project(all_pts, w, h)

        if not vis.all():
            return

        margin = 60
        # remap to bottom-left corner mini viewport
        cx, cy = margin, h - margin
        scale = 40.0

        view = self._build_view_matrix()
        dirs = np.array([
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len],
        ], dtype=np.float32)
        rot_dirs = (view[:3, :3] @ dirs.T).T
        labels = ["X", "Y", "Z"]

        for i in range(3):
            d = rot_dirs[i]
            ex = cx + d[0] * scale / axis_len
            ey = cy - d[1] * scale / axis_len
            pen = QPen(self._axis_colors[i], 2)
            painter.setPen(pen)
            painter.drawLine(QPointF(cx, cy), QPointF(ex, ey))

            font = QFont()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QPointF(ex + 3, ey - 3), labels[i])

    def _draw_points(self, painter: QPainter, w: int, h: int):
        """Project and draw all points via QImage pixel buffer."""
        screen, depth, visible = self._project(self._points, w, h)

        mask = visible.copy()
        mask &= (screen[:, 0] >= 0) & (screen[:, 0] < w)
        mask &= (screen[:, 1] >= 0) & (screen[:, 1] < h)

        if not mask.any():
            return

        sx = screen[mask, 0].astype(np.int32)
        sy = screen[mask, 1].astype(np.int32)
        d = depth[mask]
        c = self._colors[mask]

        order = np.argsort(-d)
        sx, sy, c = sx[order], sy[order], c[order]

        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)

        ps = self._point_size
        r = (c[:, 0] * 255).astype(np.uint8)
        g = (c[:, 1] * 255).astype(np.uint8)
        b = (c[:, 2] * 255).astype(np.uint8)

        pt_painter = QPainter(img)
        pt_painter.setPen(Qt.NoPen)

        for i in range(len(sx)):
            pt_painter.setBrush(QColor(int(r[i]), int(g[i]), int(b[i])))
            pt_painter.drawRect(int(sx[i]), int(sy[i]), ps, ps)

        pt_painter.end()
        painter.drawImage(0, 0, img)

    # ── Mouse interaction ─────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()

        if event.buttons() & Qt.LeftButton:
            self._rot_y += dx * 0.5
            self._rot_x += dy * 0.5
            self._rot_x = max(-90, min(90, self._rot_x))
            self.update()
        elif event.buttons() & Qt.MiddleButton:
            scale = self._distance * 0.002
            view = self._build_view_matrix()
            R_inv = view[:3, :3].T
            pan_world = R_inv @ np.array(
                [dx * scale, -dy * scale, 0], dtype=np.float32
            )
            self._center -= pan_world
            self.update()

        self._last_pos = pos

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._distance = max(0.1, self._distance * factor)
        self.update()
