"""Non-metric capture motion demonstration, rendered entirely with QPainter."""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractScrollArea, QComboBox, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from ..theme import ACCENT, BG_MID, BORDER, TEXT_DIM, TEXT_HINT, WARNING, app_icon
from .board_motion import BoardTilt, TILT_ACTIONS, tilt_point
from .stereo_camera_model import draw_stereo_camera
from .board_view import EDGES, paint_board_view

def motion_description(target):
    if target is None:
        return "暂无采图动作"
    if target.tilt in TILT_ACTIONS:
        return TILT_ACTIONS[target.tilt][1]
    if target.scale != 'medium':
        move = '向镜头靠近' if target.scale == 'large' else '远离镜头'
        return f'保持当前高度，{move}；双路完整可见，到位后静止'
    if (target.zone_row, target.zone_col) == (1, 1):
        return "板面正对镜头，放在画面中央并保持静止"
    return "按左目预览方向平移至建议区域，保持板面朝向"


def board_point(x, y, progress, target, motion='plan'):
    """Camera axes: X right, Y down, Z forward; pivot edge stays fixed."""
    if target is None:
        return x, y, 4.7
    if motion in ('approach', 'retreat'):
        end_z = 4.7 + {'small': 1.6, 'medium': 0., 'large': -1.}.get(target.scale, 0.)
        start_offset = 1. if motion == 'approach' else -1.
        return (x + (target.zone_col - 1) * 1.25,
                y + (target.zone_row - 1) * .8, end_z + start_offset * (1. - progress))
    if motion == 'hold':
        progress = 1.
    tilt = target.tilt
    x, y, z = tilt_point(x, y, progress, tilt)
    # Tilt demonstrations start at the suggested image position; they do not
    # combine a translation with a rotation around a supposedly fixed edge.
    shift = progress if tilt == BoardTilt.FRONT else 1.
    # The board is a fixed-size object: vary camera-axis distance, not its dimensions.
    depth_shift = {'small': 1.6, 'medium': 0., 'large': -1.}.get(target.scale, 0.)
    depth_shift *= progress if tilt == BoardTilt.FRONT else 1.
    return (x + (target.zone_col - 1) * 1.25 * shift,
            y + (target.zone_row - 1) * .8 * shift, z + depth_shift)


VIEW_EYE = (6., -6., 10.)
_forward = (-6., 6., -7.4)
_length = math.sqrt(sum(v * v for v in _forward))
_forward = tuple(v / _length for v in _forward)
_length = math.hypot(_forward[2], _forward[0])
_right = (_forward[2] / _length, 0., -_forward[0] / _length)
_down = (_forward[1] * _right[2], _forward[2] * _right[0] - _forward[0] * _right[2],
         -_forward[1] * _right[0])


def project_point(point):
    """Front-above view exposes lenses and top cover, with camera Y pointing down."""
    v = tuple(a - b for a, b in zip(point, VIEW_EYE))
    depth = sum(a * b for a, b in zip(v, _forward))
    return QPointF(154 + 360 * sum(a * b for a, b in zip(v, _right)) / depth,
                   100 + 360 * sum(a * b for a, b in zip(v, _down)) / depth)


class MotionCanvas(QWidget):
    def __init__(self, guide):
        super().__init__(guide)
        self.guide = guide
        self.compact = False
        self._base = None
        self._cache_key = None
        self.cache_builds = 0
        self.setMinimumSize(360, 210)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("固定一体式双目相机与标定板动作对照")
        self.setToolTip("左侧沿相机朝向看板，不镜像。右侧默认固定立体斜视，也可切换俯视/侧视。白色为固定边、黄色为移动边、蓝色虚线为目标；放大构图非真实视场/角度。")
        self.setAccessibleDescription(self.toolTip())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        key = (self.size().width(), self.size().height(), self.devicePixelRatioF(),
               self.guide._key, self.guide.view_mode, self.compact,
               self.palette().cacheKey(), self.font().toString())
        if key != self._cache_key:
            ratio = self.devicePixelRatioF()
            self._base = QPixmap(round(self.width() * ratio), round(self.height() * ratio))
            self._base.setDevicePixelRatio(ratio)
            self._base.fill(Qt.transparent)
            base = QPainter(self._base)
            base.setRenderHint(QPainter.Antialiasing)
            base.setFont(self.font())
            if self.guide.view_mode != 'spatial':
                paint_board_view(base, self.rect(), self.guide.target, 0., self._board_point,
                                 self.compact, 'static', self._depth_mode())
            for rect, moving in (self._panes() if self.guide.view_mode == 'spatial' else []):
                base.setPen(QPen(QColor(BORDER), 1))
                base.setBrush(QColor(BG_MID))
                base.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 6, 6)
                base.setPen(QColor(TEXT_DIM))
                title = "动作 · 相机固定" if moving else "起始 · 相机固定"
                base.drawText(rect.adjusted(10, 4, -10, 0), Qt.AlignTop, title)
                if self.guide.target is None:
                    base.drawText(rect, Qt.AlignCenter, "等待采图计划")
                else:
                    self._draw_scene(base, rect, 0., False, 'static' if moving else 'all')
            base.end()
            self._cache_key = key
            self.cache_builds += 1
        p.drawPixmap(0, 0, self._base)
        if self.guide.view_mode != 'spatial':
            paint_board_view(p, self.rect(), self.guide.target, self.guide.progress,
                             self._board_point, self.compact, depth_mode=self._depth_mode())
            return
        if self.guide.target is not None:
            for rect, moving in self._panes():
                if moving:
                    self._draw_scene(p, rect, self.guide.progress, True, 'dynamic')

    def _board_point(self, x, y, progress):
        return board_point(x, y, progress, self.guide.target, self.guide.motion)

    def _depth_mode(self):
        return 'profile' if self.guide.view_mode == 'profile' else 'oblique'

    def _panes(self):
        if self.compact:
            return [(QRectF(self.rect()), True)]
        width = (self.width() - 12) / 2
        return [(QRectF(0, 0, width, self.height()), False),
                (QRectF(width + 12, 0, width, self.height()), True)]

    def _draw_scene(self, painter, rect, progress, moving, layer):
        painter.save()
        interior = rect.adjusted(4, 28, -4, -4)
        painter.setClipRect(interior)
        scale = min(interior.width() / 320, interior.height() / 221)
        painter.translate(interior.center().x() - 160 * scale,
                          interior.center().y() - 142.5 * scale)
        painter.scale(scale, scale)
        self._scene(painter, progress, moving, layer)
        painter.restore()

    def _scene(self, p, progress, moving, layer='all'):
        target = self.guide.target

        def path(points, color, width=1., dashed=False, fill=None):
            polygon = QPolygonF([project_point(pt) for pt in points])
            p.setPen(QPen(QColor(color), width, Qt.DashLine if dashed else Qt.SolidLine))
            p.setBrush(QColor(fill) if fill else Qt.NoBrush)
            if fill:
                p.drawPolygon(polygon)
            else:
                p.drawPolyline(polygon)

        if layer != 'dynamic':
            for x in (-3, 0, 3):
                path([(x, 1.7, -1), (x, 1.7, 7)], BORDER, .5)
            for z in (0, 3, 6):
                path([(-3, 1.7, z), (3, 1.7, z)], BORDER, .5)
            draw_stereo_camera(p, project_point, VIEW_EYE)
        corners = [(-1.45, -1), (1.45, -1), (1.45, 1), (-1.45, 1), (-1.45, -1)]

        def board(t, ghost=False):
            pts = [self._board_point(x, y, t) for x, y in corners]
            path(pts, ACCENT if ghost else '#68d7cf', 1.6, ghost,
                 None if ghost else '#304d49')
            if not ghost:
                for k in range(1, 7):
                    x = -1.45 + 2.9 * k / 7
                    path([self._board_point(x, -1, t), self._board_point(x, 1, t)], '#68a69f', .7)
                for k in range(1, 5):
                    y = -1 + 2 * k / 5
                    path([self._board_point(-1.45, y, t), self._board_point(1.45, y, t)], '#68a69f', .7)

        if layer != 'static':
            board(progress)
        if moving:
            # Draw the target outline last so it remains visible at the endpoint.
            board(1., True)
            edge = {
                BoardTilt.TILT_LEFT: [(-1.45, -1), (-1.45, 1)],
                BoardTilt.TILT_RIGHT: [(1.45, -1), (1.45, 1)],
                BoardTilt.TILT_UP: [(-1.45, -1), (1.45, -1)],
                BoardTilt.TILT_DOWN: [(-1.45, 1), (1.45, 1)],
            }.get(target.tilt)
            if edge:
                path([self._board_point(x, y, progress) for x, y in edge], WARNING, 3.5)
            # Arrow points from starting board centre to target centre.
            a = project_point(self._board_point(0, 0, 0))
            b = project_point(self._board_point(0, 0, 1))
            if math.hypot(b.x() - a.x(), b.y() - a.y()) > 5:
                p.setPen(QPen(QColor(WARNING), 1.8))
                p.drawLine(a, b)
                angle = math.atan2(b.y() - a.y(), b.x() - a.x())
                for delta in (-.45, .45):
                    p.drawLine(b, b - QPointF(8 * math.cos(angle + delta), 8 * math.sin(angle + delta)))

class StereoSchematic(QWidget):
    """Time-based animation; no camera reads, detectors, or capture side effects."""

    def __init__(self):
        super().__init__()
        self.target = None
        self.motion = 'plan'
        self.view_mode = 'board'
        self.progress = 0.
        self._key = object()
        self._enabled = False
        self._instruction = None
        self._paused = False
        self._elapsed = 0.
        self._scroll_area = None
        self._preferred_compact = False
        self._last = time.monotonic()
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self._visibility_timer = QTimer(self)
        self._visibility_timer.setSingleShot(True)
        self._visibility_timer.timeout.connect(self._sync_timer)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)
        toolbar = QHBoxLayout()
        self.description = QLabel("暂无采图动作")
        self.description.setWordWrap(True)
        self.description.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        toolbar.addWidget(self.description, 1)
        self.expand = QToolButton()
        self.expand.setCheckable(True)
        self.expand.setIcon(app_icon('maximize-2'))
        self.expand.setToolTip("展开起始与动作对照")
        self.expand.setAccessibleName("展开起始与动作对照")
        self.expand.toggled.connect(self._apply_compact)
        self.pause = QToolButton()
        self.pause.setCheckable(True)
        self.pause.toggled.connect(self._pause)
        self.replay = QToolButton()
        self.replay.setIcon(app_icon('rotate-ccw'))
        self.replay.setToolTip("重播动作示意")
        self.replay.setAccessibleName("重播动作示意")
        self.replay.clicked.connect(self.restart)
        toolbar.addWidget(self.pause)
        toolbar.addWidget(self.replay)
        toolbar.addWidget(self.expand)
        layout.addLayout(toolbar)
        self.canvas = MotionCanvas(self)
        layout.addWidget(self.canvas, 1)
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet(f"color:{TEXT_HINT}; font-size:12px;")
        self.view_options = QWidget()
        view_row = QHBoxLayout(self.view_options)
        view_row.setContentsMargins(0, 0, 0, 0)
        self.view_selector = QComboBox()
        self.view_selector.addItem('板面＋立体', 'board')
        self.view_selector.addItem('板面＋俯/侧', 'profile')
        self.view_selector.addItem('空间对照', 'spatial')
        self.view_selector.setAccessibleName('动作示意观察视角')
        self.view_selector.setToolTip('左侧保持板面视角，右侧可选固定立体斜视或俯视/侧视；空间对照保留原示意')
        self.view_selector.currentIndexChanged.connect(self._change_view)
        view_row.addWidget(self.view_selector)
        view_row.addWidget(self.caption, 1)
        layout.addWidget(self.view_options)
        self._pause(False)
        self.set_target(None)

    def _change_view(self, _index):
        self.view_mode = self.view_selector.currentData()
        self._refresh_caption()
        self.canvas.update()

    def _refresh_caption(self):
        if self._instruction is not None and not self._enabled:
            text = '目标姿态参考（非实测） · 当前请按上方提示操作'
        elif self.view_mode == 'spatial':
            text = '镜头前侧观察 · 非左目画面方向\n蓝色虚线为目标，黄色为移动侧；尺寸仅示意'
        else:
            edges = EDGES.get(self.target.tilt) if self.target is not None else None
            action = f'固定{edges[2]} · {edges[3]}靠近镜头' if edges else '板面保持朝向镜头'
            text = action + '\n白色为固定边 · 黄色为移动边 · 蓝色虚线为目标（非实测）'
        self.caption.setText(text)

    def set_target(self, target, *, active=True, instruction=None, motion='plan'):
        key = (target.zone_row, target.zone_col, target.tilt, target.scale, motion) if target else None
        enabled = bool(active and target is not None and motion != 'hold')
        description = instruction if instruction is not None else motion_description(target)
        if (key == self._key and enabled == self._enabled
                and description == self.description.text()):
            return
        changed = key != self._key
        self.target, self._key = target, key
        self.motion = motion
        self._enabled = enabled
        self._instruction = instruction
        self._refresh_caption()
        self.description.setText(description)
        self.pause.setEnabled(self._enabled and self.has_motion)
        self.replay.setEnabled(self._enabled and self.has_motion)
        if changed:
            self._elapsed = self.progress = 0.
            self._last = time.monotonic()
        if not enabled:
            self.progress = 1.
        self._sync_timer()
        self.canvas.update()

    def set_compact(self, compact):
        self._preferred_compact = compact
        self._apply_compact()

    def _apply_compact(self):
        compact = self._preferred_compact and not self.expand.isChecked()
        self.canvas.compact = compact
        self.canvas.setMinimumWidth(280 if self._preferred_compact else 360)
        self.canvas.setMinimumHeight(150 if compact else 210)
        self.canvas.setMaximumHeight(175 if compact else 16777215)
        self.caption.setVisible(not compact)
        self.view_options.setVisible(not compact)
        self.view_options.layout().setDirection(
            QHBoxLayout.TopToBottom if self._preferred_compact else QHBoxLayout.LeftToRight)
        label = '收起示意与视角选项' if self.expand.isChecked() else '展开示意与视角选项'
        self.expand.setToolTip(label)
        self.expand.setAccessibleName(label)
        self.canvas.update()

    def _sync_timer(self, *args):
        if (self.isVisible() and not self.canvas.visibleRegion().isEmpty()
                and self._enabled and self.has_motion and not self._paused):
            if not self.timer.isActive():
                self._last = time.monotonic()
                self.timer.start()
        else:
            self.timer.stop()

    @property
    def has_motion(self):
        if self.motion in ('approach', 'retreat'):
            return self.target is not None
        if self.motion == 'hold':
            return False
        return self.target is not None and (
            self.target.tilt != BoardTilt.FRONT
            or (self.target.zone_row, self.target.zone_col) != (1, 1)
            or self.target.scale != 'medium'
        )

    def _pause(self, paused):
        self._paused = paused
        label = "继续动作示意" if paused else "暂停动作示意"
        self.pause.setIcon(app_icon('play' if paused else 'pause'))
        self.pause.setToolTip(label)
        self.pause.setAccessibleName(label)
        self._sync_timer()

    def restart(self):
        self._elapsed = self.progress = 0.
        self._last = time.monotonic()
        self.pause.setChecked(False)
        self._sync_timer()
        self.canvas.update()

    def _tick(self):
        if self.canvas.visibleRegion().isEmpty():
            self.timer.stop()
            return
        now = time.monotonic()
        self._elapsed += min(now - self._last, .1)
        self._last = now
        phase = self._elapsed % 6
        if phase < .7:
            value = 0.
        elif phase < 3.4:
            value = (phase - .7) / 2.7
        elif phase < 4.7:
            value = 1.
        else:
            value = 1 - (phase - 4.7) / 1.3
        self.progress = value * value * (3 - 2 * value)
        self.canvas.update()

    def showEvent(self, event):
        super().showEvent(event)
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QAbstractScrollArea):
            parent = parent.parentWidget()
        if parent is not self._scroll_area:
            if self._scroll_area is not None:
                self._scroll_area.verticalScrollBar().valueChanged.disconnect(self._sync_timer)
                self._scroll_area.horizontalScrollBar().valueChanged.disconnect(self._sync_timer)
                self._scroll_area.viewport().removeEventFilter(self)
            self._scroll_area = parent
            if parent is not None:
                parent.verticalScrollBar().valueChanged.connect(self._sync_timer)
                parent.horizontalScrollBar().valueChanged.connect(self._sync_timer)
                parent.viewport().installEventFilter(self)
        self._sync_timer()
        self._visibility_timer.start(0)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Resize, QEvent.Show, QEvent.Hide):
            self._visibility_timer.start(0)
        return super().eventFilter(watched, event)

    def hideEvent(self, event):
        self.timer.stop()
        self._visibility_timer.stop()
        super().hideEvent(event)
