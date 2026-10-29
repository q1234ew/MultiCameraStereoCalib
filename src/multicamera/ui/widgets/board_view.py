"""Camera-facing board view and qualitative top/side depth explanation."""

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainterPath, QPen, QPolygonF

from ..theme import ACCENT, BG_MID, BORDER, TEXT, TEXT_DIM, WARNING
from .board_motion import BoardTilt
from .stereo_camera_model import draw_stereo_camera


EDGES = {
    BoardTilt.TILT_LEFT: (((1.45, -1), (1.45, 1)), ((-1.45, -1), (-1.45, 1)), '右边', '左边'),
    BoardTilt.TILT_RIGHT: (((-1.45, -1), (-1.45, 1)), ((1.45, -1), (1.45, 1)), '左边', '右边'),
    BoardTilt.TILT_UP: (((-1.45, 1), (1.45, 1)), ((-1.45, -1), (1.45, -1)), '下边', '上边'),
    BoardTilt.TILT_DOWN: (((-1.45, -1), (1.45, -1)), ((-1.45, 1), (1.45, 1)), '上边', '下边'),
}
BOARD_COLOR = '#68d7cf'
START_COLOR = '#82939f'
BOARD_CORNERS = ((-1.45, -1), (1.45, -1), (1.45, 1), (-1.45, 1), (-1.45, -1))


def board_projection(point):
    """Same signs as an unmirrored image: X right, Y down, positive Z ahead."""
    x, y, z = point
    if z <= 0:
        raise ValueError('Board must be in front of the schematic camera')
    return QPointF(x / z, y / z)


def depth_projection(point, tilt):
    x, y, z = point
    return QPointF(z, y) if tilt in (BoardTilt.TILT_UP, BoardTilt.TILT_DOWN) else QPointF(x, -z)


def oblique_projection(point):
    """Fixed orthographic basis: camera behind/below the board, no orbit animation."""
    x, y, z = point
    return QPointF(.8 * x + .6 * z, .3 * x + math.sqrt(.75) * y - .4 * z)


def _framing(rect, points):
    x0, x1 = min(p.x() for p in points), max(p.x() for p in points)
    y0, y1 = min(p.y() for p in points), max(p.y() for p in points)
    focal = min(rect.width() * .72 / max(x1 - x0, .01),
                rect.height() * .72 / max(y1 - y0, .01))
    center = rect.center() - QPointF((x0 + x1) / 2, (y0 + y1) / 2) * focal
    return center, focal


def board_framing(rect, point):
    """Fit the whole motion once per target, never auto-zoom within an animation."""
    points = [board_projection(point(x, y, t)) for t in (0., .25, .5, .75, 1.)
              for x, y in BOARD_CORNERS[:-1]]
    return _framing(rect, points)


def oblique_framing(rect, point):
    points = [oblique_projection(point(x, y, t)) for t in (0., .25, .5, .75, 1.)
              for x, y in BOARD_CORNERS[:-1]]
    points.extend(oblique_projection((x * .45, y * .45, z * .45))
                  for x in (-1.3, 1.3) for y in (-.42, .42) for z in (-.78, .68))
    return _framing(rect, points)


def _line(p, points, color, width=1., dashed=False, fill=None):
    p.setPen(QPen(QColor(color), width, Qt.DashLine if dashed else Qt.SolidLine,
                  Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(QColor(fill) if fill else Qt.NoBrush)
    polygon = QPolygonF(points)
    p.drawPolygon(polygon) if fill else p.drawPolyline(polygon)


def _arrow(p, a, b, color=WARNING):
    if math.hypot(b.x() - a.x(), b.y() - a.y()) < 4:
        return
    _line(p, [a, b], color, 1.6)
    angle = math.atan2(b.y() - a.y(), b.x() - a.x())
    for delta in (-.5, .5):
        _line(p, [b, b - QPointF(6 * math.cos(angle + delta), 6 * math.sin(angle + delta))], color, 1.6)


def _text(p, rect, text, color=TEXT_DIM, size=11, alignment=Qt.AlignCenter):
    font = QFont(p.font())
    font.setPixelSize(size)
    p.setFont(font)
    p.setPen(QColor(color))
    p.drawText(rect, alignment | Qt.TextWordWrap, text)


def view_rects(rect, compact):
    gap = 8 if compact else 12
    main_width = rect.width() * (.62 if compact else .67)
    return (QRectF(rect.x(), rect.y(), main_width - gap / 2, rect.height()),
            QRectF(rect.x() + main_width + gap / 2, rect.y(),
                   rect.width() - main_width - gap / 2, rect.height()))


def paint_board_view(p, rect, target, progress, point, compact=False, layer='dynamic',
                     depth_mode='oblique'):
    """Static background is cached by MotionCanvas; only board geometry animates."""
    main, depth = view_rects(QRectF(rect), compact)
    side = target is not None and target.tilt in (BoardTilt.TILT_UP, BoardTilt.TILT_DOWN)
    heading = 24 if compact else 30
    main_image = main.adjusted(10, heading, -10, -24)
    depth_image = depth.adjusted(12, heading + 8, -12, -28)
    center, focal = board_framing(main_image, point) if target is not None else (main_image.center(), 1.)
    oblique = depth_mode == 'oblique'
    depth_center, depth_scale = (oblique_framing(depth_image, point) if target is not None and oblique
                                 else (depth_image.center(), 1.))

    def front(x, y, t):
        uv = board_projection(point(x, y, t))
        return center + uv * focal

    def profile(x, y, t):
        if oblique:
            return depth_center + oblique_projection(point(x, y, t)) * depth_scale
        uv = depth_projection(point(x, y, t), target.tilt)
        if side:
            return QPointF(depth_image.left() + (uv.x() + 1) / 7 * depth_image.width(),
                           depth_image.center().y() + uv.y() / 5 * depth_image.height())
        return QPointF(depth_image.center().x() + uv.x() / 7 * depth_image.width(),
                       depth_image.bottom() + uv.y() / 7 * depth_image.height())

    if layer == 'static':
        for panel in (main, depth):
            p.setPen(QPen(QColor(BORDER), 1))
            p.setBrush(QColor(BG_MID))
            p.drawRoundedRect(panel.adjusted(1, 1, -1, -1), 6, 6)
        _text(p, QRectF(main.x() + 4, main.y(), main.width() - 8, heading),
              '相机视角 · 不镜像', TEXT, 11 if compact else 12)
        _text(p, QRectF(depth.x() + 4, depth.y(), depth.width() - 8, heading),
              ('位置示意 · 相机固定' if oblique else '侧视 · 镜头在左' if side else '俯视 · 镜头在下'),
              TEXT, 10 if compact else 12)
        if target is None:
            _text(p, main_image, '等待采图计划')
            return
        zone = ('中央' if (target.zone_row, target.zone_col) == (1, 1)
                else ('左', '', '右')[target.zone_col] + ('上', '', '下')[target.zone_row])
        _text(p, QRectF(main.x() + 6, main.bottom() - 22, main.width() - 12, 20),
              f'画面左  ←   {zone}目标   →  画面右', TEXT_DIM, 10)
        # The camera stays fixed in the depth panel; no metric baseline or FOV is implied.
        if oblique:
            p.save()
            p.setClipRect(depth_image)
            def camera_project(pt):
                return depth_center + oblique_projection(tuple(v * .45 for v in pt)) * depth_scale
            draw_stereo_camera(p, camera_project, (6., -6., -8.), show_roles=False)
            p.restore()
            _text(p, QRectF(depth.x() + 4, depth.bottom() - 22, depth.width() - 8, 20),
                  '固定视角 · 仅板面运动', BOARD_COLOR, 10)
            return
        camera = (QRectF(depth_image.left(), depth_image.center().y() - 8, 20, 16) if side
                  else QRectF(depth_image.center().x() - 18, depth_image.bottom() - 8, 36, 16))
        p.setPen(QPen(QColor(BOARD_COLOR), 1.5))
        p.setBrush(QColor('#263d47'))
        p.drawRoundedRect(camera, 3, 3)
        if side:
            p.drawRect(QRectF(camera.right(), camera.center().y() - 4, 6, 8))
            _arrow(p, QPointF(camera.right() + 8, camera.center().y()),
                   QPointF(camera.right() + 25, camera.center().y()), BOARD_COLOR)
        else:
            for offset in (-10, 6):
                p.drawRect(QRectF(camera.center().x() + offset, camera.top() - 5, 5, 5))
            _arrow(p, QPointF(camera.center().x(), camera.top() - 8),
                   QPointF(camera.center().x(), camera.top() - 23), BOARD_COLOR)
        _text(p, QRectF(depth.x() + 4, depth.bottom() - 22, depth.width() - 8, 20),
              '相机固定', BOARD_COLOR, 10)
        return
    if target is None:
        return
    corners = BOARD_CORNERS
    p.save()
    clip = QPainterPath()
    clip.addRect(main_image)
    clip.addRect(depth_image)
    p.setClipPath(clip)
    for projection in (front, profile):
        _line(p, [projection(x, y, 0.) for x, y in corners], START_COLOR, 1., True)
    # Plane cells follow the exact same rigid demonstration geometry as the edges.
    for row in range(5):
        for col in range(7):
            x0, x1 = -1.45 + col * 2.9 / 7, -1.45 + (col + 1) * 2.9 / 7
            y0, y1 = -1 + row * .4, -1 + (row + 1) * .4
            color = '#345a55' if (row + col) % 2 else '#203936'
            for projection in ((front, profile) if oblique else (front,)):
                cell = [projection(x, y, progress)
                        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
                _line(p, cell, '#527c76', .5, fill=color)
    _line(p, [front(x, y, progress) for x, y in corners], BOARD_COLOR, 1.8)
    _line(p, [profile(x, y, progress) for x, y in corners], BOARD_COLOR, 2.)
    for projection in (front, profile):
        _line(p, [projection(x, y, 1.) for x, y in corners], ACCENT, 1.4, True)
    top = front(0, -.72, progress)
    _text(p, QRectF(top.x() - 12, top.y() - 9, 24, 18), '上', TEXT, 11)
    edges = EDGES.get(target.tilt)
    if edges is not None:
        fixed, moving, _, _ = edges
        for projection in (front, profile):
            _line(p, [projection(x, y, progress) for x, y in moving], WARNING, 3.)
            pins = [projection(x, y, progress) for x, y in fixed]
            _line(p, pins, TEXT, 2.)
            p.setBrush(QColor(BG_MID))
            p.setPen(QPen(QColor(TEXT), 1.5))
            for pin in pins:
                p.drawEllipse(pin, 2.5, 2.5)
            mx, my = [(a + b) / 2 for a, b in zip(*moving)]
            _arrow(p, projection(mx, my, 0.), projection(mx, my, 1.))
    else:
        _arrow(p, front(0, 0, 0.), front(0, 0, 1.))
    p.restore()
