"""Small schematic stereo mesh, painted only when the scene cache is rebuilt."""

from dataclasses import dataclass
from functools import lru_cache
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPen, QPolygonF

from ..theme import ACCENT, TEXT_DIM


LENS_CENTERS = ((-.69, 0., .66), (.69, 0., .66))


@dataclass(frozen=True)
class CameraFace:
    points: tuple
    color: str
    part: str

    @property
    def normal(self):
        a, b, c = self.points[:3]
        u, v = tuple(y - x for x, y in zip(a, b)), tuple(y - x for x, y in zip(a, c))
        return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0])

    @property
    def center(self):
        return tuple(sum(p[i] for p in self.points) / len(self.points) for i in range(3))


def ring(center, radius, segments=32):
    x, y, z = center
    return tuple((x + radius * math.cos(i * math.tau / segments),
                  y + radius * math.sin(i * math.tau / segments), z) for i in range(segments))


@lru_cache(maxsize=1)
def camera_faces():
    """One chamfered body and two forward-pointing barrels; no metric rig data."""
    profile = ((-1.18, -.42), (1.18, -.42), (1.3, -.3), (1.3, .3),
               (1.18, .42), (-1.18, .42), (-1.3, .3), (-1.3, -.3))
    rear = tuple((x, y, -.78) for x, y in profile)
    front = tuple((x, y, 0.) for x, y in profile)
    faces = [CameraFace(front, '#344550', 'body_front'),
             CameraFace(rear[::-1], '#202c34', 'body_rear')]
    for i in range(len(profile)):
        j = (i + 1) % len(profile)
        color = '#26333d' if i in (0, 1, 2) else '#2d3c47'
        faces.append(CameraFace((rear[i], rear[j], front[j], front[i]), color, 'body_side'))
    for x, y, z in LENS_CENTERS:
        back, tip = ring((x, y, .03), .32), ring((x, y, z), .35)
        for i in range(len(back)):
            j = (i + 1) % len(back)
            faces.append(CameraFace((back[i], back[j], tip[j], tip[i]), '#273946', 'barrel'))
        faces.append(CameraFace(tip, '#586b78', 'lens_rim'))
    return tuple(faces)


def visible_faces(eye):
    """Cull rear-facing surfaces and order far-to-near along this fixed view."""
    faces = [face for face in camera_faces() if sum(
        n * (e - c) for n, e, c in zip(face.normal, eye, face.center)
    ) > 1e-9]
    forward = (-eye[0], -eye[1], 2.6 - eye[2])
    return sorted(faces, key=lambda face: sum(
        (c - e) * f for c, e, f in zip(face.center, eye, forward)
    ), reverse=True)


def draw_stereo_camera(painter, project, eye, *, show_roles=True):
    """Draw hardware silhouette, optical axes and role badges without a 3D engine."""
    painter.save()

    def path(points, color, width=1., fill=None, dashed=False):
        painter.setPen(QPen(QColor(color), width, Qt.DashLine if dashed else Qt.SolidLine,
                            Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QColor(fill) if fill else Qt.NoBrush)
        polygon = QPolygonF([project(point) for point in points])
        if fill:
            painter.drawPolygon(polygon)
        else:
            painter.drawPolyline(polygon)

    # Parallel axes indicate direction only: no guessed FOV or real baseline.
    for x, y, z in LENS_CENTERS:
        path(((x, y, z), (x, y, 3.1)), '#557b8d', .8, dashed=True)
        a, b = project((x, y, 2.75)), project((x, y, 3.1))
        angle = math.atan2(b.y() - a.y(), b.x() - a.x())
        painter.setPen(QPen(QColor('#729caf'), 1.))
        for delta in (-.45, .45):
            painter.drawLine(b, b - QPointF(5 * math.cos(angle + delta), 5 * math.sin(angle + delta)))

    for face in visible_faces(eye):
        path(face.points, '#526c7c' if face.part == 'barrel' else '#8b9fac',
             .45 if face.part == 'barrel' else .85, face.color)

    for x, y, z in LENS_CENTERS:
        path(ring((x, y, z + .004), .285), '#1d2d39', .8, '#101d27')
        path(ring((x, y, z + .008), .205), ACCENT, 1., '#183d50')
        path(ring((x, y, z + .012), .115), '#51899e', .65, '#102b3b')
        path(ring((x - .065, y - .06, z + .016), .035, 12), '#9bcad5', .4, '#9bcad5')

    if not show_roles:
        painter.restore()
        return
    # Keep role badges in screen-space so compact layouts do not shrink the text.
    transform = painter.worldTransform()
    anchors = [transform.map(project((x, .49, .08))) for x, _, _ in LENS_CENTERS]
    painter.resetTransform()
    painter.setPen(QColor(TEXT_DIM))
    font = QFont(painter.font())
    font.setPixelSize(10)
    painter.setFont(font)
    for point, label in zip(anchors, ('L', 'R')):
        painter.drawText(QRectF(point.x() - 9, point.y(), 18, 15), Qt.AlignCenter, label)
    painter.restore()
