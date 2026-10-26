"""Shared qualitative motion: image X right, Y down, camera Z forward."""

import math
from enum import Enum, auto


class BoardTilt(Enum):
    FRONT = auto()
    TILT_LEFT = auto()
    TILT_RIGHT = auto()
    TILT_UP = auto()
    TILT_DOWN = auto()


TILT_ACTIONS = {
    BoardTilt.TILT_LEFT: ("左边缘靠近镜头", "保持右边缘不动，将左边缘朝镜头转近"),
    BoardTilt.TILT_RIGHT: ("右边缘靠近镜头", "保持左边缘不动，将右边缘朝镜头转近"),
    BoardTilt.TILT_UP: ("上边缘靠近镜头", "保持下边缘不动，将上边缘朝镜头转近"),
    BoardTilt.TILT_DOWN: ("下边缘靠近镜头", "保持上边缘不动，将下边缘朝镜头转近"),
}
TILT_LABELS = {BoardTilt.FRONT: "正面", **{k: v[0] for k, v in TILT_ACTIONS.items()}}


def tilt_point(x, y, progress, tilt):
    """Rotate about the opposite fixed edge; dimensions are illustrative."""
    z = 4.7
    if tilt in (BoardTilt.TILT_LEFT, BoardTilt.TILT_RIGHT):
        side = 1 if tilt == BoardTilt.TILT_RIGHT else -1
        pivot = -side * 1.45
        delta = x - pivot
        angle = progress * math.radians(30)
        x = pivot + delta * math.cos(angle)
        z -= side * delta * math.sin(angle)
    elif tilt in (BoardTilt.TILT_UP, BoardTilt.TILT_DOWN):
        side = 1 if tilt == BoardTilt.TILT_DOWN else -1
        pivot = -side
        delta = y - pivot
        angle = progress * math.radians(25)
        y = pivot + delta * math.cos(angle)
        z -= side * delta * math.sin(angle)
    return x, y, z
