"""Draw Chinese/Unicode on OpenCV BGR frames — batch-optimised.

``cv2.putText`` only supports Hershey glyphs; CJK renders as ``?``.
We render via QPainter + system CJK fonts.

**Key optimisation**: ``TextPainter`` converts BGR→QImage once, draws all
text strings, then converts back once — avoiding per-string full-frame copies.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter


_FONT_CACHE: dict[int, QFont] = {}


def _scale_to_pixel_size(scale: float) -> int:
    return max(10, min(40, int(round(32 * float(scale)))))


def _cjk_font(pixel_size: int) -> QFont:
    cached = _FONT_CACHE.get(pixel_size)
    if cached is not None:
        return cached
    f = QFont()
    f.setPixelSize(pixel_size)
    f.setFamilies(
        [
            "PingFang SC",
            "Hiragino Sans GB",
            "STHeiti",
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "Source Han Sans SC",
            "Noto Sans CJK SC",
            "sans-serif",
        ]
    )
    _FONT_CACHE[pixel_size] = f
    return f


# ═══════════════════════════════════════════════════════════════
#  Batch painter — convert once, draw many, write-back once
# ═══════════════════════════════════════════════════════════════

class TextPainter:
    """Context-manager that batches multiple text draws on one frame.

    Usage::

        with TextPainter(bgr_frame) as tp:
            tp.text(x, y, "角点: 5/126", scale=0.45)
            tp.text(x2, y2, "覆盖率 80%", scale=0.50, color=(0,255,200))
    """

    def __init__(self, img: np.ndarray):
        self._img = img
        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self._qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._painter = QPainter(self._qimg)
        self._painter.setRenderHint(QPainter.TextAntialiasing)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self._painter.end()
        qimg = self._qimg.convertToFormat(QImage.Format.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        bpl = qimg.bytesPerLine()
        ptr = qimg.constBits()
        total = qimg.sizeInBytes()
        rgb = np.frombuffer(memoryview(ptr)[:total], dtype=np.uint8).reshape(h, bpl)
        rgb = rgb[:, : w * 3].reshape(h, w, 3)
        self._img[:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def text(
        self,
        x: int,
        y: int,
        text: str,
        *,
        scale: float = 0.5,
        color: tuple[int, int, int] = (255, 255, 255),
        bg: tuple[int, int, int] | None = (0, 0, 0),
        anchor_right: bool = False,
        anchor_center: bool = False,
    ) -> None:
        if not text:
            return
        px = _scale_to_pixel_size(scale)
        font = _cjk_font(px)
        self._painter.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(text)
        asc = fm.ascent()
        desc = fm.descent()

        bx = x
        if anchor_right:
            bx = x - tw
        elif anchor_center:
            bx = x - tw // 2

        pad = 4
        if bg is not None:
            self._painter.fillRect(
                QRect(int(bx - pad), int(y - asc - pad), int(tw + 2 * pad), int(asc + desc + 2 * pad)),
                QColor(bg[2], bg[1], bg[0]),
            )
        self._painter.setPen(QColor(color[2], color[1], color[0]))
        self._painter.drawText(int(bx), int(y), text)


# ═══════════════════════════════════════════════════════════════
#  Legacy one-shot helpers (kept for call-sites that draw once)
# ═══════════════════════════════════════════════════════════════

def draw_text_baseline_bgr(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] | None = (0, 0, 0),
    anchor_right: bool = False,
    anchor_center: bool = False,
) -> None:
    if not text:
        return
    with TextPainter(img) as tp:
        tp.text(x, y, text, scale=scale, color=color, bg=bg,
                anchor_right=anchor_right, anchor_center=anchor_center)


def draw_text_with_bg_bgr(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
) -> None:
    del thickness
    if not text:
        return
    with TextPainter(img) as tp:
        tp.text(x, y, text, scale=scale, color=color, bg=bg)
