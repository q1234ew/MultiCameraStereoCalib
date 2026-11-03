"""Image encoding with Python filesystem I/O for Unicode paths on Windows."""

from pathlib import Path

import cv2
import numpy as np


def write_png(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ok, encoded = cv2.imencode('.png', image)
    if not ok or encoded is None or encoded.size == 0:
        raise OSError(f"无法编码图片：{path.name}")
    payload = encoded.tobytes()
    if path.write_bytes(payload) != len(payload):
        raise OSError(f"图片写入不完整：{path.name}")
