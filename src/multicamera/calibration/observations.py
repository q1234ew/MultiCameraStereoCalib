"""Validate ChArUco observations before they enter a calibration dataset."""

from __future__ import annotations

import numpy as np

from ..board.charuco_board import CharucoBoard
from ..board.detector import DetectionResult


def similar_composition(corners_a, ids_a, size_a, corners_b, ids_b, size_b):
    """Require sub-board-scale displacement, not merely a small image displacement."""
    if tuple(size_a) != tuple(size_b):
        return False
    _, a, b = np.intersect1d(np.asarray(ids_a).ravel(), np.asarray(ids_b).ravel(),
                            return_indices=True)
    if len(a) < max(6, .6 * max(len(ids_a), len(ids_b))):
        return False
    pa, pb = np.asarray(corners_a).reshape(-1, 2)[a], np.asarray(corners_b).reshape(-1, 2)[b]
    if not np.isfinite(pa).all() or not np.isfinite(pb).all():
        return False
    footprint = min(np.linalg.norm(np.ptp(pa, axis=0)), np.linalg.norm(np.ptp(pb, axis=0)))
    threshold = min(.005 * max(size_a), .006 * footprint)
    rms = np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=1)))
    return bool(threshold > 0 and rms < threshold)


def prepare_observation(
    detection: DetectionResult,
    board: CharucoBoard,
    expected_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    size = detection.image_size
    if (not isinstance(size, (tuple, list)) or len(size) != 2 or any(isinstance(v, (bool, np.bool_)) or
                            not isinstance(v, (int, np.integer)) or v <= 0 for v in size)):
        raise ValueError("图像分辨率必须是正整数宽、高")
    size = tuple(int(v) for v in size)
    if expected_size is not None and size != expected_size:
        raise ValueError(f"图像分辨率 {size} 与已有观测 {expected_size} 不一致")
    corners, ids = np.asarray(detection.charuco_corners), np.asarray(detection.charuco_ids)
    if corners.ndim not in (2, 3) or corners.shape[1:] not in ((2,), (1, 2)):
        raise ValueError("角点坐标形状无效")
    if ids.ndim not in (1, 2) or (ids.ndim == 2 and ids.shape[1] != 1):
        raise ValueError("角点 ID 形状无效")
    if len(corners) != len(ids):
        raise ValueError("角点坐标与 ID 数量不一致")
    if corners.dtype.kind not in 'fiu' or not np.isfinite(corners).all():
        raise ValueError("角点坐标包含非有限数值")
    if np.any(np.abs(corners) > np.finfo(np.float32).max):
        raise ValueError("角点坐标超出标定浮点范围")
    if (not np.issubdtype(ids.dtype, np.integer) or
            np.any(ids < 0) or np.any(ids >= board.num_corners)):
        raise ValueError("角点 ID 超出当前标定板范围或不是整数")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("角点 ID 重复")
    return (np.array(corners.reshape(-1, 1, 2), dtype=np.float32, order="C", copy=True),
            np.array(ids.reshape(-1, 1), dtype=np.int32, order="C", copy=True), size)
