"""Reversible image-grid numbering, never cross-camera physical correspondence.

Only row/column reversals are considered. Square-grid transposes and 90-degree
axis swaps are intentionally not identified with these four permutations. A
canonical image origin does not establish a shared physical board origin.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .planar import PatternBoardConfig, PlanarDetectionResult


_ORDERS = ('identity', 'reverse_rows', 'reverse_cols', 'reverse_both')


def _dimensions(rows, cols):
    for value in (rows, cols):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not 2 <= value <= 100):
            raise ValueError('Grid dimensions must be integer point counts between 2 and 100')
    return int(rows), int(cols)


def _points(value, count, dimensions):
    array = np.asarray(value)
    if (array.shape != (count, dimensions) or array.dtype.kind not in 'fiu'
            or not np.isfinite(array).all()):
        raise ValueError('Expected a complete finite numeric grid')
    return array


def _permuted(points, rows, cols, order):
    if order not in _ORDERS:
        raise ValueError('Unsupported grid order; transposes are not row/column reversals')
    grid = points.reshape(rows, cols, -1)
    if order in ('reverse_rows', 'reverse_both'):
        grid = grid[::-1]
    if order in ('reverse_cols', 'reverse_both'):
        grid = grid[:, ::-1]
    return grid.reshape(points.shape).copy()


def allowed_orders(object_points: np.ndarray, rows: int, cols: int) -> list[str]:
    """Return grid reversals realizable by a proper rigid 3D transformation.

    Kabsch SVD fits each permutation with no scaling or point-distance matrix;
    working memory is linear in the number of points (including 100 x 100 boards).
    A planar reflection can be a proper 180-degree rotation about an in-plane axis.
    """
    rows, cols = _dimensions(rows, cols)
    objects = _points(object_points, rows * cols, 3).astype(np.float64)
    if len(np.unique(objects, axis=0)) != len(objects):
        raise ValueError('Object grid contains duplicate points')
    centered = objects - objects.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-10:
        raise ValueError('Object grid is degenerate')
    tolerance = float(np.max(np.linalg.norm(centered, axis=1))) * 1e-6
    orders = []
    for order in _ORDERS:
        target = _permuted(centered, rows, cols, order)
        u, _, vt = np.linalg.svd(centered.T @ target)
        # Enforce det(R)=+1: a mirror of a nonplanar grid is not a rigid rotation.
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        rotation = u @ vt
        if np.max(np.linalg.norm(centered @ rotation - target, axis=1)) <= tolerance:
            orders.append(order)
    return orders


def reorder_points(image_points: np.ndarray, rows: int, cols: int, order: str) -> np.ndarray:
    """Copy and permute an N x 2 grid without changing any coordinate values.

    Every supported order is its own inverse. This primitive does not establish
    whether an order is physically allowed; use allowed_orders for that check.
    """
    rows, cols = _dimensions(rows, cols)
    return _permuted(_points(image_points, rows * cols, 2), rows, cols, order)


def _fingerprint(detection):
    try:
        digest = hashlib.sha256(str(tuple(detection.image_size)).encode('ascii'))
        for points in (detection.object_points, detection.image_points):
            array = np.asarray(points, dtype='<f8')
            digest.update(str(array.shape).encode('ascii'))
            digest.update(array.tobytes(order='C'))
        return digest.hexdigest()
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError):
        return None


def canonicalize_detection(detection: PlanarDetectionResult,
                           board: PatternBoardConfig) -> PlanarDetectionResult:
    """Copy a complete observation into a stable image-only grid convention.

    Minimize x/w+y/h of point zero, then its y and x to break ties. Coordinates,
    image pixels and the board object frame are never adjusted. Malformed or
    incomplete observations remain invalid and receive no normalization order.
    """
    from .planar import PlanarPatternDetector

    result = replace(detection, object_points=np.array(detection.object_points, copy=True),
                     image_points=np.array(detection.image_points, copy=True),
                     detection_info=deepcopy(detection.detection_info))
    info = result.detection_info
    raw_fingerprint = _fingerprint(result)
    try:
        rows, cols = _dimensions(board.rows, board.cols)
        if not detection.valid or detection.pattern_type != board.pattern_type:
            raise ValueError('Invalid detection or mismatched board pattern')
        size = np.asarray(detection.image_size)
        if (size.shape != (2,) or size.dtype.kind not in 'fiu' or not np.isfinite(size).all()
                or np.any(size <= 0) or np.any(size != np.floor(size))):
            raise ValueError('Image size must contain positive integer dimensions')
        points = _points(result.image_points, rows * cols, 2)
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError('Image grid contains duplicate points')
        objects = _points(result.object_points, rows * cols, 3)
        expected = PlanarPatternDetector._make_object_points(board)
        scale = float(np.max(np.abs(expected)))
        if not np.allclose(objects, expected, rtol=1e-6, atol=scale * 1e-7):
            raise ValueError('Object points do not match the current board')
        orders = allowed_orders(objects, rows, cols)
        if (info.get('normalization_version') == 1 and info.get('scope') == 'image_grid_only'
                and info.get('normalization_order') in orders
                and info.get('normalized_fingerprint') == raw_fingerprint
                and info.get('physical_correspondence_verified') is False):
            return result
        candidates = [(order, reorder_points(points, rows, cols, order)) for order in orders]
        order, normalized = min(candidates, key=lambda item: (
            float(item[1][0, 0]) / size[0] + float(item[1][0, 1]) / size[1],
            float(item[1][0, 1]), float(item[1][0, 0])))
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        info.update(normalization_order=None, normalization_version=1,
                    normalization_status='not_applied', normalization_reason=str(exc),
                    raw_fingerprint=raw_fingerprint, normalized_fingerprint=raw_fingerprint,
                    scope='image_grid_only', physical_correspondence_verified=False)
        return replace(result, valid=False)
    result.image_points = normalized
    info.pop('normalization_reason', None)
    info.update(normalization_order=order, normalization_version=1,
                normalization_status='applied', raw_fingerprint=raw_fingerprint,
                normalized_fingerprint=_fingerprint(result),
                scope='image_grid_only', physical_correspondence_verified=False)
    return result
