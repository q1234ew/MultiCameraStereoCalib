"""Generic planar calibration board detection for auxiliary mono cameras."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Tuple

import cv2
import numpy as np

from .grid_order import canonicalize_detection


class PatternType(str, Enum):
    CHESSBOARD = "chessboard"
    CIRCLES_GRID = "circles_grid"


@dataclass
class PatternBoardConfig:
    """Planar OpenCV calibration board configuration.

    ``cols`` and ``rows`` are detected point counts, not physical square counts.
    ``square_size`` is measured in metres.
    """

    cols: int = 9
    rows: int = 6
    square_size: float = 0.04
    pattern_type: PatternType = PatternType.CHESSBOARD
    asymmetric: bool = False

    def __post_init__(self):
        if not isinstance(self.pattern_type, PatternType):
            self.pattern_type = PatternType(self.pattern_type)
        for name in ("cols", "rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be an integer point count")
            if not math.isfinite(value) or int(value) != value or not 2 <= value <= 100:
                raise ValueError(f"{name} must be an integer between 2 and 100")
            setattr(self, name, int(value))
        if self.pattern_type == PatternType.CHESSBOARD and min(self.cols, self.rows) < 3:
            raise ValueError("Chessboards need at least 3 inner corners in each direction")
        if (isinstance(self.square_size, bool) or
                not isinstance(self.square_size, (int, float)) or
                not math.isfinite(self.square_size) or not 0 < self.square_size <= 10):
            raise ValueError("square_size must be a positive spacing in metres (at most 10 m)")
        if not isinstance(self.asymmetric, bool):
            raise ValueError("asymmetric must be true or false")
        if self.asymmetric and self.pattern_type != PatternType.CIRCLES_GRID:
            raise ValueError("asymmetric is only valid for circles_grid")

    @classmethod
    def ir_chessboard(cls) -> "PatternBoardConfig":
        """Confirmed IR board: 7 x 5 squares, 6 x 4 inner corners, 80 mm spacing."""
        return cls(cols=6, rows=4, square_size=0.08)

    def to_dict(self) -> dict:
        return {
            "cols": self.cols, "rows": self.rows, "square_size": self.square_size,
            "pattern_type": self.pattern_type.value, "asymmetric": self.asymmetric,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatternBoardConfig":
        if not isinstance(data, dict):
            raise ValueError("Board configuration must be an object")
        # Legacy ustreamer-ir fields are inner-corner counts and millimetres.
        def count(key, legacy):
            value = data.get(key, data.get(legacy))
            if value is None:
                raise ValueError(f"Missing {key}/{legacy} (detected point count)")
            if key in data and legacy in data and data[key] != data[legacy]:
                raise ValueError(f"Conflicting {key} and {legacy}")
            return value

        spacing = data.get("square_size")
        if "square_mm" in data:
            mm = data["square_mm"]
            if isinstance(mm, bool) or not isinstance(mm, (int, float)):
                raise ValueError("square_mm must be numeric")
            metres = mm / 1000.0
            if spacing is not None and (
                isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or
                not math.isclose(spacing, metres, rel_tol=1e-9, abs_tol=1e-12)
            ):
                raise ValueError("Conflicting square_size (m) and square_mm")
            spacing = metres
        if spacing is None:
            raise ValueError("Missing square_size (m) or square_mm (mm)")
        return cls(
            cols=count("cols", "board_cols"), rows=count("rows", "board_rows"),
            square_size=spacing, pattern_type=data.get("pattern_type", "chessboard"),
            asymmetric=data.get("asymmetric", False),
        )

    @classmethod
    def from_pattern_name(cls, pattern_name: str, **kwargs) -> "PatternBoardConfig":
        return cls(pattern_type=PatternType(pattern_name), **kwargs)


@dataclass
class PlanarDetectionResult:
    valid: bool
    object_points: np.ndarray
    image_points: np.ndarray
    image_size: Tuple[int, int]
    pattern_type: PatternType
    detection_info: dict = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        return int(len(self.image_points))


class PlanarPatternDetector:
    """Detect chessboard or circles-grid planar board points."""

    def __init__(self, config: PatternBoardConfig):
        self.config = config
        self._object_points = self._make_object_points(config)

    def detect(self, image: np.ndarray, *, adaptive_subpixel: bool = False) -> PlanarDetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        image_size = (int(gray.shape[1]), int(gray.shape[0]))
        pattern_size = (self.config.cols, self.config.rows)

        if self.config.pattern_type == PatternType.CHESSBOARD:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
            if found and corners is not None and corners.size == self.config.rows * self.config.cols * 2:
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                radius = 11
                if adaptive_subpixel:
                    grid = corners.reshape(self.config.rows, self.config.cols, 2)
                    spacing = min(np.median(np.linalg.norm(np.diff(grid, axis=0), axis=2)),
                                  np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)))
                    radius = max(1, min(11, int(spacing * .25)))
                corners = cv2.cornerSubPix(gray, corners, (radius, radius), (-1, -1), criteria)
        else:
            flags = (
                cv2.CALIB_CB_ASYMMETRIC_GRID
                if self.config.asymmetric
                else cv2.CALIB_CB_SYMMETRIC_GRID
            )
            found, corners = cv2.findCirclesGrid(gray, pattern_size, flags=flags)

        if not found or corners is None:
            return canonicalize_detection(PlanarDetectionResult(
                valid=False,
                object_points=np.empty((0, 3), dtype=np.float32),
                image_points=np.empty((0, 2), dtype=np.float32),
                image_size=image_size,
                pattern_type=self.config.pattern_type,
                detection_info={'method': 'classic_native', 'coordinate_space': 'original_pixels',
                    'source_detector': ('cv2.findChessboardCorners' if self.config.pattern_type ==
                                        PatternType.CHESSBOARD else 'cv2.findCirclesGrid')},
            ), self.config)

        return canonicalize_detection(PlanarDetectionResult(
            valid=True,
            object_points=self._object_points.copy(),
            image_points=corners.reshape(-1, 2).astype(np.float32),
            image_size=image_size,
            pattern_type=self.config.pattern_type,
            detection_info={'method': 'classic_native', 'coordinate_space': 'original_pixels',
                'source_detector': ('cv2.findChessboardCorners' if self.config.pattern_type ==
                                    PatternType.CHESSBOARD else 'cv2.findCirclesGrid')},
        ), self.config)

    def detect_offline(self, image: np.ndarray, *, modality: str = 'rgb'):
        """Quality-first native detection, then bounded board-region fallbacks.

        Selection never uses calibration RMS. Quality is always measured on the
        original image; enlargement adds no new physical observations.
        """
        from .aux_capture_quality import assess_aux_frame

        if self.config.pattern_type != PatternType.CHESSBOARD:
            return self.detect(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        native = [('sb_native', 1, None, False), ('sb_inverse', 1, None, True)]
        variants = native + [('classic_native', 1, None, False)] if modality == 'ir' else [
            ('classic_native', 1, None, False), *native]
        if modality == 'ir':
            roi = self._thermal_roi(gray)
            if roi is not None:
                variants.extend([('sb_roi_native', 1, roi, False),
                                 ('sb_roi_inverse_native', 1, roi, True)])
            variants.append(('sb_2x', 2, None, False))
            if roi is not None:
                variants.extend([('sb_roi_2x', 2, roi, False),
                                 ('sb_roi_inverse_2x', 2, roi, True)])
        fallback = None
        attempts = []
        for method, scale, roi, inverted in variants:
            if method == 'classic_native':
                result = self.detect(image, adaptive_subpixel=True)
            else:
                result = self._detect_sb(gray, scale, roi, inverted)
            result.detection_info.update(method=method, scale=scale, roi=roi, inverted=inverted,
                coordinate_space='original_pixels', pixel_mapping='(p+0.5)/scale-0.5+offset')
            quality = assess_aux_frame(image, result, self.config, modality=modality)
            attempts.append({'method': method, 'valid': result.valid,
                             'image_accepted': quality.accepted})
            if fallback is None or (result.valid and not fallback.valid):
                fallback = result
            if quality.accepted:
                result.detection_info['attempts'] = attempts
                return result
        # Locate afresh on this image; never reuse a region from a different frame.
        # These proposals only restrict search. Detection and admission still use
        # native pixels, the full board, and quality measured on the original.
        if modality == 'rgb':
            for roi in self._rgb_board_rois(gray):
                result = self._detect_sb(gray, 1, roi, False)
                result.detection_info.update(method='sb_rgb_roi_native')
                quality = assess_aux_frame(image, result, self.config, modality=modality)
                attempts.append({'method': 'sb_rgb_roi_native', 'roi': roi,
                                 'valid': result.valid, 'image_accepted': quality.accepted})
                if fallback is None or (result.valid and not fallback.valid):
                    fallback = result
                if quality.accepted:
                    result.detection_info['attempts'] = attempts
                    return result
        fallback.detection_info['attempts'] = attempts
        return fallback

    def _detect_sb(self, gray, scale, roi, inverted):
        x, y, w, h = roi if roi is not None else (0, 0, gray.shape[1], gray.shape[0])
        work = gray[y:y+h, x:x+w]
        if inverted:
            work = 255 - work
        if scale != 1:
            work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(
            work, (self.config.cols, self.config.rows), flags=flags)
        found = bool(found and corners is not None)
        points = (self.map_scaled_points(corners, scale, (x, y)) if found else
                  np.empty((0, 2), np.float32))
        method = 'sb' + ('_roi' if roi is not None else '') + ('_inverse' if inverted else '')
        method += f'_{scale}x' if scale != 1 else ('_native' if roi is None and not inverted else '')
        return canonicalize_detection(PlanarDetectionResult(
            found, self._object_points.copy(), points, gray.shape[1::-1], self.config.pattern_type,
            detection_info=dict(method=method, source_detector='cv2.findChessboardCornersSB',
                scale=scale, roi=roi, inverted=inverted, coordinate_space='original_pixels',
                pixel_mapping='(p+0.5)/scale-0.5+offset')), self.config)

    @staticmethod
    def map_scaled_points(points, scale, offset=(0, 0)):
        return ((np.asarray(points).reshape(-1, 2) + .5) / scale - .5
                + np.asarray(offset)).astype(np.float32)

    @staticmethod
    def _thermal_roi(gray):
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes = [box for box in boxes if box[2] >= 20 and box[3] >= 20]
        if not boxes:
            return None
        x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
        x0, y0 = max(0, x - 12), max(0, y - 12)
        return x0, y0, min(gray.shape[1], x+w+12)-x0, min(gray.shape[0], y+h+12)-y0

    def _rgb_board_rois(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Bounded proposals from nearby square-like cells, not corner observations.

        A reduced image is used only to locate candidate regions. SB receives
        crops of the untouched original, and maps corners back to that image.
        Distractor regions cannot bypass the independent full-grid quality gate.
        """
        height, width = gray.shape
        scale = min(1., 960. / max(height, width))
        work = (cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                if scale < 1 else gray)
        if min(work.shape) < 32:
            return []
        mask = cv2.adaptiveThreshold(work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cells = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 16 <= area <= work.size * .12:
                continue
            quad = cv2.approxPolyDP(contour, .04 * cv2.arcLength(contour, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            x, y, w, h = cv2.boundingRect(quad)
            if min(w, h) >= 4 and max(w, h) <= 4 * min(w, h) and area > .5 * w * h:
                cells.append((x, y, w, h))
        # Bound grouping cost even for heavily textured backgrounds.
        cells = sorted(cells, key=lambda b: b[2] * b[3], reverse=True)[:256]
        if not cells:
            return []
        boxes = np.asarray(cells, dtype=float)
        centres = boxes[:, :2] + boxes[:, 2:] / 2
        sizes = np.sqrt(boxes[:, 2] * boxes[:, 3])
        distances = np.linalg.norm(centres[:, None] - centres[None, :], axis=2)
        ratios = sizes[:, None] / sizes[None, :]
        neighbours = ((distances <= 1.5 * (sizes[:, None] + sizes[None, :])) &
                      (ratios >= .25) & (ratios <= 4))
        unseen = set(range(len(cells)))
        groups = []
        while unseen:
            pending = [min(unseen)]
            group = []
            while pending:
                index = pending.pop()
                if index not in unseen:
                    continue
                unseen.remove(index)
                group.append(index)
                pending.extend(i for i in np.flatnonzero(neighbours[index]) if i in unseen)
            if len(group) >= max(4, min(12, self.config.rows * self.config.cols // 4)):
                groups.append(group)
        rois = []
        for group in sorted(groups, key=len, reverse=True)[:3]:
            selected = boxes[group]
            padding = 1.25 * float(np.median(sizes[group]))
            low = np.floor((selected[:, :2].min(axis=0) - padding) / scale).astype(int)
            high = np.ceil((selected[:, :2] + selected[:, 2:]).max(axis=0) / scale
                           + padding / scale).astype(int)
            x0, y0 = max(0, int(low[0])), max(0, int(low[1]))
            x1, y1 = min(width, int(high[0])), min(height, int(high[1]))
            roi = (x0, y0, x1 - x0, y1 - y0)
            if roi != (0, 0, width, height) and roi not in rois:
                rois.append(roi)
        return rois

    @staticmethod
    def _make_object_points(config: PatternBoardConfig) -> np.ndarray:
        points = np.zeros((config.rows * config.cols, 3), dtype=np.float32)
        idx = 0
        for row in range(config.rows):
            for col in range(config.cols):
                if config.pattern_type == PatternType.CIRCLES_GRID and config.asymmetric:
                    x = (2 * col + row % 2) * config.square_size
                    y = row * config.square_size
                else:
                    x = col * config.square_size
                    y = row * config.square_size
                points[idx] = (x, y, 0.0)
                idx += 1
        return points
