"""Generic planar calibration board detection for auxiliary mono cameras."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import cv2
import numpy as np


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

    @property
    def num_points(self) -> int:
        return int(len(self.image_points))


class PlanarPatternDetector:
    """Detect chessboard or circles-grid planar board points."""

    def __init__(self, config: PatternBoardConfig):
        self.config = config
        self._object_points = self._make_object_points(config)

    def detect(self, image: np.ndarray) -> PlanarDetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        image_size = (int(gray.shape[1]), int(gray.shape[0]))
        pattern_size = (self.config.cols, self.config.rows)

        if self.config.pattern_type == PatternType.CHESSBOARD:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        else:
            flags = (
                cv2.CALIB_CB_ASYMMETRIC_GRID
                if self.config.asymmetric
                else cv2.CALIB_CB_SYMMETRIC_GRID
            )
            found, corners = cv2.findCirclesGrid(gray, pattern_size, flags=flags)

        if not found or corners is None:
            return PlanarDetectionResult(
                valid=False,
                object_points=np.empty((0, 3), dtype=np.float32),
                image_points=np.empty((0, 2), dtype=np.float32),
                image_size=image_size,
                pattern_type=self.config.pattern_type,
            )

        return PlanarDetectionResult(
            valid=True,
            object_points=self._object_points.copy(),
            image_points=corners.reshape(-1, 2).astype(np.float32),
            image_size=image_size,
            pattern_type=self.config.pattern_type,
        )

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
