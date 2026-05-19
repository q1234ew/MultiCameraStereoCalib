"""ChArUco board definition, generation, and image export."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np


class ArucoDictType(IntEnum):
    DICT_4X4_50 = cv2.aruco.DICT_4X4_50
    DICT_4X4_100 = cv2.aruco.DICT_4X4_100
    DICT_4X4_250 = cv2.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv2.aruco.DICT_4X4_1000
    DICT_5X5_50 = cv2.aruco.DICT_5X5_50
    DICT_5X5_100 = cv2.aruco.DICT_5X5_100
    DICT_5X5_250 = cv2.aruco.DICT_5X5_250
    DICT_5X5_1000 = cv2.aruco.DICT_5X5_1000
    DICT_6X6_50 = cv2.aruco.DICT_6X6_50
    DICT_6X6_100 = cv2.aruco.DICT_6X6_100
    DICT_6X6_250 = cv2.aruco.DICT_6X6_250
    DICT_6X6_1000 = cv2.aruco.DICT_6X6_1000


@dataclass
class CharucoBoardConfig:
    """Configuration for a ChArUco calibration board.

    cols/rows: number of *squares* (not inner corners).
    square_length: physical side length of a chessboard square (metres).
    marker_length: physical side length of an ArUco marker (metres).
    """

    cols: int = 15
    rows: int = 10
    square_length: float = 0.04
    marker_length: float = 0.03
    dict_type: ArucoDictType = ArucoDictType.DICT_4X4_1000


class CharucoBoard:
    """Wrapper around cv2.aruco.CharucoBoard with helper utilities."""

    def __init__(self, config: CharucoBoardConfig):
        self.config = config
        self._dictionary = cv2.aruco.getPredefinedDictionary(config.dict_type.value)
        self._board = cv2.aruco.CharucoBoard(
            (config.cols, config.rows),
            config.square_length,
            config.marker_length,
            self._dictionary,
        )

    @property
    def board(self) -> cv2.aruco.CharucoBoard:
        return self._board

    @property
    def dictionary(self):
        return self._dictionary

    @property
    def num_corners(self) -> int:
        return (self.config.cols - 1) * (self.config.rows - 1)

    @property
    def object_points(self) -> np.ndarray:
        return self._board.getChessboardCorners()

    def generate_image(
        self,
        pixel_size: int = 200,
        margin: int = 20,
    ) -> np.ndarray:
        """Render a high-resolution board image for printing."""
        img_w = self.config.cols * pixel_size + 2 * margin
        img_h = self.config.rows * pixel_size + 2 * margin
        return self._board.generateImage((img_w, img_h), marginSize=margin)

    def save_image(
        self,
        path: str | Path,
        pixel_size: int = 200,
        margin: int = 20,
    ) -> Path:
        path = Path(path)
        img = self.generate_image(pixel_size, margin)
        cv2.imwrite(str(path), img)
        return path

    def to_dict(self) -> dict:
        cfg = self.config
        return {
            "cols": cfg.cols,
            "rows": cfg.rows,
            "square_length": cfg.square_length,
            "marker_length": cfg.marker_length,
            "dict_type": cfg.dict_type.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CharucoBoard":
        cfg = CharucoBoardConfig(
            cols=d["cols"],
            rows=d["rows"],
            square_length=d["square_length"],
            marker_length=d["marker_length"],
            dict_type=ArucoDictType[d["dict_type"]],
        )
        return cls(cfg)
