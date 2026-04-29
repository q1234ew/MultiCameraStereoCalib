"""ChArUco corner detection with sub-pixel refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .charuco_board import CharucoBoard


@dataclass
class DetectionResult:
    """Result of ChArUco detection on a single frame."""

    charuco_corners: Optional[np.ndarray]  # (N, 1, 2) float32
    charuco_ids: Optional[np.ndarray]  # (N, 1) int32
    marker_corners: list  # raw ArUco marker corners
    marker_ids: Optional[np.ndarray]
    image_size: Tuple[int, int]  # (width, height)

    @property
    def valid(self) -> bool:
        return (
            self.charuco_corners is not None
            and self.charuco_ids is not None
            and len(self.charuco_ids) >= 4
        )

    @property
    def num_corners(self) -> int:
        if self.charuco_ids is None:
            return 0
        return len(self.charuco_ids)


class CharucoDetector:
    """Detects ChArUco board corners in images."""

    def __init__(self, board: CharucoBoard):
        self.board = board

        self._detector_params = cv2.aruco.DetectorParameters()
        self._detector_params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
        )

        charuco_params = cv2.aruco.CharucoParameters()
        self._detector = cv2.aruco.CharucoDetector(
            board.board,
            charuco_params,
            self._detector_params,
        )

    def detect(self, image: np.ndarray) -> DetectionResult:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape[:2]

        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            self._detector.detectBoard(gray)
        )

        return DetectionResult(
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
            marker_corners=marker_corners if marker_corners else [],
            marker_ids=marker_ids,
            image_size=(w, h),
        )

    def draw_detected(
        self,
        image: np.ndarray,
        result: DetectionResult,
    ) -> np.ndarray:
        """Draw detected corners on an image copy."""
        vis = image.copy()

        if result.marker_corners:
            cv2.aruco.drawDetectedMarkers(vis, result.marker_corners, result.marker_ids)

        if result.valid:
            cv2.aruco.drawDetectedCornersCharuco(
                vis, result.charuco_corners, result.charuco_ids
            )

        return vis
