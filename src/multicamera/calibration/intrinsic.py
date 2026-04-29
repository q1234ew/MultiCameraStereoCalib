"""Single camera intrinsic calibration: pinhole and fisheye models."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..board.charuco_board import CharucoBoard
from ..board.detector import CharucoDetector, DetectionResult
from .models import CameraIntrinsics, CameraModel

logger = logging.getLogger(__name__)

MIN_FRAMES = 10
MIN_CORNERS_PER_FRAME = 6


class IntrinsicCalibrator:
    """Collects ChArUco detections and calibrates a single camera.

    Supports both pinhole and fisheye (equidistant) camera models.
    """

    def __init__(
        self,
        board: CharucoBoard,
        model: CameraModel = CameraModel.PINHOLE,
        min_frames: int = MIN_FRAMES,
    ):
        self.board = board
        self.model = model
        self.detector = CharucoDetector(board)
        self.min_frames = min_frames

        self._all_corners: List[np.ndarray] = []
        self._all_ids: List[np.ndarray] = []
        self._images: List[np.ndarray] = []  # keep for quality eval
        self._image_size: Optional[Tuple[int, int]] = None

    @property
    def num_frames(self) -> int:
        return len(self._all_corners)

    @property
    def ready(self) -> bool:
        return self.num_frames >= self.min_frames

    def add_frame(self, image: np.ndarray) -> DetectionResult:
        """Detect ChArUco in image and store if valid."""
        result = self.detector.detect(image)
        if result.valid and result.num_corners >= MIN_CORNERS_PER_FRAME:
            self._all_corners.append(result.charuco_corners)
            self._all_ids.append(result.charuco_ids)
            self._images.append(image.copy())
            self._image_size = result.image_size
        return result

    def pop_last_frame(self) -> bool:
        """Remove the most recently stored frame. Returns False if empty."""
        if not self._all_corners:
            return False
        self._all_corners.pop()
        self._all_ids.pop()
        if self._images:
            self._images.pop()
        if not self._all_corners:
            self._image_size = None
        return True

    def calibrate(self, flags: int = 0) -> CameraIntrinsics:
        """Run camera calibration using all collected frames."""
        if not self.ready:
            raise ValueError(
                f"Need at least {self.min_frames} frames, have {self.num_frames}"
            )

        if self.model == CameraModel.FISHEYE:
            return self._calibrate_fisheye(flags)
        return self._calibrate_pinhole(flags)

    def _calibrate_pinhole(self, flags: int) -> CameraIntrinsics:
        rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            self._all_corners,
            self._all_ids,
            self.board.board,
            self._image_size,
            None,
            None,
            flags=flags,
        )
        logger.info("Pinhole intrinsic RMS: %.4f", rms)

        per_frame = self._compute_per_frame_errors_pinhole(K, dist, rvecs, tvecs)

        return CameraIntrinsics(
            camera_matrix=K,
            dist_coeffs=dist,
            image_size=self._image_size,
            model=CameraModel.PINHOLE,
            rms_error=rms,
            per_frame_errors=per_frame,
            rvecs=list(rvecs),
            tvecs=list(tvecs),
        )

    def _calibrate_fisheye(self, flags: int) -> CameraIntrinsics:
        board_corners = self.board.object_points
        obj_points = []
        img_points = []

        for corners, ids in zip(self._all_corners, self._all_ids):
            ids_flat = ids.flatten()
            obj = board_corners[ids_flat].reshape(-1, 1, 3).astype(np.float64)
            img = corners.reshape(-1, 1, 2).astype(np.float64)
            obj_points.append(obj)
            img_points.append(img)

        K = np.eye(3, dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)

        fish_flags = (
            flags
            | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_FIX_SKEW
        )

        rms, K, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_points,
            img_points,
            self._image_size,
            K,
            dist,
            flags=fish_flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
        logger.info("Fisheye intrinsic RMS: %.4f", rms)

        per_frame = self._compute_per_frame_errors_fisheye(
            K, dist, rvecs, tvecs, obj_points, img_points
        )

        return CameraIntrinsics(
            camera_matrix=K,
            dist_coeffs=dist,
            image_size=self._image_size,
            model=CameraModel.FISHEYE,
            rms_error=rms,
            per_frame_errors=per_frame,
            rvecs=list(rvecs),
            tvecs=list(tvecs),
        )

    # ── Per-frame error computation ───────────────────────────

    def _compute_per_frame_errors_pinhole(
        self, K, dist, rvecs, tvecs
    ) -> List[float]:
        board_corners = self.board.object_points
        errors = []
        for i, (corners, ids) in enumerate(zip(self._all_corners, self._all_ids)):
            ids_flat = ids.flatten()
            obj = board_corners[ids_flat].astype(np.float32)
            proj, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], K, dist)
            err = np.sqrt(
                np.mean((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2)
            )
            errors.append(float(err))
        return errors

    def _compute_per_frame_errors_fisheye(
        self, K, dist, rvecs, tvecs, obj_points, img_points
    ) -> List[float]:
        errors = []
        for i in range(len(obj_points)):
            proj, _ = cv2.fisheye.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], K, dist
            )
            err = np.sqrt(
                np.mean(
                    (proj.reshape(-1, 2) - img_points[i].reshape(-1, 2)) ** 2
                )
            )
            errors.append(float(err))
        return errors

    # ── Data quality helpers ──────────────────────────────────

    def reset(self):
        self._all_corners.clear()
        self._all_ids.clear()
        self._images.clear()
        self._image_size = None

    def compute_coverage(self) -> float:
        """Fraction of board corners seen across all collected frames."""
        if not self._all_ids:
            return 0.0
        seen = set()
        for ids in self._all_ids:
            for id_val in ids.flatten():
                seen.add(int(id_val))
        return len(seen) / self.board.num_corners

    def get_corner_distribution(self) -> Optional[np.ndarray]:
        """Compute a 2D histogram of detected corner positions over the image.

        Returns (grid_rows, grid_cols) count array, or None if no data.
        """
        if not self._all_corners or self._image_size is None:
            return None

        w, h = self._image_size
        grid_r, grid_c = 4, 4
        hist = np.zeros((grid_r, grid_c), dtype=np.int32)

        for corners in self._all_corners:
            pts = corners.reshape(-1, 2)
            for px, py in pts:
                ci = min(int(px / w * grid_c), grid_c - 1)
                ri = min(int(py / h * grid_r), grid_r - 1)
                hist[ri, ci] += 1

        return hist

    @property
    def stored_images(self) -> List[np.ndarray]:
        return self._images
