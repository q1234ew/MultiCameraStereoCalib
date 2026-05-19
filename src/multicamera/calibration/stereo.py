"""Stereo pair calibration: pinhole and fisheye models."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..board.charuco_board import CharucoBoard
from ..board.detector import CharucoDetector, DetectionResult
from ..perf import perf_timer
from .models import CameraIntrinsics, CameraModel, StereoCalibResult

logger = logging.getLogger(__name__)

MIN_COMMON_CORNERS = 6
MIN_FRAMES = 10


class StereoCalibrator:
    """Collects synchronised left/right detections and performs stereo calibration.

    Supports both pinhole and fisheye camera models.
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

        self._obj_points: List[np.ndarray] = []
        self._img_points_left: List[np.ndarray] = []
        self._img_points_right: List[np.ndarray] = []
        self._image_size: Optional[Tuple[int, int]] = None

    @property
    def num_frames(self) -> int:
        return len(self._obj_points)

    @property
    def ready(self) -> bool:
        return self.num_frames >= self.min_frames

    def add_frame_pair(
        self, left: np.ndarray, right: np.ndarray
    ) -> Tuple[DetectionResult, DetectionResult]:
        with perf_timer("stereo add_frame_pair", threshold_ms=160.0):
            det_l = self.detector.detect(left)
            det_r = self.detector.detect(right)
        self.add_detection_pair(det_l, det_r)
        return det_l, det_r

    def add_detection_pair(
        self,
        det_l: DetectionResult,
        det_r: DetectionResult,
    ) -> bool:
        """Store an already computed left/right ChArUco detection pair."""
        if det_l.valid and det_r.valid:
            obj_pts, img_l, img_r = self._find_common_points(det_l, det_r)
            if obj_pts is not None and len(obj_pts) >= MIN_COMMON_CORNERS:
                self._obj_points.append(obj_pts)
                self._img_points_left.append(img_l)
                self._img_points_right.append(img_r)
                self._image_size = det_l.image_size
                return True
        return False

    def pop_last_frame(self) -> bool:
        """Remove the last accepted stereo pair. Returns False if empty."""
        if not self._obj_points:
            return False
        self._obj_points.pop()
        self._img_points_left.pop()
        self._img_points_right.pop()
        if not self._obj_points:
            self._image_size = None
        return True

    def calibrate(
        self,
        left_intrinsics: CameraIntrinsics,
        right_intrinsics: CameraIntrinsics,
        flags: int = 0,
    ) -> StereoCalibResult:
        if not self.ready:
            raise ValueError(
                f"Need at least {self.min_frames} frames, have {self.num_frames}"
            )

        if self.model == CameraModel.FISHEYE:
            with perf_timer("fisheye stereo calibrate", threshold_ms=500.0):
                return self._calibrate_fisheye(left_intrinsics, right_intrinsics, flags)
        with perf_timer("pinhole stereo calibrate", threshold_ms=500.0):
            return self._calibrate_pinhole(left_intrinsics, right_intrinsics, flags)

    # ── Pinhole ───────────────────────────────────────────────

    def _calibrate_pinhole(
        self, left: CameraIntrinsics, right: CameraIntrinsics, flags: int
    ) -> StereoCalibResult:
        if flags == 0:
            flags = cv2.CALIB_FIX_INTRINSIC

        rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
            self._obj_points,
            self._img_points_left,
            self._img_points_right,
            left.camera_matrix,
            left.dist_coeffs,
            right.camera_matrix,
            right.dist_coeffs,
            self._image_size,
            flags=flags,
        )
        logger.info("Pinhole stereo RMS: %.4f", rms)

        result = StereoCalibResult(
            R=R, T=T, E=E, F=F, rms_error=rms, model=CameraModel.PINHOLE
        )
        self._rectify_pinhole(result, left, right)
        return result

    def _rectify_pinhole(
        self, result: StereoCalibResult, left: CameraIntrinsics, right: CameraIntrinsics
    ):
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left.camera_matrix, left.dist_coeffs,
            right.camera_matrix, right.dist_coeffs,
            left.image_size, result.R, result.T, alpha=0,
        )
        result.R1, result.R2 = R1, R2
        result.P1, result.P2 = P1, P2
        result.Q = Q

        result.map1_left, result.map2_left = cv2.initUndistortRectifyMap(
            left.camera_matrix, left.dist_coeffs, R1, P1,
            left.image_size, cv2.CV_32FC1,
        )
        result.map1_right, result.map2_right = cv2.initUndistortRectifyMap(
            right.camera_matrix, right.dist_coeffs, R2, P2,
            right.image_size, cv2.CV_32FC1,
        )

    # ── Fisheye ───────────────────────────────────────────────

    def _calibrate_fisheye(
        self, left: CameraIntrinsics, right: CameraIntrinsics, flags: int
    ) -> StereoCalibResult:
        obj_fish = [o.reshape(-1, 1, 3).astype(np.float64) for o in self._obj_points]
        img_l = [p.reshape(-1, 1, 2).astype(np.float64) for p in self._img_points_left]
        img_r = [p.reshape(-1, 1, 2).astype(np.float64) for p in self._img_points_right]

        K1 = left.camera_matrix.copy()
        d1 = left.dist_coeffs.copy().reshape(4, 1)
        K2 = right.camera_matrix.copy()
        d2 = right.dist_coeffs.copy().reshape(4, 1)

        fish_flags = (
            cv2.fisheye.CALIB_FIX_INTRINSIC
            | cv2.fisheye.CALIB_FIX_SKEW
        )

        rms, K1, d1, K2, d2, R, T = cv2.fisheye.stereoCalibrate(
            obj_fish, img_l, img_r,
            K1, d1, K2, d2,
            self._image_size,
            flags=fish_flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
        logger.info("Fisheye stereo RMS: %.4f", rms)

        E = self._essential_from_rt(R, T)
        F = np.zeros((3, 3), dtype=np.float64)

        result = StereoCalibResult(
            R=R, T=T, E=E, F=F, rms_error=rms, model=CameraModel.FISHEYE
        )
        self._rectify_fisheye(result, left, right)
        return result

    def _rectify_fisheye(
        self, result: StereoCalibResult, left: CameraIntrinsics, right: CameraIntrinsics
    ):
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            left.camera_matrix, left.dist_coeffs.reshape(4, 1),
            right.camera_matrix, right.dist_coeffs.reshape(4, 1),
            left.image_size, result.R, result.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
        )
        result.R1, result.R2 = R1, R2
        result.P1, result.P2 = P1, P2
        result.Q = Q

        result.map1_left, result.map2_left = cv2.fisheye.initUndistortRectifyMap(
            left.camera_matrix, left.dist_coeffs.reshape(4, 1),
            R1, P1, left.image_size, cv2.CV_32FC1,
        )
        result.map1_right, result.map2_right = cv2.fisheye.initUndistortRectifyMap(
            right.camera_matrix, right.dist_coeffs.reshape(4, 1),
            R2, P2, right.image_size, cv2.CV_32FC1,
        )

    @staticmethod
    def _essential_from_rt(R: np.ndarray, T: np.ndarray) -> np.ndarray:
        tx = np.array([
            [0, -T[2, 0], T[1, 0]],
            [T[2, 0], 0, -T[0, 0]],
            [-T[1, 0], T[0, 0], 0],
        ], dtype=np.float64)
        return tx @ R

    # ── Common ────────────────────────────────────────────────

    def _find_common_points(
        self, det_l: DetectionResult, det_r: DetectionResult
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        ids_l = det_l.charuco_ids.flatten()
        ids_r = det_r.charuco_ids.flatten()
        common_ids = set(ids_l) & set(ids_r)

        if len(common_ids) < MIN_COMMON_CORNERS:
            return None, None, None

        board_corners = self.board.object_points
        obj_pts, img_l, img_r = [], [], []
        id_to_idx_l = {int(v): i for i, v in enumerate(ids_l)}
        id_to_idx_r = {int(v): i for i, v in enumerate(ids_r)}

        for cid in sorted(common_ids):
            obj_pts.append(board_corners[cid])
            img_l.append(det_l.charuco_corners[id_to_idx_l[cid]].flatten())
            img_r.append(det_r.charuco_corners[id_to_idx_r[cid]].flatten())

        return (
            np.array(obj_pts, dtype=np.float32),
            np.array(img_l, dtype=np.float32),
            np.array(img_r, dtype=np.float32),
        )

    def reset(self):
        self._obj_points.clear()
        self._img_points_left.clear()
        self._img_points_right.clear()
        self._image_size = None
