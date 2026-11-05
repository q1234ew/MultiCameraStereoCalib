"""Stereo pair calibration: pinhole and fisheye models."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..board.charuco_board import CharucoBoard
from ..board.detector import CharucoDetector, DetectionResult
from ..perf import perf_timer
from .models import CameraIntrinsics, CameraModel, StereoCalibResult
from .observations import prepare_observation
from .solver_settings import PINHOLE_CALIBRATION_CRITERIA

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
        self._corner_ids: List[np.ndarray] = []
        self._image_size: Optional[Tuple[int, int]] = None
        self.frame_ids: List[str] = []

    @property
    def num_frames(self) -> int:
        return len(self._obj_points)

    @property
    def frame_common_corner_counts(self) -> List[int]:
        return [int(len(obj)) for obj in self._obj_points]

    @property
    def frame_corner_ids(self) -> List[np.ndarray]:
        return [ids.copy() for ids in self._corner_ids]

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
        frame_id: str | None = None,
    ) -> bool:
        """Store an already computed left/right ChArUco detection pair."""
        if det_l.valid and det_r.valid:
            corners_l, ids_l, size = prepare_observation(det_l, self.board, self._image_size)
            corners_r, ids_r, _ = prepare_observation(det_r, self.board, size)
            det_l = replace(det_l, charuco_corners=corners_l, charuco_ids=ids_l)
            det_r = replace(det_r, charuco_corners=corners_r, charuco_ids=ids_r)
            obj_pts, img_l, img_r = self._find_common_points(det_l, det_r)
            if obj_pts is not None and len(obj_pts) >= MIN_COMMON_CORNERS:
                common_ids = np.intersect1d(ids_l.ravel(), ids_r.ravel())
                self._obj_points.append(obj_pts)
                self._img_points_left.append(img_l)
                self._img_points_right.append(img_r)
                self._corner_ids.append(common_ids.copy())
                self.frame_ids.append(str(frame_id) if frame_id is not None else str(self.num_frames))
                self._image_size = size
                return True
        return False

    def pop_last_frame(self) -> bool:
        """Remove the last accepted stereo pair. Returns False if empty."""
        if not self._obj_points:
            return False
        self._obj_points.pop()
        self._img_points_left.pop()
        self._img_points_right.pop()
        self._corner_ids.pop()
        self.frame_ids.pop()
        if not self._obj_points:
            self._image_size = None
        return True

    def copy_with_indices(self, indices: List[int]) -> "StereoCalibrator":
        self._require_corner_ids()
        clone = StereoCalibrator(self.board, self.model, self.min_frames)
        for idx in indices:
            if idx < 0 or idx >= self.num_frames:
                continue
            clone._obj_points.append(self._obj_points[idx].copy())
            clone._img_points_left.append(self._img_points_left[idx].copy())
            clone._img_points_right.append(self._img_points_right[idx].copy())
            clone._corner_ids.append(self._corner_ids[idx].copy())
            clone.frame_ids.append(self.frame_ids[idx])
        clone._image_size = self._image_size if clone._obj_points else None
        return clone

    def _require_corner_ids(self):
        if len(self._corner_ids) != self.num_frames:
            raise ValueError("缺少逐帧角点 ID，无法按真实 ID 复制观测，请重新采集或导入")
        for i, ids in enumerate(self._corner_ids):
            if (ids is None or np.asarray(ids).ndim != 1
                    or len(ids) != len(self._obj_points[i])
                    or not np.issubdtype(np.asarray(ids).dtype, np.integer)
                    or len(np.unique(ids)) != len(ids)):
                raise ValueError(f"第 {i + 1} 帧角点 ID 缺失或无效，无法按真实 ID 复制观测")

    def copy_with_common_ids(self) -> "StereoCalibrator":
        """Explicitly select identical board IDs across all views on a copy."""
        if not self.num_frames:
            raise ValueError("没有双目观测，无法计算跨帧共同角点")
        self._require_corner_ids()
        common_ids = self._corner_ids[0].copy()
        for ids in self._corner_ids[1:]:
            common_ids = np.intersect1d(common_ids, ids)
        if len(common_ids) < MIN_COMMON_CORNERS:
            raise ValueError(
                f"跨帧共同角点仅 {len(common_ids)} 个，至少需要 {MIN_COMMON_CORNERS} 个；"
                "请补拍具有足够共同角点的标定板观测"
            )
        clone = self.copy_with_indices(list(range(self.num_frames)))
        for i, ids in enumerate(self._corner_ids):
            _, take, _ = np.intersect1d(ids, common_ids, return_indices=True)
            clone._obj_points[i] = self._obj_points[i][take].copy()
            clone._img_points_left[i] = self._img_points_left[i][take].copy()
            clone._img_points_right[i] = self._img_points_right[i][take].copy()
            clone._corner_ids[i] = np.sort(common_ids).copy()
        return clone

    def calibrate(
        self,
        left_intrinsics: CameraIntrinsics,
        right_intrinsics: CameraIntrinsics,
        flags: int = 0,
    ) -> StereoCalibResult:
        """Solve and update both intrinsics from the solver's returned K/D.

        Zero flags preserve the fixed-intrinsic default. Nonzero flags belong
        to the selected model's OpenCV namespace. Changed intrinsics invalidate
        single-camera fit metrics and poses; their original observations are
        not available here to recompute them.
        """
        if not self.ready:
            raise ValueError(
                f"Need at least {self.min_frames} frames, have {self.num_frames}"
            )

        if self.model not in (CameraModel.PINHOLE, CameraModel.FISHEYE):
            raise ValueError(f"Unsupported camera model: {self.model}")
        self._validate_intrinsics(left_intrinsics)
        self._validate_intrinsics(right_intrinsics)
        size = self._validate_image_size(self._image_size)
        if left_intrinsics.image_size != size or right_intrinsics.image_size != size:
            raise ValueError("Intrinsics and stereo observations must have the same image size")

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
            np.array(left.camera_matrix, dtype=np.float64, copy=True),
            np.array(left.dist_coeffs, dtype=np.float64, copy=True),
            np.array(right.camera_matrix, dtype=np.float64, copy=True),
            np.array(right.dist_coeffs, dtype=np.float64, copy=True),
            self._image_size,
            flags=flags,
            criteria=PINHOLE_CALIBRATION_CRITERIA,
        )
        logger.info("Pinhole stereo RMS: %.4f flags=%d criteria=%s",
                    rms, flags, PINHOLE_CALIBRATION_CRITERIA)

        result = StereoCalibResult(
            R=R, T=T, E=E, F=F, rms_error=rms, model=CameraModel.PINHOLE
        )
        solved_left = self._solved_intrinsics(left, K1, d1)
        solved_right = self._solved_intrinsics(right, K2, d2)
        self._rectify_pinhole(result, solved_left, solved_right)
        self._apply_intrinsics(left, solved_left)
        self._apply_intrinsics(right, solved_right)
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
        counts = self.frame_common_corner_counts
        if len(set(counts)) > 1:
            details = "、".join(
                f"{self.frame_ids[i] if i < len(self.frame_ids) else i + 1}: {count} 点"
                for i, count in enumerate(counts)
            )
            raise ValueError(
                "鱼眼双目标定要求跨帧共同角点数量一致，当前观测为："
                f"{details}。请选择“鱼眼兼容：跨帧共同角点”后重试；原始观测未修改。"
            )
        obj_fish = [o.reshape(-1, 1, 3).astype(np.float64) for o in self._obj_points]
        img_l = [p.reshape(-1, 1, 2).astype(np.float64) for p in self._img_points_left]
        img_r = [p.reshape(-1, 1, 2).astype(np.float64) for p in self._img_points_right]

        K1 = np.array(left.camera_matrix, dtype=np.float64, copy=True)
        d1 = np.array(left.dist_coeffs, dtype=np.float64, copy=True).reshape(4, 1)
        K2 = np.array(right.camera_matrix, dtype=np.float64, copy=True)
        d2 = np.array(right.dist_coeffs, dtype=np.float64, copy=True).reshape(4, 1)

        fish_flags = flags or (
            cv2.fisheye.CALIB_FIX_INTRINSIC
            | cv2.fisheye.CALIB_FIX_SKEW
        )

        # Newer bindings also return per-view board poses after R/T.
        rms, K1, d1, K2, d2, R, T, *_ = cv2.fisheye.stereoCalibrate(
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
        solved_left = self._solved_intrinsics(left, K1, d1)
        solved_right = self._solved_intrinsics(right, K2, d2)
        self._rectify_fisheye(result, solved_left, solved_right)
        self._apply_intrinsics(left, solved_left)
        self._apply_intrinsics(right, solved_right)
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

    @staticmethod
    def _validate_image_size(size) -> Tuple[int, int]:
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or any(isinstance(v, (bool, np.bool_))
                       or not isinstance(v, (int, np.integer)) or v <= 0 for v in size)):
            raise ValueError("Image size must contain two positive integer dimensions")
        return tuple(int(v) for v in size)

    def _validate_intrinsics(self, intrinsics: CameraIntrinsics):
        if intrinsics.model != self.model:
            raise ValueError("Intrinsic model must match the stereo camera model")
        self._validate_image_size(intrinsics.image_size)
        k = np.asarray(intrinsics.camera_matrix)
        if (k.shape != (3, 3) or not np.isfinite(k).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0., 0., 1.], rtol=0, atol=1e-12)
                or k[1, 0] != 0):
            raise ValueError("Camera matrix must be finite 3x3 with positive focal lengths")
        d = np.asarray(intrinsics.dist_coeffs)
        counts = (4,) if self.model == CameraModel.FISHEYE else (4, 5, 8, 12, 14)
        if (d.ndim not in (1, 2) or (d.ndim == 2 and 1 not in d.shape)
                or d.size not in counts or not np.isfinite(d).all()):
            raise ValueError("Invalid distortion coefficient vector for the camera model")

    def _solved_intrinsics(self, original: CameraIntrinsics, k, d) -> CameraIntrinsics:
        solved = replace(
            original,
            camera_matrix=np.array(k, dtype=np.float64, copy=True),
            dist_coeffs=np.array(d, dtype=np.float64, copy=True),
        )
        self._validate_intrinsics(solved)
        changed = (not np.array_equal(original.camera_matrix, solved.camera_matrix)
                   or not np.array_equal(np.asarray(original.dist_coeffs).ravel(),
                                         solved.dist_coeffs.ravel()))
        if changed:
            solved.rms_error = float("nan")
            solved.per_frame_errors = None
            solved.rvecs = None
            solved.tvecs = None
        return solved

    @staticmethod
    def _apply_intrinsics(target: CameraIntrinsics, solved: CameraIntrinsics):
        # Commit only after rectification succeeds, never during OpenCV's solve.
        target.camera_matrix = solved.camera_matrix
        target.dist_coeffs = solved.dist_coeffs
        target.rms_error = solved.rms_error
        target.per_frame_errors = solved.per_frame_errors
        target.rvecs = solved.rvecs
        target.tvecs = solved.tvecs

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
        self.frame_ids.clear()
        self._corner_ids.clear()
        self._obj_points.clear()
        self._img_points_left.clear()
        self._img_points_right.clear()
        self._image_size = None
