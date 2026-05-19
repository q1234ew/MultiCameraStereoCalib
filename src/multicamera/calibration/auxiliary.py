"""Auxiliary mono-camera intrinsics and cross-modal extrinsics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from ..perf import perf_timer
from .models import CameraIntrinsics, CameraModel
from .planar import PatternBoardConfig, PlanarDetectionResult, PlanarPatternDetector

logger = logging.getLogger(__name__)


@dataclass
class CrossModalExtrinsicResult:
    T_rgb_left_aux: np.ndarray
    rms_error: float
    frame_count: int
    per_frame_errors: List[float]


class AuxIntrinsicCalibrator:
    """Calibrate one auxiliary mono camera from planar-board detections."""

    def __init__(
        self,
        board_config: PatternBoardConfig,
        model: CameraModel = CameraModel.PINHOLE,
        min_frames: int = 10,
    ):
        self.board_config = board_config
        self.model = model
        self.min_frames = min_frames
        self.detector = PlanarPatternDetector(board_config)
        self._object_points: List[np.ndarray] = []
        self._image_points: List[np.ndarray] = []
        self._image_size: Optional[tuple[int, int]] = None

    @property
    def num_frames(self) -> int:
        return len(self._image_points)

    @property
    def ready(self) -> bool:
        return self.num_frames >= self.min_frames

    def add_frame(self, image: np.ndarray) -> PlanarDetectionResult:
        result = self.detector.detect(image)
        self.add_detection(result)
        return result

    def add_detection(self, result: PlanarDetectionResult) -> bool:
        if not result.valid:
            return False
        if result.num_points != self.board_config.rows * self.board_config.cols:
            return False
        self._object_points.append(result.object_points.astype(np.float32))
        self._image_points.append(result.image_points.astype(np.float32))
        self._image_size = result.image_size
        return True

    def calibrate(self, flags: int = 0) -> CameraIntrinsics:
        if not self.ready:
            raise ValueError(
                f"Need at least {self.min_frames} frames, have {self.num_frames}"
            )
        if self.model == CameraModel.FISHEYE:
            with perf_timer("aux fisheye intrinsic calibrate", threshold_ms=500.0):
                return self._calibrate_fisheye(flags)
        with perf_timer("aux pinhole intrinsic calibrate", threshold_ms=500.0):
            return self._calibrate_pinhole(flags)

    def _calibrate_pinhole(self, flags: int) -> CameraIntrinsics:
        obj = [p.reshape(-1, 1, 3) for p in self._object_points]
        img = [p.reshape(-1, 1, 2) for p in self._image_points]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj, img, self._image_size, None, None, flags=flags
        )
        per_frame = _per_frame_errors(obj, img, rvecs, tvecs, K, dist)
        logger.info("Auxiliary pinhole intrinsic RMS: %.4f", rms)
        return CameraIntrinsics(
            camera_matrix=K,
            dist_coeffs=dist,
            image_size=self._image_size,
            model=CameraModel.PINHOLE,
            rms_error=float(rms),
            per_frame_errors=per_frame,
            rvecs=list(rvecs),
            tvecs=list(tvecs),
        )

    def _calibrate_fisheye(self, flags: int) -> CameraIntrinsics:
        obj = [p.reshape(-1, 1, 3).astype(np.float64) for p in self._object_points]
        img = [p.reshape(-1, 1, 2).astype(np.float64) for p in self._image_points]
        K = np.eye(3, dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)
        fish_flags = (
            flags
            | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_FIX_SKEW
        )
        rms, K, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            obj,
            img,
            self._image_size,
            K,
            dist,
            flags=fish_flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
        per_frame = _per_frame_errors_fisheye(obj, img, rvecs, tvecs, K, dist)
        logger.info("Auxiliary fisheye intrinsic RMS: %.4f", rms)
        return CameraIntrinsics(
            camera_matrix=K,
            dist_coeffs=dist,
            image_size=self._image_size,
            model=CameraModel.FISHEYE,
            rms_error=float(rms),
            per_frame_errors=per_frame,
            rvecs=list(rvecs),
            tvecs=list(tvecs),
        )


class CrossModalExtrinsicCalibrator:
    """Estimate AUX-to-RGB_L transform from paired planar-board observations."""

    def __init__(self, board_config: PatternBoardConfig, min_frames: int = 3):
        self.board_config = board_config
        self.min_frames = min_frames
        self._pairs: List[tuple[PlanarDetectionResult, PlanarDetectionResult]] = []

    @property
    def num_frames(self) -> int:
        return len(self._pairs)

    def add_detection_pair(
        self,
        rgb_left: PlanarDetectionResult,
        aux: PlanarDetectionResult,
    ) -> bool:
        expected = self.board_config.rows * self.board_config.cols
        if not rgb_left.valid or not aux.valid:
            return False
        if rgb_left.num_points != expected or aux.num_points != expected:
            return False
        self._pairs.append((rgb_left, aux))
        return True

    def calibrate(
        self,
        rgb_left_intrinsics: CameraIntrinsics,
        aux_intrinsics: CameraIntrinsics,
    ) -> CrossModalExtrinsicResult:
        if self.num_frames < self.min_frames:
            raise ValueError(
                f"Need at least {self.min_frames} paired frames, have {self.num_frames}"
            )

        transforms: list[np.ndarray] = []
        errors: list[float] = []
        with perf_timer("cross-modal extrinsic calibrate", threshold_ms=500.0):
            for rgb_det, aux_det in self._pairs:
                ok_rgb, rvec_rgb, tvec_rgb = cv2.solvePnP(
                    rgb_det.object_points,
                    rgb_det.image_points,
                    rgb_left_intrinsics.camera_matrix,
                    rgb_left_intrinsics.dist_coeffs,
                )
                ok_aux, rvec_aux, tvec_aux = cv2.solvePnP(
                    aux_det.object_points,
                    aux_det.image_points,
                    aux_intrinsics.camera_matrix,
                    aux_intrinsics.dist_coeffs,
                )
                if not ok_rgb or not ok_aux:
                    continue

                T_rgb_board = _rt_from_rvec_tvec(rvec_rgb, tvec_rgb)
                T_aux_board = _rt_from_rvec_tvec(rvec_aux, tvec_aux)
                transforms.append(T_rgb_board @ np.linalg.inv(T_aux_board))
                errors.append(
                    0.5
                    * (
                        _reprojection_error(rgb_det, rvec_rgb, tvec_rgb, rgb_left_intrinsics)
                        + _reprojection_error(aux_det, rvec_aux, tvec_aux, aux_intrinsics)
                    )
                )

        if len(transforms) < self.min_frames:
            raise ValueError("Not enough valid solvePnP frames for cross-modal extrinsics")

        selected = _select_inliers(errors, min_keep=self.min_frames)
        from scipy.spatial.transform import Rotation

        rotations = Rotation.from_matrix([transforms[i][:3, :3] for i in selected])
        R_avg = rotations.mean().as_matrix()
        t_avg = np.mean([transforms[i][:3, 3] for i in selected], axis=0)
        T_avg = np.eye(4, dtype=np.float64)
        T_avg[:3, :3] = R_avg
        T_avg[:3, 3] = t_avg
        selected_errors = [float(errors[i]) for i in selected]
        rms = float(np.sqrt(np.mean(np.square(selected_errors))))
        return CrossModalExtrinsicResult(
            T_rgb_left_aux=T_avg,
            rms_error=rms,
            frame_count=len(selected),
            per_frame_errors=selected_errors,
        )


def _rt_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def _reprojection_error(
    det: PlanarDetectionResult,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intr: CameraIntrinsics,
) -> float:
    if intr.model == CameraModel.FISHEYE:
        obj = det.object_points.reshape(-1, 1, 3).astype(np.float64)
        proj, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, intr.camera_matrix, intr.dist_coeffs)
        projected = proj.reshape(-1, 2)
    else:
        proj, _ = cv2.projectPoints(
            det.object_points, rvec, tvec, intr.camera_matrix, intr.dist_coeffs
        )
        projected = proj.reshape(-1, 2)
    return float(np.sqrt(np.mean((projected - det.image_points.reshape(-1, 2)) ** 2)))


def _per_frame_errors(obj, img, rvecs, tvecs, K, dist) -> list[float]:
    errors = []
    for o, i, rvec, tvec in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rvec, tvec, K, dist)
        errors.append(float(np.sqrt(np.mean((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2))))
    return errors


def _per_frame_errors_fisheye(obj, img, rvecs, tvecs, K, dist) -> list[float]:
    errors = []
    for o, i, rvec, tvec in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(o, rvec, tvec, K, dist)
        errors.append(float(np.sqrt(np.mean((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2))))
    return errors


def _select_inliers(errors: list[float], min_keep: int) -> list[int]:
    if len(errors) <= min_keep:
        return list(range(len(errors)))
    median = float(np.median(errors))
    mad = float(np.median(np.abs(np.asarray(errors) - median)))
    threshold = median + max(2.5 * mad, 1.0)
    selected = [idx for idx, err in enumerate(errors) if err <= threshold]
    if len(selected) >= min_keep:
        return selected
    return sorted(range(len(errors)), key=lambda idx: errors[idx])[:min_keep]
