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
        detection = _validated_detection(result, self.board_config)
        if detection is None:
            return False
        if self._image_size is not None and detection.image_size != self._image_size:
            return False
        self._object_points.append(detection.object_points)
        self._image_points.append(detection.image_points)
        self._image_size = detection.image_size
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
        obj = [p.reshape(-1, 1, 3).astype(np.float32) for p in self._object_points]
        img = [p.reshape(-1, 1, 2).astype(np.float32) for p in self._image_points]
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
            per_frame_error_metric="euclidean_rms_px",
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
            per_frame_error_metric="euclidean_rms_px",
            rvecs=list(rvecs),
            tvecs=list(tvecs),
        )


class CrossModalExtrinsicCalibrator:
    """Estimate AUX-to-RGB_L transform from paired planar-board observations.

    Both detections must use the same physical corner ordering. Symmetric boards
    can have cross-modal ordering ambiguity even with low reprojection errors;
    this solver cannot resolve that ambiguity and never auto-flips corners.
    """

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
        rgb = _validated_detection(rgb_left, self.board_config)
        auxiliary = _validated_detection(aux, self.board_config)
        if rgb is None or auxiliary is None:
            return False
        if self._pairs and (
            rgb.image_size != self._pairs[0][0].image_size
            or auxiliary.image_size != self._pairs[0][1].image_size
        ):
            return False
        self._pairs.append((rgb, auxiliary))
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

        _validate_intrinsics(rgb_left_intrinsics, self._pairs[0][0].image_size)
        _validate_intrinsics(aux_intrinsics, self._pairs[0][1].image_size)

        transforms: list[np.ndarray] = []
        errors: list[float] = []
        with perf_timer("cross-modal extrinsic calibrate", threshold_ms=500.0):
            for rgb_det, aux_det in self._pairs:
                try:
                    rgb_pose = _solve_board_pose(rgb_det, rgb_left_intrinsics)
                    aux_pose = _solve_board_pose(aux_det, aux_intrinsics)
                    if rgb_pose is None or aux_pose is None:
                        continue
                    rvec_rgb, tvec_rgb = rgb_pose
                    rvec_aux, tvec_aux = aux_pose
                    rgb_error = _reprojection_error(
                        rgb_det, rvec_rgb, tvec_rgb, rgb_left_intrinsics
                    )
                    aux_error = _reprojection_error(aux_det, rvec_aux, tvec_aux, aux_intrinsics)
                except cv2.error:
                    logger.warning("Skipping failed cross-modal board pose", exc_info=True)
                    continue
                error = float(np.sqrt((rgb_error ** 2 + aux_error ** 2) / 2))
                if not np.isfinite(error):
                    continue

                T_rgb_board = _rt_from_rvec_tvec(rvec_rgb, tvec_rgb)
                T_aux_board = _rt_from_rvec_tvec(rvec_aux, tvec_aux)
                transforms.append(T_rgb_board @ np.linalg.inv(T_aux_board))
                errors.append(error)

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


def _valid_image_size(size) -> bool:
    return (
        isinstance(size, (tuple, list)) and len(size) == 2
        and all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                and v > 0 for v in size)
    )


def _validated_detection(det, config) -> Optional[PlanarDetectionResult]:
    if not det.valid or not _valid_image_size(det.image_size):
        return None
    if det.pattern_type != config.pattern_type:
        return None
    count = config.rows * config.cols
    try:
        obj = np.asarray(det.object_points, dtype=np.float64)
        img = np.asarray(det.image_points, dtype=np.float64)
        if obj.shape not in ((count, 3), (count, 1, 3)):
            return None
        if img.shape not in ((count, 2), (count, 1, 2)):
            return None
        obj, img = obj.reshape(count, 3), img.reshape(count, 2)
        if not np.isfinite(obj).all() or not np.isfinite(img).all():
            return None
        if np.max(np.abs(img)) > np.finfo(np.float32).max:
            return None
        expected = PlanarPatternDetector._make_object_points(config)
        # Fix scale and board frame as well as rejecting collapsed/nonplanar data.
        if not np.allclose(obj, expected, rtol=1e-6, atol=config.square_size * 1e-6):
            return None
        if np.linalg.matrix_rank(obj - obj.mean(axis=0)) < 2:
            return None
        if np.linalg.matrix_rank(img - img.mean(axis=0)) < 2:
            return None
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return None
    return PlanarDetectionResult(
        True, obj.copy(), img.copy(), tuple(det.image_size), det.pattern_type
    )


def _validate_intrinsics(intr: CameraIntrinsics, image_size) -> None:
    if not _valid_image_size(intr.image_size) or tuple(intr.image_size) != image_size:
        raise ValueError("Intrinsics image_size must match the detection image_size")
    try:
        K = np.asarray(intr.camera_matrix, dtype=np.float64)
        dist = np.asarray(intr.dist_coeffs, dtype=np.float64)
        valid = (
            K.shape == (3, 3) and np.isfinite(K).all()
            and K[0, 0] > 0 and K[1, 1] > 0
            and np.allclose(K[2], [0, 0, 1])
            and K[0, 1] == 0 and K[1, 0] == 0
            and dist.ndim in (1, 2) and (dist.ndim == 1 or 1 in dist.shape)
            and np.isfinite(dist).all()
        )
        counts = {CameraModel.PINHOLE: (4, 5, 8, 12, 14), CameraModel.FISHEYE: (4,)}
        valid = valid and intr.model in counts and dist.size in counts[intr.model]
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("Invalid camera matrix, distortion coefficients or camera model")


def _solve_board_pose(det, intr):
    K = np.ascontiguousarray(intr.camera_matrix, dtype=np.float64)
    dist = np.ascontiguousarray(intr.dist_coeffs, dtype=np.float64).reshape(-1, 1)
    points = det.image_points.reshape(-1, 1, 2)
    if intr.model == CameraModel.FISHEYE:
        # Ordinary solvePnP only understands pinhole distortion. Use normalized
        # rays for the solve, but original distorted pixels for error reporting.
        points = cv2.fisheye.undistortPoints(points, K, dist)
        K, dist = np.eye(3, dtype=np.float64), None
    if not np.isfinite(points).all():
        return None
    ok, rvec, tvec = cv2.solvePnP(det.object_points, points, K, dist)
    if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    R, _ = cv2.Rodrigues(rvec)
    camera_points = det.object_points @ R.T + tvec.reshape(3)
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
        return None
    return rvec, tvec


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
    K = np.ascontiguousarray(intr.camera_matrix, dtype=np.float64)
    dist = np.ascontiguousarray(intr.dist_coeffs, dtype=np.float64).reshape(-1, 1)
    if intr.model == CameraModel.FISHEYE:
        obj = det.object_points.reshape(-1, 1, 3).astype(np.float64)
        proj, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, dist)
        projected = proj.reshape(-1, 2)
    else:
        proj, _ = cv2.projectPoints(
            det.object_points, rvec, tvec, K, dist
        )
        projected = proj.reshape(-1, 2)
    return _point_rms(projected, det.image_points)


def _point_rms(projected, observed) -> float:
    residuals = projected.reshape(-1, 2) - observed.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))


def _per_frame_errors(obj, img, rvecs, tvecs, K, dist) -> list[float]:
    errors = []
    for o, i, rvec, tvec in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rvec, tvec, K, dist)
        errors.append(_point_rms(proj, i))
    return errors


def _per_frame_errors_fisheye(obj, img, rvecs, tvecs, K, dist) -> list[float]:
    errors = []
    for o, i, rvec, tvec in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(o, rvec, tvec, K, dist)
        errors.append(_point_rms(proj, i))
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
