"""Single camera intrinsic calibration: pinhole and fisheye models."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..board.charuco_board import CharucoBoard
from ..board.detector import CharucoDetector, DetectionResult
from ..perf import perf_timer
from .models import CameraIntrinsics, CameraModel
from .observations import prepare_observation

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
        self._images: List[Optional[np.ndarray]] = []  # aligned with accepted observations
        self._image_size: Optional[Tuple[int, int]] = None
        self._data_revision = 0
        self.last_rejection_reason = ""
        self.frame_ids: List[str] = []

    @property
    def data_revision(self) -> int:
        return self._data_revision

    @property
    def num_frames(self) -> int:
        return len(self._all_corners)

    @property
    def frame_corner_counts(self) -> List[int]:
        return [int(len(ids)) for ids in self._all_ids]

    @property
    def ready(self) -> bool:
        return self.num_frames >= self.min_frames

    def add_frame(self, image: np.ndarray, frame_id: str | None = None) -> DetectionResult:
        """Detect ChArUco in image and store if valid."""
        with perf_timer("intrinsic add_frame", threshold_ms=120.0):
            result = self.detector.detect(image)
        self.add_detection(result, image=image, frame_id=frame_id)
        return result

    def add_detection(
        self,
        result: DetectionResult,
        image: Optional[np.ndarray] = None,
        frame_id: str | None = None,
    ) -> bool:
        """Store an already computed ChArUco detection."""
        self.last_rejection_reason = ""
        if result.valid and result.num_corners >= MIN_CORNERS_PER_FRAME:
            try:
                corners, ids, size = prepare_observation(result, self.board, self._image_size)
                if image is not None and (image.ndim < 2 or image.shape[1::-1] != size):
                    raise ValueError("原图分辨率与检测结果不一致")
            except (ValueError, TypeError) as exc:
                self.last_rejection_reason = str(exc)
                logger.warning("Rejected intrinsic observation: %s", exc)
                return False
            self._all_corners.append(corners)
            self._all_ids.append(ids)
            self._images.append(image.copy() if image is not None else None)
            self.frame_ids.append(str(frame_id) if frame_id is not None else str(self.num_frames))
            self._image_size = size
            self._data_revision += 1
            return True
        return False

    def pop_last_frame(self) -> bool:
        """Remove the most recently stored frame. Returns False if empty."""
        if not self._all_corners:
            return False
        self._all_corners.pop()
        self._all_ids.pop()
        self.frame_ids.pop()
        self._data_revision += 1
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
            with perf_timer("fisheye intrinsic calibrate", threshold_ms=500.0):
                return self._calibrate_fisheye(flags)
        with perf_timer("pinhole intrinsic calibrate", threshold_ms=500.0):
            return self._calibrate_pinhole(flags)

    def _calibrate_pinhole(self, flags: int) -> CameraIntrinsics:
        board_corners = self.board.object_points
        # ChArUco IDs index board corners; detections can be partial and unordered.
        obj_points = [
            np.ascontiguousarray(board_corners[ids.flatten()].reshape(-1, 1, 3), dtype=np.float32)
            for ids in self._all_ids
        ]
        img_points = [
            np.ascontiguousarray(corners.reshape(-1, 1, 2), dtype=np.float32)
            for corners in self._all_corners
        ]
        logger.info(
            "Pinhole intrinsic calibration: OpenCV=%s api=cv2.calibrateCamera frames=%d size=%s",
            cv2.__version__, self.num_frames, self._image_size,
        )
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
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
            frame_ids=list(self.frame_ids),
            per_frame_error_metric="euclidean_rms_px",
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
            frame_ids=list(self.frame_ids),
            per_frame_error_metric="euclidean_rms_px",
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
                np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))
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
                np.mean(np.sum(
                    (proj.reshape(-1, 2) - img_points[i].reshape(-1, 2)) ** 2, axis=1,
                ))
            )
            errors.append(float(err))
        return errors

    # ── Data quality helpers ──────────────────────────────────

    def recompute_errors(self, intrinsics: CameraIntrinsics) -> None:
        """Re-estimate board poses and errors after joint stereo intrinsic refinement."""
        errors, rotations, translations = [], [], []
        point_count, squared_error = 0, 0.
        for corners, ids in zip(self._all_corners, self._all_ids):
            obj = self.board.object_points[ids.ravel()].astype(np.float64)
            image = corners.astype(np.float64)
            if intrinsics.model == CameraModel.FISHEYE:
                normalized = cv2.fisheye.undistortPoints(
                    image, intrinsics.camera_matrix, intrinsics.dist_coeffs,
                )
                ok, rv, tv = cv2.solvePnP(obj, normalized, np.eye(3), None)
                projected = cv2.fisheye.projectPoints(
                    obj.reshape(-1, 1, 3), rv, tv,
                    intrinsics.camera_matrix, intrinsics.dist_coeffs,
                )[0] if ok else None
            else:
                ok, rv, tv = cv2.solvePnP(
                    obj, image, intrinsics.camera_matrix, intrinsics.dist_coeffs,
                )
                projected = cv2.projectPoints(
                    obj, rv, tv, intrinsics.camera_matrix, intrinsics.dist_coeffs,
                )[0] if ok else None
            if not ok or not np.isfinite(projected).all():
                raise ValueError("联合精修后逐帧重投影检查失败")
            total = float(np.sum((projected.reshape(-1, 2) - image.reshape(-1, 2)) ** 2))
            errors.append(float(np.sqrt(total / len(obj))))
            rotations.append(rv)
            translations.append(tv)
            squared_error += total
            point_count += len(obj)
        if not point_count:
            raise ValueError("没有可供误差复核的观测")
        intrinsics.per_frame_errors = errors
        intrinsics.rvecs, intrinsics.tvecs = rotations, translations
        intrinsics.rms_error = float(np.sqrt(squared_error / point_count))
        intrinsics.frame_ids = list(self.frame_ids)
        intrinsics.per_frame_error_metric = "euclidean_rms_px"

    def reset(self):
        self._data_revision += 1
        self._all_corners.clear()
        self._all_ids.clear()
        self._images.clear()
        self.frame_ids.clear()
        self._image_size = None

    def compute_coverage(self) -> float:
        """Legacy API: board corner completeness, NOT image-area coverage."""
        if not self._all_ids:
            return 0.0
        seen = set()
        for ids in self._all_ids:
            for id_val in ids.flatten():
                seen.add(int(id_val))
        return len(seen) / self.board.num_corners

    def compute_image_coverage(self) -> float:
        """Occupied cells in the 4x4 image-space corner histogram."""
        histogram = self.get_corner_distribution()
        return float(np.count_nonzero(histogram) / histogram.size) if histogram is not None else 0.

    def find_similar_detection(self, detection: DetectionResult, *, exclude_last=False) -> Optional[int]:
        """Advisory only: compare common IDs in image space against recent samples."""
        if not detection.valid or detection.image_size != self._image_size:
            return None
        stop = self.num_frames - int(exclude_last)
        ids = detection.charuco_ids.ravel()
        points = detection.charuco_corners.reshape(-1, 2)
        for index in range(stop - 1, max(-1, stop - 51), -1):
            old_ids = self._all_ids[index].ravel()
            common, old_idx, new_idx = np.intersect1d(old_ids, ids, return_indices=True)
            if len(common) < max(6, .6 * max(len(ids), len(old_ids))):
                continue
            delta = self._all_corners[index].reshape(-1, 2)[old_idx] - points[new_idx]
            rms = np.sqrt(np.mean(np.sum(delta ** 2, axis=1))) / max(detection.image_size)
            if np.isfinite(rms) and rms < .01:
                return index
        return None

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
    def stored_images(self) -> List[Optional[np.ndarray]]:
        return self._images

    def copy_with_indices(self, indices: List[int], *, keep_images: bool = True) -> "IntrinsicCalibrator":
        clone = IntrinsicCalibrator(self.board, self.model, self.min_frames)
        for idx in indices:
            if idx < 0 or idx >= self.num_frames:
                continue
            clone._all_corners.append(self._all_corners[idx].copy())
            clone._all_ids.append(self._all_ids[idx].copy())
            clone.frame_ids.append(self.frame_ids[idx])
            if idx < len(self._images):
                image = self._images[idx]
                clone._images.append(image.copy() if keep_images and image is not None else None)
        clone._image_size = self._image_size if clone._all_corners else None
        return clone
