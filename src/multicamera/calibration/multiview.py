"""Multi-view joint calibration: compute global extrinsics for all cameras."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from .models import (
    CameraExtrinsics,
    CameraIntrinsics,
    MultiCameraRig,
    StereoPairCalibration,
    StereoCalibResult,
)

logger = logging.getLogger(__name__)


def compose_transforms(
    R1: np.ndarray, T1: np.ndarray, R2: np.ndarray, T2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Chain two rigid transforms: T_world_to_cam2 = T_cam1_to_cam2 @ T_world_to_cam1."""
    R = R2 @ R1
    T = R2 @ T1 + T2
    return R, T


def invert_transform(
    R: np.ndarray, T: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    R_inv = R.T
    T_inv = -R.T @ T
    return R_inv, T_inv


class MultiViewCalibrator:
    """Computes global extrinsics for a multi-stereo-camera rig.

    Pipeline:
    1. Accept per-pair stereo calibration results (each gives R, T between left/right).
    2. Accept inter-pair observations: ChArUco detections visible from cameras in
       different pairs allow computing the relative pose between pairs.
    3. Build a spanning tree of camera poses rooted at a reference camera.
    4. Optionally refine with global bundle adjustment.
    """

    def __init__(self, reference_camera: str = ""):
        self.reference_camera = reference_camera
        self._pair_calibrations: Dict[str, StereoPairCalibration] = {}
        self._inter_pair_observations: List[_InterPairObs] = []

    def add_pair_calibration(self, pair: StereoPairCalibration):
        self._pair_calibrations[pair.pair_name] = pair

    def add_inter_pair_observation(
        self,
        cam_a: str,
        cam_b: str,
        obj_points: np.ndarray,
        img_points_a: np.ndarray,
        img_points_b: np.ndarray,
        intrinsics_a: CameraIntrinsics,
        intrinsics_b: CameraIntrinsics,
    ):
        """Register an observation linking two cameras from different pairs."""
        self._inter_pair_observations.append(
            _InterPairObs(
                cam_a=cam_a,
                cam_b=cam_b,
                obj_points=obj_points,
                img_points_a=img_points_a,
                img_points_b=img_points_b,
                intrinsics_a=intrinsics_a,
                intrinsics_b=intrinsics_b,
            )
        )

    def calibrate(self) -> MultiCameraRig:
        """Compute global extrinsics for all cameras."""
        if not self._pair_calibrations:
            raise ValueError("No stereo pair calibrations provided")

        if not self.reference_camera:
            first_pair = next(iter(self._pair_calibrations.values()))
            self.reference_camera = first_pair.left_id

        extrinsics = self._build_extrinsics_graph()

        rig = MultiCameraRig(
            pairs=dict(self._pair_calibrations),
            extrinsics=extrinsics,
            reference_camera=self.reference_camera,
        )

        if self._inter_pair_observations:
            self._refine_bundle_adjustment(rig)

        return rig

    def _build_extrinsics_graph(self) -> Dict[str, CameraExtrinsics]:
        """Build camera extrinsics via BFS on the stereo pair graph."""
        extrinsics: Dict[str, CameraExtrinsics] = {}

        adj: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
        for pair in self._pair_calibrations.values():
            R = pair.stereo.R
            T = pair.stereo.T

            adj.setdefault(pair.left_id, []).append((pair.right_id, R, T))

            R_inv, T_inv = invert_transform(R, T)
            adj.setdefault(pair.right_id, []).append((pair.left_id, R_inv, T_inv))

        for obs in self._inter_pair_observations:
            R_ab, T_ab = self._solve_relative_pose(obs)
            if R_ab is not None:
                adj.setdefault(obs.cam_a, []).append((obs.cam_b, R_ab, T_ab))
                R_inv, T_inv = invert_transform(R_ab, T_ab)
                adj.setdefault(obs.cam_b, []).append((obs.cam_a, R_inv, T_inv))

        extrinsics[self.reference_camera] = CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            T=np.zeros((3, 1), dtype=np.float64),
        )

        queue = [self.reference_camera]
        visited = {self.reference_camera}

        while queue:
            current = queue.pop(0)
            current_ext = extrinsics[current]

            for neighbor, R_rel, T_rel in adj.get(current, []):
                if neighbor in visited:
                    continue

                R_global, T_global = compose_transforms(
                    current_ext.R, current_ext.T, R_rel, T_rel
                )
                extrinsics[neighbor] = CameraExtrinsics(R=R_global, T=T_global)
                visited.add(neighbor)
                queue.append(neighbor)

        all_cameras = set()
        for pair in self._pair_calibrations.values():
            all_cameras.add(pair.left_id)
            all_cameras.add(pair.right_id)

        missing = all_cameras - visited
        if missing:
            logger.warning("Could not reach cameras: %s", missing)

        return extrinsics

    def _solve_relative_pose(
        self, obs: "_InterPairObs"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve relative pose between two cameras using shared ChArUco observations."""
        K_a = obs.intrinsics_a.camera_matrix
        d_a = obs.intrinsics_a.dist_coeffs
        K_b = obs.intrinsics_b.camera_matrix
        d_b = obs.intrinsics_b.dist_coeffs

        ok_a, rvec_a, tvec_a = cv2.solvePnP(
            obs.obj_points, obs.img_points_a, K_a, d_a
        )
        ok_b, rvec_b, tvec_b = cv2.solvePnP(
            obs.obj_points, obs.img_points_b, K_b, d_b
        )

        if not ok_a or not ok_b:
            return None, None

        R_a, _ = cv2.Rodrigues(rvec_a)
        R_b, _ = cv2.Rodrigues(rvec_b)

        R_a_inv, T_a_inv = invert_transform(R_a, tvec_a)
        R_ab = R_b @ R_a_inv
        T_ab = R_b @ T_a_inv + tvec_b

        return R_ab, T_ab

    def _refine_bundle_adjustment(self, rig: MultiCameraRig):
        """Global refinement of all camera extrinsics via reprojection error minimisation."""
        camera_ids = sorted(rig.extrinsics.keys())
        if len(camera_ids) < 2:
            return

        ref_idx = camera_ids.index(self.reference_camera)

        intrinsics_map: Dict[str, CameraIntrinsics] = {}
        for pair in rig.pairs.values():
            intrinsics_map[pair.left_id] = pair.left_intrinsics
            intrinsics_map[pair.right_id] = pair.right_intrinsics

        param_list = []
        for cid in camera_ids:
            ext = rig.extrinsics[cid]
            rvec, _ = cv2.Rodrigues(ext.R)
            param_list.extend(rvec.flatten().tolist())
            param_list.extend(ext.T.flatten().tolist())

        x0 = np.array(param_list, dtype=np.float64)

        def residual_fn(x):
            residuals = []
            for obs in self._inter_pair_observations:
                idx_a = camera_ids.index(obs.cam_a)
                idx_b = camera_ids.index(obs.cam_b)

                rvec_a = x[idx_a * 6 : idx_a * 6 + 3]
                tvec_a = x[idx_a * 6 + 3 : idx_a * 6 + 6]
                rvec_b = x[idx_b * 6 : idx_b * 6 + 3]
                tvec_b = x[idx_b * 6 + 3 : idx_b * 6 + 6]

                K_a = obs.intrinsics_a.camera_matrix
                d_a = obs.intrinsics_a.dist_coeffs
                K_b = obs.intrinsics_b.camera_matrix
                d_b = obs.intrinsics_b.dist_coeffs

                proj_a, _ = cv2.projectPoints(
                    obs.obj_points, rvec_a, tvec_a, K_a, d_a
                )
                proj_b, _ = cv2.projectPoints(
                    obs.obj_points, rvec_b, tvec_b, K_b, d_b
                )

                err_a = (proj_a.reshape(-1, 2) - obs.img_points_a.reshape(-1, 2)).flatten()
                err_b = (proj_b.reshape(-1, 2) - obs.img_points_b.reshape(-1, 2)).flatten()
                residuals.extend(err_a.tolist())
                residuals.extend(err_b.tolist())

            return np.array(residuals)

        result = least_squares(residual_fn, x0, method="lm", max_nfev=200)

        if result.success:
            logger.info("Bundle adjustment converged, cost: %.4f", result.cost)
            for i, cid in enumerate(camera_ids):
                rvec = result.x[i * 6 : i * 6 + 3]
                tvec = result.x[i * 6 + 3 : i * 6 + 6]
                R, _ = cv2.Rodrigues(rvec)
                rig.extrinsics[cid] = CameraExtrinsics(
                    R=R, T=tvec.reshape(3, 1)
                )
        else:
            logger.warning("Bundle adjustment did not converge: %s", result.message)


class _InterPairObs:
    __slots__ = (
        "cam_a",
        "cam_b",
        "obj_points",
        "img_points_a",
        "img_points_b",
        "intrinsics_a",
        "intrinsics_b",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
