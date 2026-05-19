"""Multi-view point cloud registration and fusion."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

from ..calibration.models import MultiCameraRig
from .pointcloud import statistical_filter, to_open3d_cloud, voxel_downsample

logger = logging.getLogger(__name__)


class MultiViewFusion:
    """Fuse point clouds from multiple stereo pairs into a unified cloud."""

    def __init__(
        self,
        voxel_size: float = 0.005,
        icp_max_distance: float = 0.02,
        icp_iterations: int = 50,
        statistical_nb: int = 20,
        statistical_std: float = 2.0,
    ):
        if o3d is None:
            raise ImportError(
                "open3d is required for point cloud fusion. "
                "Install the pointcloud extra or build Windows package with --pointcloud."
            )

        self.voxel_size = voxel_size
        self.icp_max_distance = icp_max_distance
        self.icp_iterations = icp_iterations
        self.statistical_nb = statistical_nb
        self.statistical_std = statistical_std

    def fuse(
        self,
        clouds: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
        rig: MultiCameraRig,
        use_icp_refinement: bool = True,
    ) -> "o3d.geometry.PointCloud":
        """Fuse point clouds from multiple cameras.

        Args:
            clouds: {pair_name: (points_Nx3, colors_Nx3_or_None)}
                    points are in the left camera's local frame for each pair.
            rig: calibrated multi-camera rig with global extrinsics.
            use_icp_refinement: apply ICP after coarse alignment.

        Returns:
            Fused Open3D point cloud in the world coordinate frame.
        """
        transformed = []

        for pair_name, (points, colors) in clouds.items():
            pair = rig.get_pair(pair_name)
            if pair is None:
                logger.warning("No calibration for pair %s, skipping", pair_name)
                continue

            ext = rig.get_extrinsics(pair.left_id)
            if ext is None:
                logger.warning(
                    "No extrinsics for camera %s, skipping pair %s",
                    pair.left_id,
                    pair_name,
                )
                continue

            pcd = to_open3d_cloud(points, colors)
            pcd = statistical_filter(pcd, self.statistical_nb, self.statistical_std)

            pcd.transform(ext.transformation_matrix)
            transformed.append(pcd)

        if not transformed:
            return o3d.geometry.PointCloud()

        fused = transformed[0]
        for pcd in transformed[1:]:
            if use_icp_refinement:
                pcd = self._icp_refine(fused, pcd)
            fused += pcd

        fused = voxel_downsample(fused, self.voxel_size)
        fused = statistical_filter(fused, self.statistical_nb, self.statistical_std)

        logger.info("Fused cloud: %d points", len(fused.points))
        return fused

    def _icp_refine(
        self,
        source: "o3d.geometry.PointCloud",
        target: "o3d.geometry.PointCloud",
    ) -> "o3d.geometry.PointCloud":
        """Refine alignment of target onto source using ICP."""
        source_down = voxel_downsample(source, self.voxel_size * 2)
        target_down = voxel_downsample(target, self.voxel_size * 2)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 4, max_nn=30
            )
        )
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 4, max_nn=30
            )
        )

        reg = o3d.pipelines.registration.registration_icp(
            target_down,
            source_down,
            self.icp_max_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_iterations
            ),
        )

        logger.info(
            "ICP fitness=%.4f, RMSE=%.6f", reg.fitness, reg.inlier_rmse
        )

        target.transform(reg.transformation)
        return target
