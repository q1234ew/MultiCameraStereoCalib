"""Depth map to point cloud conversion and single-view cloud utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


def depth_to_pointcloud(
    points_3d: np.ndarray,
    color_image: Optional[np.ndarray] = None,
    max_depth: float = 10.0,
    min_depth: float = 0.1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert (H, W, 3) reprojected points to (N, 3) point cloud.

    Returns (points, colors) where colors is (N, 3) float in [0, 1] or None.
    """
    mask = np.isfinite(points_3d[:, :, 2])
    mask &= points_3d[:, :, 2] > min_depth
    mask &= points_3d[:, :, 2] < max_depth

    points = points_3d[mask].reshape(-1, 3)

    colors = None
    if color_image is not None:
        if len(color_image.shape) == 3:
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.stack([color_image] * 3, axis=-1)
        colors = rgb[mask].reshape(-1, 3).astype(np.float64) / 255.0

    return points, colors


def to_open3d_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> "o3d.geometry.PointCloud":
    """Create an Open3D point cloud from numpy arrays."""
    if o3d is None:
        raise ImportError(
            "open3d is required for point cloud operations. "
            "Install the pointcloud extra or build Windows package with --pointcloud."
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def statistical_filter(
    pcd: "o3d.geometry.PointCloud",
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> "o3d.geometry.PointCloud":
    """Remove statistical outliers."""
    if o3d is None:
        raise ImportError(
            "open3d is required. Install the pointcloud extra or build Windows package with --pointcloud."
        )
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return filtered


def voxel_downsample(
    pcd: "o3d.geometry.PointCloud",
    voxel_size: float = 0.005,
) -> "o3d.geometry.PointCloud":
    """Downsample point cloud using a voxel grid."""
    if o3d is None:
        raise ImportError(
            "open3d is required. Install the pointcloud extra or build Windows package with --pointcloud."
        )
    return pcd.voxel_down_sample(voxel_size=voxel_size)
