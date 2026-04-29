"""Export calibration results and point clouds to various formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

from ..calibration.models import MultiCameraRig


def export_rig_json(rig: MultiCameraRig, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(rig.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def export_rig_yaml(rig: MultiCameraRig, path: str | Path) -> Path:
    """Export using OpenCV FileStorage (YAML format)."""
    path = Path(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)

    fs.write("reference_camera", rig.reference_camera)

    for pair_name, pair in rig.pairs.items():
        prefix = f"pair_{pair_name}"
        fs.write(f"{prefix}_left_K", pair.left_intrinsics.camera_matrix)
        fs.write(f"{prefix}_left_dist", pair.left_intrinsics.dist_coeffs)
        fs.write(f"{prefix}_right_K", pair.right_intrinsics.camera_matrix)
        fs.write(f"{prefix}_right_dist", pair.right_intrinsics.dist_coeffs)
        fs.write(f"{prefix}_R", pair.stereo.R)
        fs.write(f"{prefix}_T", pair.stereo.T)
        if pair.stereo.Q is not None:
            fs.write(f"{prefix}_Q", pair.stereo.Q)

    for cam_id, ext in rig.extrinsics.items():
        fs.write(f"ext_{cam_id}_R", ext.R)
        fs.write(f"ext_{cam_id}_T", ext.T)

    fs.release()
    return path


def export_pointcloud_ply(
    pcd: "o3d.geometry.PointCloud",
    path: str | Path,
    write_ascii: bool = False,
) -> Path:
    if o3d is None:
        raise ImportError("open3d is required for PLY export")
    path = Path(path)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=write_ascii)
    return path


def export_pointcloud_pcd(
    pcd: "o3d.geometry.PointCloud",
    path: str | Path,
    write_ascii: bool = False,
) -> Path:
    if o3d is None:
        raise ImportError("open3d is required for PCD export")
    path = Path(path)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=write_ascii)
    return path


def export_pointcloud_numpy(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    path: str | Path,
) -> Path:
    """Export point cloud as compressed .npz for easy loading."""
    path = Path(path)
    data = {"points": points}
    if colors is not None:
        data["colors"] = colors
    np.savez_compressed(str(path), **data)
    return path
