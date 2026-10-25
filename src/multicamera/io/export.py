"""Export calibration results and point clouds to various formats."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..calibration.models import (
    CameraIntrinsics,
    CameraModel,
    MultiCameraRig,
    StereoPairCalibration,
)

logger = logging.getLogger(__name__)


def export_rig_json(rig: MultiCameraRig, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(rig.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Exported calibration JSON: path=%s pairs=%d cameras=%d aux=%d",
        path,
        len(rig.pairs),
        len(_camera_intrinsics(rig)),
        len(rig.aux_cameras),
    )
    return path


def export_rig_opencv_yaml(rig: MultiCameraRig, path: str | Path) -> Path:
    """Export a rig as OpenCV FileStorage YAML.

    The output is intentionally flat because OpenCV's Python FileStorage API
    reads matrix nodes most reliably when each calibration item has a stable key.
    """
    path = Path(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)

    fs.write("reference_camera", rig.reference_camera)
    fs.write("camera_count", len(_camera_intrinsics(rig)))
    fs.write("pair_count", len(rig.pairs))
    fs.write("aux_camera_count", len(rig.aux_cameras))

    primary_pair = next(iter(rig.pairs.values()), None)
    if primary_pair is not None:
        _write_opencv_stereo_aliases(fs, primary_pair)

    for cam_id, intr in _camera_intrinsics(rig).items():
        prefix = f"camera_{_safe_key(cam_id)}"
        fs.write(f"{prefix}_id", cam_id)
        fs.write(f"{prefix}_model", intr.model.value)
        fs.write(f"{prefix}_image_width", int(intr.image_size[0]))
        fs.write(f"{prefix}_image_height", int(intr.image_size[1]))
        fs.write(f"{prefix}_camera_matrix", intr.camera_matrix)
        fs.write(f"{prefix}_distortion_coefficients", intr.dist_coeffs)
        fs.write(f"{prefix}_rms_error", float(intr.rms_error))

    for pair_name, pair in rig.pairs.items():
        prefix = f"pair_{_safe_key(pair_name)}"
        fs.write(f"{prefix}_name", pair_name)
        fs.write(f"{prefix}_left_id", pair.left_id)
        fs.write(f"{prefix}_right_id", pair.right_id)
        _write_opencv_stereo_aliases(fs, pair, prefix=prefix)
        fs.write(f"{prefix}_R", pair.stereo.R)
        fs.write(f"{prefix}_T", pair.stereo.T)
        fs.write(f"{prefix}_E", pair.stereo.E)
        fs.write(f"{prefix}_F", pair.stereo.F)
        fs.write(f"{prefix}_rms_error", float(pair.stereo.rms_error))
        if pair.stereo.Q is not None:
            fs.write(f"{prefix}_Q", pair.stereo.Q)
        if pair.stereo.R1 is not None:
            fs.write(f"{prefix}_R1", pair.stereo.R1)
            fs.write(f"{prefix}_R2", pair.stereo.R2)
            fs.write(f"{prefix}_P1", pair.stereo.P1)
            fs.write(f"{prefix}_P2", pair.stereo.P2)

    for cam_id, ext in rig.extrinsics.items():
        prefix = f"extrinsic_{_safe_key(cam_id)}"
        fs.write(f"{prefix}_id", cam_id)
        fs.write(f"{prefix}_R", ext.R)
        fs.write(f"{prefix}_T", ext.T)
        fs.write(f"{prefix}_T_world_cam", ext.transformation_matrix)

    for aux_id, aux in rig.aux_cameras.items():
        prefix = f"aux_{_safe_key(aux_id)}"
        fs.write(f"{prefix}_id", aux.camera_id)
        fs.write(f"{prefix}_modality", aux.modality)
        fs.write(f"{prefix}_board_type", aux.board_type)
        if aux.board_config is not None:
            fs.write(f"{prefix}_board_cols", int(aux.board_config["cols"]))
            fs.write(f"{prefix}_board_rows", int(aux.board_config["rows"]))
            fs.write(f"{prefix}_board_square_size_m", float(aux.board_config["square_size"]))
            fs.write(f"{prefix}_board_asymmetric", int(aux.board_config.get("asymmetric", False)))
        fs.write(f"{prefix}_frame_count", int(aux.frame_count))
        fs.write(f"{prefix}_rms_error", float(aux.rms_error))
        fs.write(f"{prefix}_T_rgb_left_aux", aux.T_rgb_left_aux)
        fs.write(f"{prefix}_R_rgb_left_aux", aux.R_rgb_left_aux)
        fs.write(f"{prefix}_t_rgb_left_aux", aux.t_rgb_left_aux)

    fs.release()
    logger.info(
        "Exported OpenCV full YAML: path=%s pairs=%d cameras=%d aux=%d",
        path,
        len(rig.pairs),
        len(_camera_intrinsics(rig)),
        len(rig.aux_cameras),
    )
    return path


def export_rig_opencv_standard_yaml(
    rig: MultiCameraRig,
    path: str | Path,
    pair_name: str | None = None,
) -> Path:
    """Export one stereo pair using common OpenCV stereo calibration keys.

    This format is intentionally small and third-party friendly. It writes the
    matrix names used by many OpenCV examples: M1/D1/M2/D2/R/T/E/F/R1/R2/P1/P2/Q.
    For rigs with multiple stereo pairs, the first pair is exported unless
    ``pair_name`` is provided.
    """
    pair = _select_pair_for_standard_export(rig, pair_name)
    path = Path(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)

    fs.write("calibration_type", "stereo")
    fs.write("pair_name", pair.pair_name)
    fs.write("left_camera_id", pair.left_id)
    fs.write("right_camera_id", pair.right_id)
    _write_opencv_standard_stereo_fields(fs, pair)

    fs.release()
    logger.info(
        "Exported OpenCV standard stereo YAML: path=%s pair=%s left=%s right=%s",
        path,
        pair.pair_name,
        pair.left_id,
        pair.right_id,
    )
    return path


def _write_opencv_stereo_aliases(
    fs: cv2.FileStorage,
    pair: StereoPairCalibration,
    prefix: str = "",
):
    """Write common OpenCV stereo calibration aliases.

    OpenCV examples often read M1/D1/M2/D2/R/T/E/F/R1/R2/P1/P2/Q directly.
    The prefixed variant lets multi-pair rigs expose the same aliases without
    losing the stable project-specific keys.
    """
    key = f"{prefix}_" if prefix else ""
    fs.write(f"{key}calibration_type", "stereo")
    fs.write(f"{key}pair_name", pair.pair_name)
    fs.write(f"{key}left_camera_id", pair.left_id)
    fs.write(f"{key}right_camera_id", pair.right_id)
    fs.write(f"{key}image_width", int(pair.left_intrinsics.image_size[0]))
    fs.write(f"{key}image_height", int(pair.left_intrinsics.image_size[1]))
    fs.write(
        f"{key}image_size",
        np.array(
            [[int(pair.left_intrinsics.image_size[0]), int(pair.left_intrinsics.image_size[1])]],
            dtype=np.int32,
        ),
    )

    fs.write(f"{key}camera_matrix", pair.left_intrinsics.camera_matrix)
    fs.write(f"{key}distortion_coefficients", pair.left_intrinsics.dist_coeffs)
    fs.write(f"{key}M1", pair.left_intrinsics.camera_matrix)
    fs.write(f"{key}D1", pair.left_intrinsics.dist_coeffs)
    fs.write(f"{key}M2", pair.right_intrinsics.camera_matrix)
    fs.write(f"{key}D2", pair.right_intrinsics.dist_coeffs)
    fs.write(f"{key}R", pair.stereo.R)
    fs.write(f"{key}T", pair.stereo.T)
    fs.write(f"{key}E", pair.stereo.E)
    fs.write(f"{key}F", pair.stereo.F)
    if pair.stereo.R1 is not None:
        fs.write(f"{key}R1", pair.stereo.R1)
        fs.write(f"{key}R2", pair.stereo.R2)
        fs.write(f"{key}P1", pair.stereo.P1)
        fs.write(f"{key}P2", pair.stereo.P2)
    if pair.stereo.Q is not None:
        fs.write(f"{key}Q", pair.stereo.Q)


def _write_opencv_standard_stereo_fields(
    fs: cv2.FileStorage,
    pair: StereoPairCalibration,
):
    width = int(pair.left_intrinsics.image_size[0])
    height = int(pair.left_intrinsics.image_size[1])
    image_size = np.array([[width, height]], dtype=np.int32)

    fs.write("image_width", width)
    fs.write("image_height", height)
    fs.write("image_size", image_size)
    fs.write("imageSize", image_size)

    fs.write("camera_matrix", pair.left_intrinsics.camera_matrix)
    fs.write("distortion_coefficients", pair.left_intrinsics.dist_coeffs)
    fs.write("M1", pair.left_intrinsics.camera_matrix)
    fs.write("D1", pair.left_intrinsics.dist_coeffs)
    fs.write("M2", pair.right_intrinsics.camera_matrix)
    fs.write("D2", pair.right_intrinsics.dist_coeffs)
    fs.write("K1", pair.left_intrinsics.camera_matrix)
    fs.write("K2", pair.right_intrinsics.camera_matrix)
    fs.write("cameraMatrix1", pair.left_intrinsics.camera_matrix)
    fs.write("distCoeffs1", pair.left_intrinsics.dist_coeffs)
    fs.write("cameraMatrix2", pair.right_intrinsics.camera_matrix)
    fs.write("distCoeffs2", pair.right_intrinsics.dist_coeffs)

    fs.write("R", pair.stereo.R)
    fs.write("T", pair.stereo.T)
    fs.write("E", pair.stereo.E)
    fs.write("F", pair.stereo.F)
    if pair.stereo.R1 is not None:
        fs.write("R1", pair.stereo.R1)
        fs.write("R2", pair.stereo.R2)
        fs.write("P1", pair.stereo.P1)
        fs.write("P2", pair.stereo.P2)
    if pair.stereo.Q is not None:
        fs.write("Q", pair.stereo.Q)


def _select_pair_for_standard_export(
    rig: MultiCameraRig,
    pair_name: str | None,
) -> StereoPairCalibration:
    if pair_name is not None:
        try:
            return rig.pairs[pair_name]
        except KeyError as exc:
            raise ValueError(f"Unknown stereo pair: {pair_name}") from exc
    try:
        return next(iter(rig.pairs.values()))
    except StopIteration as exc:
        raise ValueError("OpenCV standard export requires at least one stereo pair") from exc


def export_rig_yaml(rig: MultiCameraRig, path: str | Path) -> Path:
    """Backward-compatible alias for OpenCV YAML export."""
    return export_rig_opencv_yaml(rig, path)


def validate_rig_kalibr_chain(rig: MultiCameraRig) -> None:
    """Reject a camera list that cannot be represented by a relative pose chain."""
    camera_ids = _ordered_camera_ids(rig)
    for previous, current in zip(camera_ids, camera_ids[1:]):
        _relative_transform(rig, previous, current)


def export_rig_kalibr_yaml(rig: MultiCameraRig, path: str | Path) -> Path:
    """Export a Kalibr-compatible camchain YAML file."""
    path = Path(path)
    intrinsics = _camera_intrinsics(rig)
    camera_ids = _ordered_camera_ids(rig)
    index_by_id = {cid: idx for idx, cid in enumerate(camera_ids)}
    overlaps = _camera_overlaps(rig, index_by_id)

    lines = [
        "# Camera calibration generated by MultiCamera.",
        "# Units: metres for translations, pixels for intrinsics.",
    ]
    for idx, cam_id in enumerate(camera_ids):
        intr = intrinsics[cam_id]
        lines.append(f"cam{idx}:")
        lines.append(f"  camera_model: {_kalibr_camera_model(intr)}")
        lines.append(f"  distortion_coeffs: {_yaml_flow(_kalibr_distortion(intr))}")
        lines.append(f"  distortion_model: {_kalibr_distortion_model(intr)}")
        lines.append(f"  intrinsics: {_yaml_flow(_kalibr_intrinsics(intr))}")
        lines.append(f"  resolution: [{int(intr.image_size[0])}, {int(intr.image_size[1])}]")
        lines.append(f"  rostopic: /{cam_id}/image_raw")
        lines.append(f"  cam_overlaps: {_yaml_flow(overlaps.get(cam_id, []), int_values=True)}")
        if idx > 0:
            lines.append("  T_cn_cnm1:")
            lines.extend(_yaml_matrix(_relative_transform(rig, camera_ids[idx - 1], cam_id), 4))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        "Exported Kalibr camchain YAML: path=%s cameras=%d pairs=%d aux=%d",
        path,
        len(camera_ids),
        len(rig.pairs),
        len(rig.aux_cameras),
    )
    return path


def export_rig_calibration_bundle(
    rig: MultiCameraRig,
    path: str | Path,
    stem: str = "calibration_result",
) -> list[Path]:
    """Export JSON, OpenCV YAML, and Kalibr camchain YAML into a directory."""
    base = Path(path)
    base.mkdir(parents=True, exist_ok=True)
    return [
        export_rig_json(rig, base / f"{stem}.json"),
        export_rig_opencv_yaml(rig, base / f"{stem}_opencv.yml"),
        export_rig_opencv_standard_yaml(rig, base / f"{stem}_opencv_standard.yml"),
        export_rig_kalibr_yaml(rig, base / f"{stem}_kalibr.yaml"),
    ]


def _camera_intrinsics(rig: MultiCameraRig) -> dict[str, CameraIntrinsics]:
    cams: dict[str, CameraIntrinsics] = {}
    for pair in rig.pairs.values():
        cams.setdefault(pair.left_id, pair.left_intrinsics)
        cams.setdefault(pair.right_id, pair.right_intrinsics)
    for aux in rig.aux_cameras.values():
        cams.setdefault(aux.camera_id, aux.intrinsics)
    return cams


def _ordered_camera_ids(rig: MultiCameraRig) -> list[str]:
    ordered: list[str] = []
    for pair in rig.pairs.values():
        for cam_id in (pair.left_id, pair.right_id):
            if cam_id not in ordered:
                ordered.append(cam_id)
    for aux_id in rig.aux_cameras:
        if aux_id not in ordered:
            ordered.append(aux_id)
    return ordered


def _camera_overlaps(rig: MultiCameraRig, index_by_id: dict[str, int]) -> dict[str, list[int]]:
    overlaps: dict[str, set[int]] = {cid: set() for cid in index_by_id}
    for pair in rig.pairs.values():
        left_idx = index_by_id[pair.left_id]
        right_idx = index_by_id[pair.right_id]
        overlaps[pair.left_id].add(right_idx)
        overlaps[pair.right_id].add(left_idx)
    ref = rig.reference_camera
    for aux in rig.aux_cameras.values():
        if ref in index_by_id and aux.camera_id in index_by_id:
            overlaps[ref].add(index_by_id[aux.camera_id])
            overlaps[aux.camera_id].add(index_by_id[ref])
    return {cid: sorted(v) for cid, v in overlaps.items()}


def _relative_transform(rig: MultiCameraRig, prev_id: str, cam_id: str) -> np.ndarray:
    prev_ext = rig.get_extrinsics(prev_id)
    cam_ext = rig.get_extrinsics(cam_id)
    if prev_ext is not None and cam_ext is not None:
        return cam_ext.transformation_matrix @ np.linalg.inv(prev_ext.transformation_matrix)

    for pair in rig.pairs.values():
        if pair.left_id == prev_id and pair.right_id == cam_id:
            return _rt_matrix(pair.stereo.R, pair.stereo.T)
        if pair.right_id == prev_id and pair.left_id == cam_id:
            return np.linalg.inv(_rt_matrix(pair.stereo.R, pair.stereo.T))

    for aux in rig.aux_cameras.values():
        if aux.camera_id == cam_id and prev_id == rig.reference_camera:
            return np.linalg.inv(aux.T_rgb_left_aux)
        if aux.camera_id == prev_id and cam_id == rig.reference_camera:
            return aux.T_rgb_left_aux
        if aux.camera_id == cam_id:
            T_ref_prev = (
                prev_ext.transformation_matrix
                if prev_ext is not None
                else _reference_to_camera_transform(rig, prev_id)
            )
            if T_ref_prev is None:
                continue
            T_ref_aux = np.linalg.inv(aux.T_rgb_left_aux)
            return T_ref_aux @ np.linalg.inv(T_ref_prev)
        if aux.camera_id == prev_id:
            T_ref_aux = np.linalg.inv(aux.T_rgb_left_aux)
            T_ref_cam = (
                cam_ext.transformation_matrix
                if cam_ext is not None
                else _reference_to_camera_transform(rig, cam_id)
            )
            if T_ref_cam is None:
                continue
            return T_ref_cam @ np.linalg.inv(T_ref_aux)

    raise ValueError(f"No relative transform available from {prev_id} to {cam_id}")


def _rt_matrix(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R
    mat[:3, 3] = T.reshape(3)
    return mat


def _reference_to_camera_transform(rig: MultiCameraRig, camera_id: str) -> np.ndarray | None:
    if camera_id == rig.reference_camera:
        return np.eye(4, dtype=np.float64)
    for pair in rig.pairs.values():
        if pair.left_id == rig.reference_camera and pair.right_id == camera_id:
            return _rt_matrix(pair.stereo.R, pair.stereo.T)
        if pair.right_id == rig.reference_camera and pair.left_id == camera_id:
            return np.linalg.inv(_rt_matrix(pair.stereo.R, pair.stereo.T))
    return None


def _kalibr_camera_model(intr: CameraIntrinsics) -> str:
    return "pinhole"


def _kalibr_distortion_model(intr: CameraIntrinsics) -> str:
    if intr.model == CameraModel.FISHEYE:
        return "equidistant"
    return "radtan"


def _kalibr_intrinsics(intr: CameraIntrinsics) -> list[float]:
    K = intr.camera_matrix
    return [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]


def _kalibr_distortion(intr: CameraIntrinsics) -> list[float]:
    dist = intr.dist_coeffs.reshape(-1).astype(float).tolist()
    if intr.model == CameraModel.FISHEYE:
        return (dist + [0.0, 0.0, 0.0, 0.0])[:4]
    k1 = dist[0] if len(dist) > 0 else 0.0
    k2 = dist[1] if len(dist) > 1 else 0.0
    p1 = dist[2] if len(dist) > 2 else 0.0
    p2 = dist[3] if len(dist) > 3 else 0.0
    return [k1, k2, p1, p2]


def _yaml_matrix(mat: np.ndarray, indent: int) -> list[str]:
    pad = " " * indent
    return [f"{pad}- {_yaml_flow(row)}" for row in np.asarray(mat).tolist()]


def _yaml_flow(values, int_values: bool = False) -> str:
    if int_values:
        return "[" + ", ".join(str(int(v)) for v in values) + "]"
    return "[" + ", ".join(_yaml_float(float(v)) for v in values) + "]"


def _yaml_float(value: float) -> str:
    return f"{value:.12g}"


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)


def export_pointcloud_ply(
    pcd,
    path: str | Path,
    write_ascii: bool = False,
) -> Path:
    o3d = _require_open3d()
    path = Path(path)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=write_ascii)
    return path


def export_pointcloud_pcd(
    pcd,
    path: str | Path,
    write_ascii: bool = False,
) -> Path:
    o3d = _require_open3d()
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


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for point cloud export. "
            "Install the pointcloud extra or build Windows package with --pointcloud."
        ) from exc
    return o3d
