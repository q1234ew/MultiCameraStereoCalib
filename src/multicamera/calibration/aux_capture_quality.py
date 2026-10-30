"""Single original-frame admission for planar RGB/IR boards, not accuracy certification.

No pairing, temporal stability or duplicate policy is applied here. In particular,
neither a full grid nor a reliable PnP fit resolves physical corner ordering on a
symmetric board. Callers must establish that correspondence across modalities.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .auxiliary import _solve_board_pose, _validate_intrinsics, _validated_detection
from .models import CameraIntrinsics, CameraModel
from .planar import PatternBoardConfig, PatternType, PlanarDetectionResult


@dataclass(frozen=True)
class AuxFrameQuality:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: dict
    pose: dict | None


REASON_LABELS = {
    'invalid_image': '原图格式或像素无效', 'invalid_board': '标定板配置无效',
    'invalid_detection': '检测数据无效', 'invalid_or_incomplete_grid': '未检测到完整网格',
    'detection_size_mismatch': '检测与原图分辨率不一致', 'points_outside_image': '检测点超出画面',
    'degenerate_grid': '网格几何退化或点顺序异常', 'grid_spacing_too_small': '角点间距不足4像素，请靠近',
    'board_margin_too_small': '板面距画面边缘不足2像素，请完整入镜',
    'low_board_contrast': '板面热/亮度对比不足，请改善板面温差或照明',
    'inconsistent_board_signal': '网格明暗信号不一致，请检查反光、遮挡或热对比',
    'blurred_board_edges': '板边缘模糊，请对焦并保持静止',
    'invalid_intrinsics': '内参模型或尺寸不匹配', 'unreliable_pose': '当前位姿不可靠',
    'invalid_modality': '传感器类型无效',
}


def quality_reason_text(code):
    return REASON_LABELS.get(code, code)


def _gray(image: np.ndarray) -> tuple[np.ndarray, str]:
    """Fixed-scale display luminance, never per-frame contrast normalization."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("invalid_image")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise ValueError("invalid_image")
    if min(image.shape[:2]) < 3 or image.dtype not in (
        np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32"), np.dtype("float64")
    ):
        raise ValueError("invalid_image")
    if not np.isfinite(image).all() or np.min(image) < 0:
        raise ValueError("invalid_image")
    if image.dtype == np.uint8:
        scale, encoding = 255.0, "uint8"
    elif image.dtype == np.uint16:
        scale, encoding = 65535.0, "uint16"
    elif np.max(image) <= 1:
        scale, encoding = 1.0, "float_0_1"
    elif np.max(image) <= 255:
        scale, encoding = 255.0, "float_0_255"
    else:
        raise ValueError("invalid_image")
    values = image.astype(np.float32) / scale
    if values.ndim == 3:
        if values.shape[2] == 1:
            values = values[:, :, 0]
        else:
            # Match the application's BGR convention. Do not maximize contrast
            # over channels: a false-color palette is not radiometric evidence.
            values = cv2.cvtColor(values, cv2.COLOR_BGR2GRAY if values.shape[2] == 3
                                  else cv2.COLOR_BGRA2GRAY)
    return values, encoding


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _geometry(points, board, size, metrics, thresholds):
    if np.any(points < 0) or np.any(points > np.asarray(size) - 1):
        return "points_outside_image", None
    if len(np.unique(points, axis=0)) != len(points):
        return "degenerate_grid", None
    grid = points.reshape(board.rows, board.cols, 2)
    horizontal = np.diff(grid, axis=1)
    vertical = np.diff(grid, axis=0)
    spacing = np.concatenate((np.linalg.norm(horizontal, axis=2).ravel(),
                              np.linalg.norm(vertical, axis=2).ravel()))
    metrics["min_grid_spacing_px"] = float(spacing.min())
    metrics["median_grid_spacing_px"] = float(np.median(spacing))
    if spacing.min() < thresholds["min_grid_spacing_px"]:
        return "grid_spacing_too_small", None
    quads = np.stack((grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]), axis=2)
    edges = np.roll(quads, -1, axis=2) - quads
    turns = _cross(edges, np.roll(edges, -1, axis=2))
    sines = turns / (np.linalg.norm(edges, axis=3)
                     * np.linalg.norm(np.roll(edges, -1, axis=2), axis=3))
    metrics["min_cell_sine"] = float(np.min(np.abs(sines)))
    minimum_sine = thresholds["min_cell_sine"]
    if not (np.all(sines >= minimum_sine) or np.all(sines <= -minimum_sine)):
        return "degenerate_grid", None

    # Extrapolate the observed perimeter locally, preserving lens curvature.
    # Chessboards have one outer square beyond inner corners. Circle configs do
    # not specify a substrate border; use a half-spacing dot-support envelope.
    pad = 1.0 if board.pattern_type == PatternType.CHESSBOARD else .5
    extended = np.empty((board.rows + 2, board.cols + 2, 2))
    extended[1:-1, 1:-1] = grid
    extended[0, 1:-1] = grid[0] + pad * (grid[0] - grid[1])
    extended[-1, 1:-1] = grid[-1] + pad * (grid[-1] - grid[-2])
    extended[:, 0] = extended[:, 1] + pad * (extended[:, 1] - extended[:, 2])
    extended[:, -1] = extended[:, -2] + pad * (extended[:, -2] - extended[:, -3])
    perimeter = np.concatenate((extended[0], extended[1:, -1],
                                extended[-1, -2::-1], extended[-2:0:-1, 0]))
    margin = np.min(np.concatenate((perimeter, np.asarray(size) - 1 - perimeter)))
    metrics["board_margin_px"] = float(margin)
    metrics["board_margin_basis"] = "extrapolated_grid_envelope"
    if margin < thresholds["min_board_margin_px"]:
        return "board_margin_too_small", None
    return None, grid


def _sample(gray, points):
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2).astype(np.float32)
    # remap dimensions must stay below SHRT_MAX, even for a 100 x 100 grid.
    sampled = [cv2.remap(gray, part[:, 0].reshape(-1, 1), part[:, 1].reshape(-1, 1),
                         cv2.INTER_LINEAR).ravel() for part in np.array_split(
                             flat, max(1, (len(flat) + 31999) // 32000))]
    return np.concatenate(sampled).reshape(shape)


def _profiles(gray, grid, board):
    if board.pattern_type == PatternType.CHESSBOARD:
        centers = (grid[:-1, :-1] + grid[:-1, 1:]
                   + grid[1:, :-1] + grid[1:, 1:]) / 4
        starts = np.concatenate((centers[:, :-1].reshape(-1, 2),
                                 centers[:-1].reshape(-1, 2)))
        ends = np.concatenate((centers[:, 1:].reshape(-1, 2),
                               centers[1:].reshape(-1, 2)))
        parity = np.indices(centers.shape[:2]).sum(axis=0) % 2
        signs = np.concatenate((1 - 2 * parity[:, :-1].ravel(),
                                1 - 2 * parity[:-1].ravel()))
    else:
        # Dot-to-background radial transitions, not chessboard polarity. The
        # asymmetric grid's horizontal distance is twice its basic spacing.
        centers = grid.reshape(-1, 2)
        radius = min(np.linalg.norm(np.diff(grid, axis=0), axis=2).min(),
                     np.linalg.norm(np.diff(grid, axis=1), axis=2).min()) * .45
        angles = np.arange(8) * (np.pi / 4)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        starts = np.repeat(centers, 8, axis=0)
        ends = (centers[:, None] + radius * directions).reshape(-1, 2)
        signs = np.ones(len(starts))
    t = np.linspace(0, 1, 65)
    profiles = _sample(gray, starts[:, None] * (1 - t[None, :, None])
                       + ends[:, None] * t[None, :, None])
    # Robust endpoints damp isolated pixel/thermal texture without applying a
    # blur to the original image. Sharpness is measured along expected edges.
    first = np.median(profiles[:, :7], axis=1)
    last = np.median(profiles[:, -7:], axis=1)
    amplitude = last - first
    signed = amplitude * signs
    polarity = 1 if np.median(signed) >= 0 else -1
    return profiles, first, amplitude, signed * polarity


def _appearance(gray, grid, board, metrics, thresholds):
    hull = cv2.convexHull(grid.reshape(-1, 2).astype(np.float32))
    x, y, w, h = cv2.boundingRect(hull)
    roi = gray[y:y + h, x:x + w]
    mask = np.zeros(roi.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull[:, 0] - [x, y]).astype(np.int32), 1)
    low, high = np.percentile(roi[mask != 0], [10, 90])
    metrics["roi_contrast"] = float(high - low)
    metrics["roi_luminance_p10"] = float(low)
    metrics["roi_luminance_p90"] = float(high)
    profiles, first, amplitude, signed = _profiles(gray, grid, board)
    minimum = thresholds["min_contrast"]
    informative = signed >= minimum
    metrics["pattern_contrast"] = float(np.median(signed))
    metrics["pattern_agreement"] = float(informative.mean())
    metrics["profile_count"] = len(profiles)
    reasons = []
    # Sparse circle dots can legitimately occupy less than 10% of the ROI;
    # their center-to-ring contrast, not area percentiles, is the signal gate.
    if ((board.pattern_type == PatternType.CHESSBOARD and high - low < minimum)
            or np.median(signed) < minimum):
        reasons.append("low_board_contrast")
    if informative.mean() < thresholds["min_pattern_agreement"]:
        reasons.append("inconsistent_board_signal")
    widths = []
    for profile, start, delta in zip(profiles[informative], first[informative],
                                     amplitude[informative]):
        normalized = (profile - start) / delta
        # Crossing interpolation includes the native pixel sampling footprint.
        crossings = []
        for level in (.1, .9):
            indices = np.flatnonzero((normalized[:-1] < level) & (normalized[1:] >= level))
            if not len(indices):
                crossings.append(None)
                continue
            i = indices[np.argmin(np.abs(indices - 32))]
            crossings.append((i + (level - normalized[i])
                              / (normalized[i + 1] - normalized[i])) / 64)
        widths.append(crossings[1] - crossings[0] if None not in crossings
                      and crossings[1] > crossings[0] else 1.0)
    width = float(np.percentile(widths, 75)) if widths else None
    metrics["edge_width_fraction_p75"] = width
    if width is None or width > thresholds["max_edge_width_fraction"]:
        reasons.append("blurred_board_edges")
    return reasons


def _project(obj, rvec, tvec, intr):
    K = np.asarray(intr.camera_matrix, np.float64)
    dist = np.asarray(intr.dist_coeffs, np.float64)
    if intr.model == CameraModel.FISHEYE:
        result, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, dist)
    else:
        result, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return result.reshape(-1, 2)


def _pose(det, intr, metrics, thresholds):
    solved = _solve_board_pose(det, intr)
    if solved is None:
        return None
    rvec, tvec = (np.asarray(v, np.float64).reshape(3) for v in solved)
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    R, _ = cv2.Rodrigues(rvec)
    obj = det.object_points
    camera_points = obj @ R.T + tvec
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
        return None
    residual = np.linalg.norm(_project(obj, rvec, tvec, intr) - det.image_points, axis=1)
    if not np.isfinite(residual).all():
        return None
    rms = float(np.sqrt(np.mean(residual ** 2)))
    limit = min(thresholds["max_pose_rms_px"], thresholds["max_pose_rms_spacing_fraction"]
                * metrics["min_grid_spacing_px"])
    metrics.update(pose_rms_px=rms, pose_max_residual_px=float(residual.max()),
                   pose_rms_limit_px=limit)
    if rms > limit or residual.max() > thresholds["max_pose_residual_rms_limit_ratio"] * limit:
        return None
    center_obj = obj.mean(axis=0)
    center = R @ center_obj + tvec
    distance = float(np.linalg.norm(center))
    if (abs(R[2, 2]) < thresholds["min_pose_normal_z_abs"]
            or not np.isfinite(distance) or distance <= 0):
        return None

    # Local sensitivity at a conservative 0.5 px noise floor. Parameterize
    # rotation about the board center and translation relative to distance.
    def evaluate(parameters):
        rotation, _ = cv2.Rodrigues(parameters[:3])
        translation = center + parameters[3:] * distance - rotation @ center_obj
        return _project(obj, parameters[:3], translation, intr).ravel()

    parameters = np.concatenate((rvec, np.zeros(3)))
    step = np.eye(6) * 1e-5
    jac = np.column_stack([(evaluate(parameters + d) - evaluate(parameters - d)) / 2e-5
                           for d in step])
    _, singular, vt = np.linalg.svd(jac, full_matrices=False)
    if (not np.isfinite(singular).all()
            or singular[-1] <= singular[0] * thresholds["min_pose_jacobian_singular_ratio"]):
        return None
    covariance = (vt.T * (max(thresholds["pose_noise_floor_px"], rms) / singular) ** 2) @ vt
    depth_std = float(np.sqrt(covariance[5, 5]) * distance / center[2])
    rotation_std = float(np.degrees(np.sqrt(np.max(np.diag(covariance)[:3]))))
    metrics.update(pose_relative_depth_std=depth_std, pose_rotation_std_deg=rotation_std)
    if (not np.isfinite(depth_std) or not np.isfinite(rotation_std)
            or depth_std > thresholds["max_pose_relative_depth_std"]
            or rotation_std > thresholds["max_pose_rotation_std_deg"]):
        return None
    normal = R[:, 2]
    return {
        "depth_m": float(center[2]), "distance_m": distance,
        "yaw_deg": float(np.degrees(np.arctan2(normal[0], normal[2]))),
        "pitch_deg": float(np.degrees(np.arctan2(-normal[1], np.hypot(normal[0], normal[2])))),
        "distance_reliable": True, "rvec": rvec.tolist(), "tvec_m": tvec.tolist(),
        "reference": "board_point_centroid", "reprojection_rms_px": rms,
        "reliability_basis": "reprojection_and_local_sensitivity_not_accuracy_guarantee",
    }


def assess_aux_frame(
    image: np.ndarray,
    detection: PlanarDetectionResult,
    board: PatternBoardConfig,
    intrinsics: CameraIntrinsics | None = None,
    *,
    modality: str = "ir",
) -> AuxFrameQuality:
    """Assess a complete planar grid in the full-resolution original image.

    Reasons are stable machine-readable codes; successful frames have no reasons.
    Metrics expose thresholds and normalized [0, 1] display-luminance contrast.
    Images may be gray/BGR/BGRA uint8, uint16 (native full scale), or float
    [0, 1]/[0, 255]. IR false color is converted to luminance, never interpreted
    as temperature. Palette-only structure cannot be distinguished from a real
    thermal board without raw data; this is a conservative visual admission.

    Minimum neighbor spacing is 4 px and extrapolated board margin is 2 px.
    IR/aux contrast needs 6/255, RGB 10/255; at least 75% of grid-aligned
    transitions must agree. The 75th-percentile 10-90% edge width must be at
    most 0.40 of adjacent cell-center distance (dot-profile radius for circles).
    An absent intrinsics argument permits image-only admission with pose=None;
    invalid intrinsics or an unreliable requested pose reject the frame.
    Pose depth/distance refer to the detected-point centroid in metres; yaw
    and pitch describe the board normal in camera coordinates, in degrees.
    Reliability is conditional on the supplied intrinsics and board dimensions.
    """
    thresholds = {
        "min_grid_spacing_px": 4.0, "min_board_margin_px": 2.0,
        "min_cell_sine": .05,
        "min_contrast": (10 if isinstance(modality, str) and modality == "rgb" else 6) / 255,
        "min_pattern_agreement": .75, "max_edge_width_fraction": .40,
        "max_pose_rms_px": 1.5, "max_pose_rms_spacing_fraction": .1,
        "max_pose_residual_rms_limit_ratio": 2.0, "min_pose_normal_z_abs": .1,
        "pose_noise_floor_px": .5, "max_pose_relative_depth_std": .1,
        "max_pose_rotation_std_deg": 10.0,
        "min_pose_jacobian_singular_ratio": 1e-8,
    }
    metrics = {"modality": modality, "thresholds": thresholds, "pose_status": "not_requested"}

    def reject(reason):
        return AuxFrameQuality(False, (reason,), metrics, None)

    if not isinstance(modality, str) or modality not in ("rgb", "ir", "aux"):
        return reject("invalid_modality")
    try:
        gray, encoding = _gray(image)
    except ValueError as exc:
        return reject(str(exc))
    size = (gray.shape[1], gray.shape[0])
    metrics.update(image_size=size, image_encoding=encoding)
    if not isinstance(board, PatternBoardConfig):
        return reject("invalid_board")
    try:
        board = PatternBoardConfig(**board.to_dict())
    except (AttributeError, TypeError, ValueError, OverflowError):
        return reject("invalid_board")
    if not isinstance(detection, PlanarDetectionResult):
        return reject("invalid_detection")
    try:
        if not isinstance(detection.valid, (bool, np.bool_)):
            return reject("invalid_or_incomplete_grid")
        if any(np.asarray(p).dtype.kind not in "iuf" for p in (
            detection.object_points, detection.image_points
        )):
            return reject("invalid_or_incomplete_grid")
        det = _validated_detection(detection, board)
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return reject("invalid_or_incomplete_grid")
    if det is None:
        return reject("invalid_or_incomplete_grid")
    if det.image_size != size:
        return reject("detection_size_mismatch")
    metrics["point_count"] = det.num_points
    reason, grid = _geometry(det.image_points, board, size, metrics, thresholds)
    if reason:
        return reject(reason)
    if intrinsics is not None:
        if not isinstance(intrinsics, CameraIntrinsics):
            return reject("invalid_intrinsics")
        try:
            if any(np.asarray(p).dtype.kind not in "iuf" for p in (
                intrinsics.camera_matrix, intrinsics.dist_coeffs
            )):
                raise ValueError("Intrinsics must be real numeric arrays")
            _validate_intrinsics(intrinsics, size)
        except (ValueError, TypeError):
            metrics["pose_status"] = "invalid_intrinsics"
            return reject("invalid_intrinsics")
    reasons = _appearance(gray, grid, board, metrics, thresholds)
    pose = None
    if intrinsics is not None and not reasons:
        try:
            pose = _pose(det, intrinsics, metrics, thresholds)
        except (cv2.error, ValueError, TypeError, OverflowError, np.linalg.LinAlgError):
            pose = None
        metrics["pose_status"] = "reliable" if pose is not None else "unreliable"
        if pose is None:
            reasons.append("unreliable_pose")
    elif intrinsics is not None:
        metrics["pose_status"] = "skipped_image_quality"
    return AuxFrameQuality(not reasons, tuple(reasons), metrics, pose)
