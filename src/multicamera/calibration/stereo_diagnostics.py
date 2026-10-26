"""Pixel-space rectification residuals for stored same-ID stereo observations."""

from __future__ import annotations

import cv2
import numpy as np

from .models import CameraIntrinsics, CameraModel, StereoCalibResult


def _matrix(value, shape, name):
    if value is None:
        raise ValueError(f"Missing {name}")
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {shape} matrix")
    return result


def _intrinsics(intrinsics, model, name):
    k = _matrix(intrinsics.camera_matrix, (3, 3), f"{name}.camera_matrix")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError(f"{name} must have positive focal lengths")
    try:
        d = np.array(intrinsics.dist_coeffs, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}.dist_coeffs") from exc
    counts = (4,) if model == CameraModel.FISHEYE else (4, 5, 8, 12, 14)
    if (d.ndim not in (1, 2) or (d.ndim == 2 and 1 not in d.shape)
            or d.size not in counts or not np.isfinite(d).all()):
        raise ValueError(f"Invalid {name}.dist_coeffs for {model}")
    return k, d


def _points(value, name):
    try:
        points = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not ((points.ndim == 2 and points.shape[1] == 2)
            or (points.ndim == 3 and points.shape[2] == 2
                and 1 in points.shape[:2])):
        raise ValueError(f"{name} must have shape (N, 2), (N, 1, 2), or (1, N, 2)")
    if not points.size:
        raise ValueError(f"{name} contains no points")
    return points.reshape(-1, 1, 2)


def _statistics(errors):
    invalid = int(np.count_nonzero(~np.isfinite(errors)))
    result = {
        "status": "invalid" if invalid or not errors.size else "ok",
        "point_count": int(errors.size),
        "valid_point_count": int(errors.size) - invalid,
        "invalid_count": invalid,
        "rms_px": None,
        "mean_abs_px": None,
        "median_abs_px": None,
        "p95_abs_px": None,
        "max_abs_px": None,
    }
    if result["status"] == "invalid":
        return result
    absolute = np.abs(errors)
    scale = float(absolute.max())
    # Scaling avoids squaring overflow without clipping large finite residuals.
    normalized = absolute / scale if scale else absolute
    result.update(
        rms_px=scale * float(np.sqrt(np.mean(normalized ** 2))),
        mean_abs_px=scale * float(np.mean(normalized)),
        median_abs_px=scale * float(np.median(normalized)),
        p95_abs_px=scale * float(np.percentile(normalized, 95)),
        max_abs_px=scale,
    )
    return result


def measure_rectification(
    calibrator,
    left_intr: CameraIntrinsics,
    right_intr: CameraIntrinsics,
    stereo: StereoCalibResult,
) -> dict:
    """Measure same-ID residuals in rectified pixels without changing inputs.

    Stored left/right rows must already refer to the same corner IDs. Axis is
    the epipolar direction; coordinate is the perpendicular component measured
    as left minus right. All points contribute, including out-of-image points.
    Any nonfinite point invalidates its frame and the aggregate statistics;
    invalid per-point residuals are represented by None for strict JSON output.
    """
    model = stereo.model
    if model not in (CameraModel.PINHOLE, CameraModel.FISHEYE):
        raise ValueError(f"Unsupported stereo model: {model}")
    if (left_intr.model != model or right_intr.model != model
            or calibrator.model != model):
        raise ValueError("Calibrator, both intrinsics, and stereo models must match")
    r1 = _matrix(stereo.R1, (3, 3), "R1")
    r2 = _matrix(stereo.R2, (3, 3), "R2")
    p1 = _matrix(stereo.P1, (3, 4), "P1")
    p2 = _matrix(stereo.P2, (3, 4), "P2")
    k1, d1 = _intrinsics(left_intr, model, "left")
    k2, d2 = _intrinsics(right_intr, model, "right")
    translation = np.abs(p2[:2, 3])
    if not np.any(translation):
        raise ValueError("P2 must contain a nonzero horizontal or vertical translation")
    horizontal = bool(translation[0] >= translation[1])
    component = 1 if horizontal else 0
    left_points = calibrator._img_points_left
    right_points = calibrator._img_points_right
    frame_ids = calibrator.frame_ids
    if len(left_points) != len(right_points) or len(frame_ids) != len(left_points):
        raise ValueError("Left/right observation lists and frame_ids must have equal lengths")
    undistort = cv2.fisheye.undistortPoints if model == CameraModel.FISHEYE else cv2.undistortPoints
    per_frame, all_errors = [], []
    for i, frame_id in enumerate(frame_ids):
        if frame_id is None:
            raise ValueError(f"Missing frame ID at index {i}")
        left = _points(left_points[i], f"left frame {frame_id}")
        right = _points(right_points[i], f"right frame {frame_id}")
        if len(left) != len(right):
            raise ValueError(f"Left/right point counts differ for frame {frame_id}")
        errors = np.full(len(left), np.nan, dtype=np.float64)
        finite = (np.isfinite(left).all(axis=(1, 2))
                  & np.isfinite(right).all(axis=(1, 2)))
        if finite.any():
            rect_l = undistort(left[finite], k1, d1, R=r1, P=p1).reshape(-1, 2)
            rect_r = undistort(right[finite], k2, d2, R=r2, P=p2).reshape(-1, 2)
            valid = np.isfinite(rect_l).all(axis=1) & np.isfinite(rect_r).all(axis=1)
            with np.errstate(over="ignore", invalid="ignore"):
                residuals = rect_l[:, component] - rect_r[:, component]
            errors[finite] = np.where(valid, residuals, np.nan)
        frame = {"frame_id": str(frame_id), **_statistics(errors)}
        frame["signed_errors_px"] = [float(e) if np.isfinite(e) else None for e in errors]
        per_frame.append(frame)
        all_errors.append(errors)
    combined = np.concatenate(all_errors) if all_errors else np.empty(0)
    return {
        "axis": "horizontal" if horizontal else "vertical",
        "coordinate": "y" if horizontal else "x",
        "unit": "px",
        "frame_count": len(per_frame),
        **_statistics(combined),
        "per_frame": per_frame,
    }
