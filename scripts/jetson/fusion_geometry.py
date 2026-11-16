"""AprilTag pose adapters for calibrated rectified stereo camera coordinates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from vendor.owen_fusion.estimate_camera_extrinsics_from_landmarks import (
    average_transforms,
    invert_t,
)


def rectify_for_tags(device, path: Path, role, image, zoom=1):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != device["calibration_sha256"]:
        raise ValueError("Calibration hash mismatch")
    if list(image.shape[1::-1]) != device["resolution"][role]:
        raise ValueError("Camera resolution differs from calibration")
    fs = cv2.FileStorage(raw.decode(), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    try:
        intr = fs.getNode("camera_intrinsics").getNode(role)
        k, d = intr.getNode("K").mat(), intr.getNode("D").mat()
        index = "1" if role == "left" else "2"
        rect = fs.getNode("stereo_rectify")
        rotation = rect.getNode("R" + index).mat()
        projection = rect.getNode("P" + index).mat()
        p2 = rect.getNode("P2").mat()
        target = projection[:3, :3].copy()
        if zoom != 1:
            target[0, 0] *= zoom
            target[1, 1] *= zoom
            target[0, 2] = image.shape[1] / 2
            target[1, 2] = image.shape[0] / 2
        builder = (
            cv2.fisheye.initUndistortRectifyMap
            if intr.getNode("model").string() == "fisheye"
            else cv2.initUndistortRectifyMap
        )
        maps = builder(k, d, rotation, target, tuple(device["resolution"][role]), cv2.CV_32FC1)
        result = cv2.remap(image, *maps, cv2.INTER_LINEAR)
        right_from_left = np.eye(4)
        right_from_left[0, 3] = float(p2[0, 3] / p2[0, 0]) * 0.001
        return result, target, right_from_left
    finally:
        fs.release()


def detect_tags(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best = {}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for refinement in (cv2.aruco.CORNER_REFINE_APRILTAG, cv2.aruco.CORNER_REFINE_SUBPIX):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = refinement
        params.minMarkerPerimeterRate = 0.01
        params.adaptiveThreshWinSizeMax = 53
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        for scale in (1, 2):
            view = (
                gray
                if scale == 1
                else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            )
            corners, ids, _ = detector.detectMarkers(view)
            if ids is None:
                continue
            flat = ids.flatten().tolist()
            for corners_one, ident in zip(corners, flat):
                if flat.count(ident) != 1:
                    continue  # Repeated physical copies of one ID are not a unique anchor.
                if ident not in best:
                    best[ident] = corners_one.reshape(4, 2).astype(np.float64) / scale
    return best


def pose_from_tag(corners, k, side_m):
    if not np.isfinite(side_m) or not 0.01 <= side_m <= 1:
        raise ValueError("Tag side must be measured in meters, 0.01–1 m")
    object_points = np.array(
        [[0, 0, 0], [side_m, 0, 0], [side_m, side_m, 0], [0, side_m, 0]], np.float64
    )
    # A single planar square has two pose branches. Iterative PnP alone can converge
    # consistently to the wrong tilt even with subpixel reprojection residuals.
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points, np.asarray(corners, np.float64), k, np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for rotation, translation in zip(rvecs, tvecs):
        rotation, translation = cv2.solvePnPRefineLM(
            object_points, np.asarray(corners, np.float64), k, np.zeros(5),
            rotation, translation)
        if translation[2, 0] <= 0:
            continue
        projected, _ = cv2.projectPoints(object_points, rotation, translation, k, np.zeros(5))
        error = float(np.linalg.norm(projected.reshape(4, 2)-corners, axis=1).mean())
        candidates.append((error, rotation, translation))
    if not candidates:
        raise ValueError("Invalid AprilTag pose")
    candidates.sort(key=lambda value: value[0])
    error, rvec, tvec = candidates[0]
    if len(candidates) > 1:
        second_error, second_rotation, _ = candidates[1]
        relative = cv2.Rodrigues(rvec)[0].T @ cv2.Rodrigues(second_rotation)[0]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))
        if angle > 10 and second_error-error < .5 and second_error < max(error*1.5, .15):
            raise ValueError("AprilTag planar pose is ambiguous; use a clearer common marker")
    camera_from_tag = np.eye(4)
    camera_from_tag[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_from_tag[:3, 3] = tvec.ravel()
    return invert_t(camera_from_tag), error


def aggregate_poses(poses, errors):
    transform, stats = average_transforms(poses)
    angles = []
    for pose in poses:
        r = transform[:3, :3].T @ pose[:3, :3]
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))))
    stats.update(
        {
            "sample_count": len(poses),
            "reprojection_median_px": float(np.median(errors)),
            "rotation_p90_deg": float(np.percentile(angles, 90)),
        }
    )
    passed = (
        len(poses) >= 3
        and stats["translation_p90_m"] <= 0.08
        and stats["rotation_p90_deg"] <= 5
        and stats["reprojection_median_px"] <= 2
    )
    return transform, stats, passed


def validate_active(calibration, devices):
    if (
        not calibration.get("quality_passed")
        or calibration.get("source") != "foundation_stereo_rectified"
        or calibration.get("translation_unit") != "m"
    ):
        raise ValueError("需要至少两台设备通过一致性检查的当前外参")
    for d in devices:
        info = calibration.get("cameras", {}).get(d["id"])
        if (
            not info
            or info.get("calibration_sha256") != d["calibration_sha256"]
            or info.get("point_frame") != d["id"] + ":C_L_rect"
        ):
            raise ValueError(d["id"] + " 缺少与当前标定匹配的外参")
        t = np.asarray(info["T_world_camera"], np.float64)
        if (
            t.shape != (4, 4)
            or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-4)
        ):
            raise ValueError("Invalid rigid camera-to-world transform")


TOLERANCE_PROFILES = {
    'strict': {'ratio': [.95, 1.05], 'depth_p90_m': .05, 'normal_deg': 10},
    'preview': {'ratio': [.90, 1.10], 'depth_p90_m': .15, 'normal_deg': 10},
}


def check_anchor_depth(captures, calibration, tolerance_profile='strict'):
    """Independent depth/marker-scale gate; stable PnP alone cannot establish alignment."""
    import json

    if tolerance_profile not in TOLERANCE_PROFILES:
        raise ValueError('Unsupported fusion tolerance profile')
    limits = TOLERANCE_PROFILES[tolerance_profile]
    if calibration.get('source') != 'foundation_stereo_rectified':
        return {'applicable': False, 'passed': True}
    side = float(calibration['tag_side_mm']) * .001
    checks = {}
    world = np.array([[x*side, y*side, 0, 1]
                      for x in (.25, .5, .75) for y in (.25, .5, .75)])
    for capture in captures:
        folder = Path(capture)
        meta = json.loads((folder/'camera.json').read_text())
        ident = meta['device']['serial_number']
        camera_world = np.linalg.inv(np.asarray(
            calibration['cameras'][ident]['T_world_camera'], np.float64))
        xyz = (camera_world @ world.T).T[:, :3]
        intr = meta['color']['intrinsic']
        k = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
        uvh = (k @ xyz.T).T
        uv = uvh[:, :2] / uvh[:, 2:]
        ratios, errors, depths, expected, surface_points = [], [], [], [], []
        for path in sorted((folder/'depth').glob('*.png')):
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype != np.uint16:
                raise ValueError('Invalid metric depth capture')
            depth = raw.astype(np.float64)*float(meta['depth_scale_m_per_unit'])
            for point, target in zip(uv, xyz[:, 2]):
                u, v = np.rint(point).astype(int)
                if target <= 0 or not (2 <= u < depth.shape[1]-2
                                       and 2 <= v < depth.shape[0]-2):
                    continue
                patch = depth[v-2:v+3, u-2:u+3]
                values = patch[(patch >= .2) & (patch <= 10)]
                if len(values) < 5:
                    continue
                measured = float(np.median(values))
                ratios.append(measured/target)
                errors.append(abs(measured-target))
                depths.append(measured)
                expected.append(float(target))
                surface_points.append(np.linalg.solve(k, [point[0], point[1], 1])*measured)
        ratio = float(np.median(ratios)) if ratios else None
        p90 = float(np.percentile(errors, 90)) if errors else None
        normal_error = None
        if len(surface_points) >= 9:
            surface = np.asarray(surface_points)
            _, singular, basis = np.linalg.svd(surface-surface.mean(axis=0), full_matrices=False)
            if singular[1] > .005:
                expected_normal = camera_world[:3, 2]
                cosine = abs(float(np.dot(basis[-1], expected_normal)))
                normal_error = float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
        strict_passed = bool(len(ratios) >= 5 and .95 <= ratio <= 1.05 and p90 <= .05
                             and normal_error is not None and normal_error <= 10)
        passed = bool(len(ratios) >= 5 and limits['ratio'][0] <= ratio <= limits['ratio'][1]
                      and p90 <= limits['depth_p90_m'] and normal_error is not None
                      and normal_error <= limits['normal_deg'])
        checks[ident] = {'passed': passed, 'strict_passed': strict_passed, 'samples': len(ratios),
                         'depth_to_tag_ratio': ratio, 'depth_error_p90_m': p90,
                         'plane_normal_error_deg': normal_error,
                         'depth_median_m': float(np.median(depths)) if depths else None,
                         'tag_expected_median_m': float(np.median(expected)) if expected else None}
    strict_all = all(c['strict_passed'] for c in checks.values()) and len(checks) >= 2
    return {'tolerance_profile': tolerance_profile, 'strict_passed': strict_all,
            'preview_only': tolerance_profile == 'preview',
            'relaxed_acceptance_used': tolerance_profile == 'preview' and not strict_all,
            'applicable': True, 'passed': all(c['passed'] for c in checks.values())
            and len(checks) >= 2, 'cameras': checks,
            'ratio_range': limits['ratio'], 'max_depth_error_p90_m': limits['depth_p90_m'],
            'max_plane_normal_error_deg': limits['normal_deg'],
            'measurement_accuracy_verified': False,
            'note': 'Projected static anchor depth consistency; not a metric accuracy certificate'}
