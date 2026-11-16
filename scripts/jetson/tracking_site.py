"""Explicit floor calibration contract for raw RGB-left ankle projection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def floor_view(camera, extent_m=20.0):
    """Raw image boundary projected onto the floor, clipped before division.

    This is calibrated geometric coverage, not an occlusion or detection guarantee.
    Homogeneous clipping handles cameras whose image straddles the floor horizon.
    """
    import cv2

    try:
        width, height = camera['image_size']
        corners = np.array([[0, 0], [width - 1, 0],
                            [width - 1, height - 1], [0, height - 1]], dtype=float)
        border = np.concatenate([a + (b - a) * np.linspace(0, 1, 17)[:-1, None]
                                 for a, b in zip(corners, np.roll(corners, -1, axis=0))])
        undistort = (cv2.fisheye.undistortPoints if camera['model'] == 'fisheye'
                     else cv2.undistortPoints)
        uv = undistort(border.reshape(-1, 1, 2), np.asarray(camera['K'], dtype=float),
                       np.asarray(camera['D'], dtype=float),
                       R=np.asarray(camera['R'], dtype=float),
                       P=np.asarray(camera['K_rect'], dtype=float)).reshape(-1, 2)
        inverse = np.asarray(camera['ground_from_rectified_pixel'], dtype=float)
        transform = np.asarray(camera['camera_from_ground'], dtype=float)
        depth_scale = transform[2, [0, 1, 3]] @ inverse[:, 2]
        if abs(depth_scale) < 1e-12:
            return None
        polygon = (inverse @ np.column_stack([uv, np.ones(len(uv))]).T).T / depth_scale
        if not np.isfinite(polygon).all():
            return None
        # w > 0 is positive camera depth for this inverse floor projection.
        planes = [[0, 0, 1], [1, 0, extent_m], [-1, 0, extent_m],
                  [0, 1, extent_m], [0, -1, extent_m]]
        for plane in planes:
            clipped = []
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                da, db = np.dot(a, plane), np.dot(b, plane)
                if da >= 0:
                    clipped.append(a)
                if (da >= 0) != (db >= 0):
                    clipped.append(a + (b - a) * da / (da - db))
            polygon = np.asarray(clipped)
            if len(polygon) < 3:
                return None
        if np.any(polygon[:, 2] <= 1e-10):
            return None
        xy = polygon[:, :2] / polygon[:, 2, None]
        area = abs(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                          - xy[:, 1] * np.roll(xy[:, 0], -1))) / 2
        if not np.isfinite(xy).all() or area < 1e-6:
            return None
        origin = np.linalg.inv(transform)[:3, 3]
        if not np.isfinite(origin).all():
            return None
        return {'polygon_xy_m': xy.tolist(), 'camera_xyz_m': origin.tolist(),
                'clip_extent_m': extent_m}
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError, cv2.error):
        return None


def load_site(path, devices):
    if not path or not Path(path).is_file():
        return None, '尚未建立并核验人体追踪地面标定'
    try:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError('Invalid site document')
        if (data.get('schema_version') != 1 or data.get('floor_verified') is not True
                or not data.get('world_frame') or data.get('translation_unit') != 'm'):
            raise ValueError('Ground plane is not verified')
        if data.get('device_config_sha256') != hashlib.sha256(
                json.dumps(devices, sort_keys=True).encode()).hexdigest():
            raise ValueError('Device configuration changed; recalibration required')
        configured = {d['id'] for d in devices}
        selected = data.get('participating_camera_ids')
        if selected is not None:
            if (not isinstance(selected, list) or any(not isinstance(x, str) for x in selected)
                    or len(selected) < 2 or len(set(selected)) != len(selected)
                    or not set(selected).issubset(configured)
                    or set(data['cameras']) != set(selected)):
                raise ValueError('Invalid participating camera subset')
        else:
            selected = configured
        for device in (d for d in devices if d['id'] in selected):
            cam = data['cameras'][device['id']]
            if cam['calibration_sha256'] != device['calibration_sha256']:
                raise ValueError('Calibration hash changed')
            for key, shape in [('K', (3, 3)), ('R', (3, 3)), ('K_rect', (3, 3)),
                               ('ground_from_rectified_pixel', (3, 3)), ('camera_from_ground', (4, 4))]:
                matrix = np.asarray(cam[key], dtype=float)
                if matrix.shape != shape or not np.isfinite(matrix).all():
                    raise ValueError('Invalid projection matrix')
                if abs(np.linalg.det(matrix)) < 1e-12:
                    raise ValueError('Singular projection matrix')
            transform = np.asarray(cam['camera_from_ground'], dtype=float)
            rotation = transform[:3, :3]
            if (not np.allclose(transform[3], [0, 0, 0, 1])
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
                raise ValueError('Invalid rigid ground transform')
            product = (np.asarray(cam['K_rect']) @ transform[:3, [0, 1, 3]]
                       @ np.asarray(cam['ground_from_rectified_pixel']))
            if abs(product[2, 2]) < 1e-10 or not np.allclose(
                    product / product[2, 2], np.eye(3), atol=1e-5):
                raise ValueError('Floor homography and extrinsics disagree')
            distortion = np.asarray(cam['D'], dtype=float)
            if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
                raise ValueError('Invalid distortion coefficients')
            if cam['image_size'] != device['resolution']['left']:
                raise ValueError('Calibration image size changed')
        return data, None
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return None, '地面标定不可用：' + str(exc)


def project_ankles(camera, keypoints, image_size):
    import cv2

    points = np.asarray(keypoints, dtype=float)
    if (points.shape != (17, 3) or not np.isfinite(points).all()
            or tuple(image_size) != tuple(camera['image_size'])
            or min(points[15:17, 2]) < 0.5):
        return None
    if (np.any(points[15:17, :2] < 0)
            or np.any(points[15:17, 0] >= image_size[0])
            or np.any(points[15:17, 1] >= image_size[1])):
        return None
    ankles = points[15:17, :2].reshape(-1, 1, 2)
    k, d = np.asarray(camera['K'], dtype=float), np.asarray(camera['D'], dtype=float)
    r, p = np.asarray(camera['R'], dtype=float), np.asarray(camera['K_rect'], dtype=float)
    undistort = cv2.fisheye.undistortPoints if camera['model'] == 'fisheye' else cv2.undistortPoints
    uv = undistort(ankles, k, d, R=r, P=p).reshape(-1, 2).mean(axis=0)
    hom = np.asarray(camera['ground_from_rectified_pixel']) @ [uv[0], uv[1], 1.0]
    if abs(hom[2]) < 1e-8:
        return None
    xy = hom[:2] / hom[2]
    # Positive depth gate: intersections behind the camera are not floor observations.
    plane_camera = np.asarray(camera['camera_from_ground']) @ [xy[0], xy[1], 0, 1]
    if (not np.isfinite(xy).all() or np.linalg.norm(xy) > 20
            or plane_camera[2] <= 0):
        return None
    return xy.tolist()
