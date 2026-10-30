"""Numerical lens and rectification checks; low training RMS is not validity."""

import cv2
import numpy as np

from .models import CameraModel

MAX_INVERSE_ERROR_PX = .25
MIN_COMMON_VIEW = .5
MIN_SOURCE_COVERAGE = .1


def _grid(size, cols=33, rows=25):
    w, h = size
    return np.stack(np.meshgrid(np.linspace(0, w - 1, cols),
                                np.linspace(0, h - 1, rows)), axis=-1)


def _project(rays, intr):
    fun = cv2.fisheye.projectPoints if intr.model == CameraModel.FISHEYE else cv2.projectPoints
    return fun(np.asarray(rays, np.float64).reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
               np.asarray(intr.camera_matrix, dtype=float),
               np.asarray(intr.dist_coeffs, dtype=float))[0].reshape(-1, 2)


def check_intrinsics(intr):
    """Check the full sensor domain, including the inverse used by stereoRectify.

    Grid and derivative sampling are conservative numerical tests, not a formal
    global proof or an independent measurement of physical calibration accuracy.
    """
    reasons, metrics = [], {}
    try:
        if any(np.asarray(value).dtype.kind not in 'fiu'
               for value in (intr.camera_matrix, intr.dist_coeffs)):
            raise ValueError('内参和畸变必须为实数')
        k = np.asarray(intr.camera_matrix, dtype=float)
        d = np.asarray(intr.dist_coeffs, dtype=float)
        if intr.model not in (CameraModel.PINHOLE, CameraModel.FISHEYE):
            raise ValueError('相机模型不受支持')
        if (len(intr.image_size) != 2 or any(isinstance(v, (bool, np.bool_))
                or not isinstance(v, (int, np.integer)) or v <= 0 for v in intr.image_size)):
            raise ValueError('图像尺寸必须为正整数')
        counts = (4,) if intr.model == CameraModel.FISHEYE else (0, 4, 5, 8, 12, 14)
        if d.size not in counts or d.ndim not in (1, 2) or (d.ndim == 2 and 1 not in d.shape):
            raise ValueError('畸变系数数量或向量形状无效')
        if (k.shape != (3, 3) or not np.isfinite(k).all() or
                min(k[0, 0], k[1, 1]) <= 0 or not np.isfinite(d).all()):
            raise ValueError('内参包含无效矩阵或焦距')
        points = _grid(intr.image_size).reshape(-1, 1, 2)
        fish = intr.model == CameraModel.FISHEYE
        default = (cv2.fisheye.undistortPoints(points, k, d) if fish else
                   cv2.undistortPoints(points, k, d)).reshape(-1, 2)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
        refined = (cv2.fisheye.undistortPoints(points, k, d, criteria=criteria) if fish else
                   cv2.undistortPointsIter(points, k, d, None, None, criteria)).reshape(-1, 2)
        for name, normalized in (('default', default), ('refined', refined)):
            errors = np.linalg.norm(_project(np.c_[normalized, np.ones(len(normalized))], intr)
                                    - points.reshape(-1, 2), axis=1)
            maximum = float(np.max(errors))
            metrics[f'{name}_inverse_max_px'] = maximum if np.isfinite(maximum) else None
            if not np.isfinite(maximum) or maximum > MAX_INVERSE_ERROR_PX:
                reasons.append(f'{name} 边界反算回投误差超过 {MAX_INVERSE_ERROR_PX:g} px')
        radius = float(np.max(np.linalg.norm(refined, axis=1)))
        if not np.isfinite(radius) or (radius > 100 and reasons):
            raise ValueError('边界反算射线异常')
        t = np.linspace(0, np.arctan(radius) if fish else radius, 4097)
        a = np.zeros(14)
        a[:d.size] = d.ravel()
        q = t * t
        if fish:
            derivative = 1 + 3*a[0]*q + 5*a[1]*q**2 + 7*a[2]*q**3 + 9*a[3]*q**4
            denominator = np.ones_like(t)
        else:
            numerator = 1 + a[0]*q + a[1]*q**2 + a[4]*q**3
            denominator = 1 + a[5]*q + a[6]*q**2 + a[7]*q**3
            # Detect poles analytically as sampling alone can miss a narrow crossing.
            roots = np.polynomial.polynomial.polyroots([1., a[5], a[6], a[7]])
            if any(abs(root.imag) < 1e-8 and 0 <= root.real <= radius**2 for root in roots):
                reasons.append('有效视场内有理分母存在零点')
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                derivative = numerator / denominator + 2*q * (
                    (a[0] + 2*a[1]*q + 3*a[4]*q**2)*denominator
                    - numerator*(a[5] + 2*a[6]*q + 3*a[7]*q**2)) / denominator**2
        minimum = float(np.min(derivative))
        metrics['radial_derivative_min'] = minimum if np.isfinite(minimum) else None
        if (not np.isfinite(derivative).all() or minimum <= 0
                or np.min(denominator) <= 1e-8):
            reasons.append('有效视场内径向映射非单调或分母不稳定')
    except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        reasons.append(str(exc))
    return {'status': 'failed' if reasons else 'ok', 'reasons': reasons, 'metrics': metrics}


def _check_rectification_geometry(stereo, require_q):
    """Validate the left-rectified coordinate frame and Q with known 3D points."""
    rotation = np.asarray(stereo.R, dtype=float)
    translation = np.asarray(stereo.T, dtype=float).reshape(3)
    r1, r2 = np.asarray(stereo.R1, dtype=float), np.asarray(stereo.R2, dtype=float)
    p1, p2 = np.array(stereo.P1, dtype=float, copy=True), np.array(stereo.P2, dtype=float, copy=True)
    if (rotation.shape != (3, 3) or r1.shape != (3, 3) or r2.shape != (3, 3)
            or p1.shape != (3, 4) or p2.shape != (3, 4)
            or not all(np.isfinite(v).all() for v in (rotation, translation, r1, r2, p1, p2))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError('校正几何包含无效R/T或矩阵形状')
    baseline = float(np.linalg.norm(translation))
    if baseline == 0:
        raise ValueError('校正基线为零')
    if not np.allclose(r2 @ rotation @ r1.T, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError('R1/R2与双目旋转R不一致')
    rectified_t = r2 @ translation
    axis = int(np.argmax(np.abs(rectified_t[:2])))
    other = [i for i in range(3) if i != axis]
    if np.linalg.norm(rectified_t[other]) > baseline * 1e-6:
        raise ValueError('校正基线未对齐水平或垂直视差轴')
    for p in (p1, p2):
        if p[2, 2] == 0:
            raise ValueError('投影矩阵齐次尺度为零')
        p /= p[2, 2]
        if not np.allclose(p[2, :3], [0., 0., 1.], atol=1e-9, rtol=0):
            raise ValueError('校正投影矩阵不是标准相机矩阵')
    if not np.allclose(p1[:, 3], 0., atol=1e-9, rtol=0):
        raise ValueError('P1不以左校正相机为原点')
    expected_t = p2[:, :3] @ rectified_t
    if not np.allclose(p2[:, 3], expected_t, rtol=1e-6,
                       atol=baseline * max(np.linalg.norm(p2[:, :3]), 1.) * 1e-8):
        raise ValueError('P2平移项与双目基线R/T不一致')
    metrics = {'axis': 'horizontal' if axis == 0 else 'vertical', 'baseline_m': baseline}
    if stereo.Q is None:
        if require_q:
            raise ValueError('缺少Q，无法验证三维重建')
        return {**metrics, 'q_status': 'unavailable'}
    q = np.asarray(stereo.Q, dtype=float)
    if q.shape != (4, 4) or not np.isfinite(q).all() or not np.any(q):
        raise ValueError('Q无效或为全零矩阵')
    q = q / np.max(np.abs(q))
    # Use several depths and off-axis points; reprojection alone cannot detect a
    # common wrong metric scale, so compare reconstructed 3D coordinates too.
    points = np.array([[x*z, y*z, z, 1.]
                       for z in baseline * np.array([5., 10., 20.])
                       for x, y in ((-.15, -.1), (0., 0.), (.2, .15))])
    lh, rh = points @ p1.T, points @ p2.T
    left, right = lh[:, :2] / lh[:, 2:], rh[:, :2] / rh[:, 2:]
    if not np.allclose(left[:, 1-axis], right[:, 1-axis], atol=1e-5, rtol=0):
        raise ValueError('P1/P2不满足校正后的同一极线约束')
    disparity = left[:, axis] - right[:, axis]
    recovered_h = np.c_[left, disparity, np.ones(len(points))] @ q.T
    if not np.isfinite(recovered_h).all() or np.any(recovered_h[:, 3] == 0):
        raise ValueError('Q三维重建产生无效齐次坐标')
    recovered = recovered_h[:, :3] / recovered_h[:, 3:]
    relative_error = float(np.max(np.linalg.norm(recovered - points[:, :3], axis=1)
                                  / points[:, 2]))
    if not np.isfinite(relative_error) or relative_error > 1e-6 or np.any(recovered[:, 2] <= 0):
        raise ValueError('Q与P1/P2或基线不一致，三维重建尺度/方向错误')
    for projection, expected in ((p1, left), (p2, right)):
        projected = np.c_[recovered, np.ones(len(points))] @ projection.T
        pixels = projected[:, :2] / projected[:, 2:]
        if not np.isfinite(pixels).all() or np.max(np.linalg.norm(pixels-expected, axis=1)) > 1e-4:
            raise ValueError('Q重建点无法回投至左右校正图像')
    return {**metrics, 'q_status': 'ok', 'reconstruction_relative_error': relative_error}


def check_stereo_health(left, right, stereo, require_rectification=True):
    result = {'version': 2, 'status': 'ok', 'reasons': [],
              'limits': '全视场数值采样，不代表外部测量精度或形式化全局证明',
              'left': check_intrinsics(left), 'right': check_intrinsics(right), 'view': {}}
    reasons = result['reasons']
    for side in ('left', 'right'):
        reasons.extend(f'{side}: {reason}' for reason in result[side]['reasons'])
    matrices = [stereo.R1, stereo.R2, stereo.P1, stereo.P2]
    if not require_rectification and all(value is None for value in matrices) and stereo.Q is None:
        result['view']['status'] = 'unavailable'
    else:
        try:
            result['geometry'] = _check_rectification_geometry(stereo, require_rectification)
        except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError) as exc:
            reasons.append(str(exc))
        try:
            valid = []
            for side, intr, rotation, projection in (
                    ('left', left, stereo.R1, stereo.P1),
                    ('right', right, stereo.R2, stereo.P2)):
                rot, p = np.asarray(rotation, float), np.asarray(projection, float)
                if (rot.shape != (3, 3) or p.shape != (3, 4) or
                        not np.isfinite(p).all() or not np.isfinite(rot).all() or
                        not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6, rtol=0) or
                        not np.isclose(np.linalg.det(rot), 1., atol=1e-6) or
                        min(p[0, 0], p[1, 1]) <= 0):
                    raise ValueError('校正矩阵无效或焦距非正')
                grid = _grid(intr.image_size)
                rays = np.c_[grid.reshape(-1, 2), np.ones(grid.shape[0]*grid.shape[1])]
                rays = rays @ np.linalg.inv(p[:, :3]).T @ rot
                mapped = _project(rays, intr).reshape(grid.shape)
                inside = (np.isfinite(mapped).all(axis=2) & (rays[:, 2].reshape(grid.shape[:2]) > 0)
                          & (mapped[:, :, 0] >= 0) & (mapped[:, :, 0] < intr.image_size[0])
                          & (mapped[:, :, 1] >= 0) & (mapped[:, :, 1] < intr.image_size[1]))
                dx, dy = mapped[:, :, 0], mapped[:, :, 1]
                jac = (np.gradient(dx, axis=1)*np.gradient(dy, axis=0)
                       - np.gradient(dx, axis=0)*np.gradient(dy, axis=1))
                points = mapped[inside].astype(np.float32)
                coverage = (float(cv2.contourArea(cv2.convexHull(points)))
                            / np.prod(intr.image_size) if len(points) >= 3 else 0.)
                result['view'][side] = {'valid_fraction': float(inside.mean()),
                                       'source_coverage': float(coverage)}
                if not np.isfinite(jac).all() or np.any(jac <= 0):
                    reasons.append(f'{side}: 校正映射折叠或无效')
                if coverage < MIN_SOURCE_COVERAGE:
                    reasons.append(f'{side}: 校正视野异常裁剪，仅覆盖原图 {coverage:.1%}')
                valid.append(inside)
            common = float(np.logical_and(*valid).mean())
            result['view']['common_valid_fraction'] = common
            if common < MIN_COMMON_VIEW:
                reasons.append('左右校正共视有效比例不足')
        except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError) as exc:
            reasons.append(str(exc))
    result['status'] = 'failed' if reasons else 'ok'
    return result
