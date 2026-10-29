"""Stateless, advisory stereo capture guidance, with no UI or persistent state.

Zones are zero-based in a 3x3 grid. Angles are degrees: positive yaw means
the right edge is nearer; positive pitch means the bottom edge is nearer.
Area is the *observed inner-corner convex hull* divided by image area, not
board completeness. Thresholds below are configurable heuristics, not an
optimal-design solver or a calibration completion criterion.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product

import cv2
import numpy as np

from ..board.detector import DetectionResult
from .intrinsic import IntrinsicCalibrator
from .models import CameraIntrinsics, CameraModel
from .observations import prepare_observation, similar_composition


SCALE_AREA_THRESHOLDS = (.04, .12)
SCALE_AREA_TARGETS = {'small': .025, 'medium': .07, 'large': .16}
SCALE_AREA_RANGES = {'small': (.012, .04), 'medium': (.04, .12), 'large': (.12, .45)}
DISTANCE_LABELS = {'small': '远距离', 'medium': '中距离', 'large': '近距离'}
# Request budgets bound cold-start stages; they are not verified distance coverage.
CENTRAL_DISTANCE_PLAN = ('medium',) * 3 + ('large',) * 2 + ('small',) * 2
DISTANCE_SAMPLE_GOALS = {'medium': 3, 'large': 2, 'small': 2}
MAX_DISTANCE_REPAIR_REQUESTS = 4
MAX_DISTANCE_REPAIR_PER_SCALE = 2
CANDIDATE_IMAGE_MARGIN = .04
MIN_CORNER_SPACING_PX = 6.
CANDIDATE_TILT_DEGREES = 25.
TILTED_FOOTPRINT_PADDING = 1.08
MIN_BOARD_HULL_FRACTION = .5
TILT_THRESHOLD_DEGREES = 12.
MAX_REPROJECTION_RMS = 3.
MAX_DISTANCE_RELATIVE_STD = .05
MAX_DISTANCE_SPACING_ERROR = .10
BOOTSTRAP_MIN_VIEWS = 6
BOOTSTRAP_MAX_VIEWS = 30
BOOTSTRAP_MAX_RMS = 2.5
_TILTS = ('front', 'left', 'right', 'up', 'down')
_SCALES = ('medium', 'small', 'large')


@dataclass
class GuidanceObservation:
    corners: np.ndarray
    ids: np.ndarray
    image_size: tuple[int, int]
    frame_id: str


@dataclass
class GuidanceRecommendation:
    """pose_available means both cameras have usable guidance intrinsics.

    It does not validate a candidate or guarantee a particular pose solve.
    """

    zone_row: int
    zone_col: int
    tilt: str
    scale: str
    reason: str
    pose_available: bool
    sample_count: int
    feasible: bool = True
    repairing_distance: bool = False
    missing_distances: tuple[str, ...] = ()


@dataclass
class GuidanceAnalysis:
    recommendation: GuidanceRecommendation
    intrinsics: dict[str, CameraIntrinsics]
    summary: str
    distance_counts: dict[str, int] = field(default_factory=dict)
    approximate_distance_counts: dict[str, int] = field(default_factory=dict)


def reliable_distance_pose(pose):
    return bool(pose is not None and pose.get('distance_reliable', False)
                and np.isfinite(pose.get('distance_area', np.nan))
                and pose['distance_area'] > 0)


def _hull_area(points):
    return float(cv2.contourArea(cv2.convexHull(np.array(points, np.float32, copy=True))))


def _geometry(board, observation):
    """Copy/validate observations and reject spatially narrow partial boards."""
    try:
        if not isinstance(observation.frame_id, str) or not observation.frame_id:
            return None
        detection = DetectionResult(observation.corners, observation.ids, [], None,
                                    observation.image_size)
        corners, ids, size = prepare_observation(detection, board)
        order = np.argsort(ids.ravel())
        corners, ids = corners[order], ids[order]
        points = corners.reshape(-1, 2).astype(np.float64)
        all_obj = np.array(board.object_points, np.float64, copy=True)
        obj = all_obj[ids.ravel()]
        if len(points) < 6 or not np.isfinite(all_obj).all():
            return None
        if np.any(points < 0) or np.any(points >= np.asarray(size)):
            return None
        full_area = _hull_area(all_obj[:, :2])
        if full_area <= 0 or _hull_area(obj[:, :2]) / full_area < MIN_BOARD_HULL_FRACTION:
            return None
        area = _hull_area(points) / float(size[0] * size[1])
        if not 1e-5 < area < 1.:
            return None
        # Normalize object coordinates so perspective terms are comparable across boards.
        midpoint = (all_obj[:, :2].min(axis=0) + all_obj[:, :2].max(axis=0)) / 2
        extent = np.ptp(all_obj[:, :2], axis=0)
        if np.any(extent <= 0):
            return None
        local = (obj[:, :2] - midpoint) / extent
        H, _ = cv2.findHomography(local, points / size, 0)
        if H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-10:
            return None
        H = H / H[2, 2]
        projected = cv2.perspectiveTransform(local.reshape(-1, 1, 2), H).reshape(-1, 2)
        residual = np.sqrt(np.mean(np.sum(((projected - points / size) * size) ** 2, axis=1)))
        center = H[:2, 2]
        reasonable = (residual < max(5., .01 * np.linalg.norm(size))
                      and np.all(center >= 0) and np.all(center <= 1)
                      and np.linalg.norm(center - points.mean(axis=0) / size) < .25)
        if not reasonable:
            if len(ids) != len(all_obj):
                return None
            center = points.mean(axis=0) / size
        clean = GuidanceObservation(corners, ids, size, observation.frame_id)
        return clean, obj, center.copy(), area, H
    except (ValueError, TypeError, AttributeError, IndexError, cv2.error):
        return None


def measure_composition(board, observation):
    """Shared live/history composition; center is the full board's mapped center."""
    geometry = _geometry(board, observation)
    if geometry is None:
        return None
    full_area = _hull_area(np.asarray(board.object_points)[:, :2])
    fraction = _hull_area(geometry[1][:, :2]) / full_area
    return {'center': geometry[2].copy(), 'area': geometry[3],
            'distance_area': geometry[3] / fraction}


def distance_feedback(scale, areas):
    """Both eyes must agree; never average opposing distance instructions."""
    low, high = SCALE_AREA_RANGES[scale]
    label = DISTANCE_LABELS[scale]
    if not areas or not all(np.isfinite(a) and a > 0 for a in areas):
        return False, f'{label} · 等待双路距离检测'
    smaller = any(a < low for a in areas)
    larger = any(a >= high if scale != 'large' else a > high for a in areas)
    if smaller and larger:
        action = '双路距离判断冲突，先检查共视和相机配置，不要继续前后移动'
    elif smaller:
        action = '向镜头靠近，增大板面；保持当前高度'
    elif larger:
        action = '远离镜头，缩小板面；保持当前高度'
    else:
        action = '距离到位，保持前后距离'
    values = ' / '.join(f'{a:.1%}' for a in areas)
    return not (smaller or larger), f'{label} · {action} · 等效板面占比 {values}；目标 {low:.1%}–{high:.0%}'


def evaluate_distance(scale, measurements):
    """One distance policy for live UI, raw validation, offline replay and coverage."""
    if not measurements or not all(m.get('valid', m.get('geometry_valid', False))
                                   for m in measurements):
        return False, '等待双路距离检测', [], 'unavailable'
    poses = [m.get('pose') for m in measurements]
    reliable = all(reliable_distance_pose(p) for p in poses)
    areas = ([p['distance_area'] for p in poses] if reliable else
             [m.get('distance_area', m.get('area', 0)) for m in measurements])
    ready, text = distance_feedback(scale, areas)
    if reliable:
        distances = []
        low, high = SCALE_AREA_RANGES[scale]
        boundary = False
        for i, p in enumerate(poses, 1):
            # A = C/Z² converts the same scale policy into camera-specific metres.
            factor = p['distance_area'] * p['depth'] ** 2
            margin = 2 * p.get('depth_std', 0.)
            boundary |= (ready and not (np.sqrt(factor / high) + margin <= p['depth']
                                       <= np.sqrt(factor / low) - margin))
            distances.append(f"相机{i}板中心深度 {p['depth']:.2f} m"
                             f"（目标 {np.sqrt(factor / high):.2f}–"
                             f"{np.sqrt(factor / low):.2f} m）")
        if boundary:
            text = '距离临界，稍向目标档位中部移动后保持静止；' + text
        return ready and not boundary, (f"{text}（PnP深度等效 · 可信）；"
                                       + '；'.join(distances)), areas, 'pnp'
    uncertain = any(m.get('pose_expected') or m.get('pose') is not None for m in measurements)
    source = '占比近似，非实测距离；先保持板面朝向'
    if uncertain:
        reasons = sorted({p.get('distance_reason', '距离可信度不足')
                          for p in poses if p is not None and not reliable_distance_pose(p)})
        reason = '、'.join(reasons) or 'PnP不可解或内参不匹配'
        return False, f'距离待确认：{reason}；{text}（{source}，暂不自动验收）', areas, 'uncertain'
    return ready, f'{text}（{source}）', areas, 'composition'


def _distance_quality(obj, image, rv, center_translation, K, d, project, rms, intrinsics):
    """Conditional local pose uncertainty; excludes intrinsic/board systematic error."""
    radius = float(np.linalg.norm(np.ptp(obj, axis=0)))
    parameters = np.r_[rv.ravel(), center_translation.ravel() / radius]
    columns = []
    for axis in range(6):
        plus, minus = parameters.copy(), parameters.copy()
        plus[axis] += 1e-5
        minus[axis] -= 1e-5
        a = project(obj.reshape(-1, 1, 3), plus[:3], plus[3:] * radius, K, d)[0]
        b = project(obj.reshape(-1, 1, 3), minus[:3], minus[3:] * radius, K, d)[0]
        columns.append(((a - b) / 2e-5).ravel())
    _, singular, vt = np.linalg.svd(np.column_stack(columns), full_matrices=False)
    if singular[-1] <= singular[0] * 1e-10:
        return False, '位姿约束退化', None
    sigma = max(.5, rms * np.sqrt(2 * len(image) / max(1, 2 * len(image) - 6)))
    covariance = (vt.T * (sigma / singular) ** 2) @ vt
    depth_std = float(radius * np.sqrt(max(0., covariance[5, 5])))
    points = image.reshape(-1, 2)
    separations = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(separations, np.inf)
    spacing = float(np.median(separations.min(axis=1)))
    reasons = []
    if intrinsics.rms_error > BOOTSTRAP_MAX_RMS:
        reasons.append('内参拟合误差偏大')
    if spacing < MIN_CORNER_SPACING_PX:
        reasons.append('角点间距过小')
    if rms > MAX_DISTANCE_SPACING_ERROR * spacing:
        reasons.append('重投影误差相对角点间距偏大')
    if depth_std / center_translation[2] > MAX_DISTANCE_RELATIVE_STD:
        reasons.append('深度估计不稳定')
    return not reasons, '、'.join(reasons) or '局部深度约束通过', depth_std


def _valid_intrinsics(intrinsics, size, model):
    try:
        if (len(size) != 2 or any(isinstance(v, (bool, np.bool_))
                                 or not isinstance(v, (int, np.integer)) or v <= 0
                                 for v in size)):
            return False
        K = np.array(intrinsics.camera_matrix, np.float64, copy=True)
        d = np.array(intrinsics.dist_coeffs, np.float64, copy=True)
        w, h = size
        return bool(
            tuple(intrinsics.image_size) == tuple(size) and intrinsics.model == model
            and K.shape == (3, 3) and np.isfinite(K).all() and np.isfinite(d).all()
            and d.size in ((4,) if model == CameraModel.FISHEYE else (4, 5, 8, 12, 14))
            and np.max(np.abs(d)) < 100
            and .08 * max(size) < K[0, 0] < 8 * max(size)
            and .08 * max(size) < K[1, 1] < 8 * max(size)
            and .25 < K[0, 0] / K[1, 1] < 4
            and -.1 * w <= K[0, 2] <= 1.1 * w
            and -.1 * h <= K[1, 2] <= 1.1 * h
            and np.allclose(K[2], (0, 0, 1))
            and abs(K[1, 0]) < 1e-8 and abs(K[0, 1]) < 1e-8
            and np.isfinite(intrinsics.rms_error) and 0 <= intrinsics.rms_error <= 5
        )
    except (ValueError, TypeError, AttributeError, IndexError):
        return False


def measure_pose(board, observation, intrinsics) -> dict | None:
    """Return yaw/pitch (degrees), depth (board units), area and normalized center.

    Camera x points right and y down. For the plane normal n, depth gradients
    are -nx/nz and -ny/nz; hence yaw=atan2(nx,nz), pitch=atan2(ny,nz).
    Fisheye points are undistorted to normalized rays before solvePnP, and
    reprojection is checked in the original distorted image for both models.
    """
    geometry = _geometry(board, observation)
    if geometry is None or intrinsics is None:
        return None
    clean, obj, center, area, _ = geometry
    try:
        model = CameraModel(intrinsics.model)
        if not _valid_intrinsics(intrinsics, clean.image_size, model):
            return None
        K = np.array(intrinsics.camera_matrix, np.float64, copy=True)
        d = np.array(intrinsics.dist_coeffs, np.float64, copy=True).reshape(-1, 1)
        image = clean.corners.astype(np.float64)
        if model == CameraModel.FISHEYE:
            rays = cv2.fisheye.undistortPoints(image, K, d)
            ok, rv, tv = cv2.solvePnP(obj, rays, np.eye(3), None)
        else:
            ok, rv, tv = cv2.solvePnP(obj, image, K, d)
        if not ok or not np.isfinite(rv).all() or not np.isfinite(tv).all():
            return None
        R = cv2.Rodrigues(rv)[0]
        all_points = np.array(board.object_points, np.float64, copy=True)
        transformed = all_points @ R.T + tv.reshape(1, 3)
        if np.min(transformed[:, 2]) <= 1e-8 or abs(R[2, 2]) < .1:
            return None
        project = cv2.fisheye.projectPoints if model == CameraModel.FISHEYE else cv2.projectPoints
        predicted = project(obj.reshape(-1, 1, 3), rv, tv, K, d)[0].reshape(-1, 2)
        rms = np.sqrt(np.mean(np.sum((predicted - image.reshape(-1, 2)) ** 2, axis=1)))
        if not np.isfinite(rms) or rms > MAX_REPROJECTION_RMS:
            return None
        normal = R[:, 2] * (1 if R[2, 2] > 0 else -1)
        midpoint = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
        board_center = R @ midpoint + tv.ravel()
        depth = float(board_center[2])
        reliable, reason, depth_std = _distance_quality(
            obj - midpoint, image, rv, board_center, K, d, project, rms, intrinsics)
        frontal_area = (_hull_area(all_points[:, :2]) * K[0, 0] * K[1, 1]
                        / (np.prod(clean.image_size) * depth ** 2))
        return dict(yaw=float(np.degrees(np.arctan2(normal[0], normal[2]))),
                    pitch=float(np.degrees(np.arctan2(normal[1], normal[2]))),
                    depth=depth, range=float(np.linalg.norm(board_center)),
                    depth_std=depth_std, reprojection_rms=float(rms),
                    distance_reliable=reliable, distance_reason=reason,
                    area=area, center=center, distance_area=float(frontal_area))
    except (ValueError, TypeError, np.linalg.LinAlgError, cv2.error):
        return None


def _similar(a, b):
    return similar_composition(a.corners, a.ids, a.image_size, b.corners, b.ids, b.image_size)


def _bootstrap(board, model, geometries):
    if len(geometries) < BOOTSTRAP_MIN_VIEWS:
        return None
    indices = np.linspace(0, len(geometries) - 1,
                          min(BOOTSTRAP_MAX_VIEWS, len(geometries)), dtype=int)
    selected = [geometries[i] for i in indices]
    perspective = np.array([g[4][2, :2] for g in selected])
    singular = np.linalg.svd(perspective - perspective.mean(axis=0), compute_uv=False)
    # Translation/scale changes alone cannot constrain a planar camera calibration.
    if singular[-1] / np.sqrt(len(selected)) < .012 or np.std(perspective, axis=0).max() < .035:
        return None
    try:
        cal = IntrinsicCalibrator(board, model, min_frames=BOOTSTRAP_MIN_VIEWS)
        for clean, *_ in selected:
            detection = DetectionResult(clean.corners.copy(), clean.ids.copy(), [], None,
                                        clean.image_size)
            if not cal.add_detection(detection, frame_id=clean.frame_id):
                return None
        result = cal.calibrate()
        if not _valid_intrinsics(result, selected[0][0].image_size, model):
            return None
        if result.rms_error > BOOTSTRAP_MAX_RMS:
            return None
        poses = [measure_pose(board, g[0], result) for g in selected]
        if any(p is None for p in poses):
            return None
        angles = np.radians([[p['yaw'], p['pitch']] for p in poses])
        spread = np.linalg.svd(angles - angles.mean(axis=0), compute_uv=False)
        if spread[-1] / np.sqrt(len(poses)) < .04:
            return None
        return deepcopy(result)
    except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError, cv2.error):
        return None


def _zone(center):
    return tuple(np.clip(np.asarray(center)[::-1] * 3, 0, 2).astype(int))


def _scale(area):
    return ('small', 'medium', 'large')[int(np.searchsorted(SCALE_AREA_THRESHOLDS, area,
                                                         side='right'))]


def _tilt(pose):
    yaw, pitch = pose['yaw'], pose['pitch']
    if max(abs(yaw), abs(pitch)) < TILT_THRESHOLD_DEGREES:
        return 'front'
    if abs(yaw) >= abs(pitch):
        return 'right' if yaw > 0 else 'left'
    return 'down' if pitch > 0 else 'up'


def _candidate_feasible(board, candidate, image_sizes):
    """Approximate footprint/spacing at a zone center in each known image size.

    Use the inner-corner extent, target hull area and modest tilt foreshortening,
    with extra footprint padding for perspective. This is not a lens projection
    or stereo co-visibility test: extrinsics, roll and actual tilt are unknown.
    Unknown image sizes cannot be checked; no global feasibility is guaranteed.
    """
    points = np.array(board.object_points, np.float64, copy=True)[:, :2]
    extent = np.ptp(points, axis=0)
    if not np.isfinite(points).all() or np.any(extent <= 0):
        return False
    spacing = []
    for axis in range(2):
        gaps = np.diff(np.unique(points[:, axis]))
        gaps = gaps[gaps > extent[axis] * 1e-6]
        if not len(gaps):
            return False
        spacing.append(float(gaps.min() / extent[axis]))
    row, col, tilt, scale = candidate
    aspect = extent[0] / extent[1]
    cosine = np.cos(np.radians(CANDIDATE_TILT_DEGREES))
    if tilt in ('left', 'right'):
        aspect *= cosine
    elif tilt in ('up', 'down'):
        aspect /= cosine
    center = np.array([(col + .5) / 3, (row + .5) / 3])
    padding = 1. if tilt == 'front' else TILTED_FOOTPRINT_PADDING
    for size in image_sizes:
        width, height = size
        area_px = SCALE_AREA_TARGETS[scale] * width * height
        rectangle = np.array([np.sqrt(area_px * aspect), np.sqrt(area_px / aspect)])
        half = rectangle / np.array(size) * padding / 2
        if (np.any(center - half < CANDIDATE_IMAGE_MARGIN)
                or np.any(center + half > 1 - CANDIDATE_IMAGE_MARGIN)
                or np.min(rectangle * spacing) < MIN_CORNER_SPACING_PX):
            return False
    return True


def analyze_guidance(board, model, observations: dict[str, list[GuidanceObservation]],
                     provided_intrinsics: dict[str, CameraIntrinsics]) -> GuidanceAnalysis:
    """Recommend one next composition from exactly two cameras, without mutation.

    Frame IDs match within the pair, not by list position. Conflicting repeated
    IDs, invalid/too-partial frames, inconsistent sizes and duplicate stereo
    compositions do not add evidence. Order follows the first camera's capture
    sequence. Replaying unique samples avoids repeating the preceding target;
    repeated calls and duplicate captures leave the recommendation unchanged.
    Returned intrinsics are independent guidance-only snapshots, never results
    to commit to a calibration session. Invalid camera count/model raises ValueError.
    Candidate fit is a per-image geometric approximation, not guaranteed stereo
    co-visibility. pose_available is an intrinsic capability flag, not validation.
    """
    if len(observations) != 2:
        raise ValueError('自适应双目引导需要恰好两台相机')
    model = CameraModel(model)
    cameras = sorted(observations)
    maps = {}
    for camera in cameras:
        frames, conflicts = {}, set()
        for observation in observations[camera]:
            g = _geometry(board, observation)
            if g is None:
                continue
            key = g[0].frame_id
            if key in frames:
                old, new = frames[key][0], g[0]
                if not (old.image_size == new.image_size
                        and np.array_equal(old.ids, new.ids)
                        and np.array_equal(old.corners, new.corners)):
                    conflicts.add(key)
            else:
                frames[key] = g
        maps[camera] = {k: v for k, v in frames.items() if k not in conflicts}
    left, right = cameras
    paired = [(g, maps[right][key]) for key, g in maps[left].items() if key in maps[right]]
    if paired:
        sizes = Counter((a[0].image_size, b[0].image_size) for a, b in paired)
        expected = sizes.most_common(1)[0][0]
        paired = [p for p in paired if tuple(g[0].image_size for g in p) == expected]
    unique = []
    for pair in paired:
        if not any(all(_similar(g[0], old[i][0]) for i, g in enumerate(pair)) for old in unique):
            unique.append(pair)
    intrinsics = {}
    for i, camera in enumerate(cameras):
        supplied = provided_intrinsics.get(camera)
        size = unique[0][i][0].image_size if unique else getattr(supplied, 'image_size', ())
        if _valid_intrinsics(supplied, size, model):
            intrinsics[camera] = deepcopy(supplied)
        else:
            estimate = _bootstrap(board, model, [p[i] for p in unique])
            if estimate is not None:
                intrinsics[camera] = estimate
    zones, scales, tilts, joint = [Counter(), Counter()], [Counter(), Counter()], [Counter(), Counter()], Counter()
    # Group heights: centre, middle row, upper row, lower row. No alternating corners.
    zone_order = [(1, 1), (1, 0), (1, 2), (0, 2), (0, 1), (0, 0), (2, 0), (2, 1), (2, 2)]
    image_sizes = []
    for i, camera in enumerate(cameras):
        if unique:
            image_sizes.append(unique[0][i][0].image_size)
        elif camera in intrinsics:
            image_sizes.append(intrinsics[camera].image_size)
        elif maps[camera]:
            image_sizes.append(next(iter(maps[camera].values()))[0].image_size)
    candidates = [candidate for (r, c), t, s in product(zone_order, _TILTS, _SCALES)
                  if _candidate_feasible(board, candidate := (r, c, t, s), image_sizes)]
    target = (1, 1, 'front', 'medium')
    if candidates and target not in candidates:
        target = candidates[0]
    pose_available = len(intrinsics) == 2
    verified = 0
    paired_distances = Counter()
    approximate_distances = Counter()
    repair_requests = Counter()
    repairing_distance = False
    requests, zone_requests, tilt_requests = Counter(), Counter(), Counter()
    for count, pair in enumerate(unique, 1):
        previous_target = target
        if repairing_distance:
            repair_requests[target[3]] += 1
        requests[target] += 1
        zone_requests[target[:2]] += 1
        tilt_requests[target[2]] += 1
        poses = [measure_pose(board, g[0], intrinsics.get(camera))
                 for camera, g in zip(cameras, pair)]
        verified += int(all(p is not None for p in poses))
        pair_scales = []
        pair_areas = []
        # Do not mix one camera's PnP with the other camera's fallback scale.
        reliable_distance = all(reliable_distance_pose(p) for p in poses)
        for i, g in enumerate(pair):
            fraction = _hull_area(g[1][:, :2]) / _hull_area(np.asarray(board.object_points)[:, :2])
            distance_area = (poses[i]['distance_area'] if reliable_distance else g[3] / fraction)
            zone, scale = _zone(g[2]), _scale(distance_area)
            pair_scales.append(scale)
            pair_areas.append(distance_area)
            zones[i][zone] += 1
            scales[i][scale] += 1
            if poses[i] is not None:
                tilt = _tilt(poses[i])
                tilts[i][tilt] += 1
                joint[(*zone, tilt, scale)] += 1
        measurements = [dict(valid=True, pose=p, pose_expected=camera in intrinsics,
                             distance_area=a)
                        for camera, p, a in zip(cameras, poses, pair_areas)]
        if (pair_scales[0] == pair_scales[1]
                and evaluate_distance(pair_scales[0], measurements)[0]):
            if reliable_distance:
                paired_distances[pair_scales[0]] += 1
            else:
                approximate_distances[pair_scales[0]] += 1
        def score(candidate):
            r, c, t, s = candidate
            coverage = sum(sc[s] + .7 * ti[t]
                           for z, sc, ti in zip(zones, scales, tilts))
            provisional = 0 if pose_available or t == _TILTS[count % len(_TILTS)] else 2
            local_tilt_requests = sum(n for (rr, cc, tt, ss), n in requests.items()
                                      if (rr, cc, tt) == (r, c, t))
            # Requests guide scheduling only: they never count as verified pose coverage.
            return (coverage + .7 * joint[candidate] + provisional
                    + 3 * local_tilt_requests + 1.5 * tilt_requests[t] + .5 * requests[candidate])
        alternatives = [c for c in candidates if c != target]
        if alternatives:
            available_zones = {c[:2] for c in alternatives}
            remaining = [z for z in zone_order if z in available_zones
                         and zone_requests[z] < (len(CENTRAL_DISTANCE_PLAN) if z == (1, 1) else 2)]
            if remaining:
                zone = remaining[0]
            else:
                active_row = target[0]
                zone = min(available_zones, key=lambda z: (
                    sum(hist[z] for hist in zones), abs(z[0] - active_row), zone_order.index(z)))
            local = [c for c in alternatives if c[:2] == zone]
            if zone == (1, 1) and zone_requests[zone] < len(CENTRAL_DISTANCE_PLAN):
                desired = CENTRAL_DISTANCE_PLAN[zone_requests[zone]]
                if reliable_distance:
                    gaps = [s for s in ('medium', 'large', 'small')
                            if paired_distances[s] < DISTANCE_SAMPLE_GOALS[s]
                            and any(c[3] == s for c in local)]
                    if gaps:
                        desired = gaps[0]
                at_distance = [c for c in local if c[3] == desired]
                if at_distance:
                    local = at_distance
            target = min(local, key=score)
        repairing_distance = False
        if (zone_requests[(1, 1)] >= len(CENTRAL_DISTANCE_PLAN)
                and sum(repair_requests.values()) < MAX_DISTANCE_REPAIR_REQUESTS):
            missing = [s for s in ('large', 'small', 'medium')
                       if paired_distances[s] < DISTANCE_SAMPLE_GOALS[s]]
            repair_candidates = [c for c in candidates if c[:2] == (1, 1)
                                 and c[3] in missing and c != previous_target
                                 and repair_requests[c[3]] < MAX_DISTANCE_REPAIR_PER_SCALE]
            if repair_candidates:
                # Group distance moves, but cap ignored gaps; vary tilt within a group.
                target = min(repair_candidates, key=lambda c: (
                    missing.index(c[3]), paired_distances[c[3]] / DISTANCE_SAMPLE_GOALS[c[3]],
                    score(c)))
                repairing_distance = True
    row, col, tilt, scale = target
    tilt_text = {'front': '正对', 'left': '左边近倾斜', 'right': '右边近倾斜',
                 'up': '上边近倾斜', 'down': '下边近倾斜'}[tilt]
    scale_text = {'small': '小', 'medium': '中', 'large': '大'}[scale]
    region = (('左上', '上方', '右上'), ('左侧', '中央', '右侧'), ('左下', '下方', '右下'))[row][col]
    status = '已有内参支持，倾角可估计' if pose_available else '冷启动：倾角未验证'
    shortage = '区域观测不足' if min(z[row, col] for z in zones) == 0 else '已有观测，补足姿态组合'
    reason = (f'{region}{shortage}；补充{tilt_text}、{scale_text}画面占比'
              f'（{DISTANCE_LABELS[scale]}）。{status}。')
    if repairing_distance:
        reason = '距离缺口补采：' + reason
    if not candidates:
        reason = f'当前分辨率下构图难以兼顾边距和角点间距；请调整分辨率或标定板。{status}。'
    distances = ' / '.join(f'{DISTANCE_LABELS[s]} {paired_distances[s]}'
                           for s in ('large', 'medium', 'small'))
    approximate = ' / '.join(f'{DISTANCE_LABELS[s]} {approximate_distances[s]}'
                             for s in ('large', 'medium', 'small'))
    summary = (f'有效双目样本 {len(unique)} 组，姿态可解 {verified} 组；引导内参 {len(intrinsics)}/2。'
               '中央中→近→远，再同高度补边缘；已建议不计姿态或距离覆盖。'
               f'双路同档可信PnP距离样本：{distances}。'
               f'占比近似样本：{approximate}。'
               '占比近似不计为可信距离覆盖；内参可用后重新评估全部历史样本。'
               '构图仅为几何近似，不保证双目共视；不代表标定完成或全局最优。')
    unavailable = [s for s in _SCALES if not any(c[:2] == (1, 1) and c[3] == s for c in candidates)]
    if unavailable:
        summary += ' 当前分辨率下受限：' + '、'.join(DISTANCE_LABELS[s] for s in unavailable) + '；不计为已完成。'
    missing = tuple(s for s in ('large', 'medium', 'small')
                    if paired_distances[s] < DISTANCE_SAMPLE_GOALS[s])
    if missing:
        summary += ' 距离待补：' + '、'.join(DISTANCE_LABELS[s] for s in missing) + '。'
        feasible_missing = [s for s in missing if s not in unavailable]
        exhausted = (sum(repair_requests.values()) >= MAX_DISTANCE_REPAIR_REQUESTS
                     or (feasible_missing and all(repair_requests[s] >= MAX_DISTANCE_REPAIR_PER_SCALE
                                                 for s in feasible_missing)))
        if exhausted:
            summary += '本轮补采建议已达上限，先完成分区；缺口未清零，不代表距离采集完成。'
    rec = GuidanceRecommendation(row, col, tilt, scale, reason, pose_available, len(unique),
                                 bool(candidates), repairing_distance, missing)
    return GuidanceAnalysis(rec, intrinsics, summary, dict(paired_distances),
                            dict(approximate_distances))
