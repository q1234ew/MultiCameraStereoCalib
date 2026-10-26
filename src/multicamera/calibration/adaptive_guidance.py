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
from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from ..board.detector import DetectionResult
from .intrinsic import IntrinsicCalibrator
from .models import CameraIntrinsics, CameraModel
from .observations import prepare_observation


SCALE_AREA_THRESHOLDS = (.04, .12)
SCALE_AREA_TARGETS = {'small': .025, 'medium': .07, 'large': .16}
CANDIDATE_IMAGE_MARGIN = .04
MIN_CORNER_SPACING_PX = 6.
CANDIDATE_TILT_DEGREES = 25.
TILTED_FOOTPRINT_PADDING = 1.08
MIN_BOARD_HULL_FRACTION = .5
TILT_THRESHOLD_DEGREES = 12.
MAX_REPROJECTION_RMS = 3.
BOOTSTRAP_MIN_VIEWS = 6
BOOTSTRAP_MAX_VIEWS = 30
BOOTSTRAP_MAX_RMS = 2.5
DUPLICATE_RMS_FRACTION = .005
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


@dataclass
class GuidanceAnalysis:
    recommendation: GuidanceRecommendation
    intrinsics: dict[str, CameraIntrinsics]
    summary: str


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
        return dict(yaw=float(np.degrees(np.arctan2(normal[0], normal[2]))),
                    pitch=float(np.degrees(np.arctan2(normal[1], normal[2]))),
                    depth=float(np.mean(transformed[:, 2])), area=area, center=center)
    except (ValueError, TypeError, cv2.error):
        return None


def _similar(a, b):
    if a.image_size != b.image_size:
        return False
    _, ia, ib = np.intersect1d(a.ids.ravel(), b.ids.ravel(), return_indices=True)
    if len(ia) < max(6, .6 * max(len(a.ids), len(b.ids))):
        return False
    delta = a.corners.reshape(-1, 2)[ia] - b.corners.reshape(-1, 2)[ib]
    return np.sqrt(np.mean(np.sum(delta ** 2, axis=1))) / max(a.image_size) < DUPLICATE_RMS_FRACTION


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
    zone_order = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]
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
    for count, pair in enumerate(unique, 1):
        poses = [measure_pose(board, g[0], intrinsics.get(camera))
                 for camera, g in zip(cameras, pair)]
        verified += int(all(p is not None for p in poses))
        for i, g in enumerate(pair):
            zone, scale = _zone(g[2]), _scale(g[3])
            zones[i][zone] += 1
            scales[i][scale] += 1
            if poses[i] is not None:
                tilt = _tilt(poses[i])
                tilts[i][tilt] += 1
                joint[(*zone, tilt, scale)] += 1
        def score(candidate):
            r, c, t, s = candidate
            coverage = sum(3 * z[r, c] + sc[s] + .7 * ti[t]
                           for z, sc, ti in zip(zones, scales, tilts))
            # Unverified poses receive no pose credit; cycle explicit bootstrap requests.
            provisional = 0 if pose_available or t == _TILTS[count % len(_TILTS)] else 2
            return coverage + .7 * joint[candidate] + provisional
        alternatives = [c for c in candidates if c != target]
        if alternatives:
            target = min(alternatives, key=score)
    row, col, tilt, scale = target
    tilt_text = {'front': '正对', 'left': '左边近倾斜', 'right': '右边近倾斜',
                 'up': '上边近倾斜', 'down': '下边近倾斜'}[tilt]
    scale_text = {'small': '小', 'medium': '中', 'large': '大'}[scale]
    region = (('左上', '上方', '右上'), ('左侧', '中央', '右侧'), ('左下', '下方', '右下'))[row][col]
    status = '已有内参支持，倾角可估计' if pose_available else '冷启动：倾角未验证'
    reason = f'{region}区域观测不足；补充{tilt_text}、{scale_text}画面占比。{status}。'
    if not candidates:
        reason = f'当前分辨率下构图难以兼顾边距和角点间距；请调整分辨率或标定板。{status}。'
    summary = (f'有效双目样本 {len(unique)} 组，姿态可解 {verified} 组；引导内参 {len(intrinsics)}/2。'
               '构图仅为几何近似，不保证双目共视；不代表标定完成或全局最优。')
    rec = GuidanceRecommendation(row, col, tilt, scale, reason, pose_available, len(unique))
    return GuidanceAnalysis(rec, intrinsics, summary)
