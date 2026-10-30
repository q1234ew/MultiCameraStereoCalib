"""Evidence-based planar guidance, separate from the final cross-modal solver."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np

from .adaptive_guidance import GuidanceObservation, measure_composition, measure_pose, _valid_intrinsics
from .auxiliary import AuxIntrinsicCalibrator, _validated_detection
from .aux_capture_quality import quality_reason_text
from .health import check_intrinsics
from .observations import similar_composition
from .planar import PlanarPatternDetector


# Distances are relative to the IR/auxiliary field of view, not the wider RGB view.
AREA_RANGES = {'large': (.18, .60), 'medium': (.06, .18), 'small': (.018, .06)}
AREA_TARGETS = {'large': .28, 'medium': .11, 'small': .035}
DISTANCE_NAMES = {'large': '近距离', 'medium': '中距离', 'small': '远距离'}
DISTANCE_GOALS = {'large': 2, 'medium': 3, 'small': 2}
MAX_REPAIRS = 4


@dataclass(frozen=True)
class AuxTarget:
    row: int = 1
    col: int = 1
    tilt: str = 'front'
    scale: str = 'medium'


def initial_plan():
    center = [('medium', 'front'), ('medium', 'left'), ('medium', 'right'),
              ('medium', 'up'), ('medium', 'down'), ('large', 'front'),
              ('large', 'left'), ('small', 'front'), ('small', 'right')]
    return tuple(AuxTarget(1, 1, tilt, scale) for scale, tilt in center) + tuple(
        AuxTarget(row, col, scale='small') for row, col in
        ((1, 0), (1, 2), (0, 2), (0, 1), (0, 0), (2, 0), (2, 1), (2, 2)))


@dataclass
class AuxGuidanceAnalysis:
    phase: str
    target: AuxTarget | None
    unique_count: int
    duplicate_count: int
    pose_count: int
    distance_counts: dict
    missing: tuple
    intrinsics: tuple = (None, None)
    repair_count: int = 0
    initial_total: int = 17
    summary: str = ''
    poses: list = field(default_factory=list)


def board_adapter(board):
    points = PlanarPatternDetector._make_object_points(board)
    return SimpleNamespace(object_points=points, num_corners=len(points))


def observation(det, key='frame'):
    return GuidanceObservation(np.array(det.image_points, copy=True).reshape(-1, 1, 2),
                               np.arange(det.num_points, dtype=np.int32).reshape(-1, 1),
                               det.image_size, str(key))


def feasible_target(board, target, size):
    """Fit the full physical grid envelope in the auxiliary view, conservatively."""
    points = board_adapter(board).object_points[:, :2]
    extent = np.ptp(points, axis=0)
    width, height = size
    area = AREA_TARGETS[target.scale] * width * height
    span = np.array([np.sqrt(area * extent[0] / extent[1]),
                     np.sqrt(area * extent[1] / extent[0])])
    spacing = span / np.maximum(1, [board.cols - 1, board.rows - 1])
    envelope = span + 2 * spacing
    center = np.array([(target.col + .5) / 3 * width, (target.row + .5) / 3 * height])
    return bool(spacing.min() >= 4 and np.all(center - envelope / 2 >= 2)
                and np.all(center + envelope / 2 <= np.array(size) - 3))


def duplicate_pair(pair, previous):
    """Image-space repetition only; never reorder solver observations."""
    def equivalent(a, b):
        return any(similar_composition(
            a.image_points, np.arange(a.num_points), a.image_size,
            points, np.arange(b.num_points), b.image_size,
        ) for points in (b.image_points, b.image_points[::-1]))

    return any(all(equivalent(a, b) for a, b in zip(pair, old)) for old in previous)


def _bootstrap(board, model, detections):
    if len(detections) < 6:
        return None
    selected = [detections[i] for i in np.linspace(0, len(detections) - 1,
                                                 min(24, len(detections)), dtype=int)]
    obj = board_adapter(board).object_points[:, :2].astype(float)
    obj = (obj - obj.mean(axis=0)) / np.ptp(obj, axis=0)
    perspective = []
    try:
        for det in selected:
            h, _ = cv2.findHomography(obj, det.image_points / det.image_size, 0)
            if h is None or abs(h[2, 2]) < 1e-10:
                return None
            perspective.append(h[2, :2] / h[2, 2])
        perspective = np.asarray(perspective)
        singular = np.linalg.svd(perspective - perspective.mean(axis=0), compute_uv=False)
        if singular[-1] / np.sqrt(len(selected)) < .012:
            return None
        cal = AuxIntrinsicCalibrator(board, model, min_frames=6)
        for det in selected:
            if not cal.add_detection(det):
                return None
        result = cal.calibrate()
        if (_valid_intrinsics(result, selected[0].image_size, model)
                and result.rms_error <= 2.5 and check_intrinsics(result)['status'] == 'ok'):
            return result
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError):
        pass
    return None


def analyze_aux_guidance(board, model, pairs, intrinsics=(None, None)):
    clean = []
    duplicates = 0
    expected = None
    for _, rgb, aux in pairs:
        pair = tuple(_validated_detection(d, board) for d in (rgb, aux))
        if any(d is None for d in pair):
            continue
        sizes = tuple(d.image_size for d in pair)
        if expected is None:
            expected = sizes
        if sizes != expected:
            continue
        if duplicate_pair(pair, clean):
            duplicates += 1
            continue
        clean.append(pair)
    fitted = []
    for i, supplied in enumerate(intrinsics):
        size = expected[i] if expected else getattr(supplied, 'image_size', ())
        if (supplied is not None and _valid_intrinsics(supplied, size, supplied.model)
                and check_intrinsics(supplied)['status'] == 'ok'):
            fitted.append(supplied)
        else:
            fitted.append(_bootstrap(board, model, [p[i] for p in clean]))
    adapter = board_adapter(board)
    counts, tilted, zones = Counter(), Counter(), Counter()
    repair_requests, repair_tilts = Counter(), Counter()
    repair_target = None
    poses = []
    verified = 0
    plan = initial_plan()
    for index, (rgb, aux) in enumerate(clean):
        if index >= len(plan) and repair_target is not None:
            repair_requests[repair_target.scale] += 1
            repair_tilts[repair_target.tilt] += 1
        current = [measure_pose(adapter, observation(det), intr)
                   for det, intr in zip((rgb, aux), fitted)]
        poses.append(current)
        verified += int(all(p is not None for p in current))
        comp = measure_composition(adapter, observation(aux))
        if comp:
            row, col = np.clip(np.array(comp['center'])[::-1] * 3, 0, 2).astype(int)
            zones[int(row), int(col)] += 1
        pose = current[1]
        if pose is not None and pose['distance_reliable']:
            for scale, (low, high) in AREA_RANGES.items():
                if low <= pose['distance_area'] < high:
                    counts[scale] += 1
                    break
        rgb_pose = current[0]
        if rgb_pose is not None:
            yaw, pitch = rgb_pose['yaw'], rgb_pose['pitch']
            tilt = ('front' if max(abs(yaw), abs(pitch)) < 15 else
                    ('right' if yaw > 0 else 'left') if abs(yaw) >= abs(pitch) else
                    ('down' if pitch > 0 else 'up'))
            tilted[tilt] += 1
        if index + 1 >= len(plan):
            missing_now = [s for s in ('large', 'small', 'medium')
                           if counts[s] < DISTANCE_GOALS[s] and repair_requests[s] < 2]
            if missing_now and sum(repair_requests.values()) < MAX_REPAIRS:
                scale = missing_now[0]
                tilt = min(('front', 'left', 'right', 'up', 'down'),
                           key=lambda t: tilted[t] + repair_tilts[t])
                repair_target = AuxTarget(1, 1, tilt, scale)
            else:
                repair_target = None
    missing = tuple(s for s in ('large', 'small', 'medium') if counts[s] < DISTANCE_GOALS[s])
    repairs = max(0, len(clean) - len(plan))
    if len(clean) < len(plan):
        phase, target = 'initial', plan[len(clean)]
    elif repair_target is not None and repairs < MAX_REPAIRS:
        phase, target = 'repair', repair_target
    else:
        phase, target = 'complete', None
    if target is not None and expected is not None and not feasible_target(board, target, expected[1]):
        phase, target = 'blocked', None
    summary = (f'非重复 {len(clean)} 组 · 重复 {duplicates} 组 · 双路姿态可解 {verified} 组\n'
               + '可信 IR 距离：' + ' / '.join(f'{DISTANCE_NAMES[s]} {counts[s]}'
                                                for s in ('large', 'medium', 'small')))
    return AuxGuidanceAnalysis(phase, target, len(clean), duplicates, verified, dict(counts),
                               missing, tuple(fitted), min(repairs, MAX_REPAIRS), len(plan),
                               summary, poses)


def guidance_action(board, target, qualities):
    """Return one current action; diagram tilt follows that action, not a future step."""
    if target is None:
        return 'complete', '本轮采集完成，可以前往标定评估；仍可继续补拍'
    if len(qualities) != 2 or not all(q.accepted for q in qualities):
        reasons = [f'{name}：' + '、'.join(quality_reason_text(r) for r in q.reasons) for name, q in
                   zip(('RGB', 'IR/辅助'), qualities) if not q.accepted]
        return 'quality', '；'.join(reasons) or '让同一块标定板同时完整进入两路画面'
    aux = qualities[1]
    area = aux.metrics.get('area', 0.)
    reliable = aux.pose is not None and aux.pose.get('distance_reliable', False)
    if reliable:
        area = aux.pose['distance_area']
    low, high = AREA_RANGES[target.scale]
    suffix = '' if reliable else '（按 IR 占比近似）'
    if area < low:
        return 'distance', '板面先正对相机，再靠近镜头；保持当前高度' + suffix
    if area >= high:
        return 'distance', '板面先正对相机，再远离镜头；保持当前高度' + suffix
    center = np.asarray(aux.metrics.get('center', (.5, .5)))
    desired = np.array([(target.col + .5) / 3, (target.row + .5) / 3])
    if np.linalg.norm(center - desired) > .16:
        zone = (('左上', '上方', '右上'), ('左侧', '中央', '右侧'), ('左下', '下方', '右下'))
        return 'position', f'移至 IR 画面{zone[target.row][target.col]}；保持距离，兼顾 RGB 共视'
    pose = qualities[0].pose
    labels = {'front': '将板面正对镜头', 'left': '让标定板左边靠近镜头',
              'right': '让标定板右边靠近镜头', 'up': '让标定板上边靠近镜头',
              'down': '让标定板下边靠近镜头'}
    if pose is None or not pose.get('distance_reliable', False):
        return 'tilt_unverified', labels[target.tilt] + '；倾角待人工确认'
    yaw, pitch = pose['yaw'], pose['pitch']
    aligned = (max(abs(yaw), abs(pitch)) < 15 if target.tilt == 'front' else
               15 <= {'left': -yaw, 'right': yaw, 'up': -pitch, 'down': pitch}[target.tilt] <= 40
               and abs(pitch if target.tilt in ('left', 'right') else yaw) < 20)
    return ('ready', '保持静止，可以采集') if aligned else ('tilt', labels[target.tilt])
