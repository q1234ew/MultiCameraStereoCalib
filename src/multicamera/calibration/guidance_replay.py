"""Prefix-only offline evaluation of the existing advisory capture recommender."""

from collections import Counter
from dataclasses import asdict

import numpy as np

from .adaptive_guidance import (
    _geometry, _scale, analyze_guidance, evaluate_distance, measure_composition, measure_pose,
)


def _snapshot(analysis):
    return {'recommendation': asdict(analysis.recommendation), 'summary': analysis.summary,
            'intrinsic_cameras': sorted(analysis.intrinsics)}


def _measurement(board, observation, intrinsics):
    geometry = _geometry(board, observation) if observation is not None else None
    if geometry is None:
        return {'geometry_valid': False, 'pose': None}
    clean = geometry[0]
    center = geometry[2]
    pose = measure_pose(board, clean, intrinsics)
    if pose is not None:
        pose = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in pose.items()}
    return {'geometry_valid': True, 'center': center.tolist(), 'area': geometry[3],
            'distance_area': measure_composition(board, clean)['distance_area'],
            'corners': len(clean.ids), 'pose': pose}


def _matches(recommendation, measurements):
    if not recommendation.feasible:
        return {'composition_match': False, 'reason': '无可用构图，请调整分辨率或标定板'}
    if not all(m['geometry_valid'] for m in measurements):
        return {'composition_match': False, 'reason': '双路检测或几何有效性不足'}
    center = np.array([(recommendation.zone_col + .5) / 3,
                       (recommendation.zone_row + .5) / 3])
    if not all(np.linalg.norm(np.array(m['center']) - center) < .16 for m in measurements):
        return {'composition_match': False, 'reason': '位置不符合此前建议'}
    distance_ready, distance_message, _, _ = evaluate_distance(recommendation.scale, measurements)
    if not distance_ready:
        return {'composition_match': False, 'reason': distance_message}
    if any(m['pose'] is None for m in measurements):
        return {'composition_match': None, 'reason': '位置和大小接近；历史内参不足或姿态不可解，倾角不评估'}
    for m in measurements:
        yaw, pitch = m['pose']['yaw'], m['pose']['pitch']
        if recommendation.tilt == 'front':
            matched = abs(yaw) < 15 and abs(pitch) < 15
        else:
            primary, secondary, sign = {'left': (yaw, pitch, -1), 'right': (yaw, pitch, 1),
                                        'up': (pitch, yaw, -1), 'down': (pitch, yaw, 1)}[recommendation.tilt]
            matched = 15 <= sign * primary <= 40 and abs(secondary) < 20
        if not matched:
            return {'composition_match': False, 'reason': '倾角或方向不符合此前建议'}
    return {'composition_match': True, 'reason': '双路构图接近此前建议范围（非自动采集合格判定）'}


def replay_observations(board, model, frames):
    """Analyze each prefix before seeing its next image; never import final intrinsics.

    frames is an iterable of (id, {left/right: observation or None}). A mismatch
    is not a rejected calibration sample: historical operators did not follow
    today's recommender. Motion, synchronization and AWB remain unassessed.
    """
    history = {'left': [], 'right': []}
    zones = {'left': Counter(), 'right': Counter()}
    scales = {'left': Counter(), 'right': Counter()}
    analysis = analyze_guidance(board, model, history, {})
    rows = []
    for frame_id, observations in frames:
        before = _snapshot(analysis)
        measured = {camera: _measurement(board, observations.get(camera), analysis.intrinsics.get(camera))
                    for camera in history}
        verdict = _matches(analysis.recommendation, list(measured.values()))
        distance_ready, distance_message, distance_areas, distance_source = evaluate_distance(
            analysis.recommendation.scale, list(measured.values()))
        verdict.update(distance_match=distance_ready, distance_message=distance_message,
                       distance_source=distance_source)
        for camera in history:
            observation = observations.get(camera)
            if observation is not None:
                if observation.frame_id != frame_id:
                    raise ValueError('回放帧编号与观测编号不一致')
                history[camera].append(observation)
        analysis = analyze_guidance(board, model, history, {})
        if all(m['geometry_valid'] for m in measured.values()):
            for (camera, m), area in zip(measured.items(), distance_areas):
                col, row = np.clip(np.array(m['center']) * 3, 0, 2).astype(int)
                zones[camera][int(row), int(col)] += 1
                scales[camera][_scale(area)] += 1
        rows.append({'frame_id': frame_id, 'before': before, 'measurement': measured,
                     'verdict': verdict, 'after': _snapshot(analysis),
                     'unique_sample_change': analysis.recommendation.sample_count - before['recommendation']['sample_count'],
                     'motion_stability': None, 'synchronization': None, 'white_balance': None})
    coverage = {camera: {'zone_counts': [[zones[camera][r, c] for c in range(3)] for r in range(3)],
                         'missing_zones': [[r, c] for r in range(3) for c in range(3) if not zones[camera][r, c]],
                         'scale_counts': dict(scales[camera])} for camera in history}
    return {'steps': rows, 'final': _snapshot(analysis), 'coverage': coverage,
            'coverage_scope': '双路几何有效的输入帧中心与等效距离统计；距离口径与客户端一致，使用此前内参，含重复构图；非精度评估',
            'limits': ['按文件编号回放，编号不证明实际拍摄时间或同步',
                       '每步内参仅来自此前观测，未载入最终标定结果',
                       '本轮不按清晰度预先筛图；有效样本数仅指引导几何有效且去重，不代表图像质量或标定精度通过',
                       '静态图片不评估运动稳定、硬件同步或白平衡准备',
                       '构图不匹配不等于图片无效；历史图片并未按当前建议拍摄',
                       '建议来自覆盖补缺启发式，不证明最优姿态或标定精度']}
