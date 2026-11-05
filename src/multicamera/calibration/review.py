"""Advisory, explainable frame review over immutable observation snapshots."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json

import numpy as np

from .adaptive_guidance import measure_composition, measure_pose


def scope_key(board, model, pair, source='', swapped=False):
    value = [board.to_dict(), model.value, pair.name, pair.left.camera_id,
             pair.right.camera_id, pair.left.url, pair.right.url, str(source), swapped]
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def frame_fingerprint(left, right):
    digest = sha256()
    for observation in (left, right):
        if observation is None:
            digest.update(b'missing')
            continue
        digest.update(str(observation.image_size).encode())
        for array in (observation.ids, observation.corners):
            digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def excluded_ids(state, fingerprints):
    """Old decisions never apply to different observations reusing a frame ID."""
    return {key for category in ('excluded', 'pending')
            for key, fingerprint in state.get(category, {}).items()
            if fingerprints.get(key) == fingerprint}


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_state(value):
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        raise ValueError('图像复核记录版本无效')
    for field in ('excluded', 'pending'):
        entries = value.get(field)
        if not isinstance(entries, dict) or not all(
                isinstance(k, str) and bool(k) and _digest(v) for k, v in entries.items()):
            raise ValueError('图像参与记录无效')
    if not isinstance(value.get('replacements'), list) or not all(
            isinstance(entry, dict) and isinstance(entry.get('new_id'), str)
            and (entry.get('old_id') is None or isinstance(entry['old_id'], str))
            and _digest(entry.get('fingerprint')) for entry in value.get('replacements', [])):
        raise ValueError('图像替换记录无效')
    return value


def empty_state():
    return {'schema_version': 1, 'excluded': {}, 'pending': {}, 'replacements': []}


def _outliers(rows, metric):
    values = [row[metric] for row in rows if row.get(metric) is not None
              and np.isfinite(row[metric]) and row[metric] >= 0]
    if len(values) < 5:
        return set()
    median = float(np.median(values))
    mad = float(np.median(np.abs(np.asarray(values) - median)))
    threshold = max(1., median + max(.25, 3 * 1.4826 * mad))
    return {row['id'] for row in rows if row.get(metric) is not None
            and np.isfinite(row[metric]) and row[metric] > threshold}


def analyze_review(board, rows, intrinsics):
    """Geometry/pose are heuristics; recommendations never certify accuracy."""
    zones = [Counter(), Counter()]
    tilts = Counter()
    for row in rows:
        row['geometry'] = []
        for side, intr in zip(('left', 'right'), intrinsics):
            obs = row.get(side)
            geometry = measure_composition(board, obs) if obs is not None else None
            pose = measure_pose(board, obs, intr) if geometry and intr is not None else None
            if geometry:
                col, r = np.clip(np.asarray(geometry['center']) * 3, 0, 2).astype(int)
                area = pose['distance_area'] if pose else geometry['distance_area']
                tilt = ('left' if pose['yaw'] < -12 else 'right' if pose['yaw'] > 12 else
                        'up' if pose['pitch'] < -12 else 'down' if pose['pitch'] > 12 else
                        'front') if pose else 'front'
                row['geometry'].append(dict(row=int(r), col=int(col), tilt=tilt,
                                            center=geometry['center'].tolist(),
                                            scale='small' if area < .04 else
                                            'large' if area >= .12 else 'medium',
                                            pose_available=pose is not None,
                                            pose={k: pose[k] for k in ('yaw', 'pitch')} if pose else None))
            else:
                row['geometry'].append(None)
        if row['participates']:
            for i, geom in enumerate(row['geometry']):
                if geom:
                    zones[i][geom['row'], geom['col']] += 1
            if all(g and g['pose_available'] for g in row['geometry']):
                tilts[row['geometry'][0]['tilt']] += 1
    outliers = {metric: _outliers(rows, metric)
                for metric in ('left_rms', 'right_rms', 'alignment_rms')}
    for row in rows:
        reasons = list(row.get('quality_reasons', ()))
        for metric, label in (('left_rms', '左内参 RMS'), ('right_rms', '右内参 RMS'),
                              ('alignment_rms', '校正残差 RMS')):
            if row['id'] in outliers[metric]:
                reasons.append(f'{label} 明显高于同组多数帧，建议核对原图')
        if row.get('common_count', 0) < 6:
            reasons.append('共同角点不足，无法作为有效双目观测')
        row['reasons'] = reasons
        row['value'] = '保留原图供复核；误差高不等于图片无效。'
        for i, geom in enumerate(row['geometry']):
            if geom and zones[i][geom['row'], geom['col']] <= int(row['participates']):
                row['value'] = ('该构图在' + ('左' if i == 0 else '右')
                                + '相机的区域缺少其他有效观测，建议补拍后再排除。')
                break
    tasks = []
    positions = (('左上', '上方', '右上'), ('左侧', '中央', '右侧'), ('左下', '下方', '右下'))
    for r in range(3):
        for c in range(3):
            missing = [label for counts, label in zip(zones, ('左', '右')) if counts[r, c] == 0]
            if missing:
                tasks.append(dict(row=r, col=c, tilt='front', scale='medium',
                                  title=f'{positions[r][c]}区域 · 正对',
                                  reason='、'.join(missing) + '相机该区域缺少有效观测', count=2))
    if tilts and (not tilts['up'] or not tilts['down']):
        tasks.append(dict(row=1, col=1, tilt='up' if not tilts['up'] else 'down',
                          scale='medium', title='中央区域 · 补充上下倾角',
                          reason='已有可解板姿中，上下倾角方向不足', count=2))
    return dict(rows=rows, tasks=tasks, zone_counts=[dict(z) for z in zones],
                flagged=sum(bool(row['reasons']) for row in rows),
                notes='基于当前计划参与的数据；区域、距离、倾角为经验建议，不代表精度合格。')
