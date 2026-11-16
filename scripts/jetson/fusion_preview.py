"""Bounded, source-labelled previews of the migrated pipeline's binary PLY output."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

VERTEX_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('camera_id', '<i4'),
])
PROPERTIES = [
    'property float x', 'property float y', 'property float z',
    'property uchar red', 'property uchar green', 'property uchar blue',
    'property int camera_id',
]


def cloud_preview(ply: Path, summary: dict, known_ids, limit: int = 12000) -> dict:
    """Keep the PLY's retained-point owner, including cameras with few points.

    Ownership is assigned before voxel deduplication by the vendor pipeline; it is
    not an estimate of all cameras observing a voxel. Never infer it from proximity.
    """
    serials = summary['serials']
    if (not isinstance(serials, list) or not 2 <= len(serials) <= 3
            or len(set(serials)) != len(serials) or not set(serials) <= set(known_ids)
            or any(not re.fullmatch(r'device_[123]', s) for s in serials)):
        raise ValueError('点云来源设备与外参不一致')
    if type(limit) is not int or not len(serials) <= limit <= 12000:
        raise ValueError('预览采样上限无效')
    with Path(ply).open('rb') as stream:
        header = []
        for _ in range(32):
            line = stream.readline(256).decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        else:
            raise ValueError('点云头部无效')
        offset = stream.tell()
    if (header[:2] != ['ply', 'format binary_little_endian 1.0']
            or len(header) != 11 or header[3:-1] != PROPERTIES
            or not re.fullmatch(r'element vertex [1-9][0-9]*', header[2])):
        raise ValueError('点云缺少可验证的设备来源标签')
    count = int(header[2].split()[-1])
    if Path(ply).stat().st_size != offset + count * VERTEX_DTYPE.itemsize:
        raise ValueError('点云顶点长度不匹配')
    vertices = np.memmap(ply, dtype=VERTEX_DTYPE, mode='r', offset=offset, shape=(count,))
    if any(not np.isfinite(vertices[k]).all() for k in ('x', 'y', 'z')):
        raise ValueError('点云包含非有限坐标')
    labels = vertices['camera_id']
    if np.any((labels < 0) | (labels >= len(serials))):
        raise ValueError('点云包含未知设备标签')
    # Split the preview budget fairly so a small contributing camera stays visible.
    quota = max(1, limit // len(serials))
    selected, devices = [], []
    for index, ident in enumerate(serials):
        owned = np.flatnonzero(labels == index)
        sample = owned[np.linspace(0, len(owned) - 1, min(quota, len(owned))).astype(int)]
        selected.extend(sample.tolist())
        devices.append({'id': ident, 'point_count': len(owned), 'preview_count': len(sample)})
    indices = np.asarray(selected, dtype=int)
    points = np.column_stack([vertices[k][indices] for k in ('x', 'y', 'z')])
    colors = np.column_stack([vertices[k][indices] for k in ('red', 'green', 'blue')])
    return {
        'point_count': count, 'points': np.round(points.astype(float), 4).tolist(),
        'colors': colors.tolist(),
        'point_device_ids': [serials[int(i)] for i in labels[indices]],
        'devices': devices, 'source_labels_available': True,
    }


def load_cloud_preview(folder: Path, known_ids) -> dict:
    folder = Path(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    return cloud_preview(folder / 'multi_orbbec_world_rgb_pointcloud.ply', summary, known_ids)
