"""Keep preview geometry, color and real retained-point ownership aligned."""
import asyncio
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from scripts.jetson.core import load_config
from scripts.jetson.fusion_controller import FusionController
from scripts.jetson.fusion_preview import cloud_preview
ROOT = Path(__file__).resolve().parents[1]


def make_ply(path, labels):
    header = '\n'.join([
        'ply', 'format binary_little_endian 1.0', f'element vertex {len(labels)}',
        'property float x', 'property float y', 'property float z',
        'property uchar red', 'property uchar green', 'property uchar blue',
        'property int camera_id', 'end_header', '',
    ]).encode()
    path.write_bytes(header + b''.join(
        struct.pack('<fffBBBi', i / 10, 0, 1, i % 256, 100, 200, label)
        for i, label in enumerate(labels)
    ))


def test_preview_preserves_labels_and_rare_camera_in_bounded_sampling(tmp_path):
    path = tmp_path / 'points.ply'
    make_ply(path, [1] * 100 + [0])
    # The vendor summary's serial order is authoritative, not extrinsics dict order.
    result = cloud_preview(path, {'serials': ['device_2', 'device_1']},
                           ['device_1', 'device_2'], limit=12)
    assert len(result['points']) <= 12
    assert result['point_count'] == 101
    assert result['devices'] == [
        {'id': 'device_2', 'point_count': 1, 'preview_count': 1},
        {'id': 'device_1', 'point_count': 100, 'preview_count': 6},
    ]
    for point, color, ident in zip(result['points'], result['colors'], result['point_device_ids']):
        original_index = round(point[0] * 10)
        assert color == [original_index % 256, 100, 200]
        assert ident == ('device_2' if original_index == 100 else 'device_1')


@pytest.mark.parametrize('fault', ['truncated', 'unknown_label', 'wrong_serial', 'nan'])
def test_preview_rejects_unverifiable_provenance(tmp_path, fault):
    path = tmp_path / 'points.ply'
    make_ply(path, [0, 9 if fault == 'unknown_label' else 1])
    summary = {'serials': ['device_1', 'device_3' if fault == 'wrong_serial' else 'device_2']}
    if fault == 'truncated':
        path.write_bytes(path.read_bytes()[:-1])
    if fault == 'nan':
        raw = path.read_bytes()
        start = raw.index(b'end_header\n') + len(b'end_header\n')
        path.write_bytes(raw[:start] + struct.pack('<f', float('nan')) + raw[start + 4:])
    with pytest.raises(ValueError):
        cloud_preview(path, summary, ['device_1', 'device_2'])


def test_result_reads_real_ply_labels_and_legacy_result_is_explicit(tmp_path):
    async def check():
        config = load_config(ROOT / 'packaging/jetson/devices.json')
        controller = FusionController(config, None, None, tmp_path, tmp_path)
        ident = 'c' * 32
        folder = tmp_path / ident
        (folder / 'result/pointcloud').mkdir(parents=True)
        cameras = {d['id']: {'calibration_sha256': d['calibration_sha256'],
                            'point_frame': d['id'] + ':C_L_rect',
                            'T_world_camera': np.eye(4).tolist()} for d in config['devices'][:2]}
        (folder / 'extrinsics.json').write_text(json.dumps({
            'cameras': cameras, 'quality_passed': True,
            'source': 'foundation_stereo_rectified', 'translation_unit': 'm',
        }))
        (folder / 'result/fusion_result.json').write_text(json.dumps({
            'points': [[0, 0, 1]], 'colors': [[255, 255, 255]], 'point_count': 1,
            'point_unit': 'm', 'world_frame': 'tag', 'measurement_accuracy_verified': False,
        }))
        app = web.Application()
        app.router.add_get('/result/{job}', controller.result)
        async with TestClient(TestServer(app)) as client:
            legacy = await (await client.get('/result/' + ident)).json()
            assert legacy['source_labels_available'] is False
            assert legacy['point_device_ids'] == []
            make_ply(folder / 'result/pointcloud/multi_orbbec_world_rgb_pointcloud.ply', [1, 0])
            (folder / 'result/pointcloud/summary.json').write_text(json.dumps({
                'serials': ['device_2', 'device_1'],
            }))
            labelled = await (await client.get('/result/' + ident)).json()
            assert labelled['source_labels_available'] is True
            assert labelled['point_device_ids'] == ['device_2', 'device_1']
            assert labelled['point_count'] == 2
            cameras['device_2']['calibration_sha256'] = '0' * 64
            (folder / 'extrinsics.json').write_text(json.dumps({
                'cameras': cameras, 'quality_passed': True,
                'source': 'foundation_stereo_rectified', 'translation_unit': 'm',
            }))
            assert (await client.get('/result/' + ident)).status == 409
    asyncio.run(check())


@pytest.mark.parametrize('limit', [1, 12001])
def test_preview_budget_must_fit_all_sources_and_stay_bounded(tmp_path, limit):
    path = tmp_path / 'points.ply'
    make_ply(path, [0, 1])
    with pytest.raises(ValueError, match='采样上限'):
        cloud_preview(path, {'serials': ['device_1', 'device_2']},
                      ['device_1', 'device_2'], limit=limit)
