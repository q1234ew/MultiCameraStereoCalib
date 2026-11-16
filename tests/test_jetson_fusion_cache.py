"""Cached RGB-D is explicit, bounded, immutable and cannot bypass geometry checks."""
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from scripts.jetson.fusion_cache import CaptureCache
from scripts.jetson.fusion_controller import FusionController
from scripts.jetson.fusion_workflow import run_fusion


def source_cache(tmp_path, frames=2):
    params = dict(devices=['device_1', 'device_2'], frames=frames,
                  resolution='640x360', quality='fine')
    folder = tmp_path / ('a' * 32)
    for ident in params['devices']:
        root = folder / 'captures' / ident
        root.mkdir(parents=True)
        (root / 'camera.json').write_text('{}')
        for kind in ('color', 'depth'):
            (root / kind).mkdir()
            for i in range(frames):
                (root / kind / f'{i:06d}.png').write_bytes(bytes([i + 1]))
    metrics = [dict(device_id=d, received_at_unix_ms=round(time.time() * 1000), inference_ms=100)
               for _ in range(frames) for d in params['devices']]
    cache = CaptureCache(tmp_path)
    cache.publish(folder, params, 'current', metrics)
    return cache, folder, params, metrics


def test_cache_reuses_subset_frames_and_preserves_original_receipt_time(tmp_path):
    cache, folder, params, metrics = source_cache(tmp_path)
    target = tmp_path / ('b' * 32) / 'captures'
    reused, provenance = cache.restore(folder.name, dict(params, frames=1), 'current', target)
    assert reused == metrics[:2]
    assert provenance['captured_at_unix_ms'] == metrics[0]['received_at_unix_ms']
    assert (target / 'device_1/depth/000000.png').read_bytes() == b'\1'
    assert not (target / 'device_1/depth/000001.png').exists()
    assert CaptureCache(tmp_path).records[0]['id'] == folder.name


@pytest.mark.parametrize('change', ['expired', 'revision', 'quality', 'file', 'devices'])
def test_cache_rejects_stale_changed_or_incompatible_inputs(tmp_path, change):
    cache, folder, params, _ = source_cache(tmp_path)
    context = 'current'
    if change == 'expired':
        cache.records[0]['captured_at_unix_ms'] -= 601000
    elif change == 'revision':
        context = 'changed'
    elif change == 'quality':
        params = dict(params, quality='fast')
    elif change == 'devices':
        params = dict(params, devices=['device_1', 'device_3'])
    else:
        (folder / 'captures/device_1/depth/000000.png').write_bytes(b'modified')
    with pytest.raises(ValueError):
        cache.restore(folder.name, params, context, tmp_path / 'target')
    assert not (tmp_path / 'target').exists()


def test_clear_preserves_historical_inputs_and_marks_context_mismatch(tmp_path):
    cache, folder, _, _ = source_cache(tmp_path)
    assert cache.summaries(lambda ids: 'new-runtime')[0]['compatible'] is False
    cache.clear()
    assert not CaptureCache(tmp_path).records
    assert (folder / 'captures/device_1/camera.json').exists()


def test_pointcloud_only_skips_tsdf_but_keeps_alignment_gate(tmp_path, monkeypatch):
    from scripts.jetson import fusion_workflow
    calls = []
    monkeypatch.setattr(fusion_workflow, 'check_anchor_depth', lambda *_: {'passed': True})
    monkeypatch.setattr(fusion_workflow, 'load_cloud_preview', lambda *_: {'point_count': 2})

    def command(cmd, **kwargs):
        calls.append(cmd)
        output = Path(cmd[cmd.index('--out') + 1])
        output.mkdir(parents=True)
        (output / 'summary.json').write_text('{}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(fusion_workflow.subprocess, 'run', command)
    extrinsics = tmp_path / 'extrinsics.json'
    extrinsics.write_text(json.dumps({'cameras': {}, 'world_frame': 'tag'}))
    result = run_fusion([], extrinsics, tmp_path / 'result', build_mesh=False)
    assert len(calls) == 1 and 'make_multicamera_pointcloud.py' in str(calls[0])
    assert result['tsdf'] is None and not result['build_mesh']
    monkeypatch.setattr(fusion_workflow, 'check_anchor_depth', lambda *_: {'passed': False})
    with pytest.raises(ValueError, match='尺度或方向'):
        run_fusion([], extrinsics, tmp_path / 'invalid', build_mesh=False)
    assert len(calls) == 1


def test_clear_requires_csrf_and_refuses_busy_job(tmp_path):
    async def check():
        settings = SimpleNamespace(csrf='token')
        controller = FusionController({}, None, settings, tmp_path, tmp_path)
        app = web.Application()
        app.router.add_post('/clear', controller.clear_cache)
        async with TestClient(TestServer(app)) as client:
            assert (await client.post('/clear')).status == 403
            headers = {'Origin': str(client.make_url('')).rstrip('/'), 'X-Camera-CSRF': 'token'}
            assert (await client.post('/clear', headers=headers)).status == 200
            controller.task = asyncio.create_task(asyncio.sleep(30))
            assert (await client.post('/clear', headers=headers)).status == 409
            controller.task.cancel()
            await asyncio.gather(controller.task, return_exceptions=True)
    asyncio.run(check())


def test_controller_cache_hit_skips_inference_and_still_runs_alignment_gate(tmp_path):
    import numpy as np
    from scripts.jetson.core import load_config

    async def check():
        _, folder, params, metrics = source_cache(tmp_path)
        config = load_config(Path(__file__).resolve().parents[1] / 'packaging/jetson/devices.json')
        devices = config['devices'][:2]

        async def forbidden(*args):
            raise AssertionError('Cache hit must not invoke depth inference')

        stereo = SimpleNamespace(_compute=forbidden, batch_active=False)
        controller = FusionController(config, stereo, None, tmp_path, tmp_path)
        active = {'quality_passed': True, 'source': 'foundation_stereo_rectified',
                  'translation_unit': 'm', 'cameras': {d['id']: {
                      'calibration_sha256': d['calibration_sha256'],
                      'point_frame': d['id'] + ':C_L_rect',
                      'T_world_camera': np.eye(4).tolist()} for d in devices}}
        controller.active = lambda: active
        controller.capture_cache.publish(folder, params, controller.cache_context(params['devices']),
                                         metrics)
        commands = []

        async def command(args, output):
            commands.append(args)
            result = output / 'result'
            result.mkdir()
            (result / 'alignment_report.json').write_text(json.dumps({'passed': False}))
            return 2

        controller.command = command
        controller.job = {'id': 'b' * 32, 'state': 'running'}
        await controller.run('fuse', dict(params, capture_cache_id=folder.name, build_mesh=False, max_depth_m=4))
        assert controller.job['capture_cache']['hit'] is True
        assert controller.job['capture_metrics'] == metrics
        assert controller.job['state'] == 'needs_adjustment'
        assert len(commands) == 1 and '--pointcloud-only' in commands[0]
        assert not stereo.batch_active
    asyncio.run(check())
