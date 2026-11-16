"""Relaxed previews never imply strict geometry or measurement accuracy passed."""
import asyncio
import json

import cv2
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from scripts.jetson.fusion_controller import FusionController
from scripts.jetson.fusion_geometry import check_anchor_depth


def inputs(root, expected_m, measured_mm):
    folders, cameras = [], {}
    for ident in ['device_1', 'device_2']:
        folder = root / ident
        (folder / 'depth').mkdir(parents=True)
        (folder / 'camera.json').write_text(json.dumps({
            'device': {'serial_number': ident}, 'depth_scale_m_per_unit': .001,
            'color': {'intrinsic': {'fx': 100, 'fy': 100, 'cx': 50, 'cy': 50}}}))
        cv2.imwrite(str(folder / 'depth/000000.png'),
                    np.full((100, 100), measured_mm, np.uint16))
        pose = np.eye(4)
        pose[2, 3] = -expected_m
        cameras[ident] = {'T_world_camera': pose.tolist()}
        folders.append(folder)
    return folders, {'source': 'foundation_stereo_rectified', 'tag_side_mm': 80,
                     'cameras': cameras}


def test_preview_window_retains_failed_strict_result(tmp_path):
    folders, calibration = inputs(tmp_path, 1., 910)
    strict = check_anchor_depth(folders, calibration)
    preview = check_anchor_depth(folders, calibration, 'preview')
    assert not strict['passed'] and strict['tolerance_profile'] == 'strict'
    assert preview['passed'] and preview['preview_only']
    assert preview['relaxed_acceptance_used'] and not preview['strict_passed']
    assert not preview['measurement_accuracy_verified']
    assert preview['max_plane_normal_error_deg'] == strict['max_plane_normal_error_deg'] == 10
    assert all(not c['strict_passed'] and c['passed'] for c in preview['cameras'].values())


@pytest.mark.parametrize('expected,measured', [(1., 880), (2., 1840)])
def test_preview_still_limits_ratio_and_absolute_depth_error(tmp_path, expected, measured):
    folders, calibration = inputs(tmp_path, expected, measured)
    report = check_anchor_depth(folders, calibration, 'preview')
    assert not report['passed']


def test_unknown_tolerance_profile_is_rejected(tmp_path):
    folders, calibration = inputs(tmp_path, 1., 1000)
    with pytest.raises(ValueError, match='tolerance'):
        check_anchor_depth(folders, calibration, 'unlimited')


def test_preview_download_is_explicitly_named(tmp_path):
    async def check():
        controller = FusionController({}, None, None, tmp_path, tmp_path)
        ident = 'a' * 32
        folder = tmp_path / ident
        (folder / 'result/pointcloud').mkdir(parents=True)
        (folder / 'job.json').write_text(json.dumps({'parameters': {'tolerance_profile': 'preview'}}))
        (folder / 'result/pointcloud/multi_orbbec_world_rgb_pointcloud.ply').write_text('ply')
        app = web.Application()
        app.router.add_get('/files/{job}/{name}', controller.artifact)
        async with TestClient(TestServer(app)) as client:
            response = await client.get('/files/' + ident + '/pointcloud.ply')
            assert response.status == 200
            assert 'preview_pointcloud.ply' in response.headers['Content-Disposition']
    asyncio.run(check())
