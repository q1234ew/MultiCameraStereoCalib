"""Scale, frame direction and calibration safety at the live fusion boundary."""
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from aiohttp import web

from scripts.jetson.core import load_config
from scripts.jetson.fusion_geometry import aggregate_poses, pose_from_tag, validate_active
from scripts.jetson.fusion_controller import FusionController
from scripts.jetson.photos import annotate_photo

ROOT = Path(__file__).resolve().parents[1]


def test_tag_scale_and_right_camera_transform_direction():
    k = np.array([[800., 0, 960], [0, 800., 540], [0, 0, 1]])
    obj = np.array([[0., 0, 0], [.1, 0, 0], [.1, .1, 0], [0, .1, 0]])
    # A tag at 1m; right camera is 50mm to the right of the left camera.
    corners, _ = cv2.projectPoints(obj, np.zeros(3), np.array([-.05, 0., 1.]), k, None)
    world_right, error = pose_from_tag(corners.reshape(4, 2), k, .1)
    right_from_left = np.eye(4)
    right_from_left[0, 3] = -.05
    world_left = world_right @ right_from_left
    assert error < 1e-5
    assert np.allclose(world_right[:3, 3], [.05, 0, -1], atol=1e-6)
    assert np.allclose(world_left[:3, 3], [0, 0, -1], atol=1e-6)
    doubled, _ = pose_from_tag(corners.reshape(4, 2), k, .2)
    assert doubled[2, 3] == pytest.approx(-2)


def test_pose_consistency_rejects_large_camera_motion():
    poses = [np.eye(4) for _ in range(3)]
    _, stats, passed = aggregate_poses(poses, [.1, .2, .3])
    assert passed and stats['sample_count'] == 3
    poses[2][0, 3] = .5
    assert not aggregate_poses(poses, [.1, .2, .3])[2]


def test_active_extrinsics_reject_wrong_bundle_or_improper_transform():
    device = {'id': 'device_1', 'calibration_sha256': 'current'}
    report = {'quality_passed': True, 'source': 'foundation_stereo_rectified',
              'translation_unit': 'm', 'cameras': {'device_1': {
                  'calibration_sha256': 'current', 'point_frame': 'device_1:C_L_rect',
                  'T_world_camera': np.eye(4).tolist()}}}
    validate_active(report, [device])
    for field, value in [('calibration_sha256', 'old'), ('point_frame', 'raw_left'),
                         ('T_world_camera', np.diag([-1, 1, 1, 1]).tolist())]:
        bad = copy.deepcopy(report)
        bad['cameras']['device_1'][field] = value
        with pytest.raises(ValueError):
            validate_active(bad, [device])


def test_photo_checks_native_dimensions_and_detects_original_tag():
    image = np.full((1080, 1920, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 3, 200)
    image[200:400, 300:500] = marker[:, :, None]
    raw = cv2.imencode('.jpg', image)[1].tobytes()
    _, ids = annotate_photo(raw, (1920, 1080))
    assert 3 in ids
    with pytest.raises(ValueError, match='原生分辨率'):
        annotate_photo(raw, (1280, 720))


def test_fusion_requires_same_origin_csrf_and_valid_current_extrinsics(tmp_path):
    async def check():
        config = load_config(ROOT/'packaging/jetson/devices.json')
        controller = FusionController(config, SimpleNamespace(photo_busy=False),
                                      SimpleNamespace(csrf='token', write_lock=asyncio.Lock()), tmp_path, tmp_path)
        controller.python = Path(__file__)  # Installed check only; command must never run.
        app = web.Application()
        app.router.add_post('/jobs', controller.start)
        async with TestClient(TestServer(app)) as client:
            payload = {'action': 'fuse', 'devices': ['device_1', 'device_2']}
            assert (await client.post('/jobs', json=payload)).status == 403
            headers = {'Origin': str(client.make_url('/')).rstrip('/'),
                       'X-Camera-CSRF': 'token'}
            response = await client.post('/jobs', json=payload, headers=headers)
            assert response.status == 400
            assert '外参' in await response.text()
            payload['action'] = 'arbitrary_command'
            assert (await client.post('/jobs', json=payload, headers=headers)).status == 400
            assert controller.task is None
    asyncio.run(check())


def test_fusion_python_keeps_venv_symlink_and_restores_latest_job(tmp_path, monkeypatch):
    import json
    monkeypatch.chdir(tmp_path)
    base = tmp_path/'.fusion-venv/bin'
    base.mkdir(parents=True)
    (base/'python').symlink_to('/usr/bin/python3')
    root = tmp_path/'data/fusion'
    job_id = 'a'*32
    (root/job_id).mkdir(parents=True)
    (root/job_id/'job.json').write_text(json.dumps({'id':job_id,'state':'completed'}))
    controller = FusionController({}, None, None, tmp_path)
    assert controller.python == base/'python'
    assert controller.job['id'] == job_id


def test_depth_scale_gate_rejects_stable_but_wrong_tag_size(tmp_path):
    import json
    from scripts.jetson.fusion_geometry import check_anchor_depth
    captures = []
    cameras = {}
    for ident in ('device_1', 'device_2'):
        folder = tmp_path/ident
        (folder/'depth').mkdir(parents=True)
        metadata = {'device': {'serial_number': ident}, 'depth_scale_m_per_unit': .001,
                    'color': {'intrinsic': {'fx': 100, 'fy': 100, 'cx': 50, 'cy': 50}}}
        (folder/'camera.json').write_text(json.dumps(metadata))
        cv2.imwrite(str(folder/'depth/000000.png'), np.full((100, 100), 1000, np.uint16))
        pose = np.eye(4)
        pose[2, 3] = -1
        cameras[ident] = {'T_world_camera': pose.tolist()}
        captures.append(folder)
    report = {'source': 'foundation_stereo_rectified', 'tag_side_mm': 80, 'cameras': cameras}
    assert check_anchor_depth(captures, report)['passed']
    # The previously observed ~9cm discrepancy must no longer be labelled successful.
    for folder in captures:
        cv2.imwrite(str(folder/'depth/000000.png'), np.full((100, 100), 910, np.uint16))
    assert not check_anchor_depth(captures, report)['passed']
    for folder in captures:
        cv2.imwrite(str(folder/'depth/000000.png'), np.full((100, 100), 1000, np.uint16))

    # A 25% larger marker size scales PnP translation without changing reprojection error.
    for camera in cameras.values():
        camera['T_world_camera'][2][3] = -1.25
    bad = check_anchor_depth(captures, report)
    assert not bad['passed']
    assert bad['cameras']['device_1']['depth_to_tag_ratio'] == pytest.approx(.8)
    # A wrong planar tilt can pass scale/depth checks; require an independent normal check.
    camera_world = np.eye(4)
    camera_world[:3, :3] = cv2.Rodrigues(np.array([np.pi/3, 0., 0.]))[0]
    camera_world[2, 3] = 1
    for camera in cameras.values():
        camera['T_world_camera'] = np.linalg.inv(camera_world).tolist()
    wrong_axes = check_anchor_depth(captures, report)
    assert not wrong_axes['passed']
    assert wrong_axes['cameras']['device_1']['plane_normal_error_deg'] == pytest.approx(60)



def test_ambiguous_planar_tilt_is_rejected_despite_low_reprojection_error():
    k = np.array([[967.6576358629056, 0, 640], [0, 967.6576358629056, 360], [0, 0, 1.]])
    corners = np.array([[494.6789856, 573.3920898], [554.1768799, 551.9315796],
                        [581.1654053, 590.4730835], [519.6485596, 608.59198]])
    with pytest.raises(ValueError, match='ambiguous'):
        pose_from_tag(corners, k, .08)


def test_photo_rejects_cross_origin_and_non_string_device_without_ssh():
    from scripts.jetson.photos import PhotoController
    async def check():
        config = load_config(ROOT/'packaging/jetson/devices.json')
        photos = PhotoController(config, None, SimpleNamespace(),
                                 SimpleNamespace(csrf='token'), SimpleNamespace(task=None))
        app = web.Application()
        app.router.add_post('/photos', photos.capture)
        async with TestClient(TestServer(app)) as client:
            body = {'device': 'device_1', 'role': 'left', 'resolution': 'native1080'}
            assert (await client.post('/photos', json=body)).status == 403
            headers = {'Origin': str(client.make_url('/')).rstrip('/'),
                       'X-Camera-CSRF': 'token'}
            for key, value in [('device', []), ('role', 'ir'), ('resolution', '4000x3000')]:
                invalid = dict(body, **{key: value})
                assert (await client.post('/photos', json=invalid, headers=headers)).status == 400
            assert photos.process is None
    asyncio.run(check())


def test_result_rejects_withdrawn_calibration_hash(tmp_path):
    import json
    async def check():
        config = load_config(ROOT/'packaging/jetson/devices.json')
        controller = FusionController(config, None, None, tmp_path, tmp_path)
        ident = 'b'*32
        folder = tmp_path/ident
        (folder/'result').mkdir(parents=True)
        result = {'points': [[0, 0, 1]], 'colors': [[255, 255, 255]], 'point_count': 1,
                  'point_unit': 'm', 'world_frame': 'tag', 'measurement_accuracy_verified': False}
        (folder/'result/fusion_result.json').write_text(json.dumps(result))
        cameras = {d['id']: {'calibration_sha256': d['calibration_sha256'],
                            'point_frame': d['id']+':C_L_rect',
                            'T_world_camera': np.eye(4).tolist()} for d in config['devices'][:2]}
        extrinsics = {'cameras': cameras, 'quality_passed': True,
                      'source': 'foundation_stereo_rectified', 'translation_unit': 'm'}
        path = folder/'extrinsics.json'
        path.write_text(json.dumps(extrinsics))
        app = web.Application()
        app.router.add_get('/result/{job}', controller.result)
        async with TestClient(TestServer(app)) as client:
            first = await client.get('/result/'+ident)
            assert first.status == 200
            assert first.headers['X-Fusion-Preview-Cache'] == 'miss'
            second = await client.get('/result/'+ident)
            assert second.headers['X-Fusion-Preview-Cache'] == 'hit'
            assert await first.json() == await second.json()
            cameras['device_2']['calibration_sha256'] = '0'*64
            path.write_text(json.dumps(extrinsics))
            response = await client.get('/result/'+ident)
            assert response.status == 409 and '失效' in await response.text()
    asyncio.run(check())


def test_status_advertises_only_existing_fusion_artifacts(tmp_path):
    import json

    async def check():
        controller = FusionController({}, None, None, tmp_path, tmp_path)
        controller.job = {'id': 'a' * 32, 'action': 'fuse', 'state': 'failed'}
        folder = tmp_path / controller.job['id']
        folder.mkdir()
        (folder / 'job.json').write_text(json.dumps(controller.job))
        app = web.Application()
        app.router.add_get('/status', controller.status)
        app.router.add_get('/files/{job}/{name}', controller.artifact)
        async with TestClient(TestServer(app)) as client:
            status = await (await client.get('/status')).json()
            assert status['job']['available_artifacts'] == ['task.json']
            response = await client.get('/files/' + controller.job['id'] + '/task.json')
            assert response.status == 200
            assert (await response.json())['state'] == 'failed'
            assert (await client.get('/files/' + controller.job['id'] + '/mesh.ply')).status == 404
    asyncio.run(check())


def test_fusion_capture_failure_reports_device_and_completed_progress(tmp_path):
    async def check():
        config = load_config(ROOT / 'packaging/jetson/devices.json')
        devices = config['devices'][:2]
        calls = []

        async def compute(ident, *args):
            calls.append(ident)
            if len(calls) == 2:
                raise web.HTTPBadGateway()
            return dict(device_id=ident, valid_fraction=.9, median_depth_m=1., inference_ms=1.,
                        receive_skew_ms=0., received_at_unix_ms=1, point_frame=ident + ':C_L_rect')

        stereo = SimpleNamespace(_compute=compute, batch_active=False)
        controller = FusionController(config, stereo, None, tmp_path, tmp_path)
        active = {'quality_passed': True, 'source': 'foundation_stereo_rectified',
                  'translation_unit': 'm', 'cameras': {d['id']: {
                      'calibration_sha256': d['calibration_sha256'],
                      'point_frame': d['id'] + ':C_L_rect',
                      'T_world_camera': np.eye(4).tolist()} for d in devices}}
        controller.active = lambda: active
        controller.job = {'id': 'b' * 32, 'state': 'running'}
        params = dict(devices=[d['id'] for d in devices], frames=1, resolution='480x270',
                      quality='fast')
        await controller.run('fuse', params)
        assert controller.job['state'] == 'failed'
        assert controller.job['error_code'] == 'capture_unavailable'
        assert controller.job['current_device'] == 'device_2'
        assert 'device_2' in controller.job['message'] and '检查设备连接' in controller.job['message']
        assert controller.job['progress'] == {'done': 1, 'total': 2}
        assert not stereo.batch_active
        assert controller.active() == active  # Failure never replaces working extrinsics.
    asyncio.run(check())
