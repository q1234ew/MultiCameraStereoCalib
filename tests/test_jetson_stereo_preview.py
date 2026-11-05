"""Metric geometry and model-backed preview contracts, no CUDA needed on test host."""
import asyncio
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from scripts.jetson.core import load_config
from scripts.jetson.preview import SnapshotHub, make_preview_app
from scripts.jetson.stereo_preview import StereoHub, StereoProcessor, depth_from_disparity

ROOT = Path(__file__).resolve().parents[1]


def test_disparity_to_metric_depth_with_principal_offset():
    p1 = np.array([[200., 0, 100, 0], [0, 200, 80, 0], [0, 0, 1, 0]])
    p2 = p1.copy()
    p2[0, 2] = 98
    p2[0, 3] = -10000  # 200 px * -50 mm
    depth = depth_from_disparity(np.array([[12., 2., -3., np.inf, np.nan]]), p1, p2)
    assert depth[0, 0] == pytest.approx(1.)
    assert np.isnan(depth[0, 1:]).all()
    p2[0, 3] = 10000
    assert depth_from_disparity(np.array([[-8.]]), p1, p2)[0, 0] == pytest.approx(1.)


def calibration(tmp_path):
    config = load_config(ROOT/'packaging/jetson/devices.json')
    device = config['devices'][0]
    path = tmp_path/'calibration.yml'
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    for k, v in {'device_id':device['id'], 'translation_unit':'mm', 'coord_frame':'C_L',
                 'stereo_transform_direction':'left_to_right',
                 'ir_transform_direction':'ir_to_left'}.items():
        fs.write(k, v)
    k = np.array([[500.,0,640],[0,500,360],[0,0,1]])
    fs.startWriteStruct('camera_intrinsics', cv2.FileNode_MAP)
    for role in ('left', 'right', 'ir'):
        fs.startWriteStruct(role, cv2.FileNode_MAP)
        fs.write('width',device['resolution'][role][0])
        fs.write('height',device['resolution'][role][1])
        fs.write('model','pinhole')
        fs.write('K',k)
        fs.write('D',np.zeros((1,5)))
        fs.endWriteStruct()
    fs.endWriteStruct()
    fs.startWriteStruct('stereo_rectify',cv2.FileNode_MAP)
    for index in ('1','2'):
        fs.write('R'+index,np.eye(3))
        projection = np.column_stack((k,np.zeros(3)))
        if index == '2':
            projection[0,3] = -25000
        fs.write('P'+index,projection)
    fs.endWriteStruct()
    fs.release()
    device['calibration_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return device,path


@pytest.mark.parametrize('quality',['fast','standard','fine'])
@pytest.mark.parametrize('resolution',[(320,180),(480,270),(640,360),(960,540)])
def test_model_disparity_uses_scaled_metric_geometry_and_bounds(tmp_path,resolution,quality):
    class Predictor:
        def predict(self, left, right, iterations=8):
            assert iterations == {'fast':8,'standard':16,'fine':32}[quality]
            assert left.shape == right.shape == (resolution[1],resolution[0],3)
            disparity = np.full(left.shape[:2],8*resolution[0]/480,np.float32)
            disparity[:,200:210] = np.nan
            return disparity, {'identity':{'model':'FoundationStereo','iterations':iterations},'inference_ms':123}
    device,path = calibration(tmp_path)
    processor = StereoProcessor(device,path,Predictor())
    image = np.full((720,1280,3),128,np.uint8)
    ok,encoded = cv2.imencode('.jpg',image)
    assert ok
    result = processor.process(encoded.tobytes(),encoded.tobytes(),1.,resolution,quality)
    assert result['algorithm'] == 'FoundationStereo CUDA'
    assert result['median_depth_m'] == pytest.approx(1.172,abs=.001)
    assert result['point_unit'] == 'm' and result['point_frame'] == 'device_1:C_L_rect'
    assert 0 < result['valid_fraction'] < 1
    points = np.asarray(result['points'])
    assert len(points) <= 6000 and len(points) == len(result['colors'])
    assert np.isfinite(points).all()
    assert np.allclose(points[:,2],1.171875,atol=.0001)
    assert not result['hardware_synchronized'] and not result['physical_accuracy_verified']
    wrong = cv2.imencode('.jpg',image[:300])[1].tobytes()
    with pytest.raises(ValueError,match='resolution'):
        processor.process(wrong,wrong,1.)


def test_async_jobs_coalesce_without_network_or_unbounded_queue():
    async def check():
        config = load_config(ROOT/'packaging/jetson/devices.json')
        hub = StereoHub(config,SnapshotHub(config),None)
        called = []
        event = asyncio.Event()
        async def compute(ident,resolution,quality):
            called.append((ident,resolution,quality))
            await event.wait()
        hub._compute = compute
        for _ in range(10):
            assert await hub.get('device_1') is None
        await asyncio.sleep(0)
        assert called == [('device_1',(480,270),'fast')]
        for _ in range(10):
            assert await hub.get('device_1',(640,360),'fine') is None
        await asyncio.sleep(0)
        assert len(called)==2 and len(hub.tasks)==2
        event.set()
        await hub.close()
    asyncio.run(check())


def test_loading_endpoint_returns_202_without_claiming_depth(monkeypatch):
    async def fake(self, ident, resolution=(480,270), quality='fast'):
        return None
    monkeypatch.setattr(StereoHub,'get',fake)
    async def check():
        config = load_config(ROOT/'packaging/jetson/devices.json')
        async with TestClient(TestServer(make_preview_app(config))) as client:
            response = await client.get('/api/stereo/device_1')
            assert response.status == 202
            assert await response.json() == {'state':'processing','model':'FoundationStereo'}
            assert (await client.get('/cloud.js')).status == 200
            assert (await client.get('/api/stereo/device_1?resolution=9999x9999')).status == 400
            assert (await client.get('/api/stereo/device_1?quality=unsupported')).status == 400
    asyncio.run(check())


def test_rgbd_export_keeps_metric_depth_and_registered_intrinsics(tmp_path):
    import json

    class Predictor:
        def predict(self, left, right, iterations=8):
            disparity = np.full(left.shape[:2], 8, np.float32)
            return disparity, {'identity': {'model': 'FoundationStereo'}, 'inference_ms': 1}

    device, path = calibration(tmp_path)
    processor = StereoProcessor(device, path, Predictor())
    raw = cv2.imencode('.jpg', np.full((720, 1280, 3), 128, np.uint8))[1].tobytes()
    folder = tmp_path/'capture'
    processor.process(raw, raw, 0, (480, 270), 'fast', folder, 0)
    meta = json.loads((folder/'camera.json').read_text())
    depth = cv2.imread(str(folder/'depth/000000.png'), cv2.IMREAD_UNCHANGED)
    color = cv2.imread(str(folder/'color/000000.png'))
    assert depth.dtype == np.uint16 and depth.shape == color.shape[:2] == (270, 480)
    assert np.median(depth[depth > 0]) == 1172  # nearest mm of 1.171875m
    assert meta['depth_scale_m_per_unit'] == .001
    assert meta['color']['intrinsic']['fx'] == 500*480/1280
    assert meta['color']['intrinsic']['cx'] == (640+.5)*480/1280-.5
    assert meta['point_frame'] == 'device_1:C_L_rect'
    with pytest.raises(ValueError, match='overwrite'):
        processor.process(raw, raw, 0, (480, 270), 'fast', folder, 0)
