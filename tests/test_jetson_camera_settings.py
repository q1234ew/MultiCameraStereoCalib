"""Camera write allowlist, CSRF checks and read-back contracts, without hardware writes."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from scripts.jetson.camera_settings import CameraSettings, validate_ir, validate_rgb
from scripts.jetson.core import load_config
from scripts.jetson.preview import make_preview_app
from scripts.jetson import camera_settings

ROOT=Path(__file__).resolve().parents[1]


def test_status_without_resource_telemetry(monkeypatch, tmp_path):
    monkeypatch.setattr(camera_settings, 'resource', None)
    stereo = SimpleNamespace(predictor=SimpleNamespace(process=None, identity=None), tasks={})
    settings = CameraSettings({'devices': []}, None, stereo, tmp_path)
    result = asyncio.run(settings.status_data())
    assert result['orin']['preview_max_rss_kb'] is None
    assert result['model']['state'] == 'not_loaded'



def controls():
    return {'brightness':{'min':-64,'max':64,'step':1,'value':0,'inactive':False,'flags':0}}


@pytest.mark.parametrize('payload',[
    {}, {'arbitrary':1}, {'brightness':100}, {'brightness':True},
    {'brightness':0.5}, [], None,
])
def test_reject_invalid_rgb(payload):
    with pytest.raises(ValueError):
        validate_rgb(payload,controls())


def test_inactive_and_read_only_cannot_be_written():
    meta=controls()
    for flag in (1,2,4,16):
        meta['brightness']['flags']=flag
        with pytest.raises(ValueError):
            validate_rgb({'brightness':1},meta)
    meta['brightness']['flags']=0
    validate_rgb({'brightness':1},meta)


@pytest.mark.parametrize('payload',[{'scale':0},{'scale':True},{'colormap':'invalid'},
                                    {'overlay':1},{'reboot':True}])
def test_ir_controls_are_restricted(payload):
    with pytest.raises(ValueError):
        validate_ir(payload)


def test_camera_post_requires_same_origin_and_readback(monkeypatch):
    writes=[]
    state=controls()
    async def request_json(self,url,payload=None):
        if payload is not None:
            writes.append((url,payload.copy()))
            for key,value in payload.items():
                state[key]['value']=value
        return {'ok':True,'device':'test','controls':state}
    async def status(self):
        return {'csrf':self.csrf}
    monkeypatch.setattr(CameraSettings,'request_json',request_json)
    monkeypatch.setattr(CameraSettings,'status_data',status)
    async def check():
        config=load_config(ROOT/'packaging/jetson/devices.json')
        async with TestClient(TestServer(make_preview_app(config))) as client:
            token=(await (await client.get('/api/status')).json())['csrf']
            url='/api/cameras/device_1/left'
            assert (await client.post(url,json={'brightness':1})).status==403
            headers={'Origin':str(client.make_url('/')).rstrip('/'),'X-Camera-CSRF':token}
            response=await client.post(url,json={'brightness':5},headers=headers)
            assert response.status==200
            result=await response.json()
            assert result['readback']['controls']['brightness']['value']==5
            assert len(writes)==1 and writes[0][0].endswith(':8082/api/rgb/config')
            assert (await client.post(url,json={'brightness':1000},headers=headers)).status==400
            assert len(writes)==1
            bad=dict(headers,Origin='http://untrusted.example')
            assert (await client.post(url,json={'brightness':2},headers=bad)).status==403
            assert (await client.post('/api/cameras/device_1/unknown',json={},headers=headers)).status==404
    asyncio.run(check())


def test_readback_mismatch_never_returns_success(monkeypatch):
    async def request_json(self,url,payload=None):
        return {'ok':True,'controls':controls()}
    async def status(self):
        return {'csrf':self.csrf}
    monkeypatch.setattr(CameraSettings,'request_json',request_json)
    monkeypatch.setattr(CameraSettings,'status_data',status)
    async def check():
        config=load_config(ROOT/'packaging/jetson/devices.json')
        async with TestClient(TestServer(make_preview_app(config))) as client:
            token=(await (await client.get('/api/status')).json())['csrf']
            headers={'Origin':str(client.make_url('/')).rstrip('/'),'X-Camera-CSRF':token}
            response=await client.post('/api/cameras/device_1/left',json={'brightness':4},headers=headers)
            assert response.status==400
            assert '回读' in await response.text()
    asyncio.run(check())
