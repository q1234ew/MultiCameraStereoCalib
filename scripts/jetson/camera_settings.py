"""Fixed-device status and explicit same-origin camera controls; no network reconfiguration."""
from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import resource
import secrets
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import ClientError, web

RGB_KEYS = {'brightness', 'contrast', 'saturation', 'gain', 'white_balance_automatic',
            'white_balance_temperature', 'sharpness', 'auto_exposure',
            'exposure_time_absolute', 'power_line_frequency', 'exposure_dynamic_framerate'}


def validate_rgb(payload, controls):
    if not isinstance(payload, dict) or not payload or not set(payload) <= RGB_KEYS:
        raise ValueError('请选择受支持的相机参数')
    for key, value in payload.items():
        meta = controls.get(key)
        if (not meta or meta.get('inactive') or meta.get('flags', 0) & 23
                or type(value) is not int):
            raise ValueError(f'{key} 当前不可写')
        if (not meta['min'] <= value <= meta['max']
                or (value-meta['min']) % max(meta.get('step', 1), 1)):
            raise ValueError(f'{key} 超出相机允许范围或步长')
        if meta.get('kind') == 'menu' and value not in {m['value'] for m in meta['menu']}:
            raise ValueError(f'{key} 不是有效选项')


def validate_ir(payload):
    if not isinstance(payload, dict) or not payload or not set(payload) <= {'colormap','overlay','scale'}:
        raise ValueError('仅支持红外色板、标注与显示倍率')
    if 'colormap' in payload and payload['colormap'] not in ('inferno','turbo','jet','viridis'):
        raise ValueError('无效红外色板')
    if 'overlay' in payload and type(payload['overlay']) is not bool:
        raise ValueError('无效标注开关')
    if 'scale' in payload and (type(payload['scale']) is not int or payload['scale'] not in (1,2,3,4)):
        raise ValueError('红外倍率范围为 1–4')


class CameraSettings:
    def __init__(self, config, snapshots, stereo, bundle_root):
        self.config, self.snapshots, self.stereo = config, snapshots, stereo
        self.bundle_root = bundle_root
        self.devices = {d['id']: d for d in config['devices']}
        self.csrf = secrets.token_urlsafe(32)
        self.started = time.monotonic()
        self.write_lock = asyncio.Lock()
        self.status_lock = asyncio.Lock()
        self.status_cache = None

    async def request_json(self, url, payload=None):
        async with self.snapshots.slots:
            method = 'GET' if payload is None else 'POST'
            async with self.snapshots.session.request(
                    method, url, json=payload, allow_redirects=False) as response:
                body = bytearray()
                async for chunk in response.content.iter_chunked(8192):
                    body.extend(chunk)
                    if len(body) > 65536:
                        raise ValueError('设备响应超出限制')
                try:
                    data = json.loads(body)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError('设备接口返回格式异常') from exc
                if response.status != 200 or data.get('ok') is False:
                    raise ValueError(data.get('error', f'HTTP {response.status}'))
                return data

    def camera_url(self, ident, role):
        if ident not in self.devices or role not in ('left','right','ir'):
            raise web.HTTPNotFound()
        device = self.devices[ident]
        path = 'ir' if role == 'ir' else 'rgb'
        return f"http://{device['host']}:{device['raw_ports'][role]}/api/{path}/config"

    async def controls(self, request):
        url = self.camera_url(request.match_info['device'],request.match_info['role'])
        try:
            return web.json_response(await self.request_json(url))
        except (ClientError, OSError, asyncio.TimeoutError, ValueError) as exc:
            raise web.HTTPBadGateway(text=f'相机参数接口不可用：{exc}') from exc

    async def apply(self, request):
        # JSON + same-origin + non-simple custom header blocks cross-site form/CSRF writes.
        if (request.headers.get('Origin') != f'{request.scheme}://{request.host}'
                or not secrets.compare_digest(request.headers.get('X-Camera-CSRF',''),self.csrf)
                or request.content_type != 'application/json'):
            raise web.HTTPForbidden(text='请从本机预览设置页提交')
        if self.stereo.batch_active:
            raise web.HTTPConflict(text='标定或融合正在采集，请完成后再调整相机参数')
        ident, role = request.match_info['device'], request.match_info['role']
        url = self.camera_url(ident,role)
        try:
            payload = await request.json()
            async with self.write_lock:
                if role == 'ir':
                    validate_ir(payload)
                else:
                    current = await self.request_json(url)
                    validate_rgb(payload,current.get('controls',{}))
                await self.request_json(url,payload)
                # Return actual read-back; partial failures are never labelled successful.
                current = await self.request_json(url)
                values = (current if role == 'ir' else
                          {k:v['value'] for k,v in current.get('controls',{}).items()})
                if any(values.get(k) != v for k,v in payload.items()):
                    raise ValueError('设备回读与申请值不同，请刷新参数确认')
                self.snapshots.cache.pop((ident,role),None)
                self.stereo.invalidate(ident)
                self.status_cache = None
            return web.json_response({'ok':True,'readback':current,'applied':payload})
        except (ValueError, TypeError, KeyError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        except (ClientError, OSError, asyncio.TimeoutError) as exc:
            raise web.HTTPBadGateway(text=f'应用或回读失败，请刷新确认当前值：{exc}') from exc

    def calibration_info(self, device):
        path = (Path(self.bundle_root)/device['id']/'calibration_mm.yml'
                if self.bundle_root else Path(device['calibration_path']))
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            fs = cv2.FileStorage(raw.decode(),cv2.FILE_STORAGE_READ|cv2.FILE_STORAGE_MEMORY)
            try:
                p = fs.getNode('stereo_rectify').getNode('P2').mat()
                baseline = None if p is None else round(abs(float(p[0,3]/p[0,0])),3)
                models = {r:fs.getNode('camera_intrinsics').getNode(r).getNode('model').string()
                          for r in ('left','right','ir')}
            finally:
                fs.release()
            return {'loaded':digest==device['calibration_sha256'],'sha256':digest,
                    'baseline_mm':baseline,'models':models,'translation_unit':'mm'}
        except (OSError,ValueError,cv2.error):
            return {'loaded':False,'error':'本地标定包缺失或格式异常'}

    async def device_status(self, device):
        ident = device['id']
        async def channel(role):
            try:
                body = await self.snapshots.snapshot((ident,role))
                def decode_size():
                    image = cv2.imdecode(np.frombuffer(body,np.uint8),cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError('图像解码失败')
                    return list(image.shape[1::-1])
                size = await asyncio.to_thread(decode_size)
                return {'role':role,'online':True,'size':size,
                        'calibrated_size':device['resolution'][role],
                        'size_matches':size==device['resolution'][role]}
            except (web.HTTPException,ClientError,OSError,ValueError,asyncio.TimeoutError):
                return {'role':role,'online':False,'calibrated_size':device['resolution'][role]}
        channels = await asyncio.gather(*(channel(r) for r in ('left','right','ir')))
        calibration = await asyncio.to_thread(self.calibration_info,device)
        return {'id':ident,'host':device['host'],'expected_mac':device['expected_mac'],
                'ports':device['raw_ports'],'source_session':device['source_session'],
                'physical_binding_verified':device['physical_binding_verified'],
                'binding_note':device['binding_note'],'calibration':calibration,'channels':channels}

    async def status(self, request):
        async with self.status_lock:
            if self.status_cache and time.monotonic()-self.status_cache[0] < 2:
                return web.json_response(self.status_cache[1])
            result = await self.status_data()
            self.status_cache = (time.monotonic(),result)
            return web.json_response(result)

    async def status_data(self):
        predictor = self.stereo.predictor
        process = predictor.process
        state = ('ready' if predictor.identity and process and process.poll() is None else
                 'loading' if process and process.poll() is None else 'not_loaded')
        gpu = {'state':state,'identity':predictor.identity,
               'jobs':[{'device_id':k[0],'resolution':list(k[1]),'quality':k[2]}
                       for k,v in self.stereo.tasks.items() if not v.done()]}
        return {'csrf':self.csrf,'updated_at_unix_ms':round(time.time()*1000),
            'orin':{'hostname':platform.node(),'machine':platform.machine(),
                    'python':platform.python_version(),'opencv':cv2.__version__,
                    'uptime_s':round(time.monotonic()-self.started),
                    'preview_max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
            'model':gpu,'devices':await asyncio.gather(
                *(self.device_status(d) for d in self.config['devices']))}
