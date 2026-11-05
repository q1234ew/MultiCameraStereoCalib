"""Explicit native still capture, isolated from streaming calibration geometry."""
from __future__ import annotations

import asyncio
import base64
import secrets
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

from .fusion_geometry import detect_tags


def annotate_photo(raw, expected):
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None or tuple(image.shape[1::-1]) != expected:
        raise ValueError('相机未返回申请的原生分辨率')
    tags = detect_tags(image)
    if tags:
        cv2.aruco.drawDetectedMarkers(image, [v.astype(np.float32)[None] for v in tags.values()],
                                     np.array(list(tags), np.int32).reshape(-1, 1))
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError('截图编码失败')
    return base64.b64encode(encoded).decode(), sorted(tags)


class PhotoController:
    def __init__(self, config, snapshots, stereo, settings, fusion):
        self.devices = {d['id']: d for d in config['devices']}
        self.snapshots, self.stereo, self.settings, self.fusion = snapshots, stereo, settings, fusion
        self.cache = {}
        self.process = None
        self.busy = False

    async def capture(self, request):
        if (request.headers.get('Origin') != f'{request.scheme}://{request.host}'
                or not secrets.compare_digest(request.headers.get('X-Camera-CSRF', ''),
                                              self.settings.csrf)
                or request.content_type != 'application/json'):
            raise web.HTTPForbidden(text='请从截图设置提交')
        try:
            data = await request.json()
        except (ValueError, TypeError) as exc:
            raise web.HTTPBadRequest(text='无效截图设置') from exc
        if (not isinstance(data, dict) or set(data) != {'device', 'role', 'resolution'}
                or not all(isinstance(data[k], str) for k in data)
                or data['device'] not in self.devices or data['role'] not in ('left', 'right')
                or data['resolution'] not in ('preview', 'native1080')):
            raise web.HTTPBadRequest(text='无效截图设置')
        if self.busy or (self.fusion.task and not self.fusion.task.done()):
            raise web.HTTPConflict(text='正在拍照或执行融合任务，请完成后重试')
        self.busy = True
        self.stereo.photo_busy = True
        ident, role = data['device'], data['role']
        device = self.devices[ident]
        started = time.monotonic()
        try:
            async with self.settings.write_lock:
                if data['resolution'] == 'native1080':
                    key = Path.home()/'.ssh/id_camera_photo'
                    known = Path.home()/'.ssh/camera_photo_known_hosts'
                    if not key.exists() or not known.exists():
                        raise ValueError('未部署相机原生拍照连接')
                    self.process = await asyncio.create_subprocess_exec(
                        'ssh', '-i', str(key), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                        '-o', 'StrictHostKeyChecking=yes', '-o', f'UserKnownHostsFile={known}',
                        '-o', 'ClearAllForwardings=yes', 'root@'+device['host'],
                        f"photo {device['raw_ports'][role]}",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                    try:
                        # Remote helper is bounded and restores streams before returning bytes.
                        raw, _ = await asyncio.wait_for(self.process.communicate(), 90)
                        if self.process.returncode != 0 or len(raw) > 4*1024*1024:
                            raise ValueError('原生拍照失败，请检查相机拍照日志与视频恢复状态')
                    finally:
                        if self.process.returncode is None:
                            self.process.terminate()
                            await self.process.wait()
                        self.process = None
                    expected = (1920, 1080)
                else:
                    self.snapshots.cache.pop((ident, role), None)
                    raw = await self.snapshots.snapshot((ident, role))
                    expected = tuple(device['resolution'][role])
                annotated, tags = await asyncio.to_thread(annotate_photo, raw, expected)
                photo_id = uuid.uuid4().hex
                self.cache[photo_id] = raw
                while len(self.cache) > 3:
                    del self.cache[next(iter(self.cache))]
                return web.json_response({'id': photo_id, 'size': list(expected), 'tag_ids': tags,
                    'annotated_jpeg': annotated, 'elapsed_s': round(time.monotonic()-started, 1),
                    'original_url': '/api/photos/'+photo_id+'/original.jpg',
                    'native': data['resolution'] == 'native1080',
                    'used_for_pose': False})
        except (ValueError, OSError, asyncio.TimeoutError) as exc:
            raise web.HTTPBadGateway(text=str(exc)) from exc
        finally:
            self.busy = False
            self.stereo.photo_busy = False
            for r in ('left', 'right', 'ir'):
                self.snapshots.cache.pop((ident, r), None)
            self.stereo.invalidate(ident)
            self.settings.status_cache = None

    async def original(self, request):
        raw = self.cache.get(request.match_info['photo'])
        if raw is None:
            raise web.HTTPNotFound(text='截图已过期，请重新拍摄')
        return web.Response(body=raw, content_type='image/jpeg', headers={
            'Content-Disposition': 'attachment; filename="camera-photo.jpg"'})
