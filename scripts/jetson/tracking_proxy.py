"""Bounded gateway with same-origin controls to the isolated localhost pose process."""
from __future__ import annotations

import asyncio
import json
import secrets

from aiohttp import ClientError, ClientTimeout, web

from .tracking_preview import TrackingPreview


async def read_bounded(response, limit):
    body = bytearray()
    async for chunk in response.content.iter_chunked(65536):
        body.extend(chunk)
        if len(body) > limit:
            raise ValueError('Tracking response too large')
    return bytes(body)


class TrackingProxy:
    def __init__(self, hub, devices, base_url='http://127.0.0.1:8093', settings=None):
        self.hub = hub
        self.ids = {device['id'] for device in devices}
        self.base_url = base_url
        self.settings = settings
        self.preview = TrackingPreview(hub, devices)

    async def state(self, request):
        try:
            async with self.hub.session.get(self.base_url + '/state',
                                           timeout=ClientTimeout(total=2)) as response:
                if response.status != 200:
                    raise ValueError('Tracking service unavailable')
                body = await read_bounded(response, 1024 * 1024)
                if len(body) > 1024 * 1024:
                    raise ValueError('Tracking status too large')
                data = json.loads(body)
                if data.get('schema_version') != 1:
                    raise ValueError('Unsupported tracking schema')
                if self.settings:
                    data['control_csrf'] = self.settings.csrf
                data['preview'] = self.preview.status()
                return web.json_response(data)
        except (ClientError, asyncio.TimeoutError, ValueError, TypeError, AttributeError):
            return web.json_response({'schema_version': 1, 'state': 'unavailable',
                                      'error': '人体追踪服务未连接', 'cameras': [], 'tracks': [],
                                      'cross_camera': {'state': 'blocked',
                                                       'reason': '请先检查人体追踪服务'}})

    async def frame(self, request):
        device = request.match_info['device']
        if device not in self.ids:
            raise web.HTTPNotFound()
        try:
            async with self.hub.session.get(self.base_url + '/frame/' + device,
                                           timeout=ClientTimeout(total=2)) as response:
                if response.status != 200:
                    raise ValueError('No fresh tracked image')
                body = await read_bounded(response, 4 * 1024 * 1024)
                if len(body) > 4 * 1024 * 1024 or not body.startswith(b'\xff\xd8'):
                    raise ValueError('Invalid tracking image')
                return web.Response(body=body, content_type='image/jpeg')
        except (ClientError, asyncio.TimeoutError, ValueError):
            raise web.HTTPServiceUnavailable(text='暂无新鲜的人体追踪画面') from None


    async def control(self, request):
        if (not self.settings or request.headers.get('Origin') != f'{request.scheme}://{request.host}'
                or not secrets.compare_digest(request.headers.get('X-Camera-CSRF', ''),
                                               self.settings.csrf)
                or request.content_type != 'application/json'):
            raise web.HTTPForbidden(text='请从统一界面调整追踪设置')
        try:
            data = await request.json()
            if not isinstance(data, dict) or set(data) != {'patch', 'revision'}:
                raise ValueError('Invalid control payload')
            async with self.hub.session.post(self.base_url + '/control', json=data,
                                            timeout=ClientTimeout(total=3)) as response:
                body = await read_bounded(response, 4096)
                status = response.status if response.status in (200, 400, 409, 503) else 503
                return web.json_response(json.loads(body), status=status)
        except (ValueError, TypeError):
            raise web.HTTPBadRequest(text='追踪设置格式无效') from None
        except (ClientError, asyncio.TimeoutError):
            raise web.HTTPServiceUnavailable(text='人体追踪服务未连接，设置未提交') from None

    async def replay(self, request):
        try:
            async with self.hub.session.get(self.base_url + '/replay-report',
                                           timeout=ClientTimeout(total=2)) as response:
                if response.status != 200:
                    raise ValueError('Replay report unavailable')
                data = json.loads(await read_bounded(response, 1024 * 1024))
                return web.json_response(data)
        except (ClientError, asyncio.TimeoutError, ValueError):
            return web.json_response({'available': False, 'note': '暂无可用的回放评测报告'})
