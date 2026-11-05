"""Read-only LAN preview; snapshots are independent and do not imply synchronization."""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, web

from .core import load_config
from .fusion_controller import FusionController
from .camera_settings import CameraSettings
from .photos import PhotoController
from .stereo_preview import QUALITY_PRESETS, RESOLUTIONS, StereoHub

MAX_IMAGE_BYTES = 4 * 1024 * 1024
ASSETS = Path(__file__).with_name('web')
ROLES = {'left': 'RGB 左目', 'right': 'RGB 右目', 'ir': '红外'}


class SnapshotHub:
    """One bounded upstream request per channel, shared across viewing clients."""

    def __init__(self, config: dict):
        self.urls = {(d['id'], role): f"http://{d['host']}:{port}/snapshot"
                     for d in config['devices'] for role, port in d['raw_ports'].items()}
        self.locks = {key: asyncio.Lock() for key in self.urls}
        self.cache: dict = {}
        self.slots = asyncio.Semaphore(3)
        self.session = None

    async def _download(self, key):
        async with self.slots:
            async with self.session.get(self.urls[key], allow_redirects=False) as response:
                if response.status != 200:
                    raise ValueError('Camera unavailable')
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    body.extend(chunk)
                    if len(body) > MAX_IMAGE_BYTES:
                        raise ValueError('Camera image exceeds limit')
                # V4L2 JPEG buffers may include zero alignment bytes after EOI.
                end = body.rfind(b'\xff\xd9')
                padding = body[end + 2:] if end >= 0 else body
                if (not body.startswith(b'\xff\xd8') or end < 2
                        or len(padding) > 64 or any(padding)):
                    raise ValueError('Camera did not return JPEG')
                return bytes(body[:end + 2])

    async def snapshot(self, key):
        if key not in self.urls:
            raise web.HTTPNotFound()
        async with self.locks[key]:
            cached = self.cache.get(key)
            if cached and time.monotonic() < cached[0]:
                body = cached[1]
            else:
                try:
                    # Includes semaphore wait, so requests cannot build an unbounded backlog.
                    body = await asyncio.wait_for(self._download(key), timeout=4)
                except (ClientError, asyncio.TimeoutError, ValueError):
                    body = None
                self.cache[key] = (time.monotonic() + (0.4 if body else 1), body)
            if body is None:
                raise web.HTTPBadGateway(text='画面暂不可用，请稍后重试')
            return body


def make_preview_app(config: dict, bundle_root: Path | None = None) -> web.Application:
    @web.middleware
    async def headers(request, handler):
        response = await handler(request)
        response.headers.update({
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Content-Security-Policy': "default-src 'self'; img-src 'self' blob:; "
                                      "object-src 'none'; frame-ancestors 'none'",
            'Referrer-Policy': 'no-referrer'})
        return response

    app = web.Application(client_max_size=4096, middlewares=[headers])
    hub = SnapshotHub(config)
    app[web.AppKey('hub', SnapshotHub)] = hub
    stereo = StereoHub(config, hub, bundle_root)
    settings = CameraSettings(config, hub, stereo, bundle_root)
    fusion = FusionController(config, stereo, settings, bundle_root)
    photos = PhotoController(config, hub, stereo, settings, fusion)

    async def lifecycle(app):
        async with ClientSession(timeout=ClientTimeout(total=3), trust_env=False) as session:
            hub.session = session
            yield
            await fusion.close()
            await stereo.close()
    app.cleanup_ctx.append(lifecycle)

    async def devices(request):
        return web.json_response({'mode': 'independent_raw_snapshots', 'devices': [
            {'id': d['id'], 'name': '设备 ' + d['id'].split('_')[-1], 'host': d['host'],
             'channels': [{'role': role, 'label': label,
                           'url': f"/api/snapshot/{d['id']}/{role}"}
                          for role, label in ROLES.items()]}
            for d in config['devices']]})

    async def snapshot(request):
        body = await hub.snapshot((request.match_info['device'], request.match_info['role']))
        return web.Response(body=body, content_type='image/jpeg')

    async def stereo_result(request):
        name = request.query.get('resolution', '480x270')
        if name not in RESOLUTIONS:
            raise web.HTTPBadRequest(text='Unsupported depth resolution')
        quality = request.query.get('quality', 'fast')
        if quality not in QUALITY_PRESETS:
            raise web.HTTPBadRequest(text='Unsupported model quality')
        result = await stereo.get(request.match_info['device'], RESOLUTIONS[name], quality)
        return web.json_response(result if result is not None else
                                 {"state": "processing", "model": "FoundationStereo"},
                                 status=200 if result is not None else 202)

    async def asset(request):
        name = request.match_info.get('asset', 'index.html')
        if name not in ('index.html', 'app.js', 'styles.css', 'cloud.js', 'settings.js', 'fusion.js', 'photos.js'):
            raise web.HTTPNotFound()
        return web.FileResponse(ASSETS / name)

    app.router.add_get('/', asset)
    app.router.add_get('/api/devices', devices)
    app.router.add_get('/api/status', settings.status)
    app.router.add_post('/api/photos', photos.capture)
    app.router.add_get('/api/photos/{photo}/original.jpg', photos.original)
    app.router.add_get('/api/fusion/status', fusion.status)
    app.router.add_post('/api/fusion/jobs', fusion.start)
    app.router.add_get('/api/fusion/result/{job}', fusion.result)
    app.router.add_get('/api/fusion/files/{job}/{name}', fusion.artifact)
    app.router.add_get('/api/cameras/{device}/{role}', settings.controls)
    app.router.add_post('/api/cameras/{device}/{role}', settings.apply)
    app.router.add_get('/api/stereo/{device}', stereo_result)
    app.router.add_get('/api/snapshot/{device}/{role}', snapshot)
    app.router.add_get('/{asset}', asset)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('packaging/jetson/devices.json'))
    parser.add_argument('--bundle-root', type=Path)
    parser.add_argument('--listen', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8092)
    args = parser.parse_args()
    web.run_app(make_preview_app(load_config(args.config), args.bundle_root), host=args.listen, port=args.port)


if __name__ == '__main__':
    main()
