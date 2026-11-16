"""Read-only preview routing, coalescing and bounded upstream handling."""
import asyncio
import hashlib
import re
from pathlib import Path

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

from scripts.jetson.core import load_config
from scripts.jetson.preview import MAX_IMAGE_BYTES, SnapshotHub, make_preview_app

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / 'packaging/jetson/devices.json')


def test_preview_assets_and_read_only_routes():
    async def check():
        async with TestClient(TestServer(make_preview_app(config()))) as client:
            response = await client.get('/')
            assert response.status == 200
            html = await response.text()
            assert '统一预览' in html
            digest = hashlib.sha256((ROOT / 'scripts/jetson/web/human.js').read_bytes()
                                    ).hexdigest()[:16]
            assert f'/human.js?v={digest}' in html
            assert response.headers['Cache-Control'] == 'no-store'
            revision = response.headers['X-MultiCamera-UI']
            assert f'name="multicamera-ui-revision" content="{revision}"' in html
            for url in re.findall(r'(?:src|href)="(/[^\"]+\?v=[a-f0-9]+)"', html):
                asset = await client.get(url)
                assert asset.status == 200
                assert asset.headers['Cache-Control'] == 'no-store'
            assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
            assert (await client.get('/app.js')).status == 200
            assert (await client.get('/styles.css')).status == 200
            assert (await client.get('/human.js')).status == 200
            assert (await client.get('/human.css')).status == 200
            response = await client.get('/api/devices')
            devices = (await response.json())['devices']
            assert len(devices) == 3 and all(len(d['channels']) == 3 for d in devices)
            assert (await client.get('/api/snapshot/unknown/left')).status == 404
            assert (await client.get('/api/snapshot/device_1/arbitrary')).status == 404
            assert (await client.get('/secrets')).status == 404
            assert (await client.post('/api/devices')).status == 405
    asyncio.run(check())


def test_coalesces_clients_and_negative_caches_failure():
    async def check():
        hub = SnapshotHub(config())
        calls = []

        async def download(key):
            calls.append(key)
            await asyncio.sleep(0)
            return b'\xff\xd8jpeg\xff\xd9'

        hub._download = download
        key = ('device_1', 'left')
        results = await asyncio.gather(*(hub.snapshot(key) for _ in range(8)))
        assert len(calls) == 1 and len(set(results)) == 1

        async def failed(key):
            calls.append(key)
            raise ValueError('bad image')

        hub._download = failed
        key = ('device_2', 'right')
        for _ in range(2):
            with pytest.raises(web.HTTPBadGateway):
                await hub.snapshot(key)
        assert len(calls) == 2
    asyncio.run(check())


@pytest.mark.parametrize('mode', ['valid', 'padded', 'oversize', 'redirect', 'invalid', 'source_offline'])
def test_upstream_validation(mode):
    async def check():
        requests = []

        async def source(request):
            requests.append(request.path)
            if mode == 'redirect':
                raise web.HTTPFound('/unexpected')
            body = {'valid': b'\xff\xd8jpeg\xff\xd9', 'oversize': b'x' * (MAX_IMAGE_BYTES + 1),
                    'padded': b'\xff\xd8jpeg\xff\xd9' + b'\x00' * 7,
                    'invalid': b'<html>not JPEG</html>',
                    'source_offline': b'\xff\xd8placeholder\xff\xd9'}.get(mode)
            return web.Response(body=body, headers={
                'X-UStreamer-Online': 'false' if mode == 'source_offline' else 'true'})

        app = web.Application()
        app.router.add_get('/snapshot', source)
        async with TestServer(app) as server, ClientSession() as session:
            hub = SnapshotHub(config())
            key = ('device_1', 'left')
            hub.urls[key] = str(server.make_url('/snapshot'))
            hub.session = session
            if mode in ('valid', 'padded'):
                assert await hub.snapshot(key) == b'\xff\xd8jpeg\xff\xd9'
            else:
                with pytest.raises(web.HTTPBadGateway):
                    await hub.snapshot(key)
            assert requests == ['/snapshot']
    asyncio.run(check())


def test_html_ids_and_script_entries_are_unique():
    from html.parser import HTMLParser
    class Entries(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids, self.scripts = [], []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs:
                self.ids.append(attrs['id'])
            if tag == 'script':
                self.scripts.append(attrs['src'])
    parser = Entries()
    parser.feed((ROOT/'scripts/jetson/web/index.html').read_text())
    assert len(parser.ids) == len(set(parser.ids))
    assert len(parser.scripts) == len(set(parser.scripts))
