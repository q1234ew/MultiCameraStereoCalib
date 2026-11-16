"""Bounded proxy preserves streamed frames and fails closed on unavailable workers."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

from scripts.jetson.tracking_proxy import TrackingProxy


@pytest.mark.parametrize('kind', ['state', 'frame'])
def test_proxy_reads_complete_chunked_response(kind):
    async def check():
        body = (json.dumps({'schema_version': 1, 'payload': 'x' * 150000}).encode()
                if kind == 'state' else b'\xff\xd8' + b'x' * 150000 + b'\xff\xd9')

        async def source(request):
            response = web.StreamResponse()
            await response.prepare(request)
            for start in range(0, len(body), 2048):
                await response.write(body[start:start + 2048])
                await asyncio.sleep(0)
            await response.write_eof()
            return response

        upstream = web.Application()
        upstream.router.add_get('/{tail:.*}', source)
        async with TestServer(upstream) as source_server, ClientSession() as session:
            proxy = TrackingProxy(SimpleNamespace(session=session), [{'id': 'device_1'}],
                                  str(source_server.make_url('')).rstrip('/'))
            app = web.Application()
            app.router.add_get('/state', proxy.state)
            app.router.add_get('/frame/{device}', proxy.frame)
            async with TestClient(TestServer(app)) as client:
                response = await client.get('/state' if kind == 'state' else '/frame/device_1')
                assert response.status == 200
                received = await response.read()
                if kind == 'state':
                    data = json.loads(received)
                    assert data.pop('preview')['device_1']['unique_fps'] is None
                    assert data == json.loads(body)
                else:
                    assert received == body
                assert (await client.get('/frame/arbitrary')).status == 404
    asyncio.run(check())


def test_control_requires_same_origin_and_csrf_and_preserves_conflict_status():
    async def check():
        calls = []

        async def source(request):
            calls.append(await request.json())
            return web.json_response({'error': 'Settings changed'}, status=409)

        upstream = web.Application()
        upstream.router.add_post('/control', source)
        async with TestServer(upstream) as source_server, ClientSession() as session:
            proxy = TrackingProxy(SimpleNamespace(session=session), [],
                                  str(source_server.make_url('')).rstrip('/'),
                                  settings=SimpleNamespace(csrf='test-token'))
            app = web.Application()
            app.router.add_post('/control', proxy.control)
            async with TestClient(TestServer(app)) as client:
                payload = {'patch': {'enabled': False}, 'revision': 'stale'}
                assert (await client.post('/control', json=payload)).status == 403
                assert (await client.post('/control', json=payload, headers={
                    'Origin': 'https://foreign.example', 'X-Camera-CSRF': 'test-token'})).status == 403
                assert calls == []
                response = await client.post('/control', json=payload, headers={
                    'Origin': str(client.make_url('')).rstrip('/'), 'X-Camera-CSRF': 'test-token'})
                assert response.status == 409 and len(calls) == 1
                assert (await response.json())['error'] == 'Settings changed'
    asyncio.run(check())
