"""Raw preview remains independent, shared, rate bounded, and fails closed."""
import asyncio
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from scripts.jetson.tracking_preview import TrackingPreview


def test_preview_coalesces_clients_and_counts_unique_images(monkeypatch):
    async def check():
        calls = []
        now = [10.0]
        monkeypatch.setattr('scripts.jetson.tracking_preview.time.monotonic', lambda: now[0])
        monkeypatch.setattr('scripts.jetson.tracking_preview.encode_preview', lambda body: body)

        async def download(key):
            calls.append(key)
            await asyncio.sleep(0)
            return b'\xff\xd8frame\xff\xd9'

        preview = TrackingPreview(SimpleNamespace(_download=download), [{'id': 'device_1'}])
        app = web.Application()
        app.router.add_get('/preview/{device}', preview.frame)
        async with TestClient(TestServer(app)) as client:
            responses = await asyncio.gather(*(client.get('/preview/device_1') for _ in range(6)))
            assert all(response.status == 200 for response in responses)
            assert len(calls) == 1
            assert len({r.headers['X-Preview-Frame'] for r in responses}) == 1
            assert responses[0].headers['Cache-Control'] == 'no-store'
            assert (await client.get('/preview/unconfigured')).status == 404
            now[0] += .041
            assert (await client.get('/preview/device_1')).status == 200
            assert len(calls) == 2
            # Two network fetches of the same JPEG do not establish two fresh images.
            assert len(preview.samples['device_1']) == 1
    asyncio.run(check())


def test_preview_failure_replaces_cached_image_and_negative_caches():
    async def check():
        calls = []

        async def failed(key):
            calls.append(key)
            raise ValueError('offline')

        preview = TrackingPreview(SimpleNamespace(_download=failed), [{'id': 'device_1'}])
        preview.cache['device_1'] = (0, b'old', {})
        app = web.Application()
        app.router.add_get('/preview/{device}', preview.frame)
        async with TestClient(TestServer(app)) as client:
            for _ in range(2):
                response = await client.get('/preview/device_1')
                assert response.status == 503
                assert b'old' not in await response.read()
            assert len(calls) == 1 and preview.cache['device_1'][1] is None
    asyncio.run(check())


def test_smooth_preview_resizes_only_preview_and_reduces_transfer_size():
    import cv2
    import numpy as np

    from scripts.jetson.tracking_preview import encode_preview

    image = np.random.default_rng(3).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    original = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    encoded = encode_preview(original)
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (540, 960)
    assert len(encoded) < len(original) / 2
    assert image.shape[:2] == (720, 1280)


def test_preview_stream_has_bounded_multipart_frames_and_stops_on_upstream_failure(monkeypatch):
    async def check():
        monkeypatch.setattr('scripts.jetson.tracking_preview.encode_preview', lambda body: body)
        calls = []
        body = b'\xff\xd8frame\xff\xd9'

        async def download(key):
            calls.append(key)
            if len(calls) > 1:
                raise ValueError('camera disconnected')
            return body

        preview = TrackingPreview(SimpleNamespace(_download=download), [{'id': 'device_1'}])
        app = web.Application()
        app.router.add_get('/stream/{device}', preview.stream)
        async with TestClient(TestServer(app)) as client:
            response = await client.get('/stream/device_1')
            assert response.status == 200
            assert response.content_type == 'multipart/x-mixed-replace'
            part = await response.read()
            assert part.count(body) == 1
            assert b'Content-Length: 9\r\n' in part
            assert b'X-Preview-Frame:' in part
            assert len(calls) == 2
    asyncio.run(check())
