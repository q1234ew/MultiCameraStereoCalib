"""Shared, bounded raw preview independent of pose inference and its pause state."""
from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque

from aiohttp import ClientError, web


def encode_preview(body):
    """Bounded CPU preview; original camera pixels and model input stay unchanged."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode preview')
    height, width = image.shape[:2]
    scale = min(1, 960 / width)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 65])
    if not ok:
        raise ValueError('Cannot encode preview')
    return encoded.tobytes()


class TrackingPreview:
    def __init__(self, hub, devices):
        import cv2

        cv2.setNumThreads(1)
        self.hub = hub
        self.locks = {device['id']: asyncio.Lock() for device in devices}
        self.cache = {}
        self.samples = {key: deque(maxlen=250) for key in self.locks}
        self.last_hash = {}

    async def frame(self, request):
        device = request.match_info['device']
        if device not in self.locks:
            raise web.HTTPNotFound()
        try:
            await asyncio.wait_for(self.locks[device].acquire(), timeout=1)
        except asyncio.TimeoutError:
            raise web.HTTPServiceUnavailable(text='预览繁忙') from None
        try:
            now = time.monotonic()
            cached = self.cache.get(device)
            if not cached or cached[0] <= now:
                try:
                    started = time.monotonic()
                    body = await asyncio.wait_for(
                        self.hub._download((device, 'left')), timeout=1.5)
                    digest = hashlib.sha256(body).hexdigest()
                    fetched = time.monotonic()
                    body = await asyncio.to_thread(encode_preview, body)
                    ended = time.monotonic()
                    headers = {'X-Preview-Frame': digest,
                               'X-Preview-Receive-Unix-Ms': str(round(time.time() * 1000)),
                               'X-Preview-Fetch-Ms': str(round((fetched - started) * 1000, 2)),
                               'X-Preview-Encode-Ms': str(round((ended - fetched) * 1000, 2))}
                    if digest != self.last_hash.get(device):
                        self.samples[device].append(ended)
                        self.last_hash[device] = digest
                    # Rate limit from request START, avoiding a download-time FPS penalty.
                    cached = (started + 1 / 25, body, headers)
                except (ClientError, asyncio.TimeoutError, ValueError):
                    cached = (time.monotonic() + 1, None, {})
                self.cache[device] = cached
            if cached[1] is None:
                raise web.HTTPServiceUnavailable(text='暂无新鲜相机原图')
            return web.Response(body=cached[1], content_type='image/jpeg',
                                headers={**cached[2], 'Cache-Control': 'no-store'})
        finally:
            self.locks[device].release()

    async def stream(self, request):
        """One HTTP connection, bounded writes, no replay queue for a slow viewer."""
        first = await self.frame(request)
        response = web.StreamResponse(headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=mcframe',
            'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no',
        })
        await response.prepare(request)
        snapshot = first
        try:
            while True:
                started = time.monotonic()
                if snapshot is None:
                    snapshot = await self.frame(request)
                body = snapshot.body
                metadata = ''.join(f'{key}: {value}\r\n'
                                   for key, value in snapshot.headers.items()
                                   if key.startswith('X-Preview-'))
                header = (f'--mcframe\r\nContent-Type: image/jpeg\r\n'
                          f'Content-Length: {len(body)}\r\n{metadata}\r\n').encode()
                await asyncio.wait_for(response.write(header + body + b'\r\n'), timeout=1)
                snapshot = None
                await asyncio.sleep(max(0, .04 - (time.monotonic() - started)))
        except (ConnectionError, asyncio.TimeoutError, web.HTTPException):
            # Close instead of ever replaying stale content after an upstream failure.
            return response

    def status(self):
        now = time.monotonic()
        output = {}
        for device, samples in self.samples.items():
            recent = [stamp for stamp in samples if now - stamp <= 10]
            output[device] = {
                'unique_fps': round((len(recent) - 1) / (recent[-1] - recent[0]), 2)
                if len(recent) >= 2 and recent[-1] > recent[0] else None,
                'age_s': round(now - samples[-1], 3) if samples else None,
                'note': 'JPEG内容更新率；接收时间不是曝光时间',
            }
        return output
