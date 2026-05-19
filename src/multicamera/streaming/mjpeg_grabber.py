"""MJPEG over HTTP stream grabber with async I/O."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import aiohttp
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..perf import perf_timer


class MJPEGGrabber(QThread):
    """Grabs frames from an MJPEG over HTTP stream in a dedicated thread.

    Each camera gets its own QThread running an asyncio event loop to parse
    the multipart/x-mixed-replace stream without blocking the UI.
    """

    frame_ready = Signal(str, np.ndarray, float)  # camera_id, frame, timestamp
    connection_lost = Signal(str, str)  # camera_id, reason
    connection_established = Signal(str)  # camera_id
    fps_updated = Signal(str, float)  # camera_id, fps

    RECONNECT_DELAY = 2.0
    READ_TIMEOUT = 10.0
    FPS_UPDATE_INTERVAL = 1.0
    DEFAULT_MAX_DECODE_FPS = 20.0

    def __init__(
        self,
        camera_id: str,
        url: str,
        max_decode_fps: float = DEFAULT_MAX_DECODE_FPS,
        parent=None,
    ):
        super().__init__(parent)
        self.camera_id = camera_id
        self.url = url
        self.max_decode_fps = max_decode_fps
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._frame_count = 0
        self._fps_time = 0.0
        self._last_decode_time = 0.0

    def run(self):
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._stream_loop())
        finally:
            self._loop.close()

    def stop(self):
        self._running = False
        self.wait(5000)

    async def _stream_loop(self):
        while self._running:
            try:
                await self._connect_and_read()
            except asyncio.CancelledError:
                break
            except Exception as e:
                reason = str(e) or e.__class__.__name__
                self.connection_lost.emit(self.camera_id, reason)
                if self._running:
                    await asyncio.sleep(self.RECONNECT_DELAY)

    async def _connect_and_read(self):
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_read=self.READ_TIMEOUT,
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(self.url) as resp:
                if resp.status != 200:
                    raise ConnectionError(f"HTTP {resp.status}")

                content_type = resp.headers.get("Content-Type", "")

                if "multipart" in content_type:
                    await self._read_multipart(resp)
                else:
                    await self._read_single_jpeg(resp)

    async def _read_multipart(self, resp: aiohttp.ClientResponse):
        self.connection_established.emit(self.camera_id)
        self._fps_time = time.monotonic()
        self._frame_count = 0

        boundary = self._extract_boundary(resp.headers.get("Content-Type", ""))
        buffer = b""

        async for chunk in resp.content.iter_any():
            if not self._running:
                break

            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")
                if start == -1:
                    if boundary:
                        bound_pos = buffer.find(boundary.encode())
                        if bound_pos != -1:
                            buffer = buffer[bound_pos + len(boundary):]
                            continue
                    break

                end = buffer.find(b"\xff\xd9", start + 2)
                if end == -1:
                    break

                jpeg_data = buffer[start : end + 2]
                buffer = buffer[end + 2 :]
                self._decode_and_emit(jpeg_data)

    async def _read_single_jpeg(self, resp: aiohttp.ClientResponse):
        """Fallback for snapshot or raw JPEG streams.

        Some embedded camera services do not set a multipart content type even
        though they keep a long-lived HTTP response open. Reading until EOF would
        block forever, so we parse JPEG SOI/EOI markers from the byte stream just
        like the multipart reader does.
        """
        self.connection_established.emit(self.camera_id)
        self._fps_time = time.monotonic()
        self._frame_count = 0

        buffer = b""
        async for chunk in resp.content.iter_any():
            if not self._running:
                break
            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")
                if start == -1:
                    # Keep a small suffix in case the SOI marker is split across chunks.
                    buffer = buffer[-1:]
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end == -1:
                    if start > 0:
                        buffer = buffer[start:]
                    break

                jpeg_data = buffer[start : end + 2]
                buffer = buffer[end + 2 :]
                self._decode_and_emit(jpeg_data)

    def _decode_and_emit(self, jpeg_data: bytes):
        now = time.monotonic()
        if self.max_decode_fps > 0:
            min_interval = 1.0 / self.max_decode_fps
            if now - self._last_decode_time < min_interval:
                return
            self._last_decode_time = now

        arr = np.frombuffer(jpeg_data, dtype=np.uint8)
        with perf_timer(f"mjpeg decode {self.camera_id}", threshold_ms=20.0):
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            ts = time.time()
            self.frame_ready.emit(self.camera_id, frame, ts)
            self._update_fps()

    def _update_fps(self):
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_time
        if elapsed >= self.FPS_UPDATE_INTERVAL:
            fps = self._frame_count / elapsed
            self.fps_updated.emit(self.camera_id, fps)
            self._frame_count = 0
            self._fps_time = now

    @staticmethod
    def _extract_boundary(content_type: str) -> str:
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                return part[len("boundary=") :]
        return ""
