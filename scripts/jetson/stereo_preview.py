"""Bounded CPU stereo preview from real, independently received camera snapshots."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from aiohttp import ClientError, web

from .core import Frame, Rectifier
from .foundation_runner import FoundationRunner

RESOLUTIONS = {'320x180': (320, 180), '480x270': (480, 270), '640x360': (640, 360), '960x540': (960, 540)}
QUALITY_PRESETS = {'fast': 8, 'standard': 16, 'fine': 32}


def depth_from_disparity(disparity, p1, p2):
    """Horizontal rectified geometry: input pixels, projection translations in mm."""
    denominator = disparity - (p1[0, 2] - p2[0, 2])
    depth = np.full(disparity.shape, np.nan, np.float32)
    np.divide(-p2[0, 3], denominator, out=depth, where=np.abs(denominator) > 1e-6)
    depth *= 0.001
    depth[(depth < 0.2) | (depth > 10) | ~np.isfinite(depth)] = np.nan
    return depth


def jpeg(image):
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError('JPEG encode failed')
    return base64.b64encode(data).decode('ascii')


class StereoProcessor:
    def __init__(self, device, path, predictor):
        self.predictor = predictor
        self.device = device
        self.rectifier = Rectifier(device, path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.rectifier.sha256:
            raise ValueError('Calibration changed during initialization')
        fs = cv2.FileStorage(raw.decode('utf-8'),
                             cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        try:
            node = fs.getNode('stereo_rectify')
            self.p1, self.p2 = (node.getNode(k).mat() for k in ('P1', 'P2'))
        finally:
            fs.release()
        if (any(p is None or p.shape != (3, 4) or not np.isfinite(p).all()
                for p in (self.p1, self.p2))
                or abs(self.p2[1, 3]) > 1e-5 or abs(self.p2[0, 3]) < 1
                or not np.allclose(self.p1[:, 3], 0)
                or not np.allclose(self.p1[:2, :2], self.p2[:2, :2])
                or not np.isclose(self.p1[1, 2], self.p2[1, 2])):
            raise ValueError('Horizontal rectified stereo projections required')

    def process(self, left_bytes, right_bytes, receive_skew_ms, resolution=(480, 270), quality='fast',
                capture_dir=None, frame_index=0):
        if resolution not in RESOLUTIONS.values():
            raise ValueError("Unsupported depth resolution")
        if quality not in QUALITY_PRESETS:
            raise ValueError('Unsupported model quality')
        start = time.monotonic()
        images = []
        masks = []
        for role, body in [('left', left_bytes), ('right', right_bytes)]:
            image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Camera JPEG decode failed')
            rect = self.rectifier.rectify(Frame(self.device['id'], role, 0, image))
            # Resize only AFTER full-resolution calibrated remap; scale geometry explicitly.
            images.append(cv2.resize(rect, resolution, interpolation=cv2.INTER_AREA))
            source_mask = np.full(image.shape[:2], 255, np.uint8)
            mask = cv2.remap(source_mask, *self.rectifier.maps[role], cv2.INTER_NEAREST)
            masks.append(cv2.resize(mask, resolution, interpolation=cv2.INTER_NEAREST) == 255)
        left, right = images
        h, w = left.shape[:2]
        p1, p2 = self.p1.copy(), self.p2.copy()
        sx, sy = w/self.device['resolution']['left'][0], h/self.device['resolution']['left'][1]
        for p in (p1, p2):
            p[0] *= sx
            p[1] *= sy
            p[0, 2] += (sx - 1)/2
            p[1, 2] += (sy - 1)/2
        dl, inference = self.predictor.predict(left, right, iterations=QUALITY_PRESETS[quality])
        y, x = np.indices((h, w))
        finite = np.isfinite(dl)
        xr = np.rint(x-np.where(finite, dl, 0)).astype(np.int32)
        inside = (xr >= 0) & (xr < w)
        safe_xr = np.clip(xr, 0, w-1)
        valid = finite & (dl > 0) & inside & masks[0] & masks[1][y, safe_xr]
        depth = depth_from_disparity(dl, p1, p2)
        depth[~valid] = np.nan
        valid = np.isfinite(depth)
        count = int(valid.sum())
        # Fixed metric color scale: near 0.2m red -> far 10m blue, invalid black.
        color = cv2.applyColorMap(np.uint8(np.clip(np.nan_to_num(
            (10-depth)/9.8, nan=0), 0, 1)*255), cv2.COLORMAP_TURBO)
        color[~valid] = 0
        pair = np.hstack((left, right))
        for row in range(30, h, 40):
            cv2.line(pair, (0, row), (w*2-1, row), (80, 220, 120), 1)
        # Regular bounded sampling; no cross-device transformations or invented depth.
        yy, xx = y[::4, ::4], x[::4, ::4]
        z = depth[::4, ::4]
        keep = np.isfinite(z)
        xyz = np.column_stack(((xx[keep]-p1[0, 2])*z[keep]/p1[0, 0],
                               (yy[keep]-p1[1, 2])*z[keep]/p1[1, 1], z[keep]))
        rgb = left[::4, ::4][keep][:, ::-1]
        if len(xyz) > 6000:
            indices = np.linspace(0, len(xyz)-1, 6000).astype(int)
            xyz, rgb = xyz[indices], rgb[indices]
        if capture_dir is not None:
            folder = Path(capture_dir)
            for kind in ('color', 'depth'):
                (folder / kind).mkdir(parents=True, exist_ok=True)
            name = f'{frame_index:06d}.png'
            color_path, depth_path = folder/'color'/name, folder/'depth'/name
            if color_path.exists() or depth_path.exists():
                raise ValueError('Refusing to overwrite an existing RGB-D frame')
            depth_mm = np.rint(np.nan_to_num(depth, nan=0)*1000).astype(np.uint16)
            if not cv2.imwrite(str(color_path), left) or not cv2.imwrite(str(depth_path), depth_mm):
                raise ValueError('RGB-D export failed')
            metadata = {'device': {'serial_number': self.device['id']},
                        'source_type': 'foundation_stereo_rectified',
                        'calibration_sha256': self.rectifier.sha256,
                        'point_frame': self.device['id']+':C_L_rect',
                        'model_identity': inference['identity'],
                        'depth_scale_m_per_unit': .001,
                        'color': {'width': w, 'height': h,
                            'intrinsic': {'fx':float(p1[0,0]), 'fy':float(p1[1,1]),
                                          'cx':float(p1[0,2]), 'cy':float(p1[1,2])},
                            'distortion': {k:0.0 for k in ('k1','k2','k3','k4','k5','k6','p1','p2')}}}
            existing = folder/'camera.json'
            if existing.exists():
                previous = json.loads(existing.read_text())
                if (previous['color'] != metadata['color']
                        or previous['calibration_sha256'] != metadata['calibration_sha256']):
                    raise ValueError('RGB-D session calibration or resolution changed')
            existing.write_text(json.dumps(metadata, indent=2)+'\n')
        depths = depth[valid]
        return {'device_id': self.device['id'], 'calibration_sha256': self.rectifier.sha256,
                'computed_at_unix_ms': round(time.time()*1000),
                'algorithm': 'FoundationStereo CUDA', 'quality': quality, 'model_identity': inference['identity'],
                'inference_ms': inference['inference_ms'],
                'resolution': [w, h], 'receive_skew_ms': round(receive_skew_ms, 2),
                'hardware_synchronized': False, 'physical_accuracy_verified': False,
                'point_frame': self.device['id']+':C_L_rect', 'point_unit': 'm',
                'valid_pixels': count, 'valid_fraction': round(count/(w*h), 4),
                'depth_range_m': [0.2, 10], 'median_depth_m':
                    None if not count else round(float(np.median(depths)), 3),
                'compute_ms': round((time.monotonic()-start)*1000, 1),
                'rectified_jpeg': jpeg(pair), 'depth_jpeg': jpeg(color),
                'points': np.round(xyz, 4).tolist(), 'colors': rgb.tolist(),
                'warnings': ['独立快照，未硬件同步；动态物体可能出现错误深度',
                             '候选标定，物理镜头绑定和距离精度尚未验收']}


class StereoHub:
    """At most 36 validated profiles queued; one inference task, shared across clients."""
    def __init__(self, config, snapshots, bundle_root):
        self.config, self.snapshots = config, snapshots
        self.devices = {d['id']: d for d in config['devices']}
        self.processors = {}
        self.tasks = {}
        self.cache = {}
        self.predictor = FoundationRunner()
        self.failures = {}
        self.revisions = {}
        self.batch_active = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stereo')
        self.slot = asyncio.Semaphore(1)
        self.bundle_root = bundle_root

    async def _compute(self, ident, resolution, quality, capture_dir=None, frame_index=0):
        key = (ident, resolution, quality)
        async with self.slot:
            device = self.devices[ident]
            async def receive(role):
                body = await asyncio.wait_for(self.snapshots._download((ident, role)), 4)
                return body, time.monotonic()
            try:
                await asyncio.get_running_loop().run_in_executor(
                    self.executor, self.predictor.ensure_ready)
                start = time.monotonic()
                revision = self.revisions.get(ident, 0)
                received_at = round(time.time()*1000)
                left, right = await asyncio.gather(receive('left'), receive('right'))
                skew = abs(left[1]-right[1])*1000
                if skew > self.config['max_skew_ms'] or time.monotonic()-start > 0.5:
                    raise ValueError('左右接收时间差或帧龄超限，请重试')
                def work():
                    if ident not in self.processors:
                        path = (Path(self.bundle_root)/ident/'calibration_mm.yml'
                                if self.bundle_root else Path(device['calibration_path']))
                        self.processors[ident] = StereoProcessor(device, path, self.predictor)
                    return self.processors[ident].process(
                        left[0], right[0], skew, resolution, quality, capture_dir, frame_index)
                result = await asyncio.get_running_loop().run_in_executor(self.executor, work)
            except (ValueError, cv2.error, OSError, ClientError, asyncio.TimeoutError) as exc:
                self.cache.pop(key, None)
                self.failures[key] = (time.monotonic(), str(exc))
                raise web.HTTPBadGateway(text=str(exc)) from exc
            if revision != self.revisions.get(ident, 0):
                return None  # Camera parameters changed while this pair was processing.
            self.failures.pop(key, None)
            result['camera_configuration_revision'] = revision
            result['received_at_unix_ms'] = received_at
            self.cache[key] = (time.monotonic(), result)
            return result

    async def get(self, ident, resolution=(480, 270), quality='fast'):
        if resolution not in RESOLUTIONS.values():
            raise web.HTTPBadRequest(text="Unsupported depth resolution")
        if quality not in QUALITY_PRESETS:
            raise web.HTTPBadRequest(text='Unsupported model quality')
        if ident not in self.devices:
            raise web.HTTPNotFound()
        key = (ident, resolution, quality)
        cached = self.cache.get(key)
        age = time.monotonic()-cached[0] if cached else float('inf')
        if cached and age < 10:
            return dict(cached[1], refreshing=False)
        task = self.tasks.get(key)
        if (task is None or task.done()) and not self.batch_active and not getattr(self, 'photo_busy', False):
            failure = self.failures.get(key)
            if failure and time.monotonic()-failure[0] < 5:
                raise web.HTTPBadGateway(text=failure[1])
            task = asyncio.create_task(self._compute(ident, resolution, quality))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            self.tasks[key] = task
        if cached and age < 60:
            return dict(cached[1], refreshing=True)
        return None

    def invalidate(self, ident):
        for mapping in (self.cache, self.failures):
            for key in list(mapping):
                if key[0] == ident:
                    del mapping[key]
        self.revisions[ident] = self.revisions.get(ident, 0) + 1

    async def close(self):
        self.predictor.close()
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.executor.shutdown(wait=True)
