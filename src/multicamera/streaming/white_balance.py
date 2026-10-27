"""Measured AWB warm-up and freeze without replaying stale temperature values."""

from dataclasses import dataclass
import json
import logging
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, build_opener, ProxyHandler

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WhiteBalanceLockError(RuntimeError):
    """The automatic-to-manual transition changed the measured image color."""


def camera_base(url):
    parts = urlsplit(url)
    path = parts.path.rstrip('/')
    for suffix in ('/stream', '/snapshot', '/api/rgb/config'):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip('/') + '/', '', ''))


@dataclass(frozen=True)
class WhiteBalancePolicy:
    warmup_seconds: float = 7.0
    timeout_seconds: float = 20.0
    sample_interval: float = .5
    window: int = 4
    stability_tolerance: float = .025
    lock_tolerance: float = .05
    lock_settle_seconds: float = 1.0


class RgbControlClient:
    MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024

    def __init__(self):
        self.http = build_opener(ProxyHandler({}))

    def config(self, base, payload=None):
        request = Request(base + 'api/rgb/config',
                          data=None if payload is None else json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        with self.http.open(request, timeout=3) as response:
            result = json.loads(response.read(128 * 1024))
        if not result.get('ok'):
            raise RuntimeError(f"{base}: RGB 参数接口失败：{result.get('error')}")
        control = result.get('controls', {}).get('white_balance_automatic')
        if control is None or control.get('inactive'):
            raise RuntimeError(f'{base}: 不支持自动白平衡控制')
        if payload is not None and control.get('value') != payload['white_balance_automatic']:
            raise RuntimeError(f'{base}: 白平衡模式回读与请求不符')
        return result

    def chroma(self, base):
        with self.http.open(base + 'snapshot', timeout=3) as response:
            data = response.read(self.MAX_SNAPSHOT_BYTES + 1)
        if len(data) > self.MAX_SNAPSHOT_BYTES:
            raise RuntimeError('白平衡检测图片过大')
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError('白平衡检测未获得有效 JPEG')
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float64)
        mask = (small.min(axis=2) > 12) & (small.max(axis=2) < 245)
        if mask.sum() < 160:
            raise RuntimeError('画面过暗或过曝，无法检查白平衡稳定性')
        b, g, r = small[mask].mean(axis=0)
        return np.array([r / g, b / g])


def stabilize_and_lock(urls, client=None, policy=None, cancelled=lambda: False,
                       clock=time.monotonic, sleep=time.sleep, progress=lambda stage: None,
                       lock=True):
    client, policy = client or RgbControlClient(), policy or WhiteBalancePolicy()
    targets = list(dict.fromkeys(camera_base(url) for url in urls))
    if not targets:
        return {}
    touched, results = [], {}

    def check_cancelled():
        if cancelled():
            raise RuntimeError('白平衡准备已取消')

    try:
        progress('读取相机参数')
        for base in targets:
            check_cancelled()
            before = client.config(base)
            logger.info('AWB prepare: url=%s before=%s', base, before.get('controls'))
            if before['controls']['white_balance_automatic']['value'] != 1:
                touched.append(base)
                client.config(base, {'white_balance_automatic': 1})
            else:
                logger.info('AWB already automatic; skipping mode write: url=%s', base)
        started = clock()
        histories = {base: [] for base in targets}
        last_stage = None
        while True:
            check_cancelled()
            stage = '白平衡预热' if clock() - started < policy.warmup_seconds else '检查颜色稳定'
            if stage != last_stage:
                progress(stage)
                last_stage = stage
            if clock() - started > policy.timeout_seconds:
                raise RuntimeError('白平衡稳定超时，请保持场景和照明稳定后重试')
            stable = True
            for base in targets:
                check_cancelled()
                histories[base].append(client.chroma(base))
                histories[base] = histories[base][-policy.window:]
                samples = np.array(histories[base])
                stable &= len(samples) == policy.window and np.max(np.ptp(samples, axis=0)) <= policy.stability_tolerance
            if stable and clock() - started >= policy.warmup_seconds:
                break
            sleep(policy.sample_interval)
        references = {base: np.mean(samples, axis=0) for base, samples in histories.items()}
        if not lock:
            progress('复核自动白平衡')
            for base in targets:
                check_cancelled()
                result = client.config(base)
                if result['controls']['white_balance_automatic']['value'] != 1:
                    raise RuntimeError(f'{base}: 自动白平衡已失效，已拒绝采图')
                current = client.chroma(base)
                delta = float(np.max(np.abs(current - references[base])))
                if not np.isfinite(delta) or delta > policy.stability_tolerance:
                    raise RuntimeError(f'{base}: 自动白平衡复核时颜色仍在变化，已拒绝采图')
                result = dict(result, preparation_mode='automatic')
                results[base] = result
                logger.info('AWB automatic stable (not locked): url=%s reference=%s current=%s',
                            base, references[base], current)
            check_cancelled()
            return results
        progress('锁定白平衡')
        for base in targets:
            check_cancelled()
            # The reported temperature can be stale while AWB is active. Never replay it.
            if base not in touched:
                touched.append(base)
            results[base] = client.config(base, {'white_balance_automatic': 0})
        sleep(policy.lock_settle_seconds)
        progress('复核锁定后颜色')
        for base in targets:
            for _ in range(2):
                check_cancelled()
                current = client.chroma(base)
                delta = float(np.max(np.abs(current - references[base])))
                logger.info('AWB lock verification: url=%s reference=%s current=%s delta=%.5f limit=%.5f',
                            base, references[base], current, delta, policy.lock_tolerance)
                if not np.isfinite(delta) or delta > policy.lock_tolerance:
                    raise WhiteBalanceLockError(f'{base}: 锁定白平衡后颜色跳变，已拒绝采图；'
                                                '可选择“自动稳定（不锁定）”后重新采集')
                sleep(policy.sample_interval)
            result = client.config(base)
            if result['controls']['white_balance_automatic']['value'] != 0:
                raise RuntimeError(f'{base}: 白平衡锁定已失效')
            results[base] = dict(result, preparation_mode='locked')
            logger.info('AWB locked: url=%s reference=%s controls=%s', base, references[base], result.get('controls'))
        return results
    except Exception:
        progress('恢复自动白平衡' if touched else '准备失败，未修改白平衡模式')
        for base in touched:
            try:
                client.config(base, {'white_balance_automatic': 1})
            except Exception:
                logger.exception('AWB recovery failed; check camera manually: %s', base)
        raise
