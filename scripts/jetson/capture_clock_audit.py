"""Read-only snapshot audit for native RGB timestamps and unchanged IR delivery.

No clock qualification, camera controls, service restarts, or identity claims.
Run on Orin to validate the same network path used by tracking.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


def native_metadata(headers):
    h = {k.lower(): v for k, v in headers.items()}
    if h.get('x-capture-protocol') != 'v4l2-monotonic-v1':
        raise ValueError('native_protocol_missing')
    stamp = int(h['x-capture-monotonic-ns'])
    epoch = int(h['x-capture-stream-ns'])
    sequence = int(h['x-capture-sequence'])
    flags = int(h['x-capture-v4l2-flags'])
    lag = (int(h['x-capture-send-monotonic-ns']) - stamp) / 1e9
    if not 0 < epoch <= stamp or not 0 <= sequence < 2**32:
        raise ValueError('invalid_native_counter')
    if h['x-capture-clock'] != 'monotonic' or flags & 0xe000 != 0x2000 or flags & 0x40:
        raise ValueError('invalid_native_flags')
    if not 0 <= lag <= .5:
        raise ValueError('stale_or_future_native_frame')
    return {'stamp_ns': stamp, 'stream_ns': epoch, 'sequence': sequence,
            'flags': flags, 'boot_id': h['x-capture-boot-id'],
            'build': h['x-capture-build'], 'stage': h['x-capture-stage'],
            'send_lag_ms': round(lag * 1000, 3)}


def audit_channel(device, channel, duration, interval, expect_native):
    import cv2
    import numpy as np

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://{device['host']}:{device['raw_ports'][channel]}/snapshot"
    result = {'device_id': device['id'], 'channel': channel, 'samples': 0,
              'errors': [], 'native_frames': 0, 'distinct_frames': 0,
              'first_native': None, 'last_native': None,
              'native_resets': 0, 'max_rtt_ms': 0}
    until = time.monotonic() + duration
    previous = None
    while time.monotonic() < until:
        started = time.monotonic()
        try:
            with opener.open(url, timeout=2) as response:
                headers = dict(response.headers)
                data = response.read(4 * 1024 * 1024 + 1)
            rtt_ms = (time.monotonic() - started) * 1000
            if len(data) > 4 * 1024 * 1024:
                raise ValueError('oversize_snapshot')
            if headers.get('X-UStreamer-Online', '').lower() not in ('true', '1'):
                raise ValueError('source_offline')
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('jpeg_decode_failed')
            size = list(image.shape[1::-1])
            expected = device.get('resolution', {}).get(channel)
            if expected and size != expected:
                raise ValueError(f'image_size_changed:{size}')
            result.update(image_size=size, image_mean=round(float(image.mean()), 3),
                          image_std=round(float(image.std()), 3))
            result['max_rtt_ms'] = round(max(result['max_rtt_ms'], rtt_ms), 3)
            if 'X-Capture-Protocol' in headers or expect_native and channel != 'ir':
                native = native_metadata(headers)
                result['native_frames'] += 1
                if result['first_native'] is None:
                    result['first_native'] = native
                if previous:
                    epoch = (native['boot_id'], native['stream_ns'])
                    old_epoch = (previous['boot_id'], previous['stream_ns'])
                    if epoch != old_epoch:
                        result['native_resets'] += 1
                    elif (native['sequence'] < previous['sequence']
                          or native['stamp_ns'] < previous['stamp_ns']):
                        raise ValueError('native_counter_regressed')
                    elif ((native['sequence'] == previous['sequence'])
                          != (native['stamp_ns'] == previous['stamp_ns'])):
                        raise ValueError('native_counter_timestamp_mismatch')
                if not previous or native['stamp_ns'] != previous['stamp_ns']:
                    result['distinct_frames'] += 1
                result['last_native'] = previous = native
            result['samples'] += 1
        except Exception as exc:
            # Keep the report bounded even if the network stays down.
            if len(result['errors']) < 20:
                result['errors'].append({'after_s': round(duration - until + started, 3),
                                         'error': str(exc)})
            result['error_count'] = result.get('error_count', 0) + 1
        time.sleep(max(.01, interval - (time.monotonic() - started)))
    result['passed'] = bool(result['samples'] >= 3 and not result.get('error_count')
                            and (not expect_native or channel == 'ir'
                                 or result['distinct_frames'] >= 3))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=15)
    parser.add_argument('--interval', type=float, default=.2)
    parser.add_argument('--channels', choices=['rgb', 'ir', 'all'], default='all')
    parser.add_argument('--expect-native', action='store_true')
    args = parser.parse_args()
    if not .1 <= args.duration <= 600 or not .05 <= args.interval <= 10:
        parser.error('duration/interval out of bounds')
    devices = json.loads(args.config.read_text())['devices']
    channels = {'rgb': ['left', 'right'], 'ir': ['ir'], 'all': ['left', 'right', 'ir']}[
        args.channels]
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
        jobs = [pool.submit(audit_channel, d, c, args.duration, args.interval,
                            args.expect_native) for d in devices for c in channels]
        reports = [job.result() for job in jobs]
    report = {'schema_version': 1, 'passed': all(r['passed'] for r in reports),
              'channels': reports, 'physical_exposure_clock_verified': False,
              'note': 'Native driver timestamp delivery only; no exposure or person-ID acceptance.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
