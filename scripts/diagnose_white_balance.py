"""Record raw JPEG/control evidence; optional bounded left-only WB transition."""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import cv2
import numpy as np

from multicamera.streaming.white_balance import RgbControlClient, camera_base


def image_metrics(frame):
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(float)
    mask = (small.min(axis=2) > 12) & (small.max(axis=2) < 245)
    if mask.sum() < 160:
        raise ValueError('Too few non-clipped pixels')
    b, g, r = small[mask].mean(axis=0)
    return {'r_over_g': float(r / g), 'b_over_g': float(b / g),
            'mean_bgr': [float(b), float(g), float(r)], 'valid_pixels': int(mask.sum())}


def summarize(rows):
    summary = {}
    for role in ('left', 'right'):
        groups = {}
        for phase in ('baseline', 'manual', 'restored'):
            selected = [r for r in rows if r['role'] == role and r['phase'] == phase]
            if selected:
                values = np.array([[r['metrics']['r_over_g'], r['metrics']['b_over_g']]
                                   for r in selected])
                groups[phase] = {'median_rg_bg': np.median(values, axis=0).tolist(),
                                 'range_rg_bg': np.ptp(values, axis=0).tolist()}
        if 'baseline' in groups:
            baseline = np.array(groups['baseline']['median_rg_bg'])
            for phase in ('manual', 'restored'):
                if phase in groups:
                    delta = np.array(groups[phase]['median_rg_bg']) - baseline
                    groups[phase]['change_from_baseline'] = delta.tolist()
                    groups[phase]['max_abs_change'] = float(np.max(np.abs(delta)))
        summary[role] = groups
    return summary


def experiment(left_url, right_url, directory, *, change_left=False, client=None, sleep=time.sleep):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    client = client or RgbControlClient()
    bases = {'left': camera_base(left_url), 'right': camera_base(right_url)}
    if bases['left'] == bases['right']:
        raise ValueError('Left and right endpoints must be different')
    report = {'endpoints': bases, 'control_change_authorized': change_left,
              'initial': {}, 'samples': [], 'writes': [], 'restoration': {},
              'limits': 'No device restart, no gray reference, sequential HTTP snapshots; '
                        'image ratios measure temporal changes, not color accuracy.'}
    touched = False

    def save():
        report['summary'] = summarize(report['samples'])
        (directory / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                               encoding='utf-8')

    def sample(phase, index):
        for role, base in bases.items():
            started = time.time()
            with client.http.open(base + 'snapshot', timeout=3) as response:
                data = response.read(client.MAX_SNAPSHOT_BYTES + 1)
            if len(data) > client.MAX_SNAPSHOT_BYTES:
                raise ValueError('Snapshot too large')
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Invalid JPEG')
            name = f'{phase}_{index:02d}_{role}.jpg'
            (directory / name).write_bytes(data)
            row = {'phase': phase, 'index': index, 'role': role, 'file': name,
                   'time_utc_epoch': started, 'request_seconds': time.time() - started,
                   'sha256': hashlib.sha256(data).hexdigest(), 'image_size': list(frame.shape[1::-1]),
                   'metrics': image_metrics(frame), 'config': client.config(base)}
            report['samples'].append(row)
            print(json.dumps({k: row[k] for k in ('phase', 'role', 'index', 'metrics')},
                             ensure_ascii=False), flush=True)
        save()

    def write_mode(value, stage):
        nonlocal touched
        touched = True  # A lost response can still mean the device accepted the write.
        payload = {'white_balance_automatic': value}
        report['writes'].append({'stage': stage, 'role': 'left', 'payload': payload})
        return client.config(bases['left'], payload)

    try:
        report['initial'] = {role: client.config(base) for role, base in bases.items()}
        save()
        if change_left and report['initial']['left']['controls']['white_balance_automatic']['value'] != 1:
            raise ValueError('Left AWB is not automatic; refusing a transition that may lose manual gains')
        for i in range(3):
            if i:
                sleep(1)
            sample('baseline', i)
        if change_left:
            write_mode(0, 'manual')
            for i, delay in enumerate((.5, 1.5)):
                sleep(delay)
                sample('manual', i)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if touched:
            try:
                # Recovery must never depend on report/image filesystem writes.
                result = write_mode(1, 'restore')
                report['restoration']['left'] = result
                report['restoration']['verified'] = (
                    client.config(bases['left'])['controls']['white_balance_automatic']['value'] == 1
                )
            except Exception as exc:
                report['restoration']['error'] = str(exc)
            save()
        else:
            save()
    if change_left and not report['restoration'].get('verified'):
        raise RuntimeError('Could not verify left AWB restoration; inspect camera immediately')
    if change_left:
        for i, delay in enumerate((1., 2., 4.)):
            sleep(delay)
            sample('restored', i)
    report['final'] = {role: client.config(base) for role, base in bases.items()}
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-url', required=True)
    parser.add_argument('--right-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--change-left-awb', action='store_true',
                        help='Explicitly allow one left-only auto/manual/auto experiment')
    args = parser.parse_args()
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('Interrupted; restoring left AWB')
    signal.signal(signal.SIGTERM, interrupted)
    report = experiment(args.left_url, args.right_url, args.output,
                        change_left=args.change_left_awb)
    print(json.dumps({'summary': report['summary'], 'restoration': report['restoration'],
                      'output': str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
