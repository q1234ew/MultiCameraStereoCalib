"""Build a dedicated tracking floor calibration from freshly observed static AprilTags.

Run in the existing .fusion-venv after the cameras are fixed and the measured
marker is laid flat on the floor. Does not modify point-cloud extrinsics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .core import load_config
from .fusion_workflow import calibrate, write_json
from .tracking_site import load_site


def build_site(report, devices, bundle_root, *, configured_devices=None):
    if not report.get('quality_passed') or len(report.get('cameras', {})) != len(devices):
        raise ValueError('All cameras must pass fresh common-marker calibration')
    floor_from_tag = np.diag([1., -1., -1., 1.])
    data = {'schema_version': 1, 'floor_verified': True, 'translation_unit': 'm',
            'world_frame': f"floor:AprilTag36h11:{report['anchor_tag_id']}:top_left_m",
            'created_at_unix_ms': round(time.time() * 1000),
            'measurement_accuracy_verified': False,
            'floor_confirmation': 'Operator confirmed a floor marker and requested calibration; pose stability checked',
            'device_config_sha256': hashlib.sha256(
                json.dumps(configured_devices if configured_devices is not None else devices,
                           sort_keys=True).encode()).hexdigest(), 'cameras': {}}
    if configured_devices is not None:
        if (len(devices) < 2 or len({d['id'] for d in devices}) != len(devices)
                or any(d not in configured_devices for d in devices)):
            raise ValueError('Invalid participating camera subset')
        data['participating_camera_ids'] = [d['id'] for d in devices]
    for device in devices:
        path = Path(bundle_root) / device['id'] / 'calibration_mm.yml'
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != device['calibration_sha256']:
            raise ValueError('Calibration bundle hash mismatch')
        info = report['cameras'][device['id']]
        if not info.get('quality_passed'):
            raise ValueError('Camera pose quality did not pass')
        fs = cv2.FileStorage(raw.decode(), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        try:
            intr = fs.getNode('camera_intrinsics').getNode('left')
            rect = fs.getNode('stereo_rectify')
            k, d = intr.getNode('K').mat(), intr.getNode('D').mat()
            r, p = rect.getNode('R1').mat(), rect.getNode('P1').mat()[:3, :3]
            floor_camera = floor_from_tag @ np.asarray(info['T_world_camera'], dtype=float)
            height = float(floor_camera[2, 3])
            if not .15 <= height <= 6:
                raise ValueError('Camera height above floor is implausible')
            camera_floor = np.linalg.inv(floor_camera)
            pixel_floor = p @ camera_floor[:3, [0, 1, 3]]
            data['cameras'][device['id']] = {
                'calibration_sha256': device['calibration_sha256'],
                'image_size': device['resolution']['left'], 'K': k.tolist(), 'D': d.tolist(),
                'R': r.tolist(), 'K_rect': p.tolist(), 'model': intr.getNode('model').string(),
                'ground_from_rectified_pixel': np.linalg.inv(pixel_floor).tolist(),
                'camera_from_ground': camera_floor.tolist(), 'camera_height_m': height,
                'statistics': info['statistics'],
            }
        finally:
            fs.release()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('packaging/jetson/devices.json'))
    parser.add_argument('--bundle-root', type=Path, default=Path('bundles'))
    parser.add_argument('--output', type=Path, default=Path('data/tracking/site.json'))
    parser.add_argument('--tag-id', type=int, default=-1)
    parser.add_argument('--tag-side-mm', type=float, default=80)
    parser.add_argument('--devices', nargs='+', help='Participating camera IDs; others stay local')
    parser.add_argument('--floor-confirmed', action='store_true')
    args = parser.parse_args()
    if not args.floor_confirmed:
        parser.error('Confirm cameras are fixed and the measured marker is flat on the floor')
    configured = load_config(args.config)['devices']
    devices = configured
    if args.devices:
        devices = [d for d in configured if d['id'] in args.devices]
        if len(devices) != len(args.devices) or len(devices) < 2:
            parser.error('Select at least two distinct configured cameras')
    candidate = args.output.parent / f'candidate_{time.time_ns()}'
    report = calibrate(args.config, args.bundle_root, candidate,
                       [d['id'] for d in devices], side_mm=args.tag_side_mm,
                       tag_id=args.tag_id, samples=5)
    data = build_site(report, devices, args.bundle_root, configured_devices=configured)
    path = candidate / 'site.json'
    write_json(path, data)
    validated, error = load_site(path, configured)
    if not validated:
        raise ValueError(error)
    # Activate atomically only after all checks; retain the fresh candidate and
    # previous calibration for review and rollback.
    if args.output.exists():
        (candidate / 'previous_site.json').write_bytes(args.output.read_bytes())
    write_json(args.output, data)
    print(json.dumps({'site': str(args.output), 'world_frame': data['world_frame'],
                      'clock_verification_still_required': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
