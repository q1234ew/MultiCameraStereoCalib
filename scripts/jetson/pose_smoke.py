"""Verify the isolated YOLO26n-pose CUDA environment without starting a service."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path


WEIGHT_SHA256 = 'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--require-person', action='store_true')
    args = parser.parse_args()
    if hashlib.sha256(args.weights.read_bytes()).hexdigest() != WEIGHT_SHA256:
        raise ValueError('Unexpected YOLO26n-pose checkpoint hash')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is not a successful GPU deployment')
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError('Cannot decode validation image')
    torch.set_num_threads(2)
    model = YOLO(str(args.weights), task='pose')
    options = dict(source=image, device=0, imgsz=640, quantize=16, verbose=False,
                   save=False, nms=False)
    with torch.inference_mode():
        for _ in range(2):
            model.predict(**options)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for _ in range(5):
            started = time.monotonic()
            result = model.predict(**options)[0]
            torch.cuda.synchronize()
            elapsed.append((time.monotonic() - started) * 1000)
    running_parameter = next(model.predictor.model.backend.model.parameters())
    if model.predictor.device.type != 'cuda' or running_parameter.device.type != 'cuda':
        raise RuntimeError('Inference backend did not stay on CUDA')
    if running_parameter.dtype != torch.float16:
        raise RuntimeError('Inference backend did not use FP16')
    keypoints = result.keypoints.data.detach().float().cpu().numpy()
    if keypoints.ndim != 3 or tuple(keypoints.shape[1:]) != (17, 3):
        raise ValueError('Unexpected COCO-17 output shape')
    if not np.isfinite(keypoints).all():
        raise ValueError('Non-finite pose output')
    count = len(result.boxes)
    if args.require_person and count == 0:
        raise ValueError('Positive-control image produced no persons')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    annotation = args.report.with_suffix('.jpg')
    if not cv2.imwrite(str(annotation), result.plot()):
        raise OSError('Cannot save validation preview')
    report = {
        'passed': True, 'hostname': platform.node(), 'machine': platform.machine(),
        'python': platform.python_version(), 'ultralytics': ultralytics.__version__,
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'opencv': cv2.__version__,
        'weight_sha256': WEIGHT_SHA256, 'input_sha256': hashlib.sha256(
            args.image.read_bytes()).hexdigest(), 'input_size': image.shape[1::-1],
        'person_count': count, 'keypoint_shape': list(keypoints.shape),
        'device': str(running_parameter.device), 'fp16': running_parameter.dtype == torch.float16,
        'warm_runs': 5, 'predict_wall_ms': elapsed,
        'predict_wall_median_ms': statistics.median(elapsed),
        'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
        'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'measurement_note': 'Warm single-image prediction; not a live or multi-camera benchmark. '
                            'CUDA and RSS accounting overlap on Jetson; do not sum them.',
        'created_at_unix_ms': round(time.time() * 1000),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
