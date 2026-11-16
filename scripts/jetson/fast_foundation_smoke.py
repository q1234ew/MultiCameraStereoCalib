"""Standalone Fast-FoundationStereo CUDA test for JetPack 5 (Python 3.8+).

Uses an isolated checkout and environment; never changes the preview model.
Inputs must already be rectified. Intrinsics use the upstream two-line format
(3x3 left K, positive baseline in metres), with equal rectified principal points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--backend', choices=('torch', 'trt', 'trt-single'), default='torch')
    parser.add_argument('--engine-dir', type=Path)
    parser.add_argument('--left', type=Path)
    parser.add_argument('--right', type=Path)
    parser.add_argument('--intrinsics', type=Path)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=352)
    parser.add_argument('--iterations', type=int, choices=(4, 8), default=8)
    parser.add_argument('--max-disp', type=int, choices=(192, 256), default=192)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if bool(args.left) != bool(args.right):
        parser.error('--left and --right must be provided together')
    if (not 32 <= args.width <= 1920 or not 32 <= args.height <= 1088
            or args.width % 32 or args.height % 32):
        parser.error('width/height must be multiples of 32 within 32..1920 / 32..1088')
    if not 1 <= args.runs <= 100 or not 1 <= args.warmup <= 10:
        parser.error('runs must be 1..100 and warmup 1..10')
    if args.backend == 'torch' and not args.checkpoint:
        parser.error('--checkpoint is required for torch')
    if args.backend in ('trt', 'trt-single') and not args.engine_dir:
        parser.error('--engine-dir is required for TensorRT')
    args.root = args.root.resolve(strict=True)
    custom = args.left is not None
    args.left = args.left or args.root / 'demo_data/left.png'
    args.right = args.right or args.root / 'demo_data/right.png'
    if not custom and not args.intrinsics:
        args.intrinsics = args.root / 'demo_data/K.txt'
    # JetPack 5 has no supported Triton compiler. Keep the official eager
    # PyTorch correlation implementation, including between the TRT engines.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TIMM_OFFLINE'] = '1'
    sys.path.insert(0, str(args.root))
    import cv2
    import numpy as np
    import psutil
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.set_grad_enabled(False)
    # Upstream decorators use torch.compile. Disable them before import rather
    # than modifying torch or the source checkout globally.
    def eager_compile(function=None, **kwargs):
        return function if function is not None else lambda value: value

    torch.compile = eager_compile
    if args.backend != 'trt-single':
        from core import foundation_stereo as fast

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback')
    arrays = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in (args.left, args.right)]
    if any(array is None for array in arrays) or arrays[0].shape != arrays[1].shape:
        raise ValueError('Expected readable, equal-sized rectified left/right images')
    original_h, original_w = arrays[0].shape[:2]
    images = [cv2.resize(array, (args.width, args.height)) for array in arrays]
    tensors = [torch.from_numpy(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)).cuda().float()
               .permute(2, 0, 1).unsqueeze(0).contiguous() for array in images]
    identity = {'source_revision': (args.root / 'UPSTREAM_REVISION').read_text().strip(),
                'backend': args.backend, 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                'eager_correlation': args.backend != 'trt-single', 'torch_compile': False,
                'input_width': args.width, 'input_height': args.height,
                'iterations': args.iterations, 'max_disp': args.max_disp}
    if args.backend == 'torch':
        identity['checkpoint'] = str(args.checkpoint.resolve())
        identity['checkpoint_sha256'] = digest(args.checkpoint)
        model = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False)
        model.args.valid_iters = args.iterations
        model.args.max_disp = args.max_disp
        model = model.cuda().eval()

        def forward():
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                return model(*tensors, iters=args.iterations, test_mode=True,
                             optimize_build_volume='pytorch1')
    elif args.backend == 'trt-single':
        import tensorrt as trt

        if __package__:
            from .fast_foundation_trt import SingleEngineRunner
        else:
            from fast_foundation_trt import SingleEngineRunner

        cfg = OmegaConf.load(args.engine_dir / 'model.yaml')
        if (list(cfg.image_size) != [args.height, args.width]
                or cfg.valid_iters != args.iterations or cfg.max_disp != args.max_disp):
            raise ValueError('TRT dimensions/iterations/max-disp must match the built engine')
        provenance = json.loads((args.engine_dir / 'build.json').read_text())
        engine_hash = digest(args.engine_dir / 'model.engine')
        if (engine_hash != provenance['engine_sha256']
                or cfg.checkpoint_sha256 != provenance['checkpoint_sha256']
                or cfg.onnx_sha256 != provenance['onnx_sha256']
                or any(OmegaConf.to_container(cfg)[key] != provenance[key] for key in (
                    'image_size', 'valid_iters', 'max_disp', 'normalization', 'source_revision'))):
            raise ValueError('TensorRT engine/config identity mismatch')
        identity.update(tensorrt=trt.__version__, engine_sha256=engine_hash,
                        checkpoint_sha256=cfg.checkpoint_sha256,
                        onnx_sha256=cfg.onnx_sha256, persistent_io_buffers=True)
        model = SingleEngineRunner(args.engine_dir / 'model.engine')
        identity['trt_context_device_memory_mib'] = model.engine.device_memory_size / 2**20
        mean = tensors[0].new_tensor(list(cfg.normalization.mean)).view(1, 3, 1, 1)
        std = tensors[0].new_tensor(list(cfg.normalization.std)).view(1, 3, 1, 1)

        def forward():
            # Include external normalization and device-to-device input copies
            # in timing; the PyTorch forward also normalizes resident RGB input.
            with torch.inference_mode():
                return model(*[(tensor - mean) / std for tensor in tensors])
    else:
        import tensorrt as trt

        cfg = OmegaConf.load(args.engine_dir / 'onnx.yaml')
        if (list(cfg.image_size) != [args.height, args.width]
                or cfg.valid_iters != args.iterations or cfg.max_disp != args.max_disp):
            raise ValueError('TRT dimensions/iterations/max-disp must match the built engine')
        trt.init_libnvinfer_plugins(None, '')
        fast.build_gwc_volume_triton = fast.build_gwc_volume_optimized_pytorch1
        names = ('feature_runner.engine', 'post_runner.engine')
        identity['tensorrt'] = trt.__version__
        identity['engines_sha256'] = {name: digest(args.engine_dir / name) for name in names}
        identity['engine_provenance'] = 'pre-existing on this Orin; export source unverified'
        model = fast.TrtRunner(cfg, *(str(args.engine_dir / name) for name in names))

        def forward():
            with torch.inference_mode():
                return model(*tensors)

    timings = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.warmup + args.runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = forward()
        torch.cuda.synchronize()
        if index >= args.warmup:
            timings.append((time.perf_counter() - start) * 1000)
    disparity = output.float().cpu().numpy().reshape(args.height, args.width)
    if not np.isfinite(disparity).all() or not (disparity > 0).any():
        raise ValueError('Model output is not finite or contains no positive disparity')
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'disparity_px.npy', disparity, allow_pickle=False)
    lo, hi = np.percentile(disparity, [2, 98])
    visual = np.clip((disparity - lo) / max(float(hi - lo), .001), 0, 1)
    visual = cv2.applyColorMap((visual * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(args.output / 'disparity_preview.png'),
                np.concatenate([images[0], images[1], visual], axis=1))
    depth_report = None
    if args.intrinsics:
        lines = args.intrinsics.read_text().strip().splitlines()
        matrix = np.array([float(v) for v in lines[0].split()]).reshape(3, 3)
        baseline = float(lines[1])
        if (not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.isfinite(baseline) or baseline <= 0):
            raise ValueError('Invalid intrinsics or baseline in metres')
        sx, sy = args.width / original_w, args.height / original_h
        matrix[0] *= sx
        matrix[1] *= sy
        matrix[0, 2] += (sx - 1) / 2
        matrix[1, 2] += (sy - 1) / 2
        xx = np.indices(disparity.shape)[1]
        valid = (disparity > 0) & (xx - disparity >= 0)
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        depth[valid] = matrix[0, 0] * baseline / disparity[valid]
        np.save(args.output / 'depth_m.npy', depth, allow_pickle=False)
        depth_report = {'intrinsics_sha256': digest(args.intrinsics),
                        'baseline_m': baseline, 'resized_K': matrix.tolist(),
                        'valid_fraction': float(valid.mean()),
                        'median_m': float(np.median(depth[valid])) if valid.any() else None,
                        'accuracy_verified': False}
    report = {'passed': True, 'identity': identity,
              'left_sha256': digest(args.left), 'right_sha256': digest(args.right),
              'input_is_official_demo': not custom,
              'warmup': args.warmup, 'runs': args.runs,
              'forward_ms': timings, 'forward_median_ms': statistics.median(timings),
              'forward_p95_ms': float(np.percentile(timings, 95)),
              'process_rss_mib': psutil.Process().memory_info().rss / 2**20,
              'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
              'system_used_gib': psutil.virtual_memory().used / 2**30,
              'positive_disparity_fraction': float((disparity > 0).mean()),
              'depth': depth_report,
              'measurement_note': 'CUDA-synchronized warm forward only; excludes resize, transfers, '
              'depth conversion and file writes. CUDA allocator excludes TensorRT allocations. '
              'Process/CUDA/system memory overlap; do not sum. Not a live camera accuracy test.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
