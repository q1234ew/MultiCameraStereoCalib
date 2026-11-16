"""Export a pinned Fast-FoundationStereo checkpoint for JetPack 5 TensorRT.

Reuses the official single-ONNX wrapper and volume builders. Opset 16 keeps
LayerNorm decomposed for the TensorRT 8.5 parser. Does not edit upstream files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--iterations', type=int, choices=(4, 8), default=8)
    parser.add_argument('--max-disp', type=int, default=192, choices=(192, 256))
    parser.add_argument('--half-trace', action='store_true',
                        help='Trace with CUDA autocast to limit export memory at high resolution')
    parser.add_argument('--dynamic-spatial', action='store_true',
                        help='Export dynamic H/W at a small trace size, then specialize off-device')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (not 32 <= args.width <= 1920 or not 32 <= args.height <= 1088
            or args.width % 32 or args.height % 32):
        parser.error('Expected width/height divisible by 32, up to 1920x1088')
    args.root = args.root.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'model.onnx').exists():
        raise FileExistsError('Choose a new output directory; ONNX already exists')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TIMM_OFFLINE'] = '1'
    sys.path.insert(0, str(args.root))
    import torch

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.compile = lambda fn=None, **kwargs: fn if fn is not None else lambda value: value
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    spec = importlib.util.spec_from_file_location(
        'official_single_export', args.root / 'scripts/make_single_onnx.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    from core import foundation_stereo as fast

    # Check the replacement operators against the deployed eager operators
    # before tracing; include both border pixels and normalization branches.
    for normalize in (False, True):
        left = torch.randn(1, 16, 4, 16, device='cuda')
        right = torch.randn_like(left)
        expected = fast.build_gwc_volume_optimized_pytorch1(left, right, 4, 4, normalize)
        actual = official._build_gwc_volume_onnx(left, right, 4, 4, normalize)
        torch.testing.assert_close(expected, actual)
        torch.testing.assert_close(
            fast.build_concat_volume_optimized_pytorch1(left, right, 4),
            official._build_concat_volume_onnx(left, right, 4))
    print('Cost-volume export equivalence passed', flush=True)
    model = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.iterations
    model.args.mixed_precision = args.half_trace
    model.dtype = torch.float32
    model = model.cuda().float().eval()
    fast.normalize_image = lambda image: image
    fast.build_gwc_volume_optimized_pytorch1 = official._build_gwc_volume_onnx
    fast.build_concat_volume_optimized_pytorch1 = official._build_concat_volume_onnx
    wrapper = official.FastFoundationStereoSingleOnnx(model).cuda().eval()
    images = tuple(torch.randn(1, 3, args.height, args.width, device='cuda') for _ in range(2))
    print('Exporting %dx%d, %d iterations, opset16' % (
        args.width, args.height, args.iterations), flush=True)
    with torch.cuda.amp.autocast(enabled=args.half_trace, dtype=torch.float16,
                                 cache_enabled=False):
        export_options = {}
        if args.dynamic_spatial:
            export_options['dynamic_axes'] = {
                name: {2: 'image_height', 3: 'image_width'}
                for name in ('left_image', 'right_image', 'disparity')}
        torch.onnx.export(wrapper, images, str(args.output / 'model.onnx'),
                          opset_version=16, input_names=['left_image', 'right_image'],
                          output_names=['disparity'], do_constant_folding=True, **export_options)
    import onnx

    onnx.checker.check_model(str(args.output / 'model.onnx'))
    import yaml

    metadata = {'source_revision': (args.root / 'UPSTREAM_REVISION').read_text().strip(),
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'image_size': [args.height, args.width], 'valid_iters': args.iterations,
                'max_disp': args.max_disp, 'opset': 16, 'torch': str(torch.__version__),
                'trace_amp': args.half_trace,
                'dynamic_spatial': args.dynamic_spatial,
                'input_names': ['left_image', 'right_image'], 'output_name': 'disparity',
                'normalization': {'mean': [123.675, 116.28, 103.53],
                                  'std': [58.395, 57.12, 57.375]},
                'onnx_sha256': hashlib.sha256((args.output / 'model.onnx').read_bytes()).hexdigest(),
                'export_cost_volume_equivalence_passed': True}
    (args.output / 'model.yaml').write_text(yaml.safe_dump(metadata))
    (args.output / 'export.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
