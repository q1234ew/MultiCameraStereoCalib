"""Python 3.8 CUDA worker; speaks private stdin/stdout JSON, never listens on LAN."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    protocol = sys.stdout
    # Upstream dependencies may print during import; keep protocol stdout isolated.
    with contextlib.redirect_stdout(sys.stderr):
        # Avoid this application's core.py shadowing upstream's namespace package.
        own_dir = Path(__file__).resolve().parent
        sys.path = [str(args.root)] + [p for p in sys.path
                                          if Path(p).resolve() != own_dir]
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        if not torch.cuda.is_available():
            raise RuntimeError('FoundationStereo requires the existing CUDA runtime')
        cfg = OmegaConf.load(args.checkpoint.parent / 'cfg.yaml')
        if 'vit_size' not in cfg:
            cfg.vit_size = 'vitl'
        model = FoundationStereo(cfg)
        checkpoint = torch.load(str(args.checkpoint), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        import gc
        import ctypes
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        model = model.cuda().eval()
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        digest = hashlib.sha256()
        with args.checkpoint.open('rb') as f:
            for chunk in iter(lambda: f.read(4*1024*1024), b''):
                digest.update(chunk)
    identity = {'model': 'FoundationStereo', 'checkpoint': str(args.checkpoint),
                'checkpoint_sha256': digest.hexdigest(),
                'torch': torch.__version__}
    print(json.dumps({'ready': True, 'identity': identity}), file=protocol, flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            iterations = request.get('iterations', 8)
            if type(iterations) is not int or iterations not in (8, 16, 32):
                raise ValueError('Unsupported iteration count')
            with contextlib.redirect_stdout(sys.stderr):
                gc.collect()
                torch.cuda.empty_cache()
                ctypes.CDLL('libc.so.6').malloc_trim(0)
                with np.load(request['input'], allow_pickle=False) as pair:
                    arrays = [pair[side].copy() for side in ('left', 'right')]
                tensors = [torch.from_numpy(im).cuda().float()[None].permute(0, 3, 1, 2)
                           for im in arrays]
                padder = InputPadder(tensors[0].shape, divis_by=32, force_square=False)
                a, b = padder.pad(*tensors)
                torch.cuda.synchronize()
                start = time.monotonic()
                with torch.inference_mode(), torch.cuda.amp.autocast(True):
                    disparity = model(a, b, iters=iterations, test_mode=True, low_memory=True)
                disparity = padder.unpad(disparity.float()).cpu().numpy().reshape(
                    arrays[0].shape[:2])
                torch.cuda.synchronize()
                elapsed = (time.monotonic()-start)*1000
                np.save(request['output'], disparity, allow_pickle=False)
                del tensors, a, b
            print(json.dumps({'ok': True, 'inference_ms': round(elapsed, 1),
                              'identity': dict(identity, iterations=iterations)}), file=protocol, flush=True)
        except Exception as exc:
            torch.cuda.empty_cache()
            print(json.dumps({'ok': False, 'error': str(exc)}), file=protocol, flush=True)


if __name__ == '__main__':
    main()
