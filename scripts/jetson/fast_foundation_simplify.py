"""Fold static ONNX shapes/constants on a separate host before TRT building.

Use a separate output directory. Numerical equivalence is checked afterwards
against PyTorch on the Orin; check_n=0 does not run random ONNX Runtime tests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    import onnx

    metadata = json.loads((args.source / 'export.json').read_text())
    data = (args.source / 'model.onnx').read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata['onnx_sha256']:
        raise ValueError('Source ONNX digest does not match export metadata')
    original = onnx.load_model_from_string(data)
    options = []
    if args.width is not None or args.height is not None:
        if (args.width is None or args.height is None or args.width % 32 or args.height % 32
                or not 32 <= args.width <= 1920 or not 32 <= args.height <= 1088):
            parser.error('Provide width/height divisible by 32, up to 1920x1088')
        if not metadata.get('dynamic_spatial'):
            raise ValueError('A different spatial size requires a dynamic export')
        options = ['--width', str(args.width), '--height', str(args.height)]
    with tempfile.TemporaryDirectory(prefix='fast-onnx-') as temporary:
        output = Path(temporary) / 'model.onnx'
        subprocess.run([sys.executable, str(Path(__file__).with_name(
            'fast_foundation_onnx_worker.py')), '--input', str(args.source / 'model.onnx'),
            '--output', str(output), *options], check=True)
        simplified = onnx.load_model(str(output))
    onnx.checker.check_model(simplified)
    args.output.mkdir(parents=True)
    onnx.save_model(simplified, str(args.output / 'model.onnx'))
    metadata.update(raw_onnx_sha256=metadata['onnx_sha256'],
                    onnxsim=importlib.metadata.version('onnxsim'), isolated_onnx_cpp_worker=True,
                    onnx_sha256=hashlib.sha256((args.output / 'model.onnx').read_bytes()).hexdigest(),
                    onnx_nodes_before=len(original.graph.node),
                    onnx_nodes_after=len(simplified.graph.node),
                    random_onnx_equivalence_runs=0)
    if options:
        metadata.update(trace_image_size=metadata['image_size'],
                        image_size=[args.height, args.width],
                        specialized_from_dynamic=True, dynamic_spatial=False)
    contents = json.dumps(metadata, indent=2) + '\n'
    (args.output / 'export.json').write_text(contents)
    # JSON is a YAML subset and can be read by OmegaConf on the Orin.
    (args.output / 'model.yaml').write_text(contents)
    print(contents, flush=True)


if __name__ == '__main__':
    main()
