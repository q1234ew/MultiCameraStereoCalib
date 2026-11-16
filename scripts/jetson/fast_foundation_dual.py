"""Pinned 640x352 Fast-FoundationStereo dual engines plus eager-equivalent CUDA GWC.

Run export, build, then validate on Orin. Does not modify upstream or system packages.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT_SHA256 = '98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692'
REVISION = '476f4249561f7c79ca707326954f9255643412a6'
FEATURES = ('features_left_04', 'features_left_08', 'features_left_16',
            'features_left_32', 'features_right_04', 'stem_2x')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def check_identity(root, checkpoint):
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Expected the pinned 20-30-48 checkpoint SHA256')
    if (root / 'UPSTREAM_REVISION').read_text().strip() != REVISION:
        raise ValueError('Unexpected upstream revision')


def check_metadata(metadata):
    expected = {'checkpoint_sha256': CHECKPOINT_SHA256, 'source_revision': REVISION,
                'image_size': [352, 640], 'valid_iters': 8, 'max_disp': 192}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Expected pinned 640x352-i8-d192 export identity')
    if (metadata.get('split_equivalence_passed') is not True
            and metadata.get('validation_strategy') != 'staged-reference'):
        raise ValueError('Missing split validation contract')
    if type(metadata.get('cv_group')) is not int or metadata['cv_group'] <= 0:
        raise ValueError('Invalid correlation group count')
    if type(metadata.get('normalize')) is not bool:
        raise ValueError('Invalid correlation normalization setting')


def setup(root):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TIMM_OFFLINE'] = '1'
    sys.path.insert(0, str(root))
    import torch
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.compile = lambda fn=None, **kw: fn if fn is not None else lambda value: value
    if not torch.cuda.is_available():
        raise RuntimeError('Orin CUDA required')
    from core import foundation_stereo as fast
    return torch, fast


def load_model(torch, checkpoint):
    model = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    model.args.valid_iters = 8
    model.args.max_disp = 192
    model.args.mixed_precision = True
    return model.cuda().float().eval()


class CudaGWC:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(library.resolve()))
        self.fn = self.library.ffs_gwc
        self.fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 6 + [
            ctypes.c_bool, ctypes.c_void_p]
        self.fn.restype = ctypes.c_int

    def __call__(self, left, right, output, normalize):
        import torch
        if (left.dtype != torch.float16 or right.dtype != left.dtype
                or output.dtype != left.dtype or not left.is_cuda
                or right.device != left.device or output.device != left.device
                or left.shape != right.shape or left.ndim != 4 or output.ndim != 5
                or not all(x.is_contiguous() for x in (left, right, output))):
            raise ValueError('GWC requires contiguous, same-device FP16 CUDA bindings')
        b, c, h, w = left.shape
        ob, g, d, oh, ow = output.shape
        if (ob, oh, ow) != (b, h, w) or g <= 0 or c % g or d <= 0:
            raise ValueError('GWC shape/group mismatch')
        with torch.cuda.device(left.device):
            code = self.fn(left.data_ptr(), right.data_ptr(), output.data_ptr(),
                           b, c, h, w, d, g, normalize,
                           torch.cuda.current_stream().cuda_stream)
        if code:
            raise RuntimeError('CUDA GWC launch error %d' % code)
        return output


def release_memory(torch):
    gc.collect()
    torch.cuda.empty_cache()
    ctypes.CDLL('libc.so.6').malloc_trim(0)


def memory_stage(label):
    import psutil
    print(json.dumps({'stage': label, 'rss_mib': psutil.Process().memory_info().rss / 2**20,
                      'available_gib': psutil.virtual_memory().available / 2**30}), flush=True)


def tiled_gwc(left, right, maxdisp, groups, normalize=True):
    """Same eager operations, one disparity slice at a time instead of a 6D temporary."""
    import torch.nn.functional as functional
    b, c, h, w = left.shape
    k = c // groups
    left = left.reshape(b, groups, k, h, w)
    right = right.reshape(b, groups, k, h, w)
    if normalize:
        left = functional.normalize(left.float(), dim=2).to(left.dtype)
        right = functional.normalize(right.float(), dim=2).to(right.dtype)
    output = left.new_zeros((b, groups, maxdisp, h, w))
    for d in range(min(maxdisp, w)):
        output[:, :, d, :, d:] = (left[..., d:] * right[..., :w-d]).sum(dim=2)
    return output


def read_images(torch, left, right, device='cuda'):
    import cv2
    arrays = [cv2.imread(str(path)) for path in (left, right)]
    if any(value is None for value in arrays) or arrays[0].shape != arrays[1].shape:
        raise ValueError('Readable equal-size rectified stereo pair required')
    return tuple(torch.from_numpy(cv2.cvtColor(cv2.resize(value, (640, 352)),
                       cv2.COLOR_BGR2RGB)).to(device).float().permute(2, 0, 1)
                 .unsqueeze(0).contiguous() for value in arrays)


def export(args):
    check_identity(args.root, args.checkpoint)
    if args.component is None:
        args.directory.mkdir(parents=True, exist_ok=False)
        for component in ('feature', 'post'):
            command = [sys.executable, str(Path(__file__).resolve()), 'export',
                            '--component', component, '--trace-device', args.trace_device,
                            '--root', str(args.root),
                            '--checkpoint', str(args.checkpoint),
                            '--directory', str(args.directory)]
            if getattr(args, 'serial_features', False):
                command.append('--serial-features')
            subprocess.run(command, check=True)
        return
    folder = args.directory / args.component
    folder.mkdir(exist_ok=False)
    torch, fast = setup(args.root)
    import numpy as np
    memory_stage('imported_' + args.component)
    # Load onto CPU first; retain only this component before moving to CUDA.
    model = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False)
    model.args.valid_iters = 8
    model.args.max_disp = 192
    model.args.mixed_precision = True
    common = {'checkpoint_sha256': CHECKPOINT_SHA256, 'source_revision': REVISION,
              'image_size': [352, 640], 'valid_iters': 8, 'max_disp': 192,
              'cv_group': int(model.cv_group), 'normalize': bool(model.args.normalize),
              'opset': 16, 'torch': str(torch.__version__),
              'validation_strategy': 'staged-reference', 'trace_device': args.trace_device,
              'normalization': 'RGB 0..255; ImageNet normalization inside feature engine'}
    if args.serial_features:
        common['serial_features'] = True
    module = fast.TrtFeatureRunner(model) if args.component == 'feature' else fast.TrtPostRunner(model)
    if args.component == 'post':
        # This unused member otherwise retains the complete feature backbone.
        del module.feature
    del model
    release_memory(torch)
    module = module.to(args.trace_device).float().eval()
    memory_stage('loaded_' + args.component)

    class Wrapper(torch.nn.Module):
        def __init__(self, runner, feature):
            super().__init__()
            self.runner, self.feature = runner, feature

        def forward(self, *inputs):
            with torch.cuda.amp.autocast(enabled=args.trace_device == 'cuda',
                                         dtype=torch.float16):
                if args.trace_device == 'cpu':
                    inputs = tuple(value.float() for value in inputs)
                if self.feature and args.serial_features:
                    image = fast.normalize_image(inputs[0])
                    values = self.runner.feature(image)
                    return tuple(x.half() for x in (*values, self.runner.stem_2(image)))
                value = self.runner(*inputs)
                return tuple(x.half() for x in value) if self.feature else value.float()

    wrapper = Wrapper(module, args.component == 'feature').eval()
    if args.component == 'feature':
        inputs = read_images(torch, args.root / 'demo_data/left.png',
                             args.root / 'demo_data/right.png', args.trace_device)
        if args.serial_features:
            left_values = wrapper(inputs[0])
            right_value = wrapper(inputs[1])[0]
            values = (*left_values[:4], right_value, left_values[-1])
            inputs = (inputs[0],)
            del left_values, right_value
            names, outputs = ('image',), (*FEATURES[:4], FEATURES[-1])
        else:
            values = wrapper(*inputs)
            names, outputs = ('left', 'right'), FEATURES
        np.savez(args.directory / 'features.npz', **{
            key: value.cpu().numpy() for key, value in zip(FEATURES, values)})
        del values
        write_json(args.directory / 'export.json', common)
    else:
        if json.loads((args.directory / 'export.json').read_text()) != common:
            raise ValueError('Feature/post export identity mismatch')
        # Both concat builders have the same eager semantics; select the ONNX-friendly one.
        spec = importlib.util.spec_from_file_location(
            'official_single_export', args.root / 'scripts/make_single_onnx.py')
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        fast.build_concat_volume_optimized_pytorch = official._build_concat_volume_onnx
        with np.load(args.directory / 'features.npz', allow_pickle=False) as values:
            inputs = tuple(torch.from_numpy(values[key]).to(args.trace_device) for key in FEATURES)
        volume = tiled_gwc(inputs[0], inputs[4], 48, common['cv_group'], common['normalize'])
        inputs = (*inputs, volume)
        del volume
        names, outputs = (*FEATURES, 'gwc_volume'), ('disp',)
        prediction = wrapper(*inputs)
        np.save(args.directory / 'split_torch_disparity.npy', prediction.cpu().numpy())
        del prediction
    release_memory(torch)
    memory_stage('before_trace_' + args.component)
    torch.onnx.export(wrapper, inputs, str(folder / 'model.onnx'), opset_version=16,
                      input_names=list(names), output_names=list(outputs),
                      do_constant_folding=True)
    # Release tracing's CUDA state before loading the ONNX checker.
    shapes = {key: list(value.shape) for key, value in zip(names, inputs)}
    del inputs, wrapper, module
    release_memory(torch)
    memory_stage('after_trace_' + args.component)
    import onnx
    onnx.checker.check_model(str(folder / 'model.onnx'))
    write_json(folder / 'export.json', dict(common, component=args.component,
               input_shapes=shapes, onnx_sha256=digest(folder / 'model.onnx')))


def reference(args):
    """Independent full PyTorch model; slice only the mathematically identical GWC op."""
    check_identity(args.root, args.checkpoint)
    if (args.directory / 'reference.json').exists():
        raise FileExistsError('Preserve previous reference')
    if args.cached_reference:
        return import_reference(args)
    torch, fast = setup(args.root)
    memory_stage('reference_imported')
    original = fast.build_gwc_volume_optimized_pytorch1
    for normalize in (False, True):
        for scale in (0., 1e-5, 1.):
            left = (torch.randn(2, 16, 3, 11, device='cuda') * scale).half()
            right = torch.randn_like(left) * scale
            torch.testing.assert_close(tiled_gwc(left, right, 13, 4, normalize),
                                       original(left, right, 13, 4, normalize), atol=0, rtol=0)
    fast.build_gwc_volume_optimized_pytorch1 = tiled_gwc
    model = load_model(torch, args.checkpoint)
    release_memory(torch)
    memory_stage('reference_loaded')
    tensors = read_images(torch, args.left, args.right)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        prediction = model(*tensors, iters=8, test_mode=True, optimize_build_volume='pytorch1')
    import numpy as np
    result = prediction.float().cpu().numpy()
    np.save(args.directory / 'reference_disparity.npy', result)
    del model, prediction, tensors
    release_memory(torch)
    report = {'checkpoint_sha256': CHECKPOINT_SHA256, 'source_revision': REVISION,
              'image_size': [352, 640], 'valid_iters': 8, 'max_disp': 192,
              'left_sha256': digest(args.left), 'right_sha256': digest(args.right),
              'disparity_sha256': digest(args.directory / 'reference_disparity.npy'),
              'tiled_gwc_small_exact': True,
              'split_comparison': error_metrics(result, np.load(
                  args.directory / 'split_torch_disparity.npy', allow_pickle=False))}
    write_json(args.directory / 'reference.json', report)
    print(json.dumps(report), flush=True)
    if not report['split_comparison']['passed']:
        raise RuntimeError('Split PyTorch/reference equivalence failed')


def import_reference(args):
    """Reuse an existing full eager run only after verifying its identity and digests."""
    import numpy as np
    folder = args.cached_reference
    if (digest(folder / 'report.json') != args.reference_report_sha256
            or digest(folder / 'disparity_px.npy') != args.reference_disparity_sha256):
        raise ValueError('Cached reference digest mismatch')
    original = json.loads((folder / 'report.json').read_text())
    identity = original['identity']
    expected = {'backend': 'torch', 'checkpoint_sha256': CHECKPOINT_SHA256,
                'source_revision': REVISION, 'input_width': 640, 'input_height': 352,
                'iterations': 8, 'max_disp': 192, 'eager_correlation': True}
    if (not original['passed'] or any(identity.get(k) != v for k, v in expected.items())
            or original['left_sha256'] != digest(args.left)
            or original['right_sha256'] != digest(args.right)):
        raise ValueError('Cached reference model/input identity mismatch')
    result = np.load(folder / 'disparity_px.npy', allow_pickle=False)
    if result.shape != (352, 640):
        raise ValueError('Unexpected cached disparity shape')
    result = result.reshape(1, 1, 352, 640)
    np.save(args.directory / 'reference_disparity.npy', result)
    report = {'checkpoint_sha256': CHECKPOINT_SHA256, 'source_revision': REVISION,
              'image_size': [352, 640], 'valid_iters': 8, 'max_disp': 192,
              'left_sha256': digest(args.left), 'right_sha256': digest(args.right),
              'disparity_sha256': digest(args.directory / 'reference_disparity.npy'),
              'cached_reference': str(folder),
              'cached_report_sha256': args.reference_report_sha256,
              'cached_disparity_sha256': args.reference_disparity_sha256,
              'split_comparison': error_metrics(result, np.load(
                  args.directory / 'split_torch_disparity.npy', allow_pickle=False))}
    write_json(args.directory / 'reference.json', report)
    print(json.dumps(report), flush=True)
    if not report['split_comparison']['passed']:
        raise RuntimeError('Split PyTorch/cached full eager comparison failed')


def build(args):
    metadata = json.loads((args.directory / 'export.json').read_text())
    check_metadata(metadata)
    source = Path(__file__).with_name('fast_foundation_gwc.cu')
    library = args.directory / 'libffs_gwc.so'
    if library.exists():
        kernel = json.loads((args.directory / 'kernel.json').read_text())
        if (digest(library) != kernel['library_sha256']
                or digest(source) != kernel['source_sha256']):
            raise ValueError('Refusing to replace an existing or changed CUDA library')
    else:
        subprocess.run(['/usr/local/cuda/bin/nvcc', '-shared', '-Xcompiler', '-fPIC',
                        '-O3', '--fmad=false', '-arch=sm_87', str(source), '-o', str(library)],
                       check=True)
        write_json(args.directory / 'kernel.json', {
            'source_sha256': digest(source), 'library_sha256': digest(library),
            'semantics': 'eager-pytorch1-fp16', 'fmad': False, 'architecture': 'sm_87'})
    for component in ('feature', 'post'):
        folder = args.directory / component
        exported = json.loads((folder / 'export.json').read_text())
        check_metadata(exported)
        if (exported['component'] != component
                or any(exported.get(key) != value for key, value in metadata.items())
                or digest(folder / 'model.onnx') != exported['onnx_sha256']):
            raise ValueError('Component export identity mismatch')
        if (folder / 'model.engine').exists():
            built = json.loads((folder / 'build.json').read_text())
            if (any(built.get(key) != value for key, value in exported.items())
                    or digest(folder / 'model.engine') != built['engine_sha256']):
                raise ValueError('Refusing to overwrite an unverified engine')
            continue
        command = [sys.executable, str(Path(__file__).with_name(
            'fast_foundation_build_trt.py')), '--directory', str(folder),
            '--workspace-mib', str(getattr(args, 'workspace_mib', 512))]
        if getattr(args, 'core_tactics_only', False):
            command.append('--core-tactics-only')
        elif getattr(args, 'precompiled_tactics_only', False):
            command.append('--precompiled-tactics-only')
        subprocess.run(command, check=True)


class StaticEngine:
    def __init__(self, folder, shared=None):
        import tensorrt as trt
        import torch
        metadata = json.loads((folder / 'build.json').read_text())
        check_metadata(metadata)
        exported = json.loads((folder / 'export.json').read_text())
        if (any(metadata.get(key) != value for key, value in exported.items())
                or metadata['component'] != folder.name
                or digest(folder / 'model.engine') != metadata['engine_sha256']):
            raise ValueError('Engine identity mismatch')
        self.metadata = metadata
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine((folder / 'model.engine').read_bytes())
        if self.engine is None:
            raise RuntimeError('Cannot deserialize engine')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Cannot allocate engine context')
        self.buffers, self.inputs = {}, []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = {trt.float32: torch.float32, trt.float16: torch.float16}[
                self.engine.get_tensor_dtype(name)]
            if any(x <= 0 for x in shape):
                raise ValueError('Static engine required')
            tensor = (shared or {}).get(name)
            if tensor is None:
                tensor = torch.empty(shape, device='cuda', dtype=dtype)
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError('Engine boundary shape/dtype mismatch: ' + name)
            self.buffers[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
        if set(self.inputs) != set(metadata['input_shapes']):
            raise ValueError('Engine input names mismatch')
        for name in self.inputs:
            if list(self.buffers[name].shape) != metadata['input_shapes'][name]:
                raise ValueError('Engine input shape mismatch')

    def run(self):
        import torch
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT enqueue failed')


class DualEngineRunner:
    """One instance, one current CUDA stream; returned output storage is reused."""
    def __init__(self, directory):
        kernel = json.loads((directory / 'kernel.json').read_text())
        if digest(directory / 'libffs_gwc.so') != kernel['library_sha256']:
            raise ValueError('CUDA library identity mismatch')
        self.gwc = CudaGWC(directory / 'libffs_gwc.so')
        self.feature = StaticEngine(directory / 'feature')
        self.serial = bool(self.feature.metadata.get('serial_features', False))
        self.post = StaticEngine(directory / 'post', None if self.serial else self.feature.buffers)
        feature_names = ({'image', *FEATURES[:4], FEATURES[-1]} if self.serial else
                         {'left', 'right', *FEATURES})
        if (set(self.feature.buffers) != feature_names
                or set(self.post.buffers) != {*FEATURES, 'gwc_volume', 'disp'}):
            raise ValueError('Unexpected dual-engine boundary names')
        # The post graph has the same binding contract for batched and serial
        # feature execution; its verified ONNX export can be reused unchanged.
        for key in ('cv_group', 'normalize'):
            if self.feature.metadata.get(key) != self.post.metadata.get(key):
                raise ValueError('Engine metadata mismatch: ' + key)

    def __call__(self, left, right):
        for name, value in (('left', left), ('right', right)):
            target = self.feature.buffers['image' if self.serial else name]
            if value.shape != target.shape or value.device != target.device:
                raise ValueError('Expected resident CUDA RGB 640x352 input')
            target.copy_(value)
            if self.serial:
                self.feature.run()
                if name == 'left':
                    for key in (*FEATURES[:4], FEATURES[-1]):
                        self.post.buffers[key].copy_(self.feature.buffers[key])
                else:
                    self.post.buffers[FEATURES[4]].copy_(self.feature.buffers[FEATURES[0]])
        if not self.serial:
            self.feature.run()
        self.gwc(self.post.buffers[FEATURES[0]], self.post.buffers[FEATURES[4]],
                 self.post.buffers['gwc_volume'], self.post.metadata['normalize'])
        self.post.run()
        return self.post.buffers['disp']


def error_metrics(reference, actual):
    import numpy as np
    if reference.shape != actual.shape:
        raise ValueError('Comparison shape mismatch')
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    if not finite:
        return {'finite': False, 'passed': False}
    error = np.abs(reference.astype(np.float32) - actual.astype(np.float32))
    result = {'finite': True, 'mae_px': float(error.mean()),
              'max_abs_px': float(error.max()), 'p95_abs_px': float(np.percentile(error, 95)),
              'fraction_gt_2px': float((error > 2).mean())}
    result['passed'] = (result['mae_px'] <= .5 and result['p95_abs_px'] <= 1
                        and result['fraction_gt_2px'] <= .01)
    return result


def check_gwc_cases(torch, fast, gwc_kernel):
    # Exercise both branches, zero/tiny norms, non-square shapes, batch > 1,
    # disparities beyond width, and every invalid left border pixel.
    checks = []
    for normalize in (False, True):
        for scale in (0., 1e-5, 1.):
            left = (torch.randn(2, 16, 3, 11, device='cuda') * scale).half()
            right = torch.randn_like(left) * scale
            expected = fast.build_gwc_volume_optimized_pytorch1(left, right, 13, 4, normalize)
            torch.testing.assert_close(tiled_gwc(left, right, 13, 4, normalize),
                                       expected, atol=0, rtol=0)
            actual = torch.empty_like(expected)
            gwc_kernel(left, right, actual, normalize)
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            checks.append({'normalize': normalize, 'scale': scale,
                           'max_abs': (actual - expected).abs().max().item()})
    return checks


def kernel_check(args):
    check_identity(args.root, args.checkpoint)
    kernel = json.loads((args.directory / 'kernel.json').read_text())
    if digest(args.directory / 'libffs_gwc.so') != kernel['library_sha256']:
        raise ValueError('CUDA library identity mismatch')
    report_path = args.directory / ('kernel-fixture-validation.json' if args.feature_fixture
                                    else 'kernel-validation.json')
    if report_path.exists():
        raise FileExistsError('Preserve previous kernel validation')
    torch, fast = setup(args.root)
    gwc = CudaGWC(args.directory / 'libffs_gwc.so')
    memory_stage('kernel_check_imported')
    checks = check_gwc_cases(torch, fast, gwc)
    fixture_result = None
    if args.feature_fixture:
        import numpy as np
        with np.load(args.feature_fixture, allow_pickle=False) as fixture:
            left = torch.from_numpy(fixture[FEATURES[0]]).cuda()
            right = torch.from_numpy(fixture[FEATURES[4]]).cuda()
        metadata = json.loads((args.directory / 'export.json').read_text())
        if tuple(left.shape) != (1, 224, 88, 160) or left.shape != right.shape:
            raise ValueError('Expected the pinned 640 feature fixture')
        expected = tiled_gwc(left, right, 48, metadata['cv_group'], metadata['normalize'])
        actual = torch.empty_like(expected)
        gwc(left, right, actual, metadata['normalize'])
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        fixture_result = {'fixture_sha256': digest(args.feature_fixture),
                          'source': 'CPU-traced pinned PyTorch feature outputs',
                          'shape': list(expected.shape), 'normalize': metadata['normalize'],
                          'max_abs': float((actual - expected).abs().max().item()),
                          'passed': True}
    torch.cuda.synchronize()
    write_json(report_path, {'passed': True, 'cases': checks, 'tiled_gwc_small_exact': True,
                            'kernel': kernel, 'torch': str(torch.__version__),
                            'feature_fixture': fixture_result})
    memory_stage('kernel_check_passed')


def validate(args):
    check_identity(args.root, args.checkpoint)
    report_path = args.directory / 'validation.json'
    if report_path.exists():
        raise FileExistsError('Preserve previous validation; choose a new artifact directory')
    torch, fast = setup(args.root)
    import numpy as np
    kernel = json.loads((args.directory / 'kernel.json').read_text())
    if digest(args.directory / 'libffs_gwc.so') != kernel['library_sha256']:
        raise ValueError('CUDA library identity mismatch')
    gwc_kernel = CudaGWC(args.directory / 'libffs_gwc.so')
    checks = check_gwc_cases(torch, fast, gwc_kernel)
    paths = (args.left, args.right)
    ref = json.loads((args.directory / 'reference.json').read_text())
    expected_identity = {'checkpoint_sha256': CHECKPOINT_SHA256, 'source_revision': REVISION,
                         'image_size': [352, 640], 'valid_iters': 8, 'max_disp': 192,
                         'left_sha256': digest(args.left), 'right_sha256': digest(args.right)}
    if (any(ref.get(k) != v for k, v in expected_identity.items())
            or not ref['split_comparison']['passed']
            or digest(args.directory / 'reference_disparity.npy') != ref['disparity_sha256']):
        raise ValueError('Independent reference identity/comparison mismatch')
    reference = np.load(args.directory / 'reference_disparity.npy', allow_pickle=False)
    tensors = read_images(torch, *paths)
    release_memory(torch)
    runner = DualEngineRunner(args.directory)
    # Compare the CUDA bridge to eager using actual feature-engine outputs.
    actual = runner(*tensors)
    torch.cuda.synchronize()
    f = runner.post.buffers
    eager = tiled_gwc(
        f[FEATURES[0]], f[FEATURES[4]], 48, runner.post.metadata['cv_group'],
        runner.post.metadata['normalize'])
    torch.testing.assert_close(runner.post.buffers['gwc_volume'], eager, atol=2e-3, rtol=2e-3)
    bridge_error = (runner.post.buffers['gwc_volume'] - eager).abs().max().item()
    cuda_result = actual.float().cpu().numpy().copy()
    runner.post.buffers['gwc_volume'].copy_(eager)
    runner.post.run()
    eager_result = runner.post.buffers['disp'].float().cpu().numpy().copy()
    timings = []
    for index in range(7):
        torch.cuda.synchronize()
        start = time.perf_counter()
        runner(*tensors)
        torch.cuda.synchronize()
        if index >= 2:
            timings.append((time.perf_counter() - start) * 1000)
    np.save(args.directory / 'dual_cuda_disparity.npy', cuda_result)
    comparison = error_metrics(reference, cuda_result)
    bridge_comparison = error_metrics(eager_result, cuda_result)
    report = {'checkpoint_sha256': CHECKPOINT_SHA256, 'source_revision': REVISION,
              'image_size': [352, 640], 'valid_iters': 8, 'max_disp': 192,
              'left_sha256': digest(paths[0]), 'right_sha256': digest(paths[1]),
              'reference': ref, 'tiled_gwc_small_exact': True,
              'engines_sha256': {name: digest(args.directory / name / 'model.engine')
                                 for name in ('feature', 'post')},
              'kernel': kernel,
              'kernel_cases': checks, 'actual_feature_gwc_max_abs': bridge_error,
              'torch_vs_dual_cuda': comparison, 'dual_eager_vs_dual_cuda': bridge_comparison,
              'forward_ms': timings, 'forward_median_ms': float(np.median(timings)),
              'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'tensorrt': str(runner.feature.metadata['tensorrt']),
              'thresholds': {'mae_px': .5, 'p95_abs_px': 1., 'fraction_gt_2px': .01},
              'passed': comparison['passed'] and bridge_comparison['passed']}
    write_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise RuntimeError('Numerical acceptance failed; retain artifacts for diagnosis')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('export', 'build', 'reference', 'validate', 'kernel-check'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--component', choices=('feature', 'post'))
    parser.add_argument('--trace-device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--serial-features', action='store_true',
                        help='Run one batch-1 feature engine twice to reduce activation memory')
    parser.add_argument('--workspace-mib', type=int, default=512)
    tactics = parser.add_mutually_exclusive_group()
    tactics.add_argument('--core-tactics-only', action='store_true')
    tactics.add_argument('--precompiled-tactics-only', action='store_true')
    parser.add_argument('--cached-reference', type=Path)
    parser.add_argument('--reference-report-sha256')
    parser.add_argument('--reference-disparity-sha256')
    parser.add_argument('--feature-fixture', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--left', type=Path)
    parser.add_argument('--right', type=Path)
    args = parser.parse_args()
    if args.cached_reference and not (args.reference_report_sha256
                                      and args.reference_disparity_sha256):
        parser.error('Cached reference requires both independently recorded file digests')
    if args.action != 'build' and (args.root is None or args.checkpoint is None):
        parser.error('export/validate require --root and --checkpoint')
    if args.action in ('validate', 'reference') and (args.left is None or args.right is None):
        parser.error('validate requires --left and --right')
    if args.root:
        args.root = args.root.resolve(strict=True)
    args.directory = args.directory.resolve()
    globals()[args.action.replace('-', '_')](args)


if __name__ == '__main__':
    main()
