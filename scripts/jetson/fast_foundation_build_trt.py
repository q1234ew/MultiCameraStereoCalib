"""Build a single, static FP16 TensorRT engine and record exact provenance."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import time
from pathlib import Path


def configure_tactic_search(trt, config, *, exhaustive=False, optimization_level=None):
    """Keep TRT8.5 heuristics; use supported builder levels on TRT10+."""
    has_levels = hasattr(config, 'builder_optimization_level')
    if optimization_level is not None and not has_levels:
        raise ValueError('This TensorRT version does not support builder optimization levels')
    if has_levels:
        level = optimization_level if optimization_level is not None else (3 if exhaustive else 2)
        config.builder_optimization_level = level
        return {'builder_optimization_level': level, 'tactic_heuristic': False}
    if not exhaustive:
        config.set_flag(trt.BuilderFlag.ENABLE_TACTIC_HEURISTIC)
        config.set_preview_feature(
            trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, True)
    return {'builder_optimization_level': None, 'tactic_heuristic': not exhaustive}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--workspace-mib', type=int, default=2048)
    parser.add_argument('--exhaustive-tactics', action='store_true',
                        help='Disable the JetPack 5 tactic heuristic (slower first build)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--optimization-level', type=int, choices=range(6),
                        help='TRT8.6+/10 builder search level; default2, or3 with exhaustive')
    tactics = parser.add_mutually_exclusive_group()
    tactics.add_argument('--core-tactics-only', action='store_true',
                        help='Use core/JIT kernels without cuBLAS/cuDNN/edge-mask tactics')
    tactics.add_argument('--precompiled-tactics-only', action='store_true',
                        help='Disable external, edge-mask and JIT tactic sources')
    args = parser.parse_args()
    if not 64 <= args.workspace_mib <= 4096:
        parser.error('workspace must be 64..4096 MiB')
    import tensorrt as trt

    directory = args.directory
    if (directory / 'model.engine').exists():
        raise FileExistsError('Refusing to overwrite an existing engine')
    metadata = json.loads((directory / 'export.json').read_text())
    if metadata.get('dynamic_spatial'):
        raise ValueError('Specialize dynamic spatial dimensions before building a static engine')
    onnx_hash = hashlib.sha256()
    with (directory / 'model.onnx').open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            onnx_hash.update(block)
    block = None
    if onnx_hash.hexdigest() != metadata['onnx_sha256']:
        raise ValueError('ONNX digest differs from recorded export')
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    builder.max_threads = 1
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(directory / 'model.onnx')):
        errors = [str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)]
        raise RuntimeError('ONNX parser failed: ' + '\n'.join(errors))
    height, width = metadata['image_size']
    input_shapes = metadata.get('input_shapes', {
        name: [1, 3, height, width] for name in ('left_image', 'right_image')})
    if {network.get_input(i).name for i in range(network.num_inputs)} != set(input_shapes):
        raise ValueError('ONNX input names do not match recorded export')
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if tuple(tensor.shape) != tuple(input_shapes[tensor.name]):
            raise ValueError('ONNX bindings do not match the recorded static image size')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 2**20)
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    # Retain FP32 for numerical range-sensitive reductions and normalization.
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    search = configure_tactic_search(
        trt, config, exhaustive=args.exhaustive_tactics,
        optimization_level=args.optimization_level)
    config.avg_timing_iterations = 1
    if args.core_tactics_only:
        if not config.set_tactic_sources(1 << int(trt.TacticSource.JIT_CONVOLUTIONS)):
            raise RuntimeError('TensorRT rejected the core/JIT tactic source selection')
    elif args.precompiled_tactics_only:
        if not config.set_tactic_sources(0):
            raise RuntimeError('TensorRT rejected the precompiled tactic selection')
    cache_path = directory.parent / 'timing.cache'
    timing_cache = config.create_timing_cache(
        cache_path.read_bytes() if cache_path.exists() else b'')
    config.set_timing_cache(timing_cache, ignore_mismatch=False)
    protected = []
    half_compute = []
    full_correlation = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.type in (trt.LayerType.SOFTMAX, trt.LayerType.REDUCE):
            layer.precision = trt.float32
            protected.append(layer.name)
        elif layer.type == trt.LayerType.MATRIX_MULTIPLY and '/Einsum' in layer.name:
            # Upstream explicitly disables autocast for the all-pairs
            # correlation matrix. Preserve that calculation in FP32.
            layer.precision = trt.float32
            full_correlation.append(layer.name)
        elif layer.type in (trt.LayerType.CONVOLUTION, trt.LayerType.DECONVOLUTION,
                            trt.LayerType.MATRIX_MULTIPLY):
            # Match the deployed CUDA-autocast compute path and avoid expensive
            # searches through FP32 convolution/GEMM layout alternatives.
            layer.precision = trt.float16
            for output_index in range(layer.num_outputs):
                layer.set_output_type(output_index, trt.float16)
            half_compute.append(layer.name)
    print('Parsed %d layers; %d protected FP32 reductions' % (
        network.num_layers, len(protected)), flush=True)
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed; inspect builder log')
    cache_bytes = bytes(config.get_timing_cache().serialize())
    tactic_sources = config.get_tactic_sources()
    max_threads = builder.max_threads
    # The serialized engine owns its data. Release the parsed graph and builder
    # before writing it, and avoid another engine-sized Python bytes allocation.
    del onnx_parser, network, config, builder
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    engine_view = memoryview(serialized)
    (directory / 'model.engine').write_bytes(engine_view)
    cache_path.write_bytes(cache_bytes)
    metadata.update(engine_sha256=hashlib.sha256(engine_view).hexdigest(),
                    tensorrt=trt.__version__, workspace_mib=args.workspace_mib,
                    build_s=time.monotonic() - started, fp16=True, tf32=False,
                    **search,
                    tactic_sources=tactic_sources,
                    core_tactics_only=args.core_tactics_only,
                    precompiled_tactics_only=args.precompiled_tactics_only,
                    builder_max_threads=max_threads,
                    fp16_compute_layers=half_compute,
                    fp16_compute_outputs=True,
                    fp32_correlation_layers=full_correlation,
                    precision_constraints='obey', fp32_protected_layers=protected)
    (directory / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('BUILD_COMPLETE ' + json.dumps({k: metadata[k] for k in (
        'image_size', 'engine_sha256', 'build_s')}), flush=True)


if __name__ == '__main__':
    main()
