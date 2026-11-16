"""Isolated onnxsim C++ worker (do not import the ONNX C++ extension here).

Some macOS builds export incompatible ONNX/protobuf C++ symbols from both
extensions. Exchange serialized protos and use only pure-Python protobuf types
in the constant executor, keeping one ONNX C++ implementation per process.
"""
from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import importlib.util
import sys
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    args = parser.parse_args()
    import numpy as np
    import onnxruntime as ort

    pb = load_module('isolated_onnx_pb', importlib.metadata.distribution('onnx').locate_file(
        'onnx/onnx_ml_pb2.py'))
    sim_dir = importlib.metadata.distribution('onnxsim').locate_file('onnxsim')
    extension = next(sim_dir.glob('onnxsim_cpp2py_export*.so'))
    cpp = load_module('onnxsim_cpp2py_export', extension)
    dtype_map = {pb.TensorProto.FLOAT: np.dtype('float32'),
                 pb.TensorProto.DOUBLE: np.dtype('float64'),
                 pb.TensorProto.FLOAT16: np.dtype('float16'),
                 pb.TensorProto.INT64: np.dtype('int64'),
                 pb.TensorProto.INT32: np.dtype('int32'),
                 pb.TensorProto.INT16: np.dtype('int16'),
                 pb.TensorProto.INT8: np.dtype('int8'),
                 pb.TensorProto.UINT64: np.dtype('uint64'),
                 pb.TensorProto.UINT32: np.dtype('uint32'),
                 pb.TensorProto.UINT16: np.dtype('uint16'),
                 pb.TensorProto.UINT8: np.dtype('uint8'),
                 pb.TensorProto.BOOL: np.dtype('bool')}

    def to_array(data):
        tensor = pb.TensorProto.FromString(data)
        dtype = dtype_map[tensor.data_type]
        if tensor.HasField('raw_data'):
            return np.frombuffer(tensor.raw_data, dtype=dtype).reshape(tensor.dims)
        if tensor.data_type == pb.TensorProto.FLOAT16:
            return np.asarray(tensor.int32_data, dtype=np.uint16).view(np.float16).reshape(
                tensor.dims)
        fields = {pb.TensorProto.FLOAT: 'float_data', pb.TensorProto.DOUBLE: 'double_data',
                  pb.TensorProto.INT64: 'int64_data', pb.TensorProto.UINT64: 'uint64_data',
                  pb.TensorProto.UINT32: 'uint64_data'}
        return np.asarray(getattr(tensor, fields.get(tensor.data_type, 'int32_data')),
                          dtype=dtype).reshape(tensor.dims)

    class Executor(cpp.ModelExecutor):
        def __init__(self):
            cpp.ModelExecutor.__init__(self)

        def Run(self, model_bytes, input_bytes):
            model = pb.ModelProto.FromString(model_bytes)
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.log_severity_level = 3
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(model_bytes, options, providers=['CPUExecutionProvider'])
            arrays = session.run(None, {value.name: to_array(data) for value, data in zip(
                model.graph.input, input_bytes)})
            results = []
            for array in arrays:
                tensor = pb.TensorProto()
                tensor.data_type = next(code for code, dtype in dtype_map.items()
                                        if dtype == array.dtype)
                tensor.dims.extend(array.shape)
                tensor.raw_data = np.ascontiguousarray(array).tobytes()
                results.append(tensor.SerializeToString())
            return results

    executor = Executor()
    cpp._set_model_executor(executor)
    model = pb.ModelProto.FromString(args.input.read_bytes())
    if args.width is not None and args.height is not None:
        for value in model.graph.input:
            for dimension, size in zip(value.type.tensor_type.shape.dim,
                                       [1, 3, args.height, args.width]):
                dimension.dim_value = size
    if 'onnx' in sys.modules:
        raise RuntimeError('ONNX C++ isolation was violated')
    # An empty skip list enables optimizers; None disables all optimizer passes.
    simplified = cpp.simplify(model.SerializeToString(), [], True, True, int(1.5 * 2**30))
    args.output.write_bytes(simplified)


if __name__ == '__main__':
    main()
