"""Static, single-engine TensorRT 8.5 runner with persistent CUDA buffers."""
from __future__ import annotations


class SingleEngineRunner:
    def __init__(self, engine_path):
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError('Unable to deserialize this Orin TensorRT engine')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Unable to allocate TensorRT execution context')
        dtype_map = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                     trt.DataType.INT32: torch.int32, trt.DataType.INT8: torch.int8,
                     trt.DataType.BOOL: torch.bool}
        self.buffers = {}
        self.input_names = []
        self.output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(dimension <= 0 for dimension in shape):
                raise ValueError('Only fixed-shape engines are supported')
            dtype = self.engine.get_tensor_dtype(name)
            if dtype not in dtype_map:
                raise ValueError('Unsupported TRT tensor type: %s' % dtype)
            self.buffers[name] = torch.empty(shape, device='cuda', dtype=dtype_map[dtype])
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if set(self.input_names) != {'left_image', 'right_image'}:
            raise ValueError('Unexpected model input names')
        if self.output_names != ['disparity']:
            raise ValueError('Unexpected model output names')

    def __call__(self, left, right):
        for name, tensor in (('left_image', left), ('right_image', right)):
            target = self.buffers[name]
            if tensor.shape != target.shape or tensor.device != target.device:
                raise ValueError('Input does not match the static CUDA engine binding')
            target.copy_(tensor)
        if not self.context.execute_async_v3(self.torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        # Returned storage is reused on the next call; caller synchronizes/copies
        # before retaining it. One instance must not run on concurrent streams.
        return self.buffers['disparity']
