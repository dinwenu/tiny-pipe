# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import os
import shutil
from enum import Enum
import torch
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed.runtime.swap_tensor.constants import *
from utils import swap_in_tensors_sync_2, swap_out_tensors_sync_2, MIN_AIO_BYTES, AIO_ALIGNED_BYTES, SwapBufferPool

import time

from copy import deepcopy
from collections import deque

def print_rank_0(message, debug=False, force=False):
    if dist.get_rank() == 0 and (debug or force):
        print(message)


class PartitionedParamStatus(Enum):
    # Partitioned parameters are present and ready for use
    AVAILABLE = 1

    # partitioned params are in some non-memory device
    NOT_AVAILABLE = 2

    # partitioned params are being read from some non-memory device.
    INFLIGHT = 3

class PinnedData:
    def __init__(self, pinned_buffer, pinned_layer):
        self.pinned_buffer = pinned_buffer
        self.pinned_layer = pinned_layer

class AsyncParameterSwapper(object):
    def __init__(self, ds_config, special_layer_id_in_rank, layer_count, model_dtype, rank, transformer_layer_info=None, special_layers_info=None, transformer_layer=None, special_layers_in_rank=None):
        aio_op = AsyncIOBuilder().load(verbose=False)
        self.aio_handle = aio_op.aio_handle
        self.dtype = model_dtype
        self.layer_count = layer_count
        self.rank = rank

        if transformer_layer_info is not None:
            self.param_idx_in_layer_to_numel = transformer_layer_info['param_idx_to_numel']
            self.param_idx_to_start_pos = transformer_layer_info['param_idx_to_start_pos']
            self.param_idx_in_layer_to_shape = transformer_layer_info['param_idx_to_shape']
            self.layer_size = transformer_layer_info['layer_size']
            self.layer_aligned_size = transformer_layer_info['layer_aligned_size']
        elif transformer_layer is not None:
            self.param_idx_in_layer_to_numel = {}
            self.param_idx_to_start_pos = {}
            self.param_idx_in_layer_to_shape = {}
            
        self.param_idx_in_special_layer_to_numel = {}
        self.param_idx_in_special_layer_to_start_pos = {}
        self.param_idx_in_special_layer_to_shape = {}
        self.special_layer_size = {}
        self.special_layer_aligned_size = {}
        self.special_layer_start_pos = {}
        if special_layers_info is not None:
            start_pos = 0
            for layer_id, layer_info in special_layers_info.items():
                if layer_id in special_layer_id_in_rank:
                    self.param_idx_in_special_layer_to_numel[layer_id] = layer_info['param_idx_to_numel']
                    self.param_idx_in_special_layer_to_start_pos[layer_id] = layer_info['param_idx_to_start_pos']
                    self.param_idx_in_special_layer_to_shape[layer_id] = layer_info['param_idx_to_shape']
                    self.special_layer_size[layer_id] = layer_info['layer_size']
                    self.special_layer_aligned_size[layer_id] = layer_info['layer_aligned_size']
                    self.special_layer_start_pos[layer_id] = start_pos
                    start_pos += self.special_layer_aligned_size[layer_id]
        elif special_layers_in_rank is not None:
            for layer_id, layer in special_layers_in_rank:
                self.param_idx_in_special_layer_to_numel[layer_id] = {}
                self.param_idx_in_special_layer_to_start_pos[layer_id] = {}
                self.param_idx_in_special_layer_to_shape[layer_id] = {}
                self.special_layer_size[layer_id] = 0
                self.special_layer_aligned_size[layer_id] = 0
                self.special_layer_start_pos[layer_id] = 0

        #set swap buffers, create aio handles
        self._configure_aio(ds_config)

        if transformer_layer_info is None and transformer_layer is not None:
            self.layer_size, self.layer_aligned_size = self.cal_transformer_layer_size(transformer_layer)
        if special_layers_info is None and special_layers_in_rank is not None:
            self.cal_special_layer_size(special_layers_in_rank)

        self.total_special_layer_size = sum(self.special_layer_aligned_size.values())

        self.pinned_transformer_layers_size = self.layer_aligned_size * layer_count

        # self.pinned_buffer = get_accelerator().pin_memory(torch.empty(int(self.pinned_transformer_layers_size+self.total_special_layer_size),
        #                                                         dtype=self.dtype,
        #                                                         requires_grad=False),
        #                                             align_bytes=0)
        self.pinned_buffer = torch.empty(int(self.pinned_transformer_layers_size+self.total_special_layer_size),
                                                                dtype=self.dtype,
                                                                requires_grad=False).pin_memory()

        self.cpu_cache_queue = deque(maxlen=self.layer_count)
        self.create_pinned_layers(transformer_layer)
        self.special_cpu_queue = self.create_special_pinned_layers(special_layers_in_rank)

        #mapping from param id to path
        self.id_to_path = {}

        #mapping from pram_id to buffer id
        self.param_id_to_buffer_id = {}

        # mapping from param_id to swap buffer
        self.param_id_to_swap_buffer = {}

        #number of elements in the param
        self.param_id_to_numel = {}

        self.pending_writes = 0
        self.pending_reads = 0

        #keep track of async swap in params and buffers
        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0

        #keep track of available params
        self.available_params = set()
        self.available_numel = 0

        # for swapping out from partitioned fp32 params
        self.partitioned_swap_buffer = None
        self.partitioned_swap_pool = None

        self.invalid_buffer = torch.tensor(1).half()

        # if self.rank == 0:
        #     exclude_list = ['aio_read_handle', 'aio_write_handle', 'buffers']
        #     print_object(obj=self, name='AsyncParameterSwapper', exclude_list=exclude_list)

    def available_swap_in_buffers(self):
        return len(self.available_buffer_ids)

    def _configure_aio(self, ds_config):
        self.swap_config = ds_config.zero_config.offload_param
        torch_dtype_string = str(self.dtype).split(".")[1] # torch.float32
        self.swap_folder = os.path.join(self.swap_config.nvme_path, 'zero_stage_3', f'{torch_dtype_string}params')
        clean_swap_folder = False  # 设置为True时清理，False时保留
        if clean_swap_folder:
            shutil.rmtree(self.swap_folder, ignore_errors=True)
        os.makedirs(self.swap_folder, exist_ok=True)

        self.swap_element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.aio_config = ds_config.aio_config
        print(f"rank:{self.rank}, aioconfig:{self.aio_config}")

        # Read/Write alignment for each thread during Intra-request parallelism
        # 
        self.min_aio_bytes = max(MIN_AIO_BYTES, self.aio_config[AIO_BLOCK_SIZE]) # 1024**2, 1048576 (都是1MB，相等)
        self.aligned_bytes = AIO_ALIGNED_BYTES * self.aio_config[AIO_THREAD_COUNT] # 1024 × 1
        self.numel_alignment = self.aligned_bytes // self.swap_element_size

        # Number of buffers in buffer pool for parameter offloading to NVMe. Default 5
        self.param_buffer_count = self.swap_config.buffer_count

        # self.nvme_layers = nvme_layers
        # self.layer_id_param_idx_to_path = {}
        # for layer_id in self.nvme_layers:
        #     self.layer_id_param_idx_to_path[layer_id] = {}

        # 📍2
        self.layer_id_to_path = {}

        self.aio_read_handle = self.aio_handle(self.aio_config[AIO_BLOCK_SIZE], self.aio_config[AIO_QUEUE_DEPTH],
                                               self.aio_config[AIO_SINGLE_SUBMIT], self.aio_config[AIO_OVERLAP_EVENTS],
                                               self.aio_config[AIO_THREAD_COUNT])

        self.aio_write_handle = self.aio_handle(self.aio_config[AIO_BLOCK_SIZE], self.aio_config[AIO_QUEUE_DEPTH],
                                                self.aio_config[AIO_SINGLE_SUBMIT],
                                                self.aio_config[AIO_OVERLAP_EVENTS], self.aio_config[AIO_THREAD_COUNT])

        # exit(0)
        self.swap_out_params = []

    def cal_special_layer_size(self, special_layers_in_rank):
        start_pos = 0
        for layer_id, layer in special_layers_in_rank:
            total_size = 0
            global_param_idx = 0
            for m in layer.modules():
                for key, param in m._parameters.items():
                    if param is not None:
                        param_size = param.numel()
                        self.param_idx_in_special_layer_to_numel[layer_id][global_param_idx] = param_size
                        self.param_idx_in_special_layer_to_start_pos[layer_id][global_param_idx] = total_size
                        self.param_idx_in_special_layer_to_shape[layer_id][global_param_idx] = param.shape
                        total_size += param_size
                        global_param_idx += 1
                for key, buf in m._buffers.items():
                    if buf is not None:
                        buf_size = buf.numel()
                        self.param_idx_in_special_layer_to_numel[layer_id][global_param_idx] = buf_size
                        self.param_idx_in_special_layer_to_start_pos[layer_id][global_param_idx] = total_size
                        self.param_idx_in_special_layer_to_shape[layer_id][global_param_idx] = buf.shape
                        total_size += buf_size
                        global_param_idx += 1
            self.special_layer_size[layer_id] = total_size
            self.special_layer_aligned_size[layer_id] = self._io_aligned_numel(total_size)
            self.special_layer_start_pos[layer_id] = start_pos
            start_pos += self.special_layer_aligned_size[layer_id]

    def cal_transformer_layer_size(self, transformer_layer):
        total_size = 0
        global_param_idx = 0
        for m in transformer_layer.modules():

            for key, param in m._parameters.items():
                if param is not None:
                    param_size = param.numel()
                    self.param_idx_in_layer_to_numel[global_param_idx] = param_size
                    self.param_idx_to_start_pos[global_param_idx] = total_size
                    self.param_idx_in_layer_to_shape[global_param_idx] = param.shape
                    total_size += param_size
                    global_param_idx += 1

            for key, buf in m._buffers.items():
                if buf is not None:
                    buf_size = buf.numel()
                    self.param_idx_in_layer_to_numel[global_param_idx] = buf_size
                    self.param_idx_to_start_pos[global_param_idx] = total_size
                    self.param_idx_in_layer_to_shape[global_param_idx] = buf.shape
                    total_size += buf_size
                    global_param_idx += 1

        aligned_total_size = self._io_aligned_numel(total_size)
        
        return total_size, aligned_total_size

    def allocate_and_return_buffer_for_special_layer_param(self, pinned_buffer, layer_id, global_param_idx):
        start_pos = self.param_idx_in_special_layer_to_start_pos[layer_id][global_param_idx]
        param_size = self.param_idx_in_special_layer_to_numel[layer_id][global_param_idx]
        narrowed_param = pinned_buffer.narrow(0, start_pos, param_size)
        return narrowed_param
    
    def create_special_pinned_layers(self, special_layers_in_rank):
        special_cpu_queue = {}
        for layer_id, layer in special_layers_in_rank:
            pinned_layer = deepcopy(layer)
            pinned_buffer = self.pinned_buffer.narrow(0, self.pinned_transformer_layers_size+self.special_layer_start_pos[layer_id], self.special_layer_aligned_size[layer_id])
            global_param_idx = 0
            
            for m in pinned_layer.modules():
                for key, param in m._parameters.items():
                    if param is not None:
                        assert param.grad is None, "convert to pinned model requires no grad in input model"
                        narrowed_param = self.allocate_and_return_buffer_for_special_layer_param(pinned_buffer, layer_id, global_param_idx)
                        param.data = narrowed_param.view(self.param_idx_in_special_layer_to_shape[layer_id][global_param_idx])
                        global_param_idx += 1
                for key, buf in m._buffers.items():
                    if buf is not None:
                        assert not buf.requires_grad
                        narrowed_buf = self.allocate_and_return_buffer_for_special_layer_param(pinned_buffer, layer_id, global_param_idx)
                        pinned_layer._buffers[key] = narrowed_buf.view(self.param_idx_in_special_layer_to_shape[layer_id][global_param_idx])
                        global_param_idx += 1
                        assert not pinned_layer._buffers[key].requires_grad
            
            special_cpu_queue[layer_id] = PinnedData(pinned_buffer, pinned_layer)
        return special_cpu_queue
    
    def get_special_pinned_data(self, layer_id):
        return self.special_cpu_queue[layer_id]

    def allocte_and_return_buffer_for_param(self, pinned_buffer, global_param_idx):
        start_pos = self.param_idx_to_start_pos[global_param_idx]
        param_size = self.param_idx_in_layer_to_numel[global_param_idx]
        narrowed_param = pinned_buffer.narrow(0, start_pos, param_size)
        return narrowed_param

    def create_pinned_layers(self, transformer_layer):
        for idx in range(self.layer_count):
            pinned_layer = deepcopy(transformer_layer)
            pinned_buffer = self.pinned_buffer.narrow(0, idx * self.layer_aligned_size, self.layer_aligned_size)
            global_param_idx = 0
            for m in pinned_layer.modules():
                for key, param in m._parameters.items():
                    if param is not None:
                        narrowed_param = self.allocte_and_return_buffer_for_param(pinned_buffer, global_param_idx)
                        param.data = narrowed_param.view(self.param_idx_in_layer_to_shape[global_param_idx])
                        global_param_idx += 1
                for key, buf in m._buffers.items():
                    if buf is not None:
                        narrowed_buf = self.allocte_and_return_buffer_for_param(pinned_buffer, global_param_idx)
                        buf.data = narrowed_buf.view(self.param_idx_in_layer_to_shape[global_param_idx])
                        global_param_idx += 1
            self.cpu_cache_queue.append(PinnedData(pinned_buffer, pinned_layer))

    def get_pinned_data(self):
        return self.cpu_cache_queue.pop()
    
    def return_pinned_data(self, pinned_data):
        self.cpu_cache_queue.append(pinned_data)

    #Check if partitioned param or numel in a tensor is swappable or not
    def swappable_tensor(self, param=None, numel=None):
        if param is not None:
            assert numel is None, "Both parma and numel cannot be provided"
            numel = param.ds_tensor.ds_numel
        if numel is not None:
            return self.min_aio_bytes <= numel * self.swap_element_size
        assert False, "Either param or numel must be provided"

    # worker7
    def _get_transformer_layer_swap_paths(self, layer_id, must_exist=False):
        if layer_id in self.layer_id_to_path.keys():
            param_path = self.layer_id_to_path[layer_id]
        else:
            assert not must_exist, f"Path for layer id {layer_id} does not exist"
            # param_path = os.path.join(self.swap_folder, f'{layer_id}_{param_idx}_param.tensor.swp')
            param_path = os.path.join(
                self.swap_config.nvme_path,
                'zero_stage_3',
                f'{str(self.dtype).split(".")[1]}params',
                f'layer_{layer_id}.tensor.swp'
            )
            self.layer_id_to_path[layer_id] = param_path
        return param_path

    def _track_numel(self, params):
        for param in params:
            assert param.ds_tensor is not None, "Partitioned tensor is None"
            self.param_id_to_numel[param.ds_id] = param.ds_tensor.ds_numel

    def swap_out_transformer_layer_sync(self, layer_id, pinned_tensor):
        swap_out_path = self._get_transformer_layer_swap_paths(layer_id)
        swap_out_tensors_sync_2(self.aio_write_handle, [pinned_tensor], [swap_out_path])

    # worker_7
    def swap_in_transformer_layer_sync(self, layer_id, pinned_tensor):
        swap_in_path = self._get_transformer_layer_swap_paths(layer_id)
        swap_in_tensors_sync_2(self.aio_read_handle, [pinned_tensor], [swap_in_path])

    def _io_aligned_numel(self, numel):
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)

    def _is_io_aligned(self, numel):
        return (numel % self.numel_alignment) == 0

    def clear_pinned_tensors(self):
        """清理所有 pinned tensor 以减少内存占用 (worker-7 版本)"""

        if hasattr(self, 'pinned_buffer') and self.pinned_buffer is not None:
            del self.pinned_buffer
            self.pinned_buffer = None

        if hasattr(self, 'cpu_cache_queue') and self.cpu_cache_queue is not None:
            while self.cpu_cache_queue:
                pinned_data = self.cpu_cache_queue.pop()
                self._clear_pinned_data(pinned_data)

        if hasattr(self, 'special_cpu_queue') and self.special_cpu_queue is not None:
            for layer_id in list(self.special_cpu_queue.keys()):
                pinned_data = self.special_cpu_queue[layer_id]
                self._clear_pinned_data(pinned_data)
                del self.special_cpu_queue[layer_id]

        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _clear_pinned_data(self, pinned_data):
        if pinned_data is None:
            return
        if hasattr(pinned_data, 'pinned_layer') and pinned_data.pinned_layer is not None:
            self._clear_layer_tensors(pinned_data.pinned_layer)
            pinned_data.pinned_layer = None
        if hasattr(pinned_data, 'pinned_buffer'):
            pinned_data.pinned_buffer = None

    def _clear_layer_tensors(self, layer):
        if layer is None:
            return
        for module in layer.modules():
            for n in list(module._parameters.keys()):
                module._parameters[n] = None
            for n in list(module._buffers.keys()):
                module._buffers[n] = None
