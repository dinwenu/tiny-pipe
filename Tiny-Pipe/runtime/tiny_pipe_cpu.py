# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division, print_function
import argparse
import os
import sys
import json
import numpy as np
import gc
from collections import OrderedDict as ODict
from copy import deepcopy

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
import torch.distributed as dist

import torch.cuda.profiler as cuda_profiler 
from torch.cuda.nvtx import mark as nvtx_mark 
from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 
from viewer.probe_cuda_mem import ProbeCudaMem 
from time import perf_counter as pc

import seeding, checker

from profiler import realize_X, realize_dX, realize_T
from task_data_struct import Medium, vTask
import shared_optim_cpu, local_model_gpu, msg_stash_x, ubatchsize_converter, swp_x, p2p
from decompose import decompose_minibatch
from tensor_helper import * 
from utils import *


import datetime

from typing import List

from collections import deque
from concurrent.futures import ThreadPoolExecutor

ray_get = lambda futs: [fut.result() for fut in futs]

# =====================================================
# === four kinds of hooks for forward and backward  ===
# =====================================================
def _append_loading_futs(_layer, futs, fwd=True):
    if fwd:
        if hasattr(_layer, "_gl_futs_loading_at_fwd"):
            _layer._gl_futs_loading_at_fwd += futs
        else:
            _layer._gl_futs_loading_at_fwd = futs
    else:
        if hasattr(_layer, "_gl_futs_loading_at_bwd"):
            _layer._gl_futs_loading_at_bwd += futs
        else:
            _layer._gl_futs_loading_at_bwd = futs


def _append_offloading_futs(_layer, futs, fwd=True):
    if fwd:
        if hasattr(_layer, "_gl_futs_offloading_at_fwd"):
            _layer._gl_futs_offloading_at_fwd += futs
        else:
            _layer._gl_futs_offloading_at_fwd = futs
    else:
        if hasattr(_layer, "_gl_futs_offloading_at_bwd"):
            _layer._gl_futs_offloading_at_bwd += futs
        else:
            _layer._gl_futs_offloading_at_bwd = futs


# 
def _layer_waiting_futs(_layer, nvprof):
    debug_info = f"\t\trank:{_layer._gl_handler.rank}, layer{_layer._gl_layer_num}, \n"
    # print(f"\t\trank:{_layer._gl_handler.rank}, layer{_layer._gl_layer_num}, \n", flush=True)

    if hasattr(_layer, "_gl_futs_loading_at_fwd"):
        ray_get(_layer._gl_futs_loading_at_fwd)
        del _layer._gl_futs_loading_at_fwd

    if hasattr(_layer, "_gl_futs_offloading_at_fwd"):
        ray_get(_layer._gl_futs_offloading_at_fwd)
        del _layer._gl_futs_offloading_at_fwd
    
    if hasattr(_layer, "_gl_futs_loading_at_bwd"):
        ray_get(_layer._gl_futs_loading_at_bwd)
        del _layer._gl_futs_loading_at_bwd

    if hasattr(_layer, "_gl_futs_offloading_at_bwd"):
        ray_get(_layer._gl_futs_offloading_at_bwd)
        del _layer._gl_futs_offloading_at_bwd

    return debug_info

def _forward_pre_hook(module, input):

    _current_layer = module
    _handler = _current_layer._gl_handler
    _layers = _current_layer._gl_layers
    _requires_grad = _handler.get_requires_grad()
    _is_first_ubatch = _handler.is_first_ubatch
    _is_last_ubatch = _handler.is_last_ubatch
    _layers_in_gpu = _handler.layers_in_gpu
    nvprof = _handler.nvprof

    _cuda_cache_queue = _handler.cuda_cache_queue
    _special_cuda_queue = _handler.special_cuda_queue

    _prefetch_model_handler = _handler.prefetch_model_handler
    _already_get = _handler.already_get

    if _is_first_ubatch and not _already_get[_current_layer._gl_layer_num]:
        layer_id = _prefetch_model_handler.get()
        _already_get[_current_layer._gl_layer_num] = True
        assert layer_id == _current_layer._gl_layer_num, f"rank:{_handler.rank}, layer_id:{layer_id} != _current_layer._gl_layer_num:{_current_layer._gl_layer_num}"

    if _is_first_ubatch and _handler.should_stay_in_gpu[_current_layer._gl_layer_num]:
        _handler.local_model[_current_layer._gl_layer_num].set_param_requiresgrad()

    if _is_first_ubatch and not _requires_grad and _current_layer._gl_which_layer_to_cuda_pre_fwd_required:
        if nvprof: nvtx_range_push("L{} fwd pre hook call cost".format(_current_layer._gl_layer_num)) 
        _which_layer_idx_to_cuda = _current_layer._gl_which_layer_idx_to_cuda_pre_fwd
        _which_layer_to_cuda = _current_layer._gl_which_layer_to_cuda_pre_fwd

        prefetched_layer_vt = _handler.layer_idx_to_vt[_which_layer_idx_to_cuda]

        if _which_layer_to_cuda not in _layers_in_gpu:

            if _which_layer_to_cuda in _special_cuda_queue.keys():
                _cache_unit = _special_cuda_queue[_which_layer_to_cuda]
            else:
                _cache_unit = _cuda_cache_queue.pop()
            # futs = [_handler._layer_to_cuda_remote(_which_layer_to_cuda, prefetched_layer_vt, _cache_unit)]
            _prefetch_model_handler.iput(_which_layer_to_cuda, prefetched_layer_vt, _cache_unit)


            _layers_in_gpu.append(_which_layer_to_cuda)
        else:
            debug_info += f"\tlayer{_which_layer_to_cuda} already in GPU \n"
            _handler.should_stay_in_gpu[_which_layer_to_cuda] = True

        if nvprof: nvtx_range_pop()

    elif _is_first_ubatch and _requires_grad and _current_layer._gl_which_layer_to_cuda_pre_recompute_required:
        if nvprof: nvtx_range_push("L{} BWD pre hook call cost".format(_current_layer._gl_layer_num)) 

        _which_layer_idx_to_cuda = _current_layer._gl_which_layer_idx_to_cuda_pre_recompute
        _which_layer_to_cuda = _current_layer._gl_which_layer_to_cuda_pre_recompute

        if _which_layer_to_cuda not in _layers_in_gpu:

            prefetched_layer_vt = _handler.layer_idx_to_vt[_which_layer_idx_to_cuda]

            if _which_layer_to_cuda in _special_cuda_queue.keys():
                _cache_unit = _special_cuda_queue[_which_layer_to_cuda]
            else:
                _cache_unit = _cuda_cache_queue.pop()
            # futs = [_handler._layer_to_cuda_remote(_which_layer_to_cuda, prefetched_layer_vt, _cache_unit)]
            _prefetch_model_handler.iput(_which_layer_to_cuda, prefetched_layer_vt, _cache_unit)
            # _append_loading_futs(_layers[_which_layer_to_cuda], futs)
            _layers_in_gpu.append(_which_layer_to_cuda)
        else:
            debug_info += f"\tlayer{_which_layer_to_cuda} already in GPU \n"
            _handler.should_stay_in_gpu[_which_layer_to_cuda] = True

        if nvprof: nvtx_range_pop()

    # _handler.running_idx += 1
    if _is_first_ubatch:
        print(debug_info, flush=True)

def _forward_post_hook(module, input, output):

    _current_layer = module
    _handler = _current_layer._gl_handler
    _layers = _current_layer._gl_layers
    _requires_grad = _handler.get_requires_grad()
    _is_last_ubatch = _handler.is_last_ubatch
    _layers_in_gpu = _handler.layers_in_gpu
    nvprof = _handler.nvprof

    _cuda_cache_queue = _handler.cuda_cache_queue
    _special_cuda_queue = _handler.special_cuda_queue
    _local_model = _handler.local_model
    _already_get = _handler.already_get


    if _is_last_ubatch and not _requires_grad and _current_layer._gl_which_layer_to_cpu_post_fwd_required:
        _which_layer_to_cpu = _current_layer._gl_layer_num

        if not _handler.should_stay_in_gpu[_which_layer_to_cpu]:

            debug_info += f"\trank:{_handler.rank}, 第{_current_layer._gl_layer_num}层的FWD post hook开始卸载 \n"

            if _which_layer_to_cpu in _special_cuda_queue.keys():
                _local_model[_which_layer_to_cpu].return_pinned_data()
            else:
                _local_model[_which_layer_to_cpu].append_cuda_cache_unit()
                _local_model[_which_layer_to_cpu].return_pinned_data()



            _layers_in_gpu.remove(_which_layer_to_cpu)
            _handler.should_stay_in_gpu[_which_layer_to_cpu] = False
            _already_get[_which_layer_to_cpu] = False
        else:
            # _which_layer_to_cuda = _current_layer._gl_which_layer_to_cuda_pre_fwd
            debug_info += f"\trank:{_handler.rank}, layer{_which_layer_to_cpu} should stay in GPU, do not offload \n"

    if _is_last_ubatch:
        print(debug_info, flush=True)

def _backward_pre_hook(module, grad_in, grad_out):

    _current_layer = module
    _handler = _current_layer._gl_handler
    _layers = _current_layer._gl_bwd_layers
    _requires_grad = _handler.get_requires_grad()
    _is_last_ubatch = _handler.is_last_ubatch
    nvprof = _handler.nvprof

    _cuda_cache_unit = _current_layer._gl_cuda_cache_unit
    _special_cuda_queue = _current_layer._gl_special_cuda_queue

    if _is_last_ubatch and _current_layer._gl_which_layer_to_cuda_pre_bwd_is_required:
        if nvprof: nvtx_range_push("L{} BWD pre hook call cost".format(_current_layer._gl_layer_num)) 

        _which_layer_to_cuda = _current_layer._gl_which_layer_to_cuda_pre_bwd

        assert _requires_grad, "requires_grad should be True"
        _suc_vt = _handler.layer_id_to_vt[_which_layer_to_cuda]['BWD']

        print(f"rank:{_handler.rank}, BWD: layer{module._gl_layer_num}预取{_which_layer_to_cuda}")

        print(f"\trank:{_handler.rank}, BWD: 要预取的layer{_which_layer_to_cuda}的fut挂在layer{_which_layer_to_cuda}上")
        futs = [_handler.layer_to_cuda(_which_layer_to_cuda, _suc_vt)]
        _append_loading_futs(_layers[_which_layer_to_cuda], futs, fwd=False)
        
        if nvprof: nvtx_range_pop()

        

def _backward_post_hook(module, grad_in, grad_out):
    
    _current_layer = module
    _handler = _current_layer._gl_handler
    _layers = _current_layer._gl_layers
    _requires_grad = _handler.get_requires_grad()
    _is_last_ubatch = _handler.is_last_ubatch
    _layers_in_gpu = _handler.layers_in_gpu
    _update_handler = _handler.update_handler
    nvprof = _handler.nvprof

    _already_get = _handler.already_get

    _current_layer = _layers[_current_layer._gl_layer_num]
    only_has_one_layer = hasattr(_current_layer, 'only_has_one_layer')

    if _is_last_ubatch and _current_layer._gl_which_layer_to_cpu_post_bwd_required and not only_has_one_layer:
        if nvprof: nvtx_range_push("L{} BWD post hook call cost".format(_current_layer._gl_layer_num)) 
        _current_layer = _layers[_current_layer._gl_layer_num+1]
        
        _which_layer_to_cpu = _current_layer._gl_layer_num
        print(f"\tto delete layer{_which_layer_to_cpu}")

        assert _requires_grad, "requires_grad should be True"
        _vt = _handler.layer_id_to_vt[_which_layer_to_cpu]['BWD']
        
        _update_handler.iput(_vt, _which_layer_to_cpu)

        _handler.should_stay_in_gpu[_which_layer_to_cpu] = False
        if _which_layer_to_cpu in _layers_in_gpu:
            _layers_in_gpu.remove(_which_layer_to_cpu)
        _already_get[_which_layer_to_cpu] = False
        if nvprof: nvtx_range_pop()

# =====================================================
# ===                 hooks ending                  ===
# =====================================================


class GL_PretrainHanlder:
    def __init__(
        self,
        args,
        local_model,
        config,
        vts,
        prefetch_model_handler,
        default_stream,
        rank,
        update_handler,
        cuda_cache_queue,
        special_cuda_queue,
        nvprof,
    ):


        self.CONFIGS = config
        self.vts = vts
        self.is_first_ubatch = False
        self.is_last_batch = False
        self.default_stream = default_stream
        self.rank = rank
        self.nvprof = nvprof
        self.verbose = args.verbose
        self.gl_window_size = args.gl_window_size
        print(f"rank:{self.rank}, ***********************gl_window_size:{self.gl_window_size}")
        self.update_handler = update_handler
        # self.running_idx = args.gl_window_size
        self.cuda_cache_queue = cuda_cache_queue
        self.special_cuda_queue = special_cuda_queue

        self.fwd_vts_in_rank = [vt for vt in self.vts if vt.type == "FWD"]
        self.num_fwd_layers = sum([len(vt.layers) for vt in self.fwd_vts_in_rank])
        self.fwd_layers = [layer for vt in self.fwd_vts_in_rank for layer in vt.layers]

        self.bwd_vts_in_rank = [vt for vt in self.vts if vt.type == "BWD"]
        self.num_bwd_layers = sum([len(vt.layers) for vt in self.bwd_vts_in_rank])
        self.bwd_layers = [layer for vt in self.bwd_vts_in_rank for layer in vt.layers]
        # for bwd_vt in self.bwd_vts_in_rank:
        #     print(f"rank:{self.rank}, bwd_vt:{bwd_vt.layers}")

        self.vts_in_rank = self.fwd_vts_in_rank + self.bwd_vts_in_rank
        self.layers_id_in_rank = [layer for vt in self.vts_in_rank for layer in vt.layers]
        print(f"rank:{self.rank}, layers_id_in_rank:{self.layers_id_in_rank}")
        self.num_layers = len(self.layers_id_in_rank)

        self.layer_id_to_vt = {}
        for j, vt in enumerate(self.vts_in_rank):
            for layer_id in vt.layers:
                if layer_id not in self.layer_id_to_vt:
                    self.layer_id_to_vt[layer_id] = {'FWD': None, 'BWD': None}
                if vt.type == 'FWD':
                    self.layer_id_to_vt[layer_id]['FWD'] = vt
                elif vt.type == 'BWD':
                    self.layer_id_to_vt[layer_id]['BWD'] = vt

        self.layer_idx_to_vt = {}
        layer_idx = 0
        for j, vt in enumerate(self.vts_in_rank):
            for layer_id in vt.layers:
                self.layer_idx_to_vt[layer_idx] = vt
                layer_idx += 1
        assert layer_idx == len(self.layers_id_in_rank), "layer_idx should be equal to len(self.layers_id_in_rank)"



        last_bwd_vt = self.bwd_vts_in_rank[-1]
        last_bwd_vt.is_last_bwd_vt = True

        # self.bwd_layers.sort()
        # print(f"rank:{self.rank}, sorted bwd_layers:{self.bwd_layers}")

        self.layers_in_gpu = []
        self.should_stay_in_gpu = {}
        for layer_id in set(self.layers_id_in_rank):
            self.should_stay_in_gpu[layer_id] = False


        self.executor = ThreadPoolExecutor(max_workers=args.gl_ray_max_concurrency)

        if not isinstance(local_model, list):
            local_model = [local_model]
        self.local_model: List[local_model_gpu.LocalModelGPU] = local_model
        self.layers = []
        for layer in self.local_model:
            self.layers.append(layer.model)
        self.prefetch_model_handler = prefetch_model_handler

        for vt in self.bwd_vts_in_rank:
            only_has_one_layer = len(vt.layers) == 1
            if only_has_one_layer:
                first_bwd_layer_id = vt.layers[0]
                # print(f"rank:{self.rank}, first_bwd_layer_id:{first_bwd_layer_id}")
                first_bwd_layer = self._get_layer(first_bwd_layer_id)
                first_bwd_layer.only_has_one_layer = True

        self.is_last_ubatch = False
        self.requires_grad = False

        self.already_get = {}
        for layer_id in set(self.layers_id_in_rank):
            self.already_get[layer_id] = False

    def set_requires_grad(self, requires_grad: bool):
        self.requires_grad = requires_grad

    def get_requires_grad(self):
        return self.requires_grad
    
    def set_is_first_ubatch(self):
        self.is_first_ubatch = True

    def reset_is_first_ubatch(self):
        self.is_first_ubatch = False

    def set_is_last_batch(self):
        self.is_last_ubatch = True

    def reset_is_last_batch(self):
        self.is_last_ubatch = False

    def print_model(self):
        print(self.model)
        # print(self.model[0].module.module.language_model.encoder)

    # =====================================================
    # == basic actions for hooks of forward and backward ==
    # =====================================================
    def layer_to_cpu_bwd_remote(self, *args, **kwargs):
        return self.executor.submit(self.layer_to_cpu_bwd, *args, **kwargs)

    @torch.no_grad()
    def layer_to_cpu_bwd(self, layer_num, vt):
        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            if self.CONFIGS["opt_offld"]:        
                if self.nvprof: nvtx_range_push("task{}(L{}) SwapOut(dW,B)".format(vt.idx, layer_num)) 
                if vt.Out['dW'][layer_num].medium == "LOC" and vt.Out['B'][layer_num].medium == "SHM":
                    if self.CONFIGS["mode"]=='vPP' or (self.CONFIGS["mode"]=='vDP' and self.rank==0):
                        self.local_model[layer_num].swapout_grad_blocking()
                        self.local_model[layer_num].swapout_buf_blocking()
                else:
                    raise NotImplementedError
                if self.verbose: print_gpu_mem(self.rank, vt, "SwapOut'ed(dW,B)")    
                if self.nvprof: nvtx_range_pop() 
                ### Delete model {W,dW,B} 
                if self.nvprof: nvtx_range_push("task{}(L{}) BWD-Del(W,dW,B)".format(vt.idx, layer_num)) 
                if vt.Out['dW'][layer_num].medium == "LOC" and vt.Out['B'][layer_num].medium == "SHM":
                    self.local_model[layer_num].zero_grad()
                else: # 'B' == PIN
                    raise NotImplementedError
                # gc.collect()
                if self.verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                if self.nvprof: nvtx_range_pop() 

                _stream.synchronize()
            else:
                raise ValueError("GPU Optimizer Underdevelopment.")
        pass

    def layer_to_cpu_fwd_remote(self, *args, **kwargs):
        return self.layer_to_cpu_fwd(*args, **kwargs)

    @torch.no_grad()
    def layer_to_cpu_fwd(self, layer_num, vt=None):
        self.local_model[layer_num].append_cuda_cache_unit()

        return self._layer_to_cuda(*args, **kwargs)

    def _layer_to_cpu_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_to_cpu, *args, **kwargs)

    def _layer_move_save_for_backward_to_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_move_save_for_backward_to, *args, **kwargs)

    def _layer_to_cpu_and_gather_grads_and_optimizer_update_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_to_cpu_and_gather_grads_and_optimizer_update, *args, **kwargs)

    def _layer_reset_save_for_backward_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_reset_save_for_backward, *args, **kwargs)

    def _layer_gather_grads_and_optimizer_update_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_gather_grads_and_optimizer_update, *args, **kwargs)


    def _model_to(self, device):
        _models = self.model if isinstance(self.model, list) else [self.model]

        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            if _is_cpu(device):
                for model_module in _models:
                    model_module.cpu()
            else:
                for model_module in _models:
                    model_module.cuda(device)

            _stream.synchronize()

    # def _assert_language_model(self):
    #     _models = self.model if isinstance(self.model, list) else [self.model]

    #     # 
    #     if self._language_model is None:
    #         for name, module in _models[0].named_modules():
    #             if name.endswith("language_model"):
    #                 self._language_model = module
    #                 break

    #     assert (
    #         self._language_model is not None
    #     ), "--- _assert_language_model training.py --- language_model not found !"

    def _get_layer(self, layer_num):
        return self.layers[layer_num]
    
    def _get_local_model(self, layer_num):
        return self.local_model[layer_num]

    def _get_layers(self):
        return self.layers

    def _layer_to(self, layer_num, device, non_blocking=True):
        with torch.no_grad():
            _layer = self._get_layer(layer_num)

            _stream = torch.cuda.Stream()
            with torch.cuda.stream(_stream):
                if _is_cpu(device):
                    _layer.cpu(non_blocking=non_blocking)
                else:
                    _layer.cuda(device, non_blocking=non_blocking)

                _stream.synchronize()
        pass

    @torch.no_grad()
    def _layer_to_cuda(self, layer_num, vt_of_layer_num=None, cache_unit=None):
        self.prefetch_model_handler.iput(layer_num, vt_of_layer_num, cache_unit)
        pass

    # NOTE
    @torch.no_grad()
    def _layer_to_cpu(self, layer_num):
        _layer = self._get_layer(layer_num)
        _local_model = self._get_local_model(layer_num)

        # _stream = torch.cuda.Stream()
        # with torch.cuda.stream(_stream):
        #     _layer.cpu(non_blocking=True)
        #     _stream.synchronize()
        # pass

        # 3.
        if not (layer_num in _layer.vt.Out['W']) and not (layer_num in _layer.vt.Out['B']):
            _local_model.del_param_grad_buf()
        elif _layer.vt.Out['W'][layer_num].medium=='PIN' and _layer.vt.Out['B'][layer_num].medium=='PIN':
            pass
        else: # P2P
            raise NotImplementedError

        

        

    def _layer_to_cpu_and_gather_grads(self, layer_num):
        with torch.no_grad():
            _layer = self._get_layer(layer_num)

            _stream = torch.cuda.Stream()
            with torch.cuda.stream(_stream):
                _layer.cpu(non_blocking=True)
                _stream.synchronize()

                if _layer._gl_fp16:
                    # 
                    for param in _layer.parameters():
                        if param is None or param.grad is None:
                            continue
                        if param.grad.data is not None:
                            param.main_grad.add_(param.grad.data)
                            # Now we can deallocate grad memory.
                            # _free_cuda_tensor(param.grad)
                            param.grad = None

                    _stream.synchronize()
        pass

    def _layer_reset_save_for_backward(self, layer_num):
        _layer = self._get_layer(layer_num)
        _cpu_cache = _layer._gl_save_for_backward_cpu_cache

        if len(_cpu_cache) > 0:
            # the layers have cpu cache
            for i, _packed in enumerate(_layer._gl_save_for_backward):
                if _packed[1] is None:
                    continue

                gl_warmup_print(
                    f"--- reset - save_for_backward: index={i}; layer={layer_num}",
                    f";\n\t id_packed={id(_packed)}; len(_cpu_cache)={len(_cpu_cache)}",
                    f";\n\t device={_packed[1].device}; size={_packed[1].size()}; id={id(_packed[1])}; \n",
                )

                # _packed.pop() # del _packed[1]
                # _packed.append(_cpu_cache[i] if i in _cpu_cache else None)
                _packed[1] = _cpu_cache[i] if i in _cpu_cache else None

        # the last last 'window_size' layers have no cpu_cache for save_for_backward tensors
        _layer._gl_save_for_backward.clear()

        pass

    # 1.gpu->cpu
    # 2.cpu->gpu
    def _layer_move_save_for_backward_to(
        self, layer_num, device, action, index, non_blocking=True
    ):
        with torch.no_grad():
            _layer = self._get_layer(layer_num)

            _stream = torch.cuda.Stream()
            with torch.cuda.stream(_stream):
                if action == 'g2c':
                    _save_for_backward = _layer._gl_save_for_backward

                    if len(_save_for_backward) == 0:
                        # no offloading actions for the first window_size layers
                        return

                    packed = _save_for_backward[index]

                    tensor_device, tensor = packed

                    if _is_cpu(tensor_device):
                        # only offloading cuda tensors to cpu.
                        # no actions for cpu tensors or None type
                        return

                    _cpu_cache = _layer._gl_save_for_backward_cpu_cache

                    if index not in _cpu_cache:
                        gl_warmup_print(
                            f"--- init cpu_cache for save_for_backward tensors ",
                            f"in layer-{_layer._gl_layer_num}",
                        )
                        _cpu_cache[index] = torch.empty(
                            tensor.size(),
                            dtype=tensor.dtype,
                            layout=tensor.layout,
                            device=torch.device("cpu"),
                            pin_memory=True,
                        )  # (torch.cuda.is_available() and not tensor.is_sparse))
                    else:
                        gl_warmup_print(
                            f"--- already allocated cpu_cache for ",
                            f"save_for_backward tensors in layer-{_layer._gl_layer_num}",
                        )
                        pass

                    # _free_cuda_tensor(packed[1])
                    _cpu_cache[index] = _get_item_from_cpu_cache(_cpu_cache, index).copy_(
                        tensor, non_blocking=True
                    )  # non_blocking=non_blocking
                    packed[1] = _cpu_cache[index]

                    # _free_cuda_tensor
                    _save_for_backward[index] = [tensor_device, None]

                    _move_item_to_nvme(_cpu_cache, index)
                    
                elif action == 'c2g':
                    _save_for_backward = _layer._gl_save_for_backward

                    if len(_save_for_backward) == 0:
                        # no offloading actions for the first window_size layers
                        return

                    packed = _save_for_backward[index]

                    tensor_device, tensor = packed

                    if not _is_cpu(tensor_device):
                        # only offloading cuda tensors to cpu.
                        # no actions for cpu tensors or None type
                        return
                 
                    _cpu_cache = _layer._gl_save_for_backward_cpu_cache
                    
                    packed[1] = _get_item_from_cpu_cache(_cpu_cache, index).to(
                        tensor_device, non_blocking=True
                    )  # non_blocking=non_blocking
                        
                    #packed[1] = _cpu_cache[index].to(
                    #    tensor_device, non_blocking=True
                    #)  # non_blocking=non_blocking
                else:
                    # sometimes, maybe an error
                    pass

                _stream.synchronize()
        pass

    def _layer_move_main_grads_to(self, layer_num, device, non_blocking=True):
        assert (
            False
        ), "todo. @gl. now, we put the main_grads of the first window layers at GPU side"
        pass

    def _layer_move_main_grads_to_cpu(self, layer_num, non_blocking=True):
        assert (
            False
        ), "todo. @gl. now, we put the main_grads of the first window layers at GPU side"
        pass

    def _layer_move_main_grads_to_cuda(self, layer_num, non_blocking=True):
        assert (
            False
        ), "todo. @gl. now, we put the main_grads of the first window layers at GPU side"
        pass

    def _layer_optimizer_update(self, layer_num, non_blocking=True):
        self.optimizer.layer_update(layer_num)

    # backward hook:
    # for the first window layers but no including layer-0
    def _layer_gather_grads_and_optimizer_update_and_offloading_grads(
        self, layer_num, non_blocking=True
    ):
        assert (
            False
        ), "todo. @gl. now, we put the main_grads of the first window layers at GPU side"
        _layer = self._get_layer(layer_num)

        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            if _layer._gl_fp16:
                for param in _layer.parameters():
                    if param is None or param.grad is None:
                        continue
                    if param.grad.data is not None:
                        param.main_grad.add_(param.grad.data)
                        # Now we can deallocate grad memory.
                        # _free_cuda_tensor(param.grad)
                        param.grad = None
                _stream.synchronize()

            self._layer_optimizer_update(layer_num)

            if _layer._gl_fp16:
                self._layer_move_main_grads_to_cpu(layer_num)
        pass

    # backward hook:
    def _layer_gather_grads_and_optimizer_update(self, layer_num, non_blocking=True):
        _layer = self._get_layer(layer_num)

        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            if _layer._gl_fp16:
                for param in _layer.parameters():
                    if param is None or param.grad is None:
                        continue
                    if param.grad.data is not None:
                        param.main_grad.add_(param.grad.data)
                        # Now we can deallocate grad memory.
                        # _free_cuda_tensor(param.grad)
                        param.grad = None
                _stream.synchronize()

            self._layer_optimizer_update(layer_num)
        pass

    # backward hook:
    # for the layers excluding the first window layers
    def _layer_to_cpu_and_gather_grads_and_optimizer_update(
        self, layer_num, non_blocking=True
    ):
        self._layer_to_cpu_and_gather_grads(layer_num)
        # print(f" ---------> {layer_num} _layer_to_cpu_and_gather_grads is done")
        if hasattr(torch, "_gl_is_last_batch"):
            self._layer_optimizer_update(layer_num)

    # =====================================================
    # ==              basic actions ending               ==
    # =====================================================

    def register_pretrain_handler_for_layers(self, handler):
        # for layer_num in range(self.num_layers):
        #     self._get_layer(layer_num)._gl_handler = handler
        for layer in self.layers:
            layer._gl_handler = handler

    def register_gl_properties_for_layers_deprecated(self):

        _num_fwd_layers = self.num_fwd_layers
        _window_size = self.gl_window_size

        for _cur_layer_id in range(_num_fwd_layers):
            _layer_id = self.fwd_layers[_cur_layer_id]
            _layer = self._get_layer(_layer_id)

            _layer._gl_cuda_device = torch.cuda.current_device()

            _layer._gl_layer_num = _layer_id

            _layer._gl_layers = self._get_layers()

            _layer._gl_fwd_layers = self.fwd_layers
            
            _gl_which_layer_to_cuda_pre_fwd = min(
                _cur_layer_id + _window_size, _num_fwd_layers - 1
            )
            _layer._gl_which_layer_to_cuda_pre_fwd = _gl_which_layer_to_cuda_pre_fwd

            _layer._gl_which_layer_to_cuda_pre_fwd_required = (
                _cur_layer_id < _num_fwd_layers - _window_size
            )

            _layer._gl_which_layer_to_cpu_post_fwd = _cur_layer_id

            _layer._gl_which_layer_to_cpu_post_fwd_required = True

        _num_bwd_layers = self.num_bwd_layers

        for _cur_layer_id in range(_num_bwd_layers):
            _layer_id = self.bwd_layers[_cur_layer_id]
            _layer = self._get_layer(_layer_id)

            _layer._gl_cuda_device = torch.cuda.current_device()

            _layer._gl_layer_num = _layer_id

            _layer._gl_layers = self._get_layers()

            _layer._gl_bwd_layers = self.bwd_layers

            _gl_which_layer_to_cuda_pre_bwd = max(
                _cur_layer_id - _window_size, 0
            )
            _layer._gl_which_layer_to_cuda_pre_bwd = _gl_which_layer_to_cuda_pre_bwd

            _layer._gl_which_layer_to_cuda_pre_bwd_required = (
                _cur_layer_id >= _window_size
            )

            _layer._gl_which_layer_to_cpu_post_bwd = _cur_layer_id

            _layer._gl_which_layer_to_cpu_post_bwd_required = True

    def register_gl_properties_for_layers(self):
        _num_layers = self.num_layers
        _window_size = self.gl_window_size
        _layers_id_in_rank = self.layers_id_in_rank

        for _cur_layer_id in range(_num_layers):
            _layer_id = _layers_id_in_rank[_cur_layer_id]
            _layer = self._get_layer(_layer_id)

            _layer._gl_cuda_device = torch.cuda.current_device()

            _layer._gl_layer_num = _layer_id

            _layer._gl_layers = self._get_layers()

            # _layer._cuda_cache_queue = self.cuda_cache_queue
            # _layer._special_cuda_queue = self.special_cuda_queue

            if _cur_layer_id < self.num_fwd_layers:
                _gl_which_layer_to_cuda_pre_fwd = min(
                    _cur_layer_id + _window_size, _num_layers - 1
                )
                _layer._gl_which_layer_idx_to_cuda_pre_fwd = _gl_which_layer_to_cuda_pre_fwd
                _layer._gl_which_layer_to_cuda_pre_fwd = _layers_id_in_rank[_gl_which_layer_to_cuda_pre_fwd]

                _layer._gl_which_layer_to_cuda_pre_fwd_required = (
                    _cur_layer_id < _num_layers - _window_size
                )

                # _layer._gl_which_layer_to_cpu_post_fwd = _cur_layer_id

                _layer._gl_which_layer_to_cpu_post_fwd_required = True
            else:
                # deprecated
                _layer._gl_bwd_layers = self.bwd_layers

                _gl_which_layer_to_cuda_pre_recompute = min(
                    _cur_layer_id + _window_size, _num_layers - 1
                )
                _layer._gl_which_layer_idx_to_cuda_pre_recompute = _gl_which_layer_to_cuda_pre_recompute
                _layer._gl_which_layer_to_cuda_pre_recompute = _layers_id_in_rank[_gl_which_layer_to_cuda_pre_recompute]

                _layer._gl_which_layer_to_cuda_pre_recompute_required = (
                    _cur_layer_id < _num_layers - _window_size
                )

                # _layer._gl_which_layer_to_cpu_post_bwd = _cur_layer_id

                _layer._gl_which_layer_to_cpu_post_bwd_required = True
            
    def register_hooks(self):
        layers_need_fwd = set(self.layers_id_in_rank)
        for layer_id in layers_need_fwd:
            _layer = self._get_layer(layer_id)
            _layer.register_forward_pre_hook(_forward_pre_hook)

            _layer.register_forward_hook(_forward_post_hook)

        for vt_id, vt in enumerate(self.bwd_vts_in_rank):
            if vt.has_criterion:
                # print(f"rank:{self.rank}, vt.layers[:-1]:{vt.layers[:-1]}")
                for layer_id in vt.layers[:-2]:
                    _layer = self._get_layer(layer_id)
                    _layer.register_full_backward_hook(_backward_post_hook)
            else:
                for layer_id in vt.layers[:-1]:
                    _layer = self._get_layer(layer_id)
                    _layer.register_full_backward_hook(_backward_post_hook)


    # =====================================================
    # ===              register ending                  ===
    # =====================================================

class PinnedData:
    def __init__(self, pinned_buffer, pinned_layer):
        self.pinned_buffer = pinned_buffer
        self.pinned_layer = pinned_layer

def convert_to_not_require_grad(local_model_cpu):
    ''' in-place convert a local model cpu to a pinned model (params and buffers: pinned, local, CPU, no grad) '''
    @torch.no_grad()
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # no grad
                assert param.grad is None, "convert to pinned model requires no grad in input model"
                param.detach_()
                assert not param.requires_grad
                # pin param
                param.data = torch.empty(0, device="cpu") # in-place update and let python do the gc 
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                assert not buf.requires_grad # buffer has no grad
                m._buffers[key] = torch.empty(0, device="cpu") # in-place update and let python do the gc 
                assert not m._buffers[key].requires_grad
    local_model_cpu.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    gc.collect()

class CpuLayerPool:
    def __init__(self, transformer_layer, layer_count, model_dtype, special_layers_in_rank, rank):
        self.dtype = model_dtype
        self.rank = rank
        self.layer_count = layer_count
        self.cpu_cache_queue = deque(maxlen=self.layer_count)
        
        self.param_idx_in_layer_to_numel = {}
        self.param_idx_to_start_pos = {}
        self.param_idx_in_layer_to_shape = {}

        self.param_idx_in_special_layer_to_numel = {}
        self.param_idx_in_special_layer_to_start_pos = {}
        self.param_idx_in_special_layer_to_shape = {}
        self.special_layer_size = {}
        self.special_layer_start_pos = {}

        self.layer_size = self.cal_transformer_layer_size(transformer_layer)
        self.cal_special_layer_size(special_layers_in_rank)
        self.total_special_layer_size = sum(self.special_layer_size.values())
        self.pinned_transformer_layers_size = self.layer_size * self.layer_count
        self.pinned_buffer = torch.empty(int(self.pinned_transformer_layers_size+self.total_special_layer_size),
                                         dtype=self.dtype,
                                         requires_grad=False).pin_memory()
        
        self.create_pinned_layers(transformer_layer)
        self.special_cpu_queue = self.create_special_pinned_layers(special_layers_in_rank)

    def cal_special_layer_size(self, special_layers_in_rank):
        start_pos = 0
        for layer_id, layer in special_layers_in_rank:
            self.param_idx_in_special_layer_to_numel[layer_id] = {}
            self.param_idx_in_special_layer_to_start_pos[layer_id] = {}
            self.param_idx_in_special_layer_to_shape[layer_id] = {}

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
            self.special_layer_start_pos[layer_id] = start_pos
            start_pos += self.special_layer_size[layer_id]

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

        return total_size

    def allocate_and_return_buffer_for_special_layer_param(self, pinned_buffer, layer_id, global_param_idx):
        start_pos = self.param_idx_in_special_layer_to_start_pos[layer_id][global_param_idx]
        param_size = self.param_idx_in_special_layer_to_numel[layer_id][global_param_idx]
        narrowed_param = pinned_buffer.narrow(0, start_pos, param_size)
        return narrowed_param

    def create_special_pinned_layers(self, special_layers_in_rank):
        special_cpu_queue = {}
        for layer_id, layer in special_layers_in_rank:
            pinned_layer = deepcopy(layer)
            convert_to_not_require_grad(pinned_layer)
            pinned_buffer = self.pinned_buffer.narrow(0, self.pinned_transformer_layers_size+self.special_layer_start_pos[layer_id], self.special_layer_size[layer_id])
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
            convert_to_not_require_grad(pinned_layer)
            pinned_buffer = self.pinned_buffer.narrow(0, idx * self.layer_size, self.layer_size)
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

class Worker(object): # each rank process
    def __init__(self, args, real_dataset, shared_model, shared_optimizer, empty_model, get_lr_sched, compute_loss, save_model, XMETA, TMETA, rTASKS, CONFIGS, rank):
        self.args = args
        self.shared_model = shared_model
        self.shared_optimizer = shared_optimizer
        self.compute_loss = compute_loss
        self.save_model = save_model
        self.XMETA, self.TMETA = XMETA, TMETA
        self.rTASKS, self.CONFIGS = rTASKS, CONFIGS
        self.rank, self.world_size = rank, CONFIGS["N"]
        self.verbose, self.nvprof = args.verbose, args.nvprof

        # for vt in rTASKS[self.rank]:
        #     print(vt)
        # exit(0)

        # worker process must re-seed
        seeding.seed(args.seed, args.seed_cudnn) 
        self.rand_state_train = seeding.RandState()

        for j, vt in enumerate(self.rTASKS[self.rank]):
            print(f"rank:{self.rank}, vt:{j}, vt.type:{vt.type}, layers:{vt.layers}")

        # per-rank configs
        if CONFIGS['mode'] == 'vPP':
            self.ubatchszs_fwd_local = CONFIGS['ubatchszs_fwd']
            print(f"ubatchszs_fwd_local:{CONFIGS['ubatchszs_fwd']}")
            self.ubatchszs_bwd_local = CONFIGS['ubatchszs_bwd']
            print(f"ubatchszs_bwd_local:{CONFIGS['ubatchszs_bwd']}")
            self.minibatchsize_local = CONFIGS['D']
            print(f"minibatchsize_local:{CONFIGS['D']}")
        elif CONFIGS['mode'] == 'vDP':
            self.ubatchszs_fwd_local = CONFIGS['ubatchszs_fwd'][self.rank]
            self.ubatchszs_bwd_local = CONFIGS['ubatchszs_bwd'][self.rank]
            self.minibatchsize_local = sum(self.ubatchszs_fwd_local)
            assert self.minibatchsize_local == sum(self.ubatchszs_bwd_local)
        else:
            raise ValueError
        print("CONFIGS[\"u_fwd\"]:", CONFIGS["u_fwd"])
        print("CONFIGS[\"u_bwd\"]:", CONFIGS["u_bwd"])
        self.is_convert_ubs = True if CONFIGS["u_fwd"] != CONFIGS["u_bwd"] else False
        
        # Initialize the Gloo world first
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = str(args.master_port)

        dist.init_process_group(backend="gloo", rank=self.rank, world_size=self.world_size)      
        assert dist.get_rank() == self.rank and dist.get_world_size() == self.world_size
        print("rank%d (pid %d): initialized Gloo world. world_size %d" % (self.rank, os.getpid(), self.world_size))
        
        # Set up GPU
        torch.cuda.set_device(self.rank)
        
        # initialize dataset (must be local to be pinned)
        if args.synthetic_data:
            self.data_loader = list(range(args.num_iters))
            self.data_ubatches, self.target_ubatches = synthesize_data(XMETA, TMETA, self.ubatchszs_fwd_local, self.ubatchszs_bwd_local, pin_memory=not args.no_pin_data)
        else:
            # self.bnames：{"is_data" = [True, False]， "name" = ["input0", "labels"]}
            self.data_loader, _, self.is_skip_minibatch, self.preprocess_minibatch, self.bnames, self.fdim, self.is_copy_minibatch = real_dataset(args, CONFIGS["D"], args.data_workers)
            self.data_ubatches, self.target_ubatches = None, None
        
        # initialize shared optimizer locally
        self.pcm = PrintCPUMem()
        self.pcm.print("rank%d: before initializing optimizer" % self.rank)
        lr_scheduler = []
        for id, optim in enumerate(shared_optimizer):
            optim.init_in_subproc(self.rank, no_pin_model=args.no_pin_model, no_pin_grad_buf=args.no_pin_grad_buf)
            if get_lr_sched is not None: # "gpt2_huggingface"      
                lr_scheduler.append(None if optim.shared_optimizer is None else 
                                    get_lr_sched(args, optim.shared_optimizer))
        self.pcm.print("rank%d: optimizer initialized" % self.rank)
        

        #################################################################################
        ############################# work6 #############################################
        num_layers = len(self.shared_model)
        special_layers = [0, num_layers - 1, num_layers - 2, num_layers - 3]
        fwd_vts_in_rank = [vt for vt in self.rTASKS[self.rank] if vt.type == "FWD"]
        bwd_vts_in_rank = [vt for vt in self.rTASKS[self.rank] if vt.type == "BWD"]
        self.vts_in_rank = fwd_vts_in_rank + bwd_vts_in_rank
        self.layers_id_in_rank = [layer for vt in self.vts_in_rank for layer in vt.layers]
        self.gl_window_size = args.gl_window_size

        special_layers_in_rank = []

        total_fwd_layer_num = 0
        max_fwd_vt_layer = 0
        if len(fwd_vts_in_rank) != 0:
            fwd_vt_layer_list = []
            for vt in fwd_vts_in_rank:
                vt_layer = []
                for layer in vt.layers:
                    if layer in special_layers:
                        special_layers_in_rank.append(layer)
                    if layer not in special_layers:
                        vt_layer.append(layer)
                fwd_vt_layer_list.append(vt_layer)
            fwd_vt_layer_num = [len(vt_layer) for vt_layer in fwd_vt_layer_list]
            max_fwd_vt_layer = max(fwd_vt_layer_num)
            total_fwd_layer_num = sum(fwd_vt_layer_num)
        total_bwd_layer_num = 0
        max_bwd_vt_layer = 0
        if len(bwd_vts_in_rank) != 0:
            bwd_vt_layer_list = []
            for vt in bwd_vts_in_rank:
                vt_layer = []
                for layer in vt.layers:
                    if layer in special_layers:
                        special_layers_in_rank.append(layer)
                    if layer not in special_layers:
                        vt_layer.append(layer)
                bwd_vt_layer_list.append(vt_layer)
            bwd_vt_layer_num = [len(vt_layer) for vt_layer in bwd_vt_layer_list]
            max_bwd_vt_layer = max(bwd_vt_layer_num)
            total_bwd_layer_num = sum(bwd_vt_layer_num)
        layer_num = max(total_fwd_layer_num, total_bwd_layer_num)
        max_vt_layer_num = max(max_fwd_vt_layer, max_bwd_vt_layer)

        window_size = max_vt_layer_num + args.gl_window_size
        self.max_len = min(window_size*2, layer_num)
        self.cuda_cache_queue = deque(maxlen=self.max_len)

        # special_layers_in_rank = [layer for layer in special_layers_in_rank if layer != num_layers - 1]
        self.special_layers_in_rank = set(special_layers_in_rank)
        self.special_cuda_queue = {}
        for layer_id in self.special_layers_in_rank:
            self.special_cuda_queue[layer_id] = ODict()
            unit = self.special_cuda_queue[layer_id]
            _layer,_,_ = self.shared_model[layer_id]
            global_param_index = 0
            for cpu_m in _layer.modules():
                for key, param in cpu_m._parameters.items():
                    if param is not None:
                        unit[global_param_index] = param.cuda(non_blocking=False)
                        unit[str(global_param_index)+".grad"] = torch.zeros_like(param, device=self.rank)
                        global_param_index += 1
                for key, buf in cpu_m._buffers.items():
                    if buf is not None:
                        unit[global_param_index] = buf.cuda(non_blocking=False)
                        global_param_index += 1

        for i in range(window_size+2):
            unit = ODict()
            _layer,_,_ = self.shared_model[1]

            global_param_index = 0
            for cpu_m in _layer.modules():
                for key, param in cpu_m._parameters.items():
                    if param is not None:
                        unit[global_param_index] = param.cuda(non_blocking=False)
                        unit[str(global_param_index)+".grad"] = torch.zeros_like(param, device=self.rank)
                        global_param_index += 1
                for key, buf in cpu_m._buffers.items():
                    if buf is not None:
                        unit[global_param_index] = buf.cuda(non_blocking=False)
                        global_param_index += 1

            self.cuda_cache_queue.append(unit)

        torch.cuda.synchronize(self.rank)


        _special_layers_in_rank = []
        for layer_id in self.special_layers_in_rank:
            _special_layers_in_rank.append((layer_id, self.shared_model[layer_id][0]))

        try:
            model_dtype = next(self.shared_model[1][0].parameters()).dtype
            print("model_dtype: ", model_dtype)
        except StopIteration:
            model_dtype = torch.float32
        self.cpu_layer_pool = CpuLayerPool(self.shared_model[1][0], self.max_len, model_dtype, _special_layers_in_rank, self.rank)
        self.pcm.print("rank%d: cpu_layer_pool initialized" % self.rank)
        #################################################################################

        # initialize local model GPU 
        #
        self.local_model = []
        for vlayer_id, (optim, (_,X_names,Y_names), empty_vlayer) in enumerate(zip(shared_optimizer, shared_model, empty_model)):
            local_vlayer = local_model_gpu.LocalModelGPU_work6_4_2(optim.pinned_model, optim.shared_model, empty_vlayer, self.cpu_layer_pool, self.cuda_cache_queue, special_layers_in_rank, vlayer_id, X_names, Y_names, self.rank, self.world_size, no_pin_model=args.no_pin_model, no_pin_grad_buf=args.no_pin_grad_buf) 
            local_vlayer.train() # shared_model/pinned_train.train() not necessary
            self.local_model.append(local_vlayer)
        self.pcm.print("rank%d: local model initialized" % self.rank)
        
        # initialize MSG stashing X on CPU
        layer_X_names = ODict()
        for vlayer_id, (_,X_names,_) in enumerate(shared_model):
            layer_X_names[vlayer_id] = X_names
        #
        # Handles gloo send/recv of stashing X between cpu processes. 
        msg_stashx = msg_stash_x.MSGStashX(self.rank, rTASKS, layer_X_names, XMETA, self.ubatchszs_bwd_local, 'pack-by-pack', pin_memory=not args.no_pin_x, nvprof=self.nvprof)
        
        swapout_stashx_output_fn = msg_stashx.isend

        if self.is_convert_ubs: # initialize Optional UBatchSize Converter on CPU
            # 
            # self.minibatchsize_local：minibatch size
            # pack_ordering=False
            # pin_memory=not args.no_pin_x（true）
            # 
            stashx_ubs_converter = ubatchsize_converter.UBatchSizeConverter(self.rank, self.minibatchsize_local, CONFIGS['u_fwd'], self.ubatchszs_fwd_local, CONFIGS['u_bwd'], self.ubatchszs_bwd_local, msg_stashx.isend, pack_ordering=False, pin_memory=not args.no_pin_x, nvprof=self.nvprof)
            swapout_stashx_output_fn = stashx_ubs_converter.isend
        
        # initialize SWP locally
        local_x = None
        if (CONFIGS['mode'] == 'vPP' and CONFIGS['N'] == 1) or (CONFIGS['mode'] == 'vDP'):
            local_x = msg_stash_x.LocalX(self.rank, list(range(CONFIGS['R'])))
            swapout_localx_output_fn = local_x.isend
            if self.is_convert_ubs: # initialize Optional UBatchSize Converter on CPU
                localx_ubs_converter = ubatchsize_converter.UBatchSizeConverter(self.rank, self.minibatchsize_local, CONFIGS['u_fwd'], self.ubatchszs_fwd_local, CONFIGS['u_bwd'], self.ubatchszs_bwd_local, local_x.isend, pack_ordering=False, pin_memory=not args.no_pin_x, nvprof=self.nvprof)
                swapout_localx_output_fn = localx_ubs_converter.isend
        
        # initialize P2P
        self.p2px_handler, self.p2pm_handler = None, None
        if CONFIGS['mode'] == 'vPP' and CONFIGS['N'] > 1:
            # 
            # 2.
            self.p2px_handler = p2p.P2PX(self.rank, self.world_size, CONFIGS['reverse_bwd'], verbose=self.verbose, nvprof=self.nvprof)
        elif CONFIGS['mode'] == 'vDP' and CONFIGS['N'] > 1:
            self.p2pm_handler = p2p.P2PModel(self.rank, self.world_size, verbose=self.verbose)

        # Get default cuda stream (already initialized by local_model_gpu)
        self.default_stream = torch.cuda.default_stream(self.rank)

        self.swapout_grad_handler = local_model_gpu.SwapOutGrad(self.local_model, self.rank, self.CONFIGS, swapout_stream=None)
        self.update_handler = shared_optim_cpu.UpdateInBkgd_worker6_4_2(shared_optimizer, lr_scheduler, self.local_model, self.swapout_grad_handler, self.cuda_cache_queue, self.special_layers_in_rank, self.rank, nvprof=self.nvprof)
        # 

        # initialize Update in Background thread
        # self.update_handler = shared_optim_cpu.UpdateInBkgd_3_2(self.default_stream, shared_optimizer, lr_scheduler, self.local_model, self.cuda_cache_queue, self.special_layers_in_rank, self.rank, nvprof=self.nvprof)

        # initialize Prefetch Model background thread

        syncpin_handler = shared_optim_cpu.SyncPinModelInBkgd_worker6_4_2(shared_optimizer, self.local_model, self.rank, nvprof=self.nvprof)

        self.prefetch_model_handler = local_model_gpu.PrefetchLocalModelGPU_worker6_4_2(syncpin_handler, self.local_model, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof)
        
        # initialize SwapIn background thread
        self.swapin_stashx_handler = swp_x.SwapIn(msg_stashx.recv, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof) 

        self.swapin_localx_handler = swp_x.SwapIn(local_x.recv, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof) if local_x is not None else None
        
        # initialize SwapOut background thread
        swapout_stream = torch.cuda.Stream(device=self.rank)
        self.swapout_stashx_handler = swp_x.SwapOut(swapout_stashx_output_fn, self.rank,    
                                    swapout_stream=swapout_stream,
                                    compute_stream=self.default_stream, 
                                    blocking=True if args.no_offload_stashx else False,
                                    pin_memory=not args.no_pin_x,
                                    nvprof=self.nvprof)
        
        self.swapout_localx_handler = swp_x.SwapOut(swapout_localx_output_fn, self.rank, 
                                    swapout_stream=swapout_stream, 
                                    compute_stream=self.default_stream, 
                                    blocking=True if args.no_offload_localx else False,
                                    pin_memory=not args.no_pin_x,
                                    nvprof=self.nvprof) \
                                    if local_x is not None else None
        
        # initialize MSG X on CPU # NOTE: tentatively only for last FWD to first BWD
        #
        # Handles gloo send/recv of Y/dX between cpu processes. 
        msg_x = msg_stash_x.MSGX(self.rank, rTASKS, layer_X_names, XMETA, self.ubatchszs_bwd_local, 'pack-by-pack', pin_memory=not args.no_pin_x, nvprof=self.nvprof)
        
        # Call by upstream thread. Nonblocking send. 
        # self.odict[layer_id].append(named_tensors)
        swapout_msgx_output_fn = msg_x.isend
        self.swapout_msgx_handler, self.swapin_msgx_handler = None, None

        if msg_x.has_no_send() and msg_x.has_no_recv():
            del msg_x; msg_x = None
        elif not msg_x.has_no_send() and msg_x.has_no_recv(): # sender only
            if self.is_convert_ubs:
                msgx_ubs_converter = ubatchsize_converter.UBatchSizeConverter(self.rank, self.minibatchsize_local, CONFIGS['u_fwd'], self.ubatchszs_fwd_local, CONFIGS['u_bwd'], self.ubatchszs_bwd_local, msg_x.isend, pack_ordering=False, pin_memory=not args.no_pin_x, nvprof=self.nvprof)
                swapout_msgx_output_fn = msgx_ubs_converter.isend
            self.swapout_msgx_handler = swp_x.SwapOut(swapout_msgx_output_fn, self.rank, 
                                        swapout_stream=swapout_stream, 
                                        compute_stream=self.default_stream, 
                                        blocking=True if args.no_offload_msgx else False,
                                        pin_memory=not args.no_pin_x,
                                        nvprof=self.nvprof)
        elif msg_x.has_no_send() and not msg_x.has_no_recv(): # recver only
            self.swapin_msgx_handler = swp_x.SwapIn(msg_x.recv, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof)
        else:
            raise NotImplementedError
        
        # initialize succesor info for all prefetch
        self.sucinfo = SucInfoForPrefetch(self.rank, rTASKS, XMETA)

        # ========= my version =============


        # for layer_id in layers_id_in_rank[0:self.gl_window_size]:
        #     optim = shared_optimizer[layer_id]
        #     optim.sync_pinned_model()

        # for layer_id in self.layers_id_in_rank[0:self.gl_window_size]:
        #     _model = self.local_model[layer_id]
        #     _model.alloc_param_buf()
        #     _model.copyin_param_buf()
        #     self.default_stream.synchronize()

        # for layer_id in self.layers_id_in_rank[0:self.gl_window_size]:
        #     _model = self.local_model[layer_id]
        #     if len(list(_model.model.parameters())) != 0:
        #         for param in local_vlayer.model.parameters():
        #             assert param.is_cuda

        # self.layer_id_to_vt = {}
        
        self.pretrain_handler = GL_PretrainHanlder(
            args,
            self.local_model,
            self.CONFIGS,
            self.rTASKS[self.rank],
            self.prefetch_model_handler,
            self.default_stream,
            self.rank,
            self.update_handler,
            self.cuda_cache_queue,
            self.special_cuda_queue,
            self.nvprof)
        self.pretrain_handler.register_pretrain_handler_for_layers(self.pretrain_handler)
        self.pretrain_handler.register_gl_properties_for_layers()
        self.pretrain_handler.register_hooks()
        # ============== end ===============

    ################### Initial Iteration ###################
    def _initial_a_pack_forward_an_ubatch(self, vt, ubatch_idx, ubatch_size, requires_grad=False, verbose=False, nvprof=False):
        if not vt.has_criterion:
            ### In {X}
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names
            X_named_tensors = realize_X(self.XMETA, ubatch_size, l, X_names, requires_grad, "cuda:%d"%self.rank, use_rand=False)
            ### Compute forward pass on GPU
            if nvprof: nvtx_range_push("task{}({}) {}(#{})".format(vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx)) 
            Y_tensors = [X_named_tensors[name] for name in X_names]
            for l in vt.layers:
                Y_tensors = self.local_model[l](*Y_tensors)
                if not isinstance(Y_tensors, tuple):
                    Y_tensors = (Y_tensors,)
                Y_tensors = list(Y_tensors)
            if verbose: print("\trank{}: task{}({}) {}(#{})".format(self.rank, vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx))
            if nvprof: nvtx_range_pop()
            ### Save Y
            l = vt.layers[-1]
            Y_names = self.local_model[l].Y_names
            Y_named_tensors = make_tensors_named(Y_names, Y_tensors)
            ### Clean up
            del Y_tensors
            if not requires_grad:
                del X_named_tensors; del Y_named_tensors
            else:
                return X_named_tensors, Y_named_tensors
        else: # criterion pack
            assert requires_grad
            ### In {X}
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names
            X_named_tensors = realize_X(self.XMETA, ubatch_size, l, X_names, requires_grad, "cuda:%d"%self.rank, use_rand=False)
            ### Recompute on GPU
            if nvprof: nvtx_range_push("task{}({}) Recompute(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            if len(vt.layers) > 1: # packed
                Y_tensors = [X_named_tensors[name] for name in X_names]
                for l in vt.layers[:-1]:
                    Y_tensors = self.local_model[l](*Y_tensors)
                    if not isinstance(Y_tensors, tuple):
                        Y_tensors = (Y_tensors,)
                    Y_tensors = list(Y_tensors)
                Y_names = self.local_model[vt.layers[-2]].Y_names
                Y_named_tensors = make_tensors_named(Y_names, Y_tensors)
            else: # only last vlayer
                Y_names = X_names
                Y_named_tensors = X_named_tensors
            ### In {T}
            T_named_tensors = realize_T(self.TMETA, ubatch_size, "cuda:%d"%self.rank, use_rand=False)
            ### Compute loss on GPU
            # 
            assert vt.layers[-1] == self.CONFIGS['R']-1
            # 
            last_vlayer = self.local_model[self.CONFIGS['R']-1]
            if self.compute_loss is not None: 
                # T_named_tensors：{"label": tensor}
                Y_tensors = self.compute_loss(last_vlayer, Y_named_tensors, Y_names, T_named_tensors)
            else:
                Y_tensors = [last_vlayer(Y_named_tensors[name],T_named_tensors["target"]) for name in Y_names]
                Y_tensors = [sum(Y_tensors)]
            if verbose: print("\trank{}: task{}({}) Recompute(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            if nvprof: nvtx_range_pop()
            ### Save Y
            Y_named_tensors = make_tensors_named(['loss'], Y_tensors)
            ### Clean up
            del T_named_tensors; del Y_tensors; 
            return X_named_tensors, Y_named_tensors

    def _initial_a_pack_backward_an_ubatch(self, vt, ubatch_idx, ubatch_size, X_named_tensors, Y_named_tensors, verbose=False, nvprof=False):
        ### In {dY}
        if vt.has_criterion:
            dY_named_tensors = ODict({ 'loss': None })
            assert Y_named_tensors['loss'].requires_grad

        else:
            l, m = vt.layers[-1], vt.In['dY'][vt.layers[-1]]
            dY_named_tensors = realize_dX(self.XMETA, ubatch_size, l+1, self.local_model[l+1].X_names, device="cuda:%d"%self.rank, use_rand=False)
        ### Compute backward pass
        if nvprof: nvtx_range_push("task{}({}) BWD(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
        Y_tensors = []
        Y_gradients = [] 
        for name in self.local_model[vt.layers[-1]].Y_names:
            Y = Y_named_tensors[name]
            if isinstance(Y,(torch.Tensor, Variable)) and (Y.requires_grad):
                Y_tensors.append(Y)
                Y_gradients.append(dY_named_tensors[name])
            elif isinstance(Y, list): 
                for i, y in enumerate(Y):
                    if isinstance(y,(torch.Tensor, Variable)) and (y.requires_grad):
                        Y_tensors.append(y)
                        Y_gradients.append(dY_named_tensors[name][i])
        torch.autograd.backward(tuple(Y_tensors), grad_tensors=tuple(Y_gradients))
        if verbose: print("\trank{}: task{}({}) BWD(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
        if nvprof: nvtx_range_pop() 
        ### Clean up {X,Y,dX,dY}
        del X_named_tensors; del Y_named_tensors
        del dY_named_tensors; del Y_tensors; del Y_gradients

    def run_initial_iteration(self, verbose=False, nvprof=False):
        if self.args.no_initial_iter:
            print("rank%d: --- No Initial Iteration ---" % self.rank)
            return

        print("rank%d: initial iteration starts"%(self.rank))
        assert dist.get_rank() == self.rank and torch.cuda.current_device() == self.rank
        # clean memory before start
        torch.cuda.synchronize(self.rank); dist.barrier()
        gc.collect(); torch.cuda.empty_cache() 
        torch.cuda.synchronize(self.rank)
        assert torch.cuda.memory_reserved(self.rank)==0 # 查看总共占用的显存
        dist.barrier()
        # task starts 
        if nvprof:
            probe_cuda_mem = ProbeCudaMem(self.rank)
            probe_cuda_mem.start()  
            cuda_profiler.start()
            nvtx_mark("cudaProfilerStart") 
            print("rank%d: cuda profiler starts" % self.rank)    

        time_start = pc()    
        #     
        for j, vt in enumerate(self.rTASKS[self.rank]): # { rank0: [task0,task2,task5,...] }
            if verbose: print("\trank{}: executing {}".format(self.rank, vt))
            if vt.type == 'FWD' and vt.is_gpu:
                # -----------------------------------------------      
                with torch.no_grad():
                    ### Swap-in model {W,B}
                    if nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                    if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                    cur_vt_idx = self.prefetch_model_handler.get(vt, None)
                    assert cur_vt_idx == vt.idx
                    if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                    if nvprof: nvtx_range_pop() 
                    ### Run through each microbatch in a data batch
                    for i, u in enumerate(vt.ubatchszs):# [u1, u2, u3] 
                        self._initial_a_pack_forward_an_ubatch(vt, i, u, requires_grad=False, verbose=verbose, nvprof=nvprof)
                        gc.collect()
                        if verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                    ### Delete model {W,B}
                    self.default_stream.synchronize() # CPU wait Compute
                    if nvprof: nvtx_range_push("task{}({}) Del(W,B)".format(vt.idx, vt.show_layers())) 
                    for l in vt.layers:
                        if not (l in vt.Out['W']) and not (l in vt.Out['B']):
                            self.local_model[l].del_param_grad_buf()
                        elif vt.Out['W'][l].medium=='PIN' and vt.Out['B'][l].medium=='PIN':
                            pass
                        else: # P2P
                            raise ValueError("Underdevelopment")
                    gc.collect()
                    if verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                    if nvprof: nvtx_range_pop() 
                # -----------------------------------------------
            elif vt.type == 'BWD' and vt.is_gpu:
                # -----------------------------------------------
                ### Swap-in model {W,B}
                if nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                cur_vt_idx = self.prefetch_model_handler.get(vt, None)
                assert cur_vt_idx == vt.idx 
                if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                if nvprof: nvtx_range_pop() 
                ### Run through each microbatch in a data batch. 
                for i, u in enumerate(vt.ubatchszs):
                    ### Recompute to create pytorch graph
                    X_named_tensors, Y_named_tensors = \
                        self._initial_a_pack_forward_an_ubatch(vt, i, u, requires_grad=True, verbose=verbose, nvprof=nvprof) 
                    ### Backward pass on recomputed graph
                    self._initial_a_pack_backward_an_ubatch(vt, i, u, X_named_tensors, Y_named_tensors, verbose=verbose, nvprof=nvprof)
                    ### Clean up
                    del X_named_tensors; del Y_named_tensors # very important!
                    gc.collect()
                    if verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                ### Swap-out model {W,dW,B}
                if self.CONFIGS["opt_offld"]:                
                    ### Delete model {W,dW,B}
                    self.default_stream.synchronize() # CPU wait for SwapOut
                    if nvprof: nvtx_range_push("task{}({}) Del(W,dW,B)".format(vt.idx, vt.show_layers())) 
                    for l in vt.layers: 
                        if vt.Out['dW'][l].medium == "LOC" and vt.Out['B'][l].medium == "SHM":
                            self.local_model[l].del_param_grad_buf()
                        else: # 'B' == PIN
                            raise ValueError("Underdevelopment")
                    gc.collect()
                    if verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                    if nvprof: nvtx_range_pop() 
                else:
                    raise ValueError("GPU Optimizer Underdevelopment.")
                # -----------------------------------------------
            elif vt.type == 'UPD' and not vt.is_gpu:
                # -----------------------------------------------
                pass
                # -----------------------------------------------
            else:
                raise ValueError("Unknown vTask.type {} with .device {} !".format(vt.type,vt.device))
        # tasks ends
        torch.cuda.synchronize(self.rank); dist.barrier()
        time_end = pc() 
        if nvprof:
            nvtx_mark("cudaProfilerStop") 
            cuda_profiler.stop()
            probe_cuda_mem.stop()
            print("rank%d: cuda profiler stops" % self.rank) 
        print("rank%d: initial iteration ends. time %.3f s"%(self.rank, time_end-time_start))
        # clean memory
        gc.collect(); torch.cuda.empty_cache() 
        torch.cuda.synchronize(self.rank)
        assert torch.cuda.memory_reserved(self.rank)==0
        dist.barrier()
        
        if self.args.initial_iter_only:
            print("rank%d: --- Initial Iteration Only ---" % self.rank)
            exit(0) 


    ################### Regular Training Loop ###################
    #
    def _a_pack_forward_an_ubatch(self, vt, ubatch_idx, ubatch_size,
                                data_ubatches, target_ubatches, 
                                requires_grad=False, 
                                prefetch_model_handler=None,
                                swapin_stashx_handler=None,
                                swapin_localx_handler=None,
                                swapin_msgx_handler=None,
                                swapout_stashx_handler=None,
                                swapout_localx_handler=None,
                                swapout_msgx_handler=None,
                                sucinfo=None,
                                delete_time=None,
                                prehook_time=None,
                                wait_prefetch_time=None):
        """ requires_grad == False: FWD (non-criterion)
            requires_grad == True: Recompute (for all) """
        is_last_ubatch = ubatch_idx == len(vt.ubatchszs)-1

        if not vt.has_criterion: # not last pack yet
            ### In {X}
            #   --...
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names
            if m.medium == "DAT": # Get one microbatch data
                # Data as X
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}Data)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                X_named_tensors = swp_x.swapin(data_ubatches[ubatch_idx])
                # print("\trank{}: task{}({}) SwapIn(#{}Data)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "P2P":
                if self.nvprof: nvtx_range_push("task{}({}) P2PIn(#{}X)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_p2p_prerecv:
                    X_named_tensors = self.p2px_handler.recv(self.XMETA.get(ubatch_size,l), src=m.rank)
                else:
                    X_named_tensors = self.p2px_handler.prerecv(self.XMETA.get(ubatch_size,l), src=m.rank, is_end=is_last_ubatch) 
                # print("\trank{}: task{}({}) P2PIn(#{}X)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            
            elif m.medium == "MSG": # message pass stashed input
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}StashX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_stashx:
                    X_named_tensors = swapin_stashx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_stashx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: task{}({}) SwapIn(#{}StashX)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}LocalX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_localx:
                    X_named_tensors = swapin_localx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_localx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: swp_recv'ed L{}-X".format(self.rank, l))
            else:
                raise NotImplementedError
            if self.nvprof: nvtx_range_pop() 


            ### Prefetch point @ FWD/Recompute (non-criterion)'s ULast 
            if is_last_ubatch:
                self.pretrain_handler.set_is_last_batch()
                if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 

                # 2.1
                if self.p2px_handler is not None and \
                    not self.args.no_p2p_prerecv and not requires_grad:
                    self.p2px_handler.prerecv_suc(sucinfo.p2pin())
                # if swapin_msgx_handler is not None and \
                #     not self.args.no_prefetch_msgx and not requires_grad:
                #     swapin_msgx_handler.prefetch_suc(sucinfo.msgx())
                # if prefetch_model_handler is not None and \
                #     not self.args.no_prefetch_model and not requires_grad:
                #     prefetch_model_handler.iput(sucinfo.model()) 

                # 2.2
                if swapin_stashx_handler is not None and \
                    not self.args.no_prefetch_stashx:
                    # sucinfo.stashx()：
                    #
                    swapin_stashx_handler.prefetch_suc(sucinfo.stashx())
                # if swapin_stashx_handler is not None and \
                #     not self.args.no_prefetch_stashx and requires_grad:
                #     swapin_stashx_handler.prefetch_suc(sucinfo.stashx())

                if swapin_localx_handler is not None and \
                    not self.args.no_prefetch_localx and not requires_grad:
                    swapin_localx_handler.prefetch_suc(sucinfo.localx())
                if self.nvprof: nvtx_range_pop() 

            ### Compute forward pass on GPU
            if requires_grad:
                turn_on_X_grad(X_named_tensors) 
            if self.nvprof: nvtx_range_push("task{}({}) {}(#{})".format(vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx)) 
            
            Y_tensors = [X_named_tensors[name] for name in X_names]

            for l in vt.layers:
                if not requires_grad and l in vt.Out['X']: ### Out {stashX}
                    if self.nvprof: nvtx_range_push("task{}(L{}) SwapOut(#{}StashX)".format(vt.idx, l, ubatch_idx)) 
                    if vt.Out['X'][l].medium == "MSG": # message pass stashed X
                        swapout_stashx_handler.offload(l, 
                                make_tensors_named(self.local_model[l].X_names, Y_tensors))
                        # print("\trank{}: task{}(L{}) SwapOut(#{}StashX)".format(self.rank, vt.idx, l, ubatch_idx))
                    else:
                        raise NotImplementedError
                    if self.nvprof: nvtx_range_pop() 
                # print("\trank{}: task{}(L{}) {}".format(self.rank, vt.idx, l, "FWD" if not requires_grad else "Recompute"))
                
                if self.nvprof: nvtx_range_push("__task{}(L{}) {}(#{})".format(vt.idx, l, "FWD" if not requires_grad else "Recompute", ubatch_idx)) 
                # for name,param in self.local_model[l].model.named_parameters():
                #     print(f"rank:{self.rank}, L({l}),Parameter {name} is {param}, is on {param.device} ({ubatch_idx})")
                Y_tensors = self.local_model[l](*Y_tensors)
                if self.nvprof: nvtx_range_pop() 

                if not isinstance(Y_tensors, tuple):
                    Y_tensors = (Y_tensors,)
                Y_tensors = list(Y_tensors)
            if self.verbose: print("\trank{}: task{}({}) {} (#{})".format(self.rank, vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx))
            if self.nvprof: nvtx_range_pop() 
            
            ### Save Y
            l = vt.layers[-1]
            Y_names = self.local_model[l].Y_names
            Y_named_tensors = make_tensors_named(Y_names, Y_tensors)

            if is_last_ubatch:
                self.pretrain_handler.reset_is_last_batch()

            if not requires_grad:


                ### Out {Y}
                m = vt.Out['Y'][l]
                if m.medium == "P2P":
                    if self.nvprof: nvtx_range_push("task{}({}) P2POut(#{}Y)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                    self.p2px_handler.isend(Y_named_tensors, dst=m.rank)
                    # print("\trank{}: task{}({}) P2POut(#{}Y)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
                elif m.medium == "MSG": # last FWD convert to first BWD
                    if self.nvprof: nvtx_range_push("task{}({}) MSGOut(#{}Y)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                    swapout_msgx_handler.offload(l+1, Y_named_tensors)
                    # print("\trank{}: task{}({}) MSGOut(#{}Y)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
                elif m.medium == "SWP": # swap locally for vDP
                    if self.nvprof: nvtx_range_push("task{}({}) SwapOut(#{}Y)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                    if self.is_convert_ubs:
                        flag_is_convert = True if vt.is_last_fwd else False
                        swapout_localx_handler.offload(l+1, Y_named_tensors, flag_is_convert)
                    else:
                        swapout_localx_handler.offload(l+1, Y_named_tensors)
                    # print("\trank{}: swp_send'ed L{}-Y".format(self.rank, l))
                else:
                    raise NotImplementedError
                if self.nvprof: nvtx_range_pop() 
            ### Clean up
            if self.nvprof: nvtx_range_push("task{}({}) FWDClean(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            # print("\trank{}: task{}({}) FWDClean(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            del Y_tensors
            if not requires_grad:
                del X_named_tensors; del Y_named_tensors
            else: # for backward pass
                return X_named_tensors, Y_named_tensors

        else: # criterion pack
            assert requires_grad # fused forward and backward
            ### In {X}
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names

            #   --...
            if m.medium == "DAT": # a single BWD task
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}Data)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                X_named_tensors = swp_x.swapin(data_ubatches[ubatch_idx])
                # print("\trank{}: task{}({}) SwapIn(#{}Data)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "P2P": # the same above
                if self.nvprof: nvtx_range_push("task{}({}) P2PIn(#{}X)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_p2p_prerecv:
                    X_named_tensors = self.p2px_handler.recv(self.XMETA.get(ubatch_size,l), src=m.rank)
                else:
                    X_named_tensors = self.p2px_handler.prerecv(self.XMETA.get(ubatch_size,l), src=m.rank, is_end=is_last_ubatch) 
                # print("\trank{}: task{}({}) P2PIn(#{}X)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "MSG": # last FWD convert to first BWD
                if self.nvprof: nvtx_range_push("task{}({}) MSGIn(#{}X)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_msgx:
                    X_named_tensors = swapin_msgx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_msgx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: task{}({}) MSGIn(#{}X)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}LocalX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_localx:
                    X_named_tensors = swapin_localx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_localx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: swp_recv'ed L{}-X".format(self.rank, l))
            else:
                raise NotImplementedError
            if self.nvprof: nvtx_range_pop() 


            ### Prefetch point @ Recompute(criterion) ULast
            if is_last_ubatch:
                self.pretrain_handler.set_is_last_batch()
                if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
                
                if self.p2px_handler is not None and \
                    not self.args.no_p2p_prerecv:
                    self.p2px_handler.prerecv_suc(sucinfo.p2pin())

                if swapin_stashx_handler is not None and \
                    not self.args.no_prefetch_stashx:
                    # sucinfo.stashx()：
                    #
                    swapin_stashx_handler.prefetch_suc(sucinfo.stashx())
                if self.nvprof: nvtx_range_pop() 

            ### Recompute on GPU
            turn_on_X_grad(X_named_tensors)
            if self.nvprof: nvtx_range_push("task{}({}) Recompute(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
           
            if len(vt.layers) > 1: # packed
                Y_tensors = [X_named_tensors[name] for name in X_names]
                for l in vt.layers[:-1]:
                    if not requires_grad and l in vt.Out['X']: ### Out {stashX}
                        if self.nvprof: nvtx_range_push("task{}(L{}) SwapOut(#{}StashX)".format(vt.idx, l, ubatch_idx)) 
                        if vt.Out['X'][l].medium == "MSG": # message pass stashed X
                            swapout_stashx_handler.offload(l, 
                                    make_tensors_named(self.local_model[l].X_names, Y_tensors))
                            # print("\trank{}: task{}(L{}) SwapOut(#{}StashX)".format(self.rank, vt.idx, l, ubatch_idx))
                        else:
                            raise NotImplementedError
                    if self.nvprof: nvtx_range_pop() 

                    Y_tensors = self.local_model[l](*Y_tensors)


                    if not isinstance(Y_tensors, tuple):
                        Y_tensors = (Y_tensors,)
                    Y_tensors = list(Y_tensors)
                Y_names = self.local_model[vt.layers[-2]].Y_names
                Y_named_tensors = make_tensors_named(Y_names, Y_tensors)
            else: # only last vlayer
                Y_names = X_names
                Y_named_tensors = X_named_tensors


            ### In {T}
            if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}T)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            T_named_tensors = swp_x.swapin(target_ubatches[ubatch_idx])
            if self.nvprof: nvtx_range_pop() 
            
            ### Compute loss on GPU
            assert vt.layers[-1] == self.CONFIGS['R']-1
            last_vlayer = self.local_model[self.CONFIGS['R']-1]        
            if self.compute_loss is not None: # "bert_thomwolf", "gpt2_2bw", "gpt2_huggingface"
                Y_tensors = self.compute_loss(last_vlayer, Y_named_tensors, Y_names, T_named_tensors)
            else:
                Y_tensors = [last_vlayer(Y_named_tensors[name],T_named_tensors["target"]) for name in Y_names]
                Y_tensors = [sum(Y_tensors)]
            if self.verbose: print("\trank{}: task{}({}) Recompute(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            if self.nvprof: nvtx_range_pop() 

            if is_last_ubatch:
                self.pretrain_handler.reset_is_last_batch()

            ### Save Y
            Y_named_tensors = make_tensors_named(['loss'], Y_tensors)
            ### Clean up
            if self.nvprof: nvtx_range_push("task{}({}) FWDClean(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            # print("\trank{}: task{}({}) FWDClean(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            del T_named_tensors; del Y_tensors; 
            return X_named_tensors, Y_named_tensors

    def _a_pack_backward_an_ubatch(self, vt, ubatch_idx, ubatch_size,
                                X_named_tensors, Y_named_tensors,
                                swapin_localx_handler=None,
                                swapout_localx_handler=None,
                                sucinfo=None,
                                iteration_num=None):
        is_last_ubatch = ubatch_idx == len(vt.ubatchszs) - 1
        ### In {dY}
        if vt.has_criterion:
            dY_named_tensors = ODict({ 'loss': None })
            assert Y_named_tensors['loss'].requires_grad

        else:
            l, m = vt.layers[-1], vt.In['dY'][vt.layers[-1]]
            dY_named_metas = make_dY_named_metas(self.XMETA, ubatch_size, l)
            if m.medium == "P2P":
                if self.nvprof: nvtx_range_push("task{}({}) P2PIn(#{}dY)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_p2p_prerecv:
                    dY_named_tensors = self.p2px_handler.recv(dY_named_metas, src=m.rank)
                else:
                    # now = datetime.datetime.now()
                    dY_named_tensors = self.p2px_handler.prerecv(dY_named_metas, src=m.rank, is_end=is_last_ubatch) 
                # print("\trank{}: task{}({}) P2PIn(#{}dY)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}dY)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_localx:
                    dY_named_tensors = swapin_localx_handler.fetch(l+1, dY_named_metas)
                else:
                    dY_named_tensors = swapin_localx_handler.prefetch(l+1, dY_named_metas, is_last_ubatch)
                # print("\trank{}: swp_recv'ed L{}-dY".format(self.rank, l))
            else:
                raise NotImplementedError
            if self.nvprof: nvtx_range_pop()         
       

        ### Prefetch point @ BWD's ULast
        if is_last_ubatch:
            self.pretrain_handler.set_is_last_batch()
            if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
            if self.p2px_handler is not None and \
                not self.args.no_p2p_prerecv and not vt.has_criterion:
                self.p2px_handler.prerecv_suc(sucinfo.p2pin())
            if swapin_localx_handler is not None and \
                not self.args.no_prefetch_localx:
                swapin_localx_handler.prefetch_suc(sucinfo.localx())
            if self.nvprof: nvtx_range_pop() 

        ### Compute backward pass
        if self.nvprof: nvtx_range_push("task{}({}) BWD(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
        Y_tensors = []
        Y_gradients = [] 
        for name in self.local_model[vt.layers[-1]].Y_names: # only tensor & required_grad can run autograd
            Y = Y_named_tensors[name]
            # print(f"rank:{self.rank}, Y.grad:{Y.grad}")
            if isinstance(Y,(torch.Tensor, Variable)) and (Y.requires_grad):
                Y_tensors.append(Y)
                Y_gradients.append(dY_named_tensors[name])
            elif isinstance(Y, list): # output tuple of bert pretrainheader
                for i, y in enumerate(Y):
                    if isinstance(y,(torch.Tensor, Variable)) and (y.requires_grad):
                        Y_tensors.append(y)
                        Y_gradients.append(dY_named_tensors[name][i])  


        torch.autograd.backward(tuple(Y_tensors), grad_tensors=tuple(Y_gradients))
        if self.verbose: print("\trank{}: task{}({}) BWD(#{},{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx, ubatch_size))
        if self.nvprof: nvtx_range_pop() 

        if is_last_ubatch:
            self.pretrain_handler.reset_is_last_batch()

        ### Out {dX}
        if vt.Out['dX']:
            ### Save dX
            dX_named_tensors = make_dX_from_X(X_named_tensors) # ref to .grad
            l, m = vt.layers[0], vt.Out['dX'][vt.layers[0]] 
            if m.medium == "P2P":
                if self.nvprof: nvtx_range_push("task{}({}) P2POut(#{}dX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                # now = datetime.datetime.now()
                self.p2px_handler.isend(dX_named_tensors, dst=m.rank)
                # print("\trank{}: task{}({}) P2POut(#{}dX)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapOut(#{}dX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.is_convert_ubs:
                    swapout_localx_handler.offload(l, dX_named_tensors, False)
                else:
                    swapout_localx_handler.offload(l, dX_named_tensors)
                # print("\trank{}: swp_send'ed L{}-dX".format(self.rank,l))
            else:
                raise NotImplementedError
            del dX_named_tensors; 
            if self.nvprof: nvtx_range_pop() 
        ### Clean up {X,Y,dX,dY}
        if self.nvprof: nvtx_range_push("task{}({}) BWDClean(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
        # print("\trank{}: task{}({}) BWDClean(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
        del X_named_tensors; del Y_named_tensors
        del dY_named_tensors; del Y_tensors; del Y_gradients

    # 
    def run_training_loop(self):
        local_losses = [] # per-minibatch
        global_losses = [] # per-minibatch
        if self.args.no_update: 
            grad_sums = [] # per-minibatch
        self.update_cnt = 0
        self.time_iters = []

        self.delete_time = []
        self.get_time = []
        self.swapout_time = []
        self.gc_time = []
        self.fwd_compute_time = []
        self.bwd_recompute_time = []
        self.bwd_compute_time = []
        self.update_time = []
        self.offload_time = []
        self.prehook_time = []
        self.delete_time_in_fwd = []
        self.fetch_at_minibatch_start = []

        self.avg_it = int(self.args.num_iters * self.args.num_epochs /2.) # from this iter to average time
        
        ### clean memory before start
        torch.cuda.synchronize(self.rank); dist.barrier()
        gc.collect(); torch.cuda.empty_cache() 
        torch.cuda.synchronize(self.rank)
        self.cgm = CheckGPUMem(self.rank)
        dist.barrier()
        print("rank%d: --- training starts ---" % self.rank)
        
        ### start
        self.rand_state_train.set()
        for epoch in range(self.args.num_epochs): # traverse epoches
            for it, minibatch in enumerate(self.data_loader): # traverse each minibatch
                if it >= self.args.num_iters:
                    break
                ### clean start
                # gc.collect() 
                if self.args.empty_cache: torch.cuda.empty_cache() 
                # torch.cuda.empty_cache()
                torch.cuda.synchronize(self.rank)
                # assert torch.cuda.memory_allocated(self.rank)==0, "iteration begins w/ alloc = {} B".format(torch.cuda.memory_allocated(self.rank)) 
                #     self.rank, epoch, self.args.num_epochs, it, self.args.num_iters, 
                # print(ps)
                dist.barrier()
                if self.nvprof and it == self.args.nvprof_iter["start"]:
                    probe_cuda_mem = ProbeCudaMem(self.rank)
                    probe_cuda_mem.start()  
                    cuda_profiler.start()
                    nvtx_mark("cudaProfilerStart") 
                    print("rank%d: cuda profiler starts"%self.rank)
                else:
                    torch.cuda.reset_peak_memory_stats(self.rank) 
                time_start = pc() 
                ### data minibatch
                if self.args.synthetic_data:
                    data_ubatches, target_ubatches = self.data_ubatches, self.target_ubatches
                else:
                    #
                    if self.is_copy_minibatch: # "gpt2_huggingface"
                        minibatch = (minibatch, deepcopy(minibatch))

                    if self.is_skip_minibatch(minibatch, self.CONFIGS['D'], self.fdim, verbose=self.verbose): # skip fractional minibatch
                        assert (not self.nvprof) or (self.nvprof and it != self.args.nvprof_iter["end"]), "Unstoped Profiling"
                        continue
                    minibatch = self.preprocess_minibatch(minibatch) # preprocess as if single GPU

                    # self.bnames：{"is_data" = [True, False]， "name" = ["input0", "labels"]}
                    #
                    data_ubatches, target_ubatches = decompose_minibatch(minibatch, self.bnames, self.ubatchszs_fwd_local, self.ubatchszs_bwd_local, self.XMETA, self.TMETA, self.CONFIGS, self.rank, pin_memory=not self.args.no_pin_data) # make microbatches
                ### task starts    
                delete_time = []
                prehook_time = []
                if self.nvprof: nvtx_range_push("rank{}, iteration:{}".format(self.rank,it))
                delete_time_in_fwd = []
                get_time = []
                swapout_time = []
                fwd_compute_time = []
                bwd_compute_time = []
                bwd_recompute_time = []
                fetch_at_minibatch_start = []
                update_time = []
                offload_time = []
                for j, vt in enumerate(self.rTASKS[self.rank]): # { rank0: [task0,task2,...] }
                    if self.verbose: print("\trank{}: executing {}".format(self.rank, vt))
                    self.sucinfo.set(vt, j)
                    if self.nvprof: nvtx_range_push("task{}({})({})".format(vt.idx, vt.show_layers(), vt.type)) 
                    
                    vt_start_time = time.time()
                    if vt.type == 'FWD' and vt.is_gpu:
                        # -----------------------------------------------      
                        with torch.no_grad():
                            ### Swap-in model {W,B}
                            # if self.nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                            # if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                            # suc_vt = None if self.args.no_prefetch_model else self.sucinfo.model()

                            # cur_vt_idx = self.prefetch_model_handler.get(vt, suc_vt)
                            # assert cur_vt_idx == vt.idx
                            # if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                            # if self.nvprof: nvtx_range_pop() 
                            ### Run through each microbatch in a data batch

                            start_time = time.time()
                            if j == 0:
                                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                                for layer_id in self.layers_id_in_rank[:self.gl_window_size]:
                                    # _model = self.local_model[layer_id]
                                    # _model.alloc_param_buf()
                                    if layer_id in self.special_layers_in_rank:
                                        _cache_unit = self.special_cuda_queue[layer_id]
                                    else:
                                        _cache_unit = self.cuda_cache_queue.pop()
                                    # _model.copyin_param_buf(_cache_unit)
                                    self.prefetch_model_handler.iput(layer_id, vt, _cache_unit)
                                    self.pretrain_handler.layers_in_gpu.append(layer_id)
                                if self.nvprof: nvtx_range_pop() 
                                end_time = time.time()
                                execution_time = end_time - start_time
                                fetch_at_minibatch_start.append(execution_time)


                            print(f"rank:{self.rank}, vt:{vt.layers}, ubatchszs:{vt.ubatchszs}")
                            for i, u in enumerate(vt.ubatchszs):
                                self.pretrain_handler.set_requires_grad(False)
                                if i == 0:
                                    self.pretrain_handler.set_is_first_ubatch()
                                start_time = time.time()
                                self._a_pack_forward_an_ubatch(vt, i, u,
                                        data_ubatches, target_ubatches, 
                                        requires_grad=False, 
                                        prefetch_model_handler=self.prefetch_model_handler,
                                        swapin_stashx_handler=self.swapin_stashx_handler,
                                        swapin_localx_handler=self.swapin_localx_handler,
                                        swapin_msgx_handler=self.swapin_msgx_handler,
                                        swapout_stashx_handler=self.swapout_stashx_handler,
                                        swapout_localx_handler=self.swapout_localx_handler,
                                        swapout_msgx_handler=self.swapout_msgx_handler,
                                        sucinfo=self.sucinfo,
                                        delete_time=delete_time_in_fwd,
                                        prehook_time=prehook_time,
                                        wait_prefetch_time=get_time)
                                end_time = time.time()
                                execution_time = end_time - start_time
                                fwd_compute_time.append(execution_time)
                                if self.verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                                if self.nvprof: nvtx_range_pop() 
                                # gc.collect()
                                if i == 0:
                                    self.pretrain_handler.reset_is_first_ubatch()

                            if self.swapin_msgx_handler is not None and not self.args.no_prefetch_msgx:
                                self.swapin_msgx_handler.prefetch_suc(self.sucinfo.msgx())
                        # -----------------------------------------------
                    elif vt.type == 'BWD' and vt.is_gpu:
                        is_last_bwd_vt = getattr(vt, 'is_last_bwd_vt', False)

                        if j == 0:
                            if self.nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                            start_time = time.time()
                            for layer_id in self.layers_id_in_rank[:self.gl_window_size]:
                                # _model.alloc_param_buf()
                                if layer_id in self.special_layers_in_rank:
                                    _cache_unit = self.special_cuda_queue[layer_id]
                                else:
                                    _cache_unit = self.cuda_cache_queue.pop()
                                self.prefetch_model_handler.iput(layer_id, vt, _cache_unit)
                                self.pretrain_handler.layers_in_gpu.append(layer_id)
                            if self.nvprof: nvtx_range_pop() 
                            end_time = time.time()
                            execution_time = end_time - start_time
                            fetch_at_minibatch_start.append(execution_time)

                        # -----------------------------------------------
                        ### Swap-in model {W,B}
                        # if self.nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                        # if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                        # suc_vt = None if self.args.no_prefetch_model else self.sucinfo.model()

                        # cur_vt_idx = self.prefetch_model_handler.get(vt, suc_vt)
                        # assert cur_vt_idx == vt.idx
                        # if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                        # if self.nvprof: nvtx_range_pop() 

                        ### Run through each microbatch in a data batch. 
                        m_loss = 0. # loss averaged across examples in this minibatch
                        for i, u in enumerate(vt.ubatchszs):
                            self.pretrain_handler.set_requires_grad(True)
                            if i == 0:
                                self.pretrain_handler.set_is_first_ubatch()
                            ### Recompute to create pytorch graph
                            start_time = time.time()
                            X_named_tensors, Y_named_tensors = \
                                self._a_pack_forward_an_ubatch(vt, i, u,
                                                            data_ubatches, target_ubatches, 
                                                            requires_grad=True,
                                                            swapin_stashx_handler=self.swapin_stashx_handler,
                                                            swapin_localx_handler=self.swapin_localx_handler,
                                                            swapin_msgx_handler=self.swapin_msgx_handler,
                                                            sucinfo=self.sucinfo,
                                                            wait_prefetch_time=get_time)
                            end_time = time.time()
                            execution_time = end_time - start_time
                            bwd_recompute_time.append(execution_time)
                            if self.nvprof: nvtx_range_pop() 
                            if 'loss' in Y_named_tensors:
                                # Y_named_tensors['loss'] /= len(vt.ubatchszs) # NOTE: ubatches need to be equal
                                Y_named_tensors['loss'] = Y_named_tensors['loss'].clone() / len(vt.ubatchszs)
                                m_loss += Y_named_tensors['loss'].item()
                            ### Backward pass on recomputed graph
                            start_time = time.time()
                            self._a_pack_backward_an_ubatch(vt, i, u,
                                                        X_named_tensors, Y_named_tensors,
                                                        swapin_localx_handler=self.swapin_localx_handler,
                                                        swapout_localx_handler=self.swapout_localx_handler,
                                                        sucinfo=self.sucinfo,
                                                        iteration_num=it)
                            end_time = time.time()
                            execution_time = end_time - start_time
                            bwd_compute_time.append(execution_time)
                            if self.verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                            if self.nvprof: nvtx_range_pop()
                            ### Clean up
                            del X_named_tensors; del Y_named_tensors # very important!
                            # gc.collect()
                            self.pretrain_handler.set_requires_grad(False)
                            if i == 0:
                                self.pretrain_handler.reset_is_first_ubatch()
                        ### Prefetch point @ AllReduce
                        # if not self.args.no_prefetch_model:
                        #     if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
                        #     self.prefetch_model_handler.iput(suc_vt) 
                        #     if self.nvprof: nvtx_range_pop() 
                        ### Optional dW aggregation (and B sync)
                        if self.CONFIGS["mode"] == 'vDP' and self.CONFIGS['N'] > 1:
                            self.default_stream.synchronize() # TODO: wait Compute by cuda event 
                            if self.nvprof: nvtx_range_push("task{}({}) AllReduce(dW,B)".format(vt.idx, vt.show_layers())) 
                            for l in vt.layers:
                                self.local_model[l].average_grad(self.p2pm_handler)
                                if self.args.average_buffer:
                                    self.local_model[l].average_buf(self.p2pm_handler) # optional: all rank average buffers (can comment out to only use rank0's buf)
                            # TODO: wait AllReduce finish by cuda event
                            if self.nvprof: nvtx_range_pop() 
                        if m_loss != 0.:
                            local_losses.append(m_loss)
                            global_losses.append(m_loss)

                        ### Swap-out model {W,dW,B}
                        if self.CONFIGS["opt_offld"]:                   
                            # ### Clip dW for "gpt2_huggingface"
                            #     self.default_stream.synchronize()
                            #     for l in vt.layers:
                            #         torch.nn.utils.clip_grad_norm_(self.local_model[l].model.parameters(), self.args.max_grad_norm) 
                            ### Out {W,dW,B}
                            self.default_stream.synchronize() # CPU wait
                            l = vt.layers[0]
                            # futs = [self.pretrain_handler.layer_to_cpu_bwd_remote(l, vt)]
                            self.pretrain_handler.should_stay_in_gpu[l] = False
                            self.pretrain_handler.layers_in_gpu.remove(l)
                            self.pretrain_handler.already_get[l] = False
                            # self.update_handler.iput(vt, l, futs)
                            self.update_handler.iput(vt, l)
                        else:
                            raise ValueError("GPU Optimizer Underdevelopment.")
                        # -----------------------------------------------
                    elif vt.type == 'UPD' and not vt.is_gpu:
                        # -----------------------------------------------
                        ### In {dW,W,K} Out {W,K}
                        pass
                        # -----------------------------------------------
                    else:
                        raise ValueError("Unknown vTask.type {} with .device {} !".format(vt.type,vt.device))
                    if self.nvprof: nvtx_range_pop() 
                    vt_end_time = time.time()
                if self.nvprof: nvtx_range_pop() 

                gpu_task_finish_time = pc()
                print(f"rank:{self.rank}, ========gpu_task_finish_time: {gpu_task_finish_time - time_start}秒==========", flush=True)

                ### tasks iteration ends
                if not self.args.no_update:
                    self.update_handler.synchronize()
                    self.update_cnt += 1
                torch.cuda.synchronize(self.rank)
                dist.barrier()
                ### statistics
                self.time_iters.append(pc()-time_start) 
                if self.nvprof and it == self.args.nvprof_iter["end"]:
                    nvtx_mark("cudaProfilerStop") 
                    cuda_profiler.stop()
                    probe_cuda_mem.stop()
                    print("rank%d: cuda profiler stops"%self.rank)
                ## if it % self.args.display_period == 0:
                ps = "rank%d: Epoch%d/%d Iter%d/%d %.3f sec, %.3f/%.3f GB" % ( 
                    self.rank, epoch, self.args.num_epochs, it, self.args.num_iters, 
                    self.time_iters[-1],
                    float(torch.cuda.memory_allocated()) / 1024**3,
                    float(torch.cuda.memory_reserved()) / 1024**3)
                # 
                if local_losses != []:
                    np.save(os.path.join(self.args.output_dir, "local_losses_rank%d.npy"%self.rank), local_losses)
                if self.CONFIGS["mode"] == 'vDP' and self.CONFIGS['N'] > 1:
                    global_losses[-1] = allreduce_cpu_loss(global_losses[-1], averaging=True)
                if self.rank == self.CONFIGS['loss_rank']:
                    ps += ", Loss %.3f"% global_losses[-1]
                    np.save(os.path.join(self.args.output_dir, "train_losses.npy"), global_losses)
                print(ps)

                self.delete_time.append(sum(delete_time))
                self.get_time.append(sum(get_time))
                self.swapout_time.append(sum(swapout_time))
                self.fwd_compute_time.append(sum(fwd_compute_time))
                self.bwd_recompute_time.append(sum(bwd_recompute_time))
                self.bwd_compute_time.append(sum(bwd_compute_time))
                self.prehook_time.append(sum(prehook_time))
                self.delete_time_in_fwd.append(sum(delete_time_in_fwd))
                self.fetch_at_minibatch_start.append(sum(fetch_at_minibatch_start))
                if self.args.no_update:
                    assert self.CONFIGS["mode"] !='vPP'
                    gs = checker.check_grad_sum_harmony(self.shared_model)
                    grad_sums.append(gs)
                    np.save(os.path.join(self.args.output_dir, "grad_sums_rank%d.npy"%self.rank), grad_sums)
                # check GPU OoM & cudaFree & cudaMalloc
                self.cgm.check(it, is_check_malloc=not self.args.empty_cache and len(self.time_iters)-1 >= self.avg_it)
        ### end training
        torch.cuda.synchronize(self.rank)
        dist.barrier()
        print("rank%d: --- done ---" % self.rank)

    # 
    def finish(self): 
        self.pretrain_handler.executor.shutdown()

        ### statistics
        if self.verbose:
            print_p2p_bytes(self.rank, self.p2px_handler, self.p2pm_handler, self.update_cnt)

        avg_iter_time = np.mean(self.time_iters[self.avg_it:]) # sec
        avg_fwd_compute_time = np.mean(self.fwd_compute_time[self.avg_it:])
        avg_bwd_recompute_time = np.mean(self.bwd_recompute_time[self.avg_it:])
        avg_bwd_compute_time = np.mean(self.bwd_compute_time[self.avg_it:])
        avg_delete_time = np.mean(self.delete_time[self.avg_it:])
        avg_get_time = np.mean(self.get_time[self.avg_it:])
        avg_swapout_time = np.mean(self.swapout_time[self.avg_it:])
        avg_prehook_time = np.mean(self.prehook_time[self.avg_it:])
        avg_delete_time_in_fwd = np.mean(self.delete_time_in_fwd[self.avg_it:])
        avg_fetch_at_minibatch_start = np.mean(self.fetch_at_minibatch_start[self.avg_it:])
        # CONFIGS["D"]：minibatchsize
        avg_throughput = self.CONFIGS['D'] / avg_iter_time # samples/sec
        gpu_reserved = gather_integer(torch.cuda.memory_reserved(), self.rank) # bytes
        if self.rank == 0:
            gpu_reserved = " ".join("%.1f"%(float(byte)/1024**3) for byte in gpu_reserved) # GB
            cpu_occupied = self.pcm.system_cpu_memory(["occupied"])
            # 
            print("[Global] Iter[%d,%d) Avg Iter Time: %.3f sec, Avg Throughput: %.3f sample/s, GPU: (%s) GB, CPU: %s, Num Updates: %d\n" % (self.avg_it, len(self.time_iters), avg_iter_time, avg_throughput, gpu_reserved, cpu_occupied, self.update_cnt))
            self.pcm.print("rank%d: eventually" % self.rank)
        ### save model
        if self.args.save_final_model and self.rank == 0 and self.save_model is not None:
            self.save_model(self.args, self.shared_model, self.update_cnt)
