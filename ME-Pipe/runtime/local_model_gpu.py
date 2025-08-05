# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import gc
import threading
from math import ceil
from collections import OrderedDict as ODict

import torch
from torch.nn import Parameter

from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

from task_data_struct import Medium, vTask
import threadsafe_data_struct

if os.environ.get('CUDA_LAUNCH_BLOCKING') in ['1','True', True]:
    MEMCPY_NONBLK = False
else:
    MEMCPY_NONBLK = True

import time

def delete_param_grad_buf(top_module, manual_gc=False):
    ''' Recursively delete all params, grads, and buffers from this module on either CPU or GPU.

        Note: nn.Parameters (namely, Variable) is wrapper of torch.Tenosr. The core tensor can be accessed by param.data, but not recommended (access .data is source of all evils ref: https://discuss.pytorch.org/t/how-to-delete-every-grad-after-training/63644) 
        Note: Delete param? 
            -. param.data/param.data.storage() can not be del'ed
            -. for param in self.local_model_gpu.parameters(): del param # doesn't affect content
            -. param.data = None # TypeError: Variable data has to be a tensor, but got NoneType
            -. del moduel._parameters[key] will leave moduel._parameters[key]=None. Then have to new Parameter(). Then del new Parameter can cause uncollectable alloc on GPU.
            +. param.data = torch.empty(0, device="cpu") # use pytorch's current behavior -- in-place update and let python do the gc, working for both GPU and CPU (equal to del tensor)
        Note: Assign grad?
            -. param.data.grad can not be assigned 
            -. param.grad.data = only tensor, not None
            +. param.grad = * instead, 
            +. param.grad = None works for both GPU and CPU (equal to del tensor)
        Note: Delete buffer?
            +. del _buffer[key] # works
            +. _buffers[key] = fn(buf) # works
            +. buffer has no grad
    '''   
    @torch.no_grad()
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # delete param
                param.data = torch.empty(0, device="cpu")
                # delete grad
                if param.grad is not None:
                    param.grad = None
                param.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
                assert not param.requires_grad
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                m._buffers[key] = torch.empty(0, device="cpu")
    top_module.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    #
    if manual_gc:
        gc.collect(); torch.cuda.empty_cache() # can block all cudaStreams

class LocalModelGPU(object):
    """ A wrapper class of process-local vlayer on GPU.
    """
    def __init__(self, pinned_model, shared_model, empty_model, id, X_names, Y_names, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this in subprocess """ 
        self.pinned_model = pinned_model
        self.shared_model = shared_model
        self.id = id # vlayer_id
        self.X_names = X_names
        self.Y_names = Y_names
        self.rank, self.world_size = rank, world_size
        assert self.rank == torch.cuda.current_device()
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        if not self.no_pin_model:
            # confirm pinned model is pinned, local, CPU, and no grad
            for param in self.pinned_model.parameters():
                assert param.data.is_pinned() and (not param.data.is_shared()) and (not param.data.is_cuda)
                assert (not param.requires_grad) and (param.grad is None)
            for buf in self.pinned_model.buffers():
                assert buf.data.is_pinned() and (not buf.data.is_shared()) and (not buf.data.is_cuda)
                assert (not buf.requires_grad) and (buf.grad is None)
        
        # use existing empty model one
        assert isinstance(empty_model, torch.nn.Module)
        self.model = empty_model
        
        # self.model.cuda() # Moves all model parameters and buffers to the GPU. (replace CPU params and buffers with newly alloacated GPU param and buffers) Return self module.
        self.swapin_param_buf(True)
        # initialize empty shell on GPU
        self.del_param_grad_buf(manual_gc=True)
        # print("[LocalModelGPU][id%d] rank%d: initialized local model on GPU (empty shell)"%(self.id, self.rank))
    
    def del_param_grad_buf(self, manual_gc=False):
        delete_param_grad_buf(self.model, manual_gc=manual_gc)
   
    @torch.no_grad()
    def swapin_param_buf(self, forward_only=True): 
        ''' Recursively allocate and copy-in all params and buffers from cpu module to self.local_model_gpu
            
            Note: if gpu_m._parameters[key] is previously del'ed to None, then swapin needs to create a new Parameter. Then it may leave uncollectable allocation on GPU after del this new'ed Parameter.
        '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()): # iterator over all modules in the network
            # print("{} <- {}".format(gpu_m, cpu_m))
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data = param.cuda(non_blocking=MEMCPY_NONBLK) # Returns a copy of this tensor object in CUDA memory.
                    # assert gpu_m._parameters[key].grad is None and (not gpu_m._parameters[key].requires_grad), "swapin requires no grad for both FP and BP"
                    # if not forward_only: 
                    #     gpu_m._parameters[key].requires_grad_(True)
                    # assert not param.is_cuda
                    # print("\t _parameter[{}]".format(key))
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    # if len(buf.shape) == 0: # Scalar buffer stays on CPU
                    #     gpu_m._buffers[key] = buf.clone().detach().requires_grad_(False)
                    # else:
                    gpu_m._buffers[key] = buf.cuda(non_blocking=MEMCPY_NONBLK) # Returns a copy of this tensor object in CUDA memory.
                    # assert not buf.is_cuda and (not gpu_m._buffers[key].requires_grad)
                    # print("\t _buffers[{}]".format(key))
        # print("[LocalModelGPU] rank{} swapin'ed params and bufs".format(self.rank))
        
        if not forward_only: # backward # move to here for 1) batching swap on GPU, 2) GPU CPU parallelism
            self.set_param_requiresgrad()
    
    @torch.no_grad()
    def set_param_requiresgrad(self, requires_grad=True): 
        ''' From def swapin_param_buf() '''
        for gpu_m in self.model.modules(): # iterator over all modules in the network
            for key, param in gpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    param.requires_grad_(requires_grad)

    @torch.no_grad()
    def alloc_param_buf(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data = torch.empty(param.shape,  dtype=param.dtype, device=self.rank, requires_grad=False)
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key] = torch.empty(buf.shape,  dtype=buf.dtype, device=self.rank, requires_grad=False)
    
    @torch.no_grad()
    def copyin_param_buf(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data.copy_(param.data, non_blocking=MEMCPY_NONBLK) # inplace copy
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key].data.copy_(buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy

    @torch.no_grad()
    def copyin_param_buf_blocking(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                   gpu_m._parameters[key].data.copy_(param.data, non_blocking=False) # inplace copy
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key].data.copy_(buf.data, non_blocking=False) # inplace copy

    def __call__(self, *input, **kwargs):
        return self.model(*input, **kwargs)
        # ref: https://github.com/pytorch/pytorch/blob/472be69a736c0b2aece4883be9f8b18e2f3dfbbd/torch/nn/modules/module.py#L487
    
    def average_grad(self, p2p_handler):
        p2p_handler.average_gradients(self.model)

    def average_buf(self, p2p_handler):
        p2p_handler.average_buffers(self.model)

    @torch.no_grad()
    def swapout_grad(self):
        ''' in-place copy gradient from local model gpu to local .grad on cpu 
            (lazily allocate local .grad if not exists)
        '''
        for gpu_param, cpu_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if gpu_param.grad is not None:
                if cpu_param.grad is None:
                    assert cpu_param.requires_grad
                    g = torch.zeros(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False)
                    if not self.no_pin_grad_buf:
                        g = g.pin_memory()
                    cpu_param.grad = g
                    # cpu_param.grad = torch.empty(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False, pin_memory=not self.no_pin_grad_buf) # NOTE: dont' use empty(pin_memory) with shared memory 
                    assert not cpu_param.grad.is_shared() # and cpu_param.grad.is_pinned()
                cpu_param.grad.data.copy_(gpu_param.grad.data, non_blocking=MEMCPY_NONBLK) # in-place copies the elements from src tensor into self tensor (and returns self)
            else:
                print(ValueError("model has grad = None to swap"))
                cpu_param.grad = None
        # print("[LocalModelGPU] rank{} swapout'ed grad".format(self.rank))

    @torch.no_grad()
    def swapout_grad_blocking(self):
        ''' in-place copy gradient from local model gpu to local .grad on cpu 
            (lazily allocate local .grad if not exists)
        '''
        for gpu_param, cpu_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if gpu_param.grad is not None:
                if cpu_param.grad is None:
                    assert cpu_param.requires_grad
                    g = torch.zeros(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False)
                    if not self.no_pin_grad_buf:
                        g = g.pin_memory()
                    cpu_param.grad = g
                    # cpu_param.grad = torch.empty(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False, pin_memory=not self.no_pin_grad_buf) # NOTE: dont' use empty(pin_memory) with shared memory 
                    assert not cpu_param.grad.is_shared() # and cpu_param.grad.is_pinned()
                cpu_param.grad.data.copy_(gpu_param.grad.data, non_blocking=False) # in-place copies the elements from src tensor into self tensor (and returns self)
            else:
                print(ValueError("model has grad = None to swap"))
                cpu_param.grad = None
        # print("[LocalModelGPU] rank{} swapout'ed grad".format(self.rank))

    @torch.no_grad()
    def swapout_buf(self): 
        ''' in-place copy buffers from local model gpu to local pinned buf or shared buf on cpu 
            (lazily allocate local pinned buf if not exists)
        '''
        if self.no_pin_grad_buf:
            for gpu_buf, cpu_buf in zip(self.model.buffers(), self.shared_model.buffers()):
                if gpu_buf is not None:
                    cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert cpu_buf is None, "local_model_gpu has unexpected None buffer to swap"
        else:
            if not hasattr(self.shared_model, "pinned_buf"):
                self.shared_model.pinned_buf = ODict()
                for name, buf in self.shared_model.named_buffers():
                    if buf is not None:
                        pin_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu", requires_grad=False).pin_memory()
                        # NOTE: dont' use torch.empty(pin_memory) with shared memory 
                        assert not pin_buf.is_shared() and pin_buf.is_pinned()
                    else:
                        pin_buf = None
                    self.shared_model.pinned_buf[name] = pin_buf
                for (gpu_name, gpu_buf), (pin_name, pin_buf) in zip(self.model.named_buffers(), self.shared_model.pinned_buf.items()):
                    assert gpu_name == pin_name
                    assert (gpu_buf is not None and pin_buf is not None) or \
                           (gpu_buf is None and pin_buf is None) 
            named_pin_buf = self.shared_model.pinned_buf
            for name, gpu_buf in self.model.named_buffers():
                if gpu_buf is not None:
                    named_pin_buf[name].data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local_model_gpu has unexpected None buffer to swap"

        # for gpu_buf, cpu_buf in zip(self.model.buffers(), self.pinned_model.buffers()):
        #     cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK) # Copies the elements from src tensor into self tensor (and returns self); in-place copy (no new tensor created)
        # print("[LocalModelGPU] rank{} swapout'ed buf".format(self.rank))

    @torch.no_grad()
    def swapout_buf_blocking(self): 
        ''' in-place copy buffers from local model gpu to local pinned buf or shared buf on cpu 
            (lazily allocate local pinned buf if not exists)
        '''
        if self.no_pin_grad_buf:
            for gpu_buf, cpu_buf in zip(self.model.buffers(), self.shared_model.buffers()):
                if gpu_buf is not None:
                    cpu_buf.data.copy_(gpu_buf.data, non_blocking=False)
                else:
                    assert cpu_buf is None, "local_model_gpu has unexpected None buffer to swap"
        else:
            if not hasattr(self.shared_model, "pinned_buf"):
                self.shared_model.pinned_buf = ODict()
                for name, buf in self.shared_model.named_buffers():
                    if buf is not None:
                        pin_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu", requires_grad=False).pin_memory()
                        # NOTE: dont' use torch.empty(pin_memory) with shared memory 
                        assert not pin_buf.is_shared() and pin_buf.is_pinned()
                    else:
                        pin_buf = None
                    self.shared_model.pinned_buf[name] = pin_buf
                for (gpu_name, gpu_buf), (pin_name, pin_buf) in zip(self.model.named_buffers(), self.shared_model.pinned_buf.items()):
                    assert gpu_name == pin_name
                    assert (gpu_buf is not None and pin_buf is not None) or \
                           (gpu_buf is None and pin_buf is None) 
            named_pin_buf = self.shared_model.pinned_buf
            for name, gpu_buf in self.model.named_buffers():
                if gpu_buf is not None:
                    named_pin_buf[name].data.copy_(gpu_buf.data, non_blocking=False)
                else:
                    assert named_pin_buf[name] is None, "local_model_gpu has unexpected None buffer to swap"

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class LocalModelGPU_work6_4_2(object):
    """ A wrapper class of process-local vlayer on GPU.
    """
    def __init__(self, pinned_model, shared_model, empty_model, layer_pool, cuda_cache_queue, special_layers_in_rank, id, X_names, Y_names, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this in subprocess """ 
        self.pinned_model = pinned_model
        self.shared_model = shared_model
        self.id = id # vlayer_id
        self.X_names = X_names
        self.Y_names = Y_names
        self.rank, self.world_size = rank, world_size
        assert self.rank == torch.cuda.current_device()
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf
        self.cuda_cache_queue = cuda_cache_queue
        self.special_layers_in_rank = special_layers_in_rank
        self.layer_pool = layer_pool
        # use existing empty model one
        assert isinstance(empty_model, torch.nn.Module)
        self.model = empty_model
        self.pinned_data = None

        # self.model.cuda() # Moves all model parameters and buffers to the GPU. (replace CPU params and buffers with newly alloacated GPU param and buffers) Return self module.
        # self.swapin_param_buf(True)
        # initialize empty shell on GPU
        # self.del_param_grad_buf(manual_gc=True)
        # print("[LocalModelGPU][id%d] rank%d: initialized local model on GPU (empty shell)"%(self.id, self.rank))
    
    def del_param_grad_buf(self, manual_gc=False):
        delete_param_grad_buf(self.model, manual_gc=manual_gc)
   
    @torch.no_grad()
    def swapin_param_buf(self, forward_only=True): 
        ''' Recursively allocate and copy-in all params and buffers from cpu module to self.local_model_gpu
            
            Note: if gpu_m._para meters[key] is previously del'ed to None, then swapin needs to create a new Parameter. Then it may leave uncollectable allocation on GPU after del this new'ed Parameter.
        '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()): # iterator over all modules in the network
            # print("{} <- {}".format(gpu_m, cpu_m))
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data = param.cuda(non_blocking=MEMCPY_NONBLK) # Returns a copy of this tensor object in CUDA memory.
                    # assert gpu_m._parameters[key].grad is None and (not gpu_m._parameters[key].requires_grad), "swapin requires no grad for both FP and BP"
                    # if not forward_only: 
                    #     gpu_m._parameters[key].requires_grad_(True)
                    # assert not param.is_cuda
                    # print("\t _parameter[{}]".format(key))
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    # if len(buf.shape) == 0: # Scalar buffer stays on CPU
                    #     gpu_m._buffers[key] = buf.clone().detach().requires_grad_(False)
                    # else:
                    gpu_m._buffers[key] = buf.cuda(non_blocking=MEMCPY_NONBLK) # Returns a copy of this tensor object in CUDA memory.
                    # assert not buf.is_cuda and (not gpu_m._buffers[key].requires_grad)
                    # print("\t _buffers[{}]".format(key))
        # print("[LocalModelGPU] rank{} swapin'ed params and bufs".format(self.rank))
        
        if not forward_only: # backward # move to here for 1) batching swap on GPU, 2) GPU CPU parallelism
            self.set_param_requiresgrad()
    
    @torch.no_grad()
    def set_param_requiresgrad(self, requires_grad=True): 
        ''' From def swapin_param_buf() '''
        for gpu_m in self.model.modules(): # iterator over all modules in the network
            for key, param in gpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    param.requires_grad_(requires_grad)

    # @torch.no_grad()
    # def alloc_param_buf(self): 
    #     ''' From def swapin_param_buf() '''
    #     cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
    #     for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
    #         for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
    #             if param is not None:
    #                 gpu_m._parameters[key].data = torch.empty(param.shape,  dtype=param.dtype, device=self.rank, requires_grad=False)
    #         for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
    #             if buf is not None:
    #                 gpu_m._buffers[key] = torch.empty(buf.shape,  dtype=buf.dtype, device=self.rank, requires_grad=False)

    @torch.no_grad()
    def copyin_param_buf(self, cache_unit=None): 
        if hasattr(self, "_gl_cuda_cache_unit"):
            print(f"--------------->rank:{self.rank}, cuda cache unit already exists")
            return
        # if self.id in self.special_layers_in_rank:
        #     _cuda_cache_unit = special_cuda_queue[self.id]
        # else:
        #     _cuda_cache_unit = cuda_cache_queue.pop()
        self._gl_cuda_cache_unit = cache_unit

        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        global_param_index = 0
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    param_applied = cache_unit[global_param_index].copy_(param.data, non_blocking=MEMCPY_NONBLK) # inplace copy
                    grad_applied = cache_unit[str(global_param_index)+".grad"]
                    param_applied.grad = grad_applied
                    gpu_m._parameters[key].data = param_applied
                    global_param_index += 1
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    buf_applied = cache_unit[global_param_index].copy_(buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy
                    gpu_m._buffers[key].data = buf_applied
                    global_param_index += 1

    def set_pinned_data(self):
        if self.pinned_data is not None:
            assert False, f"rank:{self.rank}, layer:{self.id}, pinned_data already exists"
        if self.id in self.special_layers_in_rank:
            pinned_data = self.layer_pool.get_special_pinned_data(self.id)
        else:
            pinned_data = self.layer_pool.get_pinned_data()
        self.pinned_data = pinned_data
        self.pinned_model = pinned_data.pinned_layer

    def get_pinned_data(self):
        assert self.pinned_data is not None, f"rank:{self.rank}, layer:{self.id}, pinned_data does not exist"
        return self.pinned_data

    def return_pinned_data(self):
        assert self.pinned_data is not None, f"rank:{self.rank}, layer:{self.id}, pinned_data does not exist"
        if self.id not in self.special_layers_in_rank:
            self.layer_pool.return_pinned_data(self.pinned_data)
        self.pinned_data = None
        self.pinned_model = None

    def __call__(self, *input, **kwargs):
        return self.model(*input, **kwargs)
        # ref: https://github.com/pytorch/pytorch/blob/472be69a736c0b2aece4883be9f8b18e2f3dfbbd/torch/nn/modules/module.py#L487
    
    def average_grad(self, p2p_handler):
        p2p_handler.average_gradients(self.model)

    def average_buf(self, p2p_handler):
        p2p_handler.average_buffers(self.model)

    @torch.no_grad()
    def swapout_grad(self):
        ''' in-place copy gradient from local model gpu to local .grad on cpu 
            (lazily allocate local .grad if not exists)
        '''
        for gpu_param, cpu_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if gpu_param.grad is not None:
                if cpu_param.grad is None:
                    assert cpu_param.requires_grad
                    g = torch.zeros(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False)
                    if not self.no_pin_grad_buf:
                        g = g.pin_memory()
                    cpu_param.grad = g
                    # cpu_param.grad = torch.empty(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False, pin_memory=not self.no_pin_grad_buf) # NOTE: dont' use empty(pin_memory) with shared memory 
                    assert not cpu_param.grad.is_shared() # and cpu_param.grad.is_pinned()
                cpu_param.grad.data.copy_(gpu_param.grad.data, non_blocking=MEMCPY_NONBLK) # in-place copies the elements from src tensor into self tensor (and returns self)
            else:
                print(ValueError(f"rank:{self.rank}, layer:{self.id}, model has grad = None to swap"))
                cpu_param.grad = None
        # print("[LocalModelGPU] rank{} swapout'ed grad".format(self.rank))

    @torch.no_grad()
    def swapout_buf(self): 
        ''' in-place copy buffers from local model gpu to local pinned buf or shared buf on cpu 
            (lazily allocate local pinned buf if not exists)
        '''
        if self.no_pin_grad_buf:
            for gpu_buf, cpu_buf in zip(self.model.buffers(), self.shared_model.buffers()):
                if gpu_buf is not None:
                    cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert cpu_buf is None, "local_model_gpu has unexpected None buffer to swap"
        else:
            if not hasattr(self.shared_model, "pinned_buf"):
                self.shared_model.pinned_buf = ODict()
                for name, buf in self.shared_model.named_buffers():
                    if buf is not None:
                        pin_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu", requires_grad=False).pin_memory()
                        # NOTE: dont' use torch.empty(pin_memory) with shared memory 
                        assert not pin_buf.is_shared() and pin_buf.is_pinned()
                    else:
                        pin_buf = None
                    self.shared_model.pinned_buf[name] = pin_buf
                for (gpu_name, gpu_buf), (pin_name, pin_buf) in zip(self.model.named_buffers(), self.shared_model.pinned_buf.items()):
                    assert gpu_name == pin_name
                    assert (gpu_buf is not None and pin_buf is not None) or \
                           (gpu_buf is None and pin_buf is None) 
            named_pin_buf = self.shared_model.pinned_buf
            for name, gpu_buf in self.model.named_buffers():
                if gpu_buf is not None:
                    named_pin_buf[name].data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local_model_gpu has unexpected None buffer to swap"

        # for gpu_buf, cpu_buf in zip(self.model.buffers(), self.pinned_model.buffers()):
        #     cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK) # Copies the elements from src tensor into self tensor (and returns self); in-place copy (no new tensor created)
        # print("[LocalModelGPU] rank{} swapout'ed buf".format(self.rank))

    def zero_grad(self):
        for gpu_param in self.model.parameters():
            if gpu_param.grad is not None:
                gpu_param.grad.zero_()
            else:
                assert False, f"rank:{self.rank}, layer:{self.id}, model has grad = None to swap"

    def append_cuda_cache_unit(self):
        if not hasattr(self, "_gl_cuda_cache_unit"):
            assert False, "cuda cache unit not exists"
        _cuda_cache_unit = self._gl_cuda_cache_unit
        self.cuda_cache_queue.append(_cuda_cache_unit)
        del self._gl_cuda_cache_unit

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

class LocalModelGPU_work7(object):
    """ A wrapper class of process-local vlayer on GPU.
    """
    def __init__(self, shared_model, empty_model, shared_model_nvme, cuda_cache_queue, special_layers_in_rank, id, X_names, Y_names, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this in subprocess """ 
        self.shared_model = shared_model
        self.id = id # vlayer_id
        self.X_names = X_names
        self.Y_names = Y_names
        self.rank, self.world_size = rank, world_size
        assert self.rank == torch.cuda.current_device()
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf
        self.cuda_cache_queue = cuda_cache_queue
        self.special_layers_in_rank = special_layers_in_rank

        # use existing empty model one
        assert isinstance(empty_model, torch.nn.Module)
        self.model = empty_model

        self.pinned_data = None
        self.pinned_model = None
        self.shared_model_nvme = shared_model_nvme

    def del_param_grad_buf(self, manual_gc=False):
        delete_param_grad_buf(self.model, manual_gc=manual_gc)

    @torch.no_grad()
    def set_param_requiresgrad(self, requires_grad=True): 
        ''' From def swapin_param_buf() '''
        for gpu_m in self.model.modules(): # iterator over all modules in the network
            for key, param in gpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    param.requires_grad_(requires_grad)

    @torch.no_grad()
    def copyin_param_buf(self, cache_unit=None): 
        if hasattr(self, "_gl_cuda_cache_unit"):
            print(f"--------------->rank:{self.rank}, cuda cache unit already exists")
            return
        # if self.id in self.special_layers_in_rank:
        #     _cuda_cache_unit = special_cuda_queue[self.id]
        # else:
        #     _cuda_cache_unit = cuda_cache_queue.pop()
        self._gl_cuda_cache_unit = cache_unit

        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        global_param_index = 0
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    param_applied = cache_unit[global_param_index].copy_(param.data, non_blocking=MEMCPY_NONBLK) # inplace copy
                    grad_applied = cache_unit[str(global_param_index)+".grad"]
                    param_applied.grad = grad_applied
                    gpu_m._parameters[key].data = param_applied
                    global_param_index += 1
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    buf_applied = cache_unit[global_param_index].copy_(buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy
                    gpu_m._buffers[key].data = buf_applied
                    global_param_index += 1

    def set_pinned_data(self):
        if self.pinned_data is not None:
            assert False, f"rank:{self.rank}, layer:{self.id}, pinned_data already exists"
        pinned_data = self.shared_model_nvme.get_pinned_data(self.id)
        self.pinned_data = pinned_data
        self.pinned_model = pinned_data.pinned_layer

    def get_pinned_data(self):
        assert self.pinned_data is not None, f"rank:{self.rank}, layer:{self.id}, pinned_data does not exist"
        return self.pinned_data

    def return_pinned_data(self):
        assert self.pinned_data is not None, f"rank:{self.rank}, layer:{self.id}, pinned_data does not exist"
        self.shared_model_nvme.return_pinned_data(self.id, self.pinned_data)
        self.pinned_data = None
        self.pinned_model = None

    def __call__(self, *input, **kwargs):
        return self.model(*input, **kwargs)
        # ref: https://github.com/pytorch/pytorch/blob/472be69a736c0b2aece4883be9f8b18e2f3dfbbd/torch/nn/modules/module.py#L487
    
    def average_grad(self, p2p_handler):
        p2p_handler.average_gradients(self.model)

    def average_buf(self, p2p_handler):
        p2p_handler.average_buffers(self.model)

    @torch.no_grad()
    def swapout_grad(self):
        ''' in-place copy gradient from local model gpu to local .grad on cpu 
            (lazily allocate local .grad if not exists)
        '''
        for gpu_param, cpu_param in zip(self.model.parameters(), self.pinned_model.parameters()):
            if gpu_param.grad is not None:
                if cpu_param.grad is None:
                    assert cpu_param.requires_grad
                    g = torch.zeros(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False)
                    if not self.no_pin_grad_buf:
                        g = g.pin_memory()
                    cpu_param.grad = g
                    # cpu_param.grad = torch.empty(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False, pin_memory=not self.no_pin_grad_buf) # NOTE: dont' use empty(pin_memory) with shared memory 
                    assert not cpu_param.grad.is_shared() # and cpu_param.grad.is_pinned()
                cpu_param.grad.data.copy_(gpu_param.grad.data, non_blocking=MEMCPY_NONBLK) # in-place copies the elements from src tensor into self tensor (and returns self)
            else:
                print(ValueError(f"rank:{self.rank}, layer:{self.id}, model has grad = None to swap"))
                cpu_param.grad = None
        # print("[LocalModelGPU] rank{} swapout'ed grad".format(self.rank))

    @torch.no_grad()
    def swapout_grad_blocking(self):
        ''' in-place copy gradient from local model gpu to local .grad on cpu 
            (lazily allocate local .grad if not exists)
        '''
        for gpu_param, cpu_param in zip(self.model.parameters(), self.pinned_model.parameters()):
            if gpu_param.grad is not None:
                if cpu_param.grad is None:
                    assert cpu_param.requires_grad
                    g = torch.zeros(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False)
                    if not self.no_pin_grad_buf:
                        g = g.pin_memory()
                    cpu_param.grad = g
                    # cpu_param.grad = torch.empty(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False, pin_memory=not self.no_pin_grad_buf) # NOTE: dont' use empty(pin_memory) with shared memory 
                    assert not cpu_param.grad.is_shared() # and cpu_param.grad.is_pinned()
                cpu_param.grad.data.copy_(gpu_param.grad.data, non_blocking=False) # in-place copies the elements from src tensor into self tensor (and returns self)
            else:
                print(ValueError(f"rank:{self.rank}, layer:{self.id}, model has grad = None to swap"))
                cpu_param.grad = None
        # print("[LocalModelGPU] rank{} swapout'ed grad".format(self.rank))

    @torch.no_grad()
    def swapout_buf(self): 
        ''' in-place copy buffers from local model gpu to local pinned buf or shared buf on cpu 
            (lazily allocate local pinned buf if not exists)
        '''
        if self.no_pin_grad_buf:
            for gpu_buf, cpu_buf in zip(self.model.buffers(), self.shared_model.buffers()):
                if gpu_buf is not None:
                    cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert cpu_buf is None, "local_model_gpu has unexpected None buffer to swap"
        else:
            if not hasattr(self.pinned_model, "pinned_buf"):
                self.pinned_model.pinned_buf = ODict()
                for name, buf in self.pinned_model.named_buffers():
                    if buf is not None:
                        pin_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu", requires_grad=False).pin_memory()
                        # NOTE: dont' use torch.empty(pin_memory) with shared memory 
                        assert not pin_buf.is_shared() and pin_buf.is_pinned()
                    else:
                        pin_buf = None
                    self.pinned_model.pinned_buf[name] = pin_buf
                for (gpu_name, gpu_buf), (pin_name, pin_buf) in zip(self.model.named_buffers(), self.pinned_model.pinned_buf.items()):
                    assert gpu_name == pin_name
                    assert (gpu_buf is not None and pin_buf is not None) or \
                           (gpu_buf is None and pin_buf is None) 
            named_pin_buf = self.pinned_model.pinned_buf
            for name, gpu_buf in self.model.named_buffers():
                if gpu_buf is not None:
                    named_pin_buf[name].data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local_model_gpu has unexpected None buffer to swap"

        # for gpu_buf, cpu_buf in zip(self.model.buffers(), self.pinned_model.buffers()):
        #     cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK) # Copies the elements from src tensor into self tensor (and returns self); in-place copy (no new tensor created)
        # print("[LocalModelGPU] rank{} swapout'ed buf".format(self.rank))

    @torch.no_grad()
    def swapout_buf_blocking(self): 
        ''' in-place copy buffers from local model gpu to local pinned buf or shared buf on cpu 
            (lazily allocate local pinned buf if not exists)
        '''
        if self.no_pin_grad_buf:
            for gpu_buf, cpu_buf in zip(self.model.buffers(), self.shared_model.buffers()):
                if gpu_buf is not None:
                    cpu_buf.data.copy_(gpu_buf.data, non_blocking=False)
                else:
                    assert cpu_buf is None, "local_model_gpu has unexpected None buffer to swap"
        else:
            if not hasattr(self.pinned_model, "pinned_buf"):
                self.pinned_model.pinned_buf = ODict()
                for name, buf in self.pinned_model.named_buffers():
                    if buf is not None:
                        pin_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu", requires_grad=False).pin_memory()
                        # NOTE: dont' use torch.empty(pin_memory) with shared memory 
                        assert not pin_buf.is_shared() and pin_buf.is_pinned()
                    else:
                        pin_buf = None
                    self.pinned_model.pinned_buf[name] = pin_buf
                for (gpu_name, gpu_buf), (pin_name, pin_buf) in zip(self.model.named_buffers(), self.shared_model.pinned_buf.items()):
                    assert gpu_name == pin_name
                    assert (gpu_buf is not None and pin_buf is not None) or \
                           (gpu_buf is None and pin_buf is None) 
            named_pin_buf = self.pinned_model.pinned_buf
            for name, gpu_buf in self.model.named_buffers():
                if gpu_buf is not None:
                    named_pin_buf[name].data.copy_(gpu_buf.data, non_blocking=False)
                else:
                    assert named_pin_buf[name] is None, "local_model_gpu has unexpected None buffer to swap"

    def zero_grad(self):
        for gpu_param in self.model.parameters():
            if gpu_param.grad is not None:
                gpu_param.grad.zero_()
            else:
                assert False, f"rank:{self.rank}, layer:{self.id}, model has grad = None to swap"

    def append_cuda_cache_unit(self):
        if not hasattr(self, "_gl_cuda_cache_unit"):
            assert False, "cuda cache unit not exists"
        _cuda_cache_unit = self._gl_cuda_cache_unit
        self.cuda_cache_queue.append(_cuda_cache_unit)
        del self._gl_cuda_cache_unit

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

""" Prefetch LocalModelGPU  """
class PrefetchLocalModelGPU(object):
    """ Handles flexible prefetch of local model (W,B) from pinned model in background thread.
        Step:
        1) main thread: allocate (W,B) on GPU
        2) prefetch thread: get sync'ed pinned model (W,B)
        3) prefetch thread: copy in (W,B) in swapin_cudaStream
        4) prefetch thread: turn on grad
        
        Assumption: 
        1) always in FIFO ordering. put each task, and get prefetched task. 
        2) use cuda event for unblocking next CPU ops

        NOTE: why main thread allocates? i) 'no memory reuse across different stream' -- PyTorch, ii) different threads use different CUDA address spaces
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, syncpin_handler, local_model, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.syncpin_handler = syncpin_handler
        self.local_model = local_model # list
        self.rank = rank
        self.swapin_stream = swapin_stream if swapin_stream is not None else torch.cuda.Stream(device=rank)
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert rank == torch.cuda.current_device()
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vt1, ...] # between main and prefetch thread
        self.get_queue = threadsafe_data_struct.Queue() # [vt1.idx, ...] # between prefetch and main thread
        self.prefetch_thread = threading.Thread(target=self._thread_func)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        self.is_running = False
        
        # print("[PrefetchLocalModelGPU] rank{} started prefetch_thread".format(self.rank))

    #
    def iput(self, vt):
        ''' Call by main thread. Blocking Alloc & Nonblocking CopyIn.
            Assumption: only one iput can be running. '''
        if vt is None:
            return
        assert not self.is_running, "the prefetch is still running"
        self.is_running = True
        assert isinstance(vt, vTask)
        # record previous compute event for swapin stream to wait
        ev_compute = self.compute_stream.record_event()
        # allocation in main thread
        if self.nvprof: nvtx_range_push("task{}({}) Alloc(W,B)".format(vt.idx, vt.show_layers())) 

        time_start = time.time()
        for l in vt.layers: # verified: vt.In['W'].keys==vt.In['B'].keys==vt.layers
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                self.local_model[l].alloc_param_buf()
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                pass
            else: # P2P
                raise ValueError("Underdevelopment")
        time_end = time.time()
        # do the rest in background thread
        self.put_queue.add((vt,ev_compute))
        if self.nvprof: nvtx_range_pop() 

    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt, ev_compute = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) CopyIn(W,B)".format(vt.idx, vt.show_layers())) 
            # get sync'ed pinned model 
            syncpin_vt_idx = self.syncpin_handler.get()
            assert syncpin_vt_idx == vt.idx
            # let swapin stream waits for this compute event 
            self.swapin_stream.wait_event(ev_compute) 
            # copy in and turn on grad
            with torch.cuda.stream(self.swapin_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
                for l in vt.layers:
                    if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                        self.local_model[l].copyin_param_buf()
                        if vt.type == 'BWD':
                            self.local_model[l].set_param_requiresgrad()
                    elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                        if vt.type == 'BWD':
                            self.local_model[l].set_param_requiresgrad()
                    else: # P2P
                        raise ValueError("Underdevelopment")
            # record this swapin event for compute stream to wait
            ev_swapin = self.swapin_stream.record_event() 
            # if MEMCPY_NONBLK: self.swapin_stream.synchronize() # Wait for all the kernels in this stream to complete.
            # ready to use
            self.get_queue.add( (vt.idx,ev_swapin) )
            if self.nvprof: nvtx_range_pop() 

    def _wait(self):
        ''' Wait for the running iput. Called in main thread.
            Assumption: only one iput can be running. '''
        assert self.is_running, "no running prefetch"
        self.is_running = False
        return self.get_queue.remove()

    
    # 5.self.compute_stream.wait_event(ev_swapin)
    def get(self, vt, suc_vt=None):
        ''' Call by main thread. Blocking wait until current prefetch finish. Then launch next sync pin. Return current vt.idx. '''
        assert isinstance(vt, vTask)
        # wait current one (if no current one, get this one)
        if not self.is_running:
            # self.put_queue.add(vt)
            self.syncpin_handler.iput(vt)
            # 
            self.iput(vt)
        cur_vt_idx, ev_swapin = self._wait()
        self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        # syncpin next one if exsits
        if suc_vt is not None:
            self.syncpin_handler.iput(suc_vt)
        return cur_vt_idx

    
class PrefetchLocalModelGPU_worker6_4_2(object):
    """ Handles flexible prefetch of local model (W,B) from pinned model in background thread.
        Step:
        1) main thread: allocate (W,B) on GPU
        2) prefetch thread: get sync'ed pinned model (W,B)
        3) prefetch thread: copy in (W,B) in swapin_cudaStream
        4) prefetch thread: turn on grad
        
        Assumption: 
        1) always in FIFO ordering. put each task, and get prefetched task. 
        2) use cuda event for unblocking next CPU ops

        NOTE: why main thread allocates? i) 'no memory reuse across different stream' -- PyTorch, ii) different threads use different CUDA address spaces
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, syncpin_handler, local_model, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.syncpin_handler = syncpin_handler
        self.local_model = local_model # list
        self.rank = rank
        self.swapin_stream = swapin_stream if swapin_stream is not None else torch.cuda.Stream(device=rank)
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert rank == torch.cuda.current_device()
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vt1, ...] # between main and prefetch thread
        self.get_queue = threadsafe_data_struct.Queue() # [vt1.idx, ...] # between prefetch and main thread
        self.prefetch_thread = threading.Thread(target=self._thread_func)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        self.is_running = False
        
        # print("[PrefetchLocalModelGPU] rank{} started prefetch_thread".format(self.rank))

    def iput(self, layer_id, vt=None, cache_unit=None):
        ''' Call by main thread. Blocking Alloc & Nonblocking CopyIn.
            Assumption: only one iput can be running. '''
        # if vt is None:
        #     return
        # assert not self.is_running, "the prefetch is still running"
        # self.is_running = True
        # do the rest in background thread
        self.put_queue.add((layer_id, vt, cache_unit))

    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            layer_id, vt, cache_unit = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) CopyIn(W,B)".format(vt.idx, vt.show_layers())) 
            # get sync'ed pinned model 
            self.syncpin_handler.sync_pinned_model(layer_id)
            # assert syncpin_vt_idx == vt.idx
            # copy in and turn on grad
            with torch.cuda.stream(self.swapin_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
                self.local_model[layer_id].copyin_param_buf(cache_unit)
                if vt.type == 'BWD':
                    self.local_model[layer_id].set_param_requiresgrad()
            # record this swapin event for compute stream to wait
            ev_swapin = self.swapin_stream.record_event() 
            # if MEMCPY_NONBLK: self.swapin_stream.synchronize() # Wait for all the kernels in this stream to complete.
            # ready to use
            self.get_queue.add((layer_id, ev_swapin))
            if self.nvprof: nvtx_range_pop() 

    def _wait(self):
        ''' Wait for the running iput. Called in main thread.
            Assumption: only one iput can be running. '''
        # assert self.is_running, "no running prefetch"
        # self.is_running = False
        return self.get_queue.remove()

    def get(self, layer_id=None, vt=None, cache_unit=None):
        ''' Call by main thread. Blocking wait until current prefetch finish. Then launch next sync pin. Return current vt.idx. '''
        # wait current one (if no current one, get this one)
        # if not self.is_running:
        #     assert self.is_running, "no running prefetch"
        #     self.iput(layer_id, vt, cache_unit)
        layer_id, ev_swapin = self._wait()
        self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        return layer_id

class PrefetchLocalModelGPU_worker7(object):
    """ Handles flexible prefetch of local model (W,B) from pinned model in background thread.
        Step:
        1) main thread: allocate (W,B) on GPU
        2) prefetch thread: get sync'ed pinned model (W,B)
        3) prefetch thread: copy in (W,B) in swapin_cudaStream
        4) prefetch thread: turn on grad
        
        Assumption: 
        1) always in FIFO ordering. put each task, and get prefetched task. 
        2) use cuda event for unblocking next CPU ops

        NOTE: why main thread allocates? i) 'no memory reuse across different stream' -- PyTorch, ii) different threads use different CUDA address spaces
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, syncpin_handler, local_model, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.syncpin_handler = syncpin_handler
        self.local_model = local_model # list
        self.rank = rank
        self.swapin_stream = swapin_stream if swapin_stream is not None else torch.cuda.Stream(device=rank)
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert rank == torch.cuda.current_device()
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vt1, ...] # between main and prefetch thread
        self.get_queue = threadsafe_data_struct.Queue() # [vt1.idx, ...] # between prefetch and main thread
        self.prefetch_thread = threading.Thread(target=self._thread_func)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        self.is_running = False
        
        # print("[PrefetchLocalModelGPU] rank{} started prefetch_thread".format(self.rank))

    def iput(self, layer_id, vt=None, cache_unit=None):
        ''' Call by main thread. Blocking Alloc & Nonblocking CopyIn.
            Assumption: only one iput can be running. '''
        # if vt is None:
        #     return
        # assert not self.is_running, "the prefetch is still running"
        # self.is_running = True
        # do the rest in background thread
        self.put_queue.add((layer_id, vt, cache_unit))

    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            layer_id, vt, cache_unit = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) CopyIn(W,B)".format(vt.idx, vt.show_layers())) 
            # get sync'ed pinned model 
            self.syncpin_handler.sync_pinned_model(layer_id, vt)
            # assert syncpin_vt_idx == vt.idx
            # copy in and turn on grad
            with torch.cuda.stream(self.swapin_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
                self.local_model[layer_id].copyin_param_buf(cache_unit)
                if vt.type == 'BWD':
                    self.local_model[layer_id].set_param_requiresgrad()
            # record this swapin event for compute stream to wait
            ev_swapin = self.swapin_stream.record_event() 
            # self.swapin_stream.synchronize()
            # if vt.type != 'BWD':
            #     self.local_model[layer_id].return_pinned_data()
            # if MEMCPY_NONBLK: self.swapin_stream.synchronize() # Wait for all the kernels in this stream to complete.
            # ready to use
            self.get_queue.add((layer_id, ev_swapin))
            if self.nvprof: nvtx_range_pop() 

    def _wait(self):
        ''' Wait for the running iput. Called in main thread.
            Assumption: only one iput can be running. '''
        # assert self.is_running, "no running prefetch"
        # self.is_running = False
        return self.get_queue.remove()

    def get(self, layer_id=None, vt=None, cache_unit=None):
        ''' Call by main thread. Blocking wait until current prefetch finish. Then launch next sync pin. Return current vt.idx. '''
        # wait current one (if no current one, get this one)
        # if not self.is_running:
        #     assert self.is_running, "no running prefetch"
        #     self.iput(layer_id, vt, cache_unit)
        layer_id, ev_swapin = self._wait()
        self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        return layer_id

class SwapOutGrad(object):
    def __init__(self, local_model, rank, configs, swapout_stream=None, nvprof=False):
        self.local_model: list[LocalModelGPU_work6] = local_model
        self.rank = rank
        self.swapout_stream = swapout_stream if swapout_stream is not None else torch.cuda.Stream(device=rank)
        self.nvprof = nvprof
        self.configs = configs
        assert rank == torch.cuda.current_device()
        
        self.put_queue = threadsafe_data_struct.Queue() # [vt1, ...] # between main and prefetch thread
        self.get_queue = threadsafe_data_struct.Queue() # [vt1.idx, ...] # between prefetch and main thread
        self.prefetch_thread = threading.Thread(target=self._thread_func)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        self.is_running = False

    def iput(self, layer_id, vt):
        self.put_queue.add((layer_id, vt))

    def _thread_func(self):
        while True:
            layer_id, vt = self.put_queue.remove()
            with torch.cuda.stream(self.swapout_stream):
                if self.configs["opt_offld"]:        
                    if self.nvprof: nvtx_range_push("task{}(L{}) SwapOut(dW,B)".format(vt.idx, layer_id)) 
                    if vt.Out['dW'][layer_id].medium == "LOC" and vt.Out['B'][layer_id].medium == "SHM":
                        if self.configs["mode"]=='vPP' or (self.configs["mode"]=='vDP' and self.rank==0):
                            self.local_model[layer_id].swapout_grad()
                            self.local_model[layer_id].swapout_buf()
                    else:
                        raise NotImplementedError
                    if self.nvprof: nvtx_range_pop() 
                    ### Delete model {W,dW,B} 
                    if self.nvprof: nvtx_range_push("task{}(L{}) BWD-Del(W,dW,B)".format(vt.idx, layer_id)) 
                    if vt.Out['dW'][layer_id].medium == "LOC" and vt.Out['B'][layer_id].medium == "SHM":
                        self.local_model[layer_id].zero_grad()
                        # self.local_model[layer_num].pop_cuda_cache_unit() # also del gradient
                    else: # 'B' == PIN
                        raise NotImplementedError
                    # gc.collect()
                    if self.nvprof: nvtx_range_pop() 
                else:
                    raise ValueError("GPU Optimizer Underdevelopment.")
            self.swapout_stream.synchronize()
            self.get_queue.add((layer_id))
            if self.nvprof: nvtx_range_pop()

    def get(self):
        return self.get_queue.remove()
