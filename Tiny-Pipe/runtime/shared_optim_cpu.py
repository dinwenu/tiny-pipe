# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import gc
import threading
import queue  # 添加这一行
from collections import OrderedDict as ODict

import torch

from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

from task_data_struct import Medium, vTask
import threadsafe_data_struct

import time

if os.environ.get('CUDA_LAUNCH_BLOCKING') in ['1','True', True]:
    MEMCPY_NONBLK = False
else:
    MEMCPY_NONBLK = True

def convert_to_pinned(local_model_cpu):
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
                param.data = param.pin_memory() # in-place update and let python do the gc 
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                assert not buf.requires_grad # buffer has no grad
                m._buffers[key] = buf.pin_memory() # in-place update and let python do the gc 
                assert not m._buffers[key].requires_grad
    local_model_cpu.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    gc.collect()

class SharedOptimCPU(object):
    """ A wrapper class of shared optimizer (referring to each shared vlayer) on CPU.
        Data structure:
        o	a multiprocess-shared vlayer and optimizer
        o	a process-local pinned .grad as input buffer
        o	a process-local pinned model (param and buffs) as output buffer
        TODO: decouple pinned_model from here
    """
    @torch.no_grad()
    def __init__(self, shared_model, optimizer, id=-1):
        """ Call this in parent process and before fork/spawn subprocess """ 
        self.id = id # vlayer_id
        if optimizer is None: # this wrapper is only for pinned model
            self.shared_model = shared_model
            self.shared_optimizer = None
            # print("[SharedOptimizer][id%d] optimizer is None; for pinned model only."%self.id)
        else:
            # confirm model and optimizer are on CPU
            for param in shared_model.parameters():
                assert not param.is_cuda and param.is_shared()
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    # print("..............:", self.id)
                    # print(k)
                    # print(v)
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
            # 1) create zero gradient 
            for param in shared_model.parameters():
                param.grad = torch.zeros_like(param.data) # == torch.zeros(input.size(), input.dtype, input.layout, input.device, requires_grad=False)
                # Note: "param.grad = tensor" works, but "param.data.grad = tensor" doesn't work
                # Note: id(param.grad) == id(param.data.grad)
            # 2) force initialization of optimizer states (Bert, GPT2, Adam, SGD)
            optimizer.step() 
            # 3) move optimzer.state to shared memory
            # print("[SharedOptimizer] sharing optimizer:")
            for param, state in optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                        # 
                        v.share_memory_(); assert v.is_shared()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))

                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage 
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64) 
                        state[k].share_memory_(); assert state[k].is_shared()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            # 4) move optimzer.hyperparams to shared memory? No. They are constant (Bert, GPT2, Adam, SGD)
            # 5) clean up gradient
            for param in shared_model.parameters():
                param.grad = None
                assert param.grad is None, "cannot None grad?"
            gc.collect()
            # optimizer.zero_grad()
            self.shared_model = shared_model
            self.shared_optimizer = optimizer
            # print(f"self.shared_optimizer.param_groups:{self.shared_optimizer.param_groups}")
            # print(f"self.shared_optimizer.param_groups[0]:{self.shared_optimizer.param_groups[0]}")
            # exit(0)
            # print("[SharedOptimizer][id%d] sharing optimizer done."%self.id)
    
    @torch.no_grad()
    def init_in_subproc(self, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this on entering subprocess """
        self.rank, self.world_size = rank, world_size
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        if self.shared_optimizer is not None:
            # confirm model and optimizer are shared     
            for param in self.shared_model.parameters():
                assert param.data.is_shared()
                assert param.requires_grad 
                # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
            # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                for k, v in state.items():
                    assert isinstance(v, torch.Tensor) and v.is_shared()
            for k, v in self.shared_optimizer.param_groups[0].items():    
                if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
                    assert (not isinstance(v, torch.Tensor)) # or (isinstance(v, torch.Tensor) and v.is_shared())
                # ref1: https://pytorch.org/docs/1.5.0/_modules/torch/optim/optimizer.html#Optimizer.state_dict
                # ref2: https://pytorch.org/docs/1.5.0/_modules/torch/optim/adam.html#Adam
            # print("[SharedOptimizer] rank{}'s optimizer is shared".format(self.rank))
            
            # initialize local pinned .grad # Trimed
            # for param in self.shared_model.parameters():
            #     assert param.requires_grad
            #     param.grad = torch.zeros(param.shape, dtype=param.dtype, device="cpu", requires_grad=False).pin_memory()
            #     assert not param.grad.is_shared() and param.grad.is_pinned()
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned .grad"%(self.id, self.rank))

        if self.no_pin_model:
            self.pinned_model = None
        else:
            # initialize local pinned model (params and buffs)
            # with torch.no_grad():
            self.pinned_model = copy.deepcopy(self.shared_model)
            convert_to_pinned(self.pinned_model) # in-place convert a local model cpu to a pinned model
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned model"%(self.id, self.rank))
            # for s_param, p_param in zip(self.shared_model.parameters(), self.pinned_model.parameters()):
            #     print("Values equal:", torch.all(s_param == p_param))
            #     print("Same object:", s_param.data_ptr() == p_param.data_ptr())
            #     print(p_param)

    # shared_model.named_buffers()
    @torch.no_grad()
    def update_buf(self):
        """ In-place copy buffer from local pinned buf to shared model """  
        if self.no_pin_grad_buf:
            return
        
        else:
            assert hasattr(self.shared_model, "pinned_buf")
            named_pin_buf = self.shared_model.pinned_buf
            for name, shared_buf in self.shared_model.named_buffers():
                if shared_buf is not None:    
                    shared_buf.data.copy_(named_pin_buf[name].data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local pinned buffer must match shared model"

        # for shared_buf, pinned_buf in zip(self.shared_model.buffers(), self.pinned_model.buffers()):
        #     shared_buf.data.copy_(pinned_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from pinned memory to shared memory # nonblocking useless
        #     assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
    
    @torch.no_grad()
    def step(self, zero_grad=False):
        if self.shared_optimizer is not None:
            self.shared_optimizer.step()
            if zero_grad:

                self.shared_optimizer.zero_grad()
        # confirm local .grad is still pinned
        # for param in self.shared_model.parameters():
        #     assert param.grad.is_pinned()
        # print("[SharedOptimizer] rank{} steped shared optimizer".format(self.rank))
    
    @torch.no_grad()
    def sync_pinned_model(self):
        """ In-place copy from shared model to local pinned model """  
        if self.no_pin_model:
            return
        #
        for pinned_param, shared_param in zip(self.pinned_model.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(self.pinned_model.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
        # print("[SharedOptimizer] rank{} synced pinned model".format(self.rank))

class SharedOptimCPU_worker6_4_2(object):
    """ A wrapper class of shared optimizer (referring to each shared vlayer) on CPU.
        Data structure:
        o	a multiprocess-shared vlayer and optimizer
        o	a process-local pinned .grad as input buffer
        o	a process-local pinned model (param and buffs) as output buffer
        TODO: decouple pinned_model from here
    """
    @torch.no_grad()
    def __init__(self, shared_model, optimizer, id=-1):
        """ Call this in parent process and before fork/spawn subprocess """ 
        self.id = id # vlayer_id
        if optimizer is None: # this wrapper is only for pinned model
            self.shared_model = shared_model
            self.shared_optimizer = None
            # print("[SharedOptimizer][id%d] optimizer is None; for pinned model only."%self.id)
        else:
            # confirm model and optimizer are on CPU
            for param in shared_model.parameters():
                assert not param.is_cuda and param.is_shared()
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    # print("..............:", self.id)
                    # print(k)
                    # print(v)
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
            # 1) create zero gradient 
            for param in shared_model.parameters():
                param.grad = torch.zeros_like(param.data) # == torch.zeros(input.size(), input.dtype, input.layout, input.device, requires_grad=False)
                # Note: "param.grad = tensor" works, but "param.data.grad = tensor" doesn't work
                # Note: id(param.grad) == id(param.data.grad)
            # 2) force initialization of optimizer states (Bert, GPT2, Adam, SGD)
            optimizer.step() 
            # 3) move optimzer.state to shared memory
            # print("[SharedOptimizer] sharing optimizer:")
            for param, state in optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                        # 
                        v.share_memory_(); assert v.is_shared()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))

                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage 
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64) 
                        state[k].share_memory_(); assert state[k].is_shared()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            # 4) move optimzer.hyperparams to shared memory? No. They are constant (Bert, GPT2, Adam, SGD)
            # 5) clean up gradient
            for param in shared_model.parameters():
                param.grad = None
                assert param.grad is None, "cannot None grad?"
            gc.collect()
            # optimizer.zero_grad()
            self.shared_model = shared_model
            self.shared_optimizer = optimizer
            # print(f"self.shared_optimizer.param_groups:{self.shared_optimizer.param_groups}")
            # print(f"self.shared_optimizer.param_groups[0]:{self.shared_optimizer.param_groups[0]}")
            # exit(0)
            # print("[SharedOptimizer][id%d] sharing optimizer done."%self.id)
    
    @torch.no_grad()
    def init_in_subproc(self, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this on entering subprocess """
        self.rank, self.world_size = rank, world_size
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        if self.shared_optimizer is not None:
            # confirm model and optimizer are shared     
            for param in self.shared_model.parameters():
                assert param.data.is_shared()
                assert param.requires_grad 
                # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
            # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                for k, v in state.items():
                    assert isinstance(v, torch.Tensor) and v.is_shared()
            for k, v in self.shared_optimizer.param_groups[0].items():    
                if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
                    assert (not isinstance(v, torch.Tensor)) # or (isinstance(v, torch.Tensor) and v.is_shared())
                # ref1: https://pytorch.org/docs/1.5.0/_modules/torch/optim/optimizer.html#Optimizer.state_dict
                # ref2: https://pytorch.org/docs/1.5.0/_modules/torch/optim/adam.html#Adam
            # print("[SharedOptimizer] rank{}'s optimizer is shared".format(self.rank))
            
            # initialize local pinned .grad # Trimed
            # for param in self.shared_model.parameters():
            #     assert param.requires_grad
            #     param.grad = torch.zeros(param.shape, dtype=param.dtype, device="cpu", requires_grad=False).pin_memory()
            #     assert not param.grad.is_shared() and param.grad.is_pinned()
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned .grad"%(self.id, self.rank))

        self.pinned_model = None


    # shared_model.named_buffers()
    @torch.no_grad()
    def update_buf(self):
        """ In-place copy buffer from local pinned buf to shared model """  
        if self.no_pin_grad_buf:
            return
        
        else:
            assert hasattr(self.shared_model, "pinned_buf")
            named_pin_buf = self.shared_model.pinned_buf
            for name, shared_buf in self.shared_model.named_buffers():
                if shared_buf is not None:    
                    shared_buf.data.copy_(named_pin_buf[name].data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local pinned buffer must match shared model"

        # for shared_buf, pinned_buf in zip(self.shared_model.buffers(), self.pinned_model.buffers()):
        #     shared_buf.data.copy_(pinned_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from pinned memory to shared memory # nonblocking useless
        #     assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
    
    @torch.no_grad()
    def step(self, zero_grad=False):
        if self.shared_optimizer is not None:
            self.shared_optimizer.step()
            if zero_grad:

                self.shared_optimizer.zero_grad()
        # confirm local .grad is still pinned
        # for param in self.shared_model.parameters():
        #     assert param.grad.is_pinned()
        # print("[SharedOptimizer] rank{} steped shared optimizer".format(self.rank))
    
    @torch.no_grad()
    def sync_pinned_model(self, pinned_layer):
        """ In-place copy from shared model to local pinned model """  
        if self.no_pin_model:
            return
        self.pinned_model = pinned_layer
        for pinned_param, shared_param in zip(self.pinned_model.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(self.pinned_model.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
        # print("[SharedOptimizer] rank{} synced pinned model".format(self.rank))

class SharedOptimCPU_worker7(object):
    """ A wrapper class of shared optimizer (referring to each shared vlayer) on CPU.
        Data structure:
        o	a multiprocess-shared vlayer and optimizer
        o	a process-local pinned .grad as input buffer
        o	a process-local pinned model (param and buffs) as output buffer
        TODO: decouple pinned_model from here
    """
    @torch.no_grad()
    def __init__(self, shared_model, optimizer, id=-1):
        """ Call this in parent process and before fork/spawn subprocess """ 
        self.id = id # vlayer_id
        if optimizer is None: # this wrapper is only for pinned model
            self.shared_model = shared_model
            self.shared_optimizer = None
            # print("[SharedOptimizer][id%d] optimizer is None; for pinned model only."%self.id)
        else:
            # confirm model and optimizer are on CPU
            for param in shared_model.parameters():
                assert not param.is_cuda and param.is_shared()
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    # print("..............:", self.id)
                    # print(k)
                    # print(v)
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
            # 1) create zero gradient 
            for param in shared_model.parameters():
                param.grad = torch.zeros_like(param.data) # == torch.zeros(input.size(), input.dtype, input.layout, input.device, requires_grad=False)
                # Note: "param.grad = tensor" works, but "param.data.grad = tensor" doesn't work
                # Note: id(param.grad) == id(param.data.grad)
            # 2) force initialization of optimizer states (Bert, GPT2, Adam, SGD)
            optimizer.step() 
            # 3) move optimzer.state to shared memory
            # print("[SharedOptimizer] sharing optimizer:")
            for param, state in optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                        # 
                        v.share_memory_(); assert v.is_shared()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))

                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage 
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64) 
                        state[k].share_memory_(); assert state[k].is_shared()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            # 4) move optimzer.hyperparams to shared memory? No. They are constant (Bert, GPT2, Adam, SGD)
            # 5) clean up gradient
            for param in shared_model.parameters():
                param.grad = None
                assert param.grad is None, "cannot None grad?"
            gc.collect()
            # optimizer.zero_grad()
            self.shared_model = shared_model
            self.shared_optimizer = optimizer
            # print(f"self.shared_optimizer.param_groups:{self.shared_optimizer.param_groups}")
            # print(f"self.shared_optimizer.param_groups[0]:{self.shared_optimizer.param_groups[0]}")
            # exit(0)
            # print("[SharedOptimizer][id%d] sharing optimizer done."%self.id)
    
    @torch.no_grad()
    def init_in_subproc(self, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this on entering subprocess """
        self.rank, self.world_size = rank, world_size
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        if self.shared_optimizer is not None:
            # confirm model and optimizer are shared     
            for param in self.shared_model.parameters():
                # assert param.data.is_shared()
                assert param.requires_grad 
                # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
            # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                for k, v in state.items():
                    assert isinstance(v, torch.Tensor) and v.is_shared()
            for k, v in self.shared_optimizer.param_groups[0].items():    
                if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
                    assert (not isinstance(v, torch.Tensor)) # or (isinstance(v, torch.Tensor) and v.is_shared())
                # ref1: https://pytorch.org/docs/1.5.0/_modules/torch/optim/optimizer.html#Optimizer.state_dict
                # ref2: https://pytorch.org/docs/1.5.0/_modules/torch/optim/adam.html#Adam
            # print("[SharedOptimizer] rank{}'s optimizer is shared".format(self.rank))
            
            # initialize local pinned .grad # Trimed
            # for param in self.shared_model.parameters():
            #     assert param.requires_grad
            #     param.grad = torch.zeros(param.shape, dtype=param.dtype, device="cpu", requires_grad=False).pin_memory()
            #     assert not param.grad.is_shared() and param.grad.is_pinned()
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned .grad"%(self.id, self.rank))

        if self.no_pin_model:
            self.pinned_model = None
        else:
            pass
             # initialize local pinned model (params and buffs)
            # with torch.no_grad():
            # self.pinned_model = copy.deepcopy(self.shared_model)
            # convert_to_pinned(self.pinned_model) # in-place convert a local model cpu to a pinned model
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned model"%(self.id, self.rank))

    def make_shared_model_point_to_pinned(self, pinned_layer):
        for cpu_param, pinned_param in zip(self.shared_model.parameters(), pinned_layer.parameters()):
            if pinned_param.grad is not None:
                cpu_param.data = pinned_param.data
                cpu_param.grad = pinned_param.grad
            else:
                cpu_param.grad = None
                assert cpu_param.grad is None, f"rank:{self.rank}, layer:{self.id}, model has grad = None when make_shared_model_point_to_pinned"

    # shared_model.named_buffers()
    @torch.no_grad()
    def update_buf(self):
        """ In-place copy buffer from local pinned buf to shared model """  
        if self.no_pin_grad_buf:
            return
        
        else:
            assert hasattr(self.shared_model, "pinned_buf")
            named_pin_buf = self.shared_model.pinned_buf
            for name, shared_buf in self.shared_model.named_buffers():
                if shared_buf is not None:    
                    shared_buf.data.copy_(named_pin_buf[name].data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local pinned buffer must match shared model"

        # for shared_buf, pinned_buf in zip(self.shared_model.buffers(), self.pinned_model.buffers()):
        #     shared_buf.data.copy_(pinned_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from pinned memory to shared memory # nonblocking useless
        #     assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)

    @torch.no_grad()
    def step(self, zero_grad=False):
        if self.shared_optimizer is not None:
            self.shared_optimizer.step()
            if zero_grad:

                self.shared_optimizer.zero_grad()
        # confirm local .grad is still pinned
        # for param in self.shared_model.parameters():
        #     assert param.grad.is_pinned()
        # print("[SharedOptimizer] rank{} steped shared optimizer".format(self.rank))
    
    @torch.no_grad()
    def sync_pinned_model(self):
        """ In-place copy from shared model to local pinned model """  
        if self.no_pin_model:
            return
        #
        for pinned_param, shared_param in zip(self.pinned_model.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(self.pinned_model.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
        # print("[SharedOptimizer] rank{} synced pinned model".format(self.rank))

    @torch.no_grad()
    def sync_pinned_layer(self, pinned_layer):
        for pinned_param, shared_param in zip(pinned_layer.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(pinned_layer.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)


class PinnedOptimCPU(object):
    """ A wrapper class of shared optimizer (referring to each shared vlayer) on CPU.
        Data structure:
        o	a multiprocess-shared vlayer and optimizer
        o	a process-local pinned .grad as input buffer
        o	a process-local pinned model (param and buffs) as output buffer
        TODO: decouple pinned_model from here
    """
    @torch.no_grad()
    def __init__(self, shared_model, optimizer, id=-1):
        """ Call this in parent process and before fork/spawn subprocess """ 
        self.id = id # vlayer_id
        if optimizer is None: # this wrapper is only for pinned model
            self.shared_model = shared_model
            self.shared_optimizer = None
            # print("[SharedOptimizer][id%d] optimizer is None; for pinned model only."%self.id)
        else:
            # confirm model and optimizer are on CPU
            for param in shared_model.parameters():
                assert not param.is_cuda and param.is_shared()
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    # print("..............:", self.id)
                    # print(k)
                    # print(v)
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
            # 1) create zero gradient 
            for param in shared_model.parameters():
                param.grad = torch.zeros_like(param.data) # == torch.zeros(input.size(), input.dtype, input.layout, input.device, requires_grad=False)
                # Note: "param.grad = tensor" works, but "param.data.grad = tensor" doesn't work
                # Note: id(param.grad) == id(param.data.grad)
            # 2) force initialization of optimizer states (Bert, GPT2, Adam, SGD)
            optimizer.step() 
            # 3) move optimzer.state to shared memory
            # print("[SharedOptimizer] sharing optimizer:")
            for param, state in optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                        # 
                        v.share_memory_(); assert v.is_shared()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))

                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage 
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64) 
                        state[k].share_memory_(); assert state[k].is_shared()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            # 4) move optimzer.hyperparams to shared memory? No. They are constant (Bert, GPT2, Adam, SGD)
            # 5) clean up gradient
            for param in shared_model.parameters():
                param.grad = None
                assert param.grad is None, "cannot None grad?"
            gc.collect()
            # optimizer.zero_grad()
            self.shared_model = shared_model
            self.shared_optimizer = optimizer
            # print(f"self.shared_optimizer.param_groups:{self.shared_optimizer.param_groups}")
            # print(f"self.shared_optimizer.param_groups[0]:{self.shared_optimizer.param_groups[0]}")
            # exit(0)
            # print("[SharedOptimizer][id%d] sharing optimizer done."%self.id)

    def from_shared_to_pinned(self):
        if self.shared_optimizer is not None:
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                        # 
                        v.pin_memory(); assert v.is_pinned()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))

                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage 
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64) 
                        state[k].pin_memory(); assert state[k].is_pinned()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            gc.collect()
            self.pinned_optimizer = self.shared_optimizer
            self.shared_optimizer = None

    # 📍
    @torch.no_grad()
    def init_in_subproc(self, id, cpu_layers, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this on entering subprocess """
        self.rank, self.world_size = rank, world_size
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        # 
        # if id not in cpu_layers:
        #     self.from_shared_to_pinned()

        if self.shared_optimizer is not None:
            # confirm model and optimizer are shared     
            for param in self.shared_model.parameters():
                assert param.data.is_shared()
                assert param.requires_grad 
                # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
            # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                for k, v in state.items():
                    assert isinstance(v, torch.Tensor) and v.is_shared()
            for k, v in self.shared_optimizer.param_groups[0].items():    
                if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
                    assert (not isinstance(v, torch.Tensor)) # or (isinstance(v, torch.Tensor) and v.is_shared())
        # elif self.pinned_optimizer is not None:
        #     for param in self.pinned_optimizer.parameters():
        #         assert param.data.is_pinned()
        #         assert param.requires_grad 
        #         # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
        #     # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
        #     for param, state in self.pinned_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
        #         for k, v in state.items():
        #             assert isinstance(v, torch.Tensor) and v.is_pinned()
        #     for k, v in self.pinned_optimizer.param_groups[0].items():    
        #         if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
        #             assert (not isinstance(v, torch.Tensor))

        if self.no_pin_model:
            self.pinned_model = None
        elif id in cpu_layers:
             # initialize local pinned model (params and buffs)
            # with torch.no_grad():
            self.pinned_model = copy.deepcopy(self.shared_model)
            convert_to_pinned(self.pinned_model) # in-place convert a local model cpu to a pinned model
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned model"%(self.id, self.rank))
        else:
            self.pinned_model = None

    # shared_model.named_buffers()
    @torch.no_grad()
    def update_buf(self):
        """ In-place copy buffer from local pinned buf to shared model """  
        if self.no_pin_grad_buf:
            return
        
        else:
            assert hasattr(self.shared_model, "pinned_buf")
            named_pin_buf = self.shared_model.pinned_buf
            for name, shared_buf in self.shared_model.named_buffers():
                if shared_buf is not None:    
                    shared_buf.data.copy_(named_pin_buf[name].data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local pinned buffer must match shared model"

        # for shared_buf, pinned_buf in zip(self.shared_model.buffers(), self.pinned_model.buffers()):
        #     shared_buf.data.copy_(pinned_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from pinned memory to shared memory # nonblocking useless
        #     assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
    
    @torch.no_grad()
    def step(self, zero_grad=False):
        if self.shared_optimizer is not None:
            self.shared_optimizer.step()
            if zero_grad:

                self.shared_optimizer.zero_grad()
        # confirm local .grad is still pinned
        # for param in self.shared_model.parameters():
        #     assert param.grad.is_pinned()
        # print("[SharedOptimizer] rank{} steped shared optimizer".format(self.rank))
    
    @torch.no_grad()
    def sync_pinned_model(self):
        """ In-place copy from shared model to local pinned model """  
        if self.no_pin_model:
            return
        #
        for pinned_param, shared_param in zip(self.pinned_model.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(self.pinned_model.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
        # print("[SharedOptimizer] rank{} synced pinned model".format(self.rank))


class UpdateInBkgd_worker6_4_2(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, lr_scheduler, local_model, swapout_grad_handler, cuda_cache_queue, special_cuda_layers, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.local_model = local_model
        self.swapout_grad_handler = swapout_grad_handler
        self.cuda_cache_queue = cuda_cache_queue
        self.special_cuda_layers = special_cuda_layers
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt, layer_id = self.put_queue.remove() # blocking
            layer_id_swapout = self.swapout_grad_handler.get()
            assert layer_id == layer_id_swapout, f"rank:{self.rank}, 要更新的layer{layer_id}与刚刚卸载的layer{layer_id_swapout}不一致"
            self._del_cache_unit(layer_id)
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            self._update_buf(vt, layer_id) # if using local pinned model for B'
            self._step(vt, layer_id)

            self.local_model[layer_id].return_pinned_data()

            self.get_queue.add(layer_id)
            if self.nvprof: nvtx_range_pop() 

    def _del_cache_unit(self, layer_id):
        if layer_id in self.special_cuda_layers:
            pass
        else:
            _cuda_cache_unit = self.local_model[layer_id]._gl_cuda_cache_unit
            self.cuda_cache_queue.append(_cuda_cache_unit)
            del self.local_model[layer_id]._gl_cuda_cache_unit

    @torch.no_grad()
    def _update_buf(self, vt, layer_id):
        """ update B of this pack """  
        self.shared_optimizer[layer_id].update_buf()
    
    @torch.no_grad()
    def _step(self, vt, layer_id):
        """ update W,K of this pack """  
        # assert vt.In['dW'][layer_id].medium == "LOC"
        # assert vt.In['W'][layer_id].medium == "SHM"  
        # assert vt.In['K'][layer_id].medium == "SHM"
        # assert vt.Out['W'][layer_id].medium == "SHM"
        # assert vt.Out['K'][layer_id].medium == "SHM" 

        self.shared_optimizer[layer_id].step() # Update shared model and optim using swap-out'ed local .grad
        if self.lr_scheduler != []: # "gpt2_huggingface"
            if self.lr_scheduler[layer_id] is not None:
                assert self.shared_optimizer[layer_id].shared_optimizer is not None
                self.lr_scheduler[layer_id].step() 
            else:
                assert self.shared_optimizer[layer_id].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    def iput(self, vt, layer_id):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.swapout_grad_handler.iput(layer_id, vt)
        self.put_queue.add((vt, layer_id))
        self.the_last_put = layer_id

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            layer_id = self.get_queue.remove()
            if layer_id == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))


class UpdateInBkgd_worker7(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, lr_scheduler, local_model, shared_model_nvme, swapout_grad_handler, cuda_cache_queue, special_cuda_layers, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.local_model = local_model
        self.shared_model_nvme = shared_model_nvme
        self.swapout_grad_handler = swapout_grad_handler
        self.cuda_cache_queue = cuda_cache_queue
        self.special_cuda_layers = special_cuda_layers
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt, layer_id = self.put_queue.remove() # blocking
            layer_id_swapout = self.swapout_grad_handler.get()
            assert layer_id == layer_id_swapout, f"rank:{self.rank}, 要更新的layer{layer_id}与刚刚卸载的layer{layer_id_swapout}不一致"
            self._del_cache_unit(layer_id)
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            # self._update_buf(vt, layer_id) # if using local pinned model for B'
            self._step(vt, layer_id)

            self.shared_model_nvme.swap_out_sync(layer_id, self.local_model[layer_id].pinned_data.pinned_buffer)
            self.local_model[layer_id].return_pinned_data()

            self.get_queue.add(layer_id)
            if self.nvprof: nvtx_range_pop() 

    def _del_cache_unit(self, layer_id):
        if layer_id in self.special_cuda_layers:
            pass
        else:
            _cuda_cache_unit = self.local_model[layer_id]._gl_cuda_cache_unit
            self.cuda_cache_queue.append(_cuda_cache_unit)
            del self.local_model[layer_id]._gl_cuda_cache_unit

    @torch.no_grad()
    def _update_buf(self, vt, layer_id):
        """ update B of this pack """  
        self.shared_optimizer[layer_id].update_buf()
    
    @torch.no_grad()
    def _step(self, vt, layer_id):
        """ update W,K of this pack """  
        # assert vt.In['dW'][layer_id].medium == "LOC"
        # assert vt.In['W'][layer_id].medium == "SHM"  
        # assert vt.In['K'][layer_id].medium == "SHM"
        # assert vt.Out['W'][layer_id].medium == "SHM"
        # assert vt.Out['K'][layer_id].medium == "SHM" 

        self.shared_optimizer[layer_id].step() # Update shared model and optim using swap-out'ed local .grad
        if self.lr_scheduler != []: # "gpt2_huggingface"
            if self.lr_scheduler[layer_id] is not None:
                assert self.shared_optimizer[layer_id].shared_optimizer is not None
                self.lr_scheduler[layer_id].step() 
            else:
                assert self.shared_optimizer[layer_id].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    def iput(self, vt, layer_id):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.swapout_grad_handler.iput(layer_id, vt)
        self.put_queue.add((vt, layer_id))
        self.the_last_put = layer_id

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            layer_id = self.get_queue.remove()
            if layer_id == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))


def delete_param_grad_buf_for_shared_model(top_module, manual_gc=False):
    @torch.no_grad()
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # delete param
                param.data = torch.empty(0, device="cpu")
                # delete grad
                if param.grad is not None:
                    param.grad = None
                # param.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
                # assert not param.requires_grad
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                m._buffers[key] = torch.empty(0, device="cpu")
    top_module.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    #
    if manual_gc:
        gc.collect(); torch.cuda.empty_cache() # can block all cudaStreams

def delete_param_buf_for_shared_model(top_module, manual_gc=False):
    @torch.no_grad()
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # delete param
                param.data = torch.empty(0, device="cpu")
                # delete grad
                # if param.grad is not None:
                #     param.grad = None
                # param.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
                # assert not param.requires_grad
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                m._buffers[key] = torch.empty(0, device="cpu")
    top_module.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    #
    if manual_gc:
        gc.collect(); torch.cuda.empty_cache() # can block all cudaStreams

class SharedModelNVMe_worker7(object):
    def __init__(self, shared_model, param_swapper, special_layer_id, rank=None):
        self.shared_model = shared_model
        self.param_swapper = param_swapper
        self.rank = rank
        self.special_layer_id = special_layer_id

    def get_pinned_data(self, layer_id):
        if layer_id in self.special_layer_id:
            return self.param_swapper.get_special_pinned_data(layer_id)
        else:
            # 
            # if self.rank == 3:
            return self.param_swapper.get_pinned_data()

    def return_pinned_data(self, layer_id, pinned_data=None):
        if layer_id not in self.special_layer_id:
            # if self.rank == 3:
            self.param_swapper.return_pinned_data(pinned_data)

    def swap_out_sync(self, layer_id, pinned_tensor):
        # start_time = time.perf_counter()
        self.param_swapper.swap_out_transformer_layer_sync(layer_id, pinned_tensor)
        # end_time = time.perf_counter()

    def swap_in_sync(self, layer_id, pinned_tensor):
        self.param_swapper.swap_in_transformer_layer_sync(layer_id, pinned_tensor)

    def _delete_param_grad_buf(self, model, manual_gc=False):
        delete_param_grad_buf_for_shared_model(model, manual_gc=manual_gc)

    def copy_shared_model_to_pinned_buffer(self, layer_id, pinned_layer):
        for cpu_m, pin_m in zip(self.shared_model[layer_id][0].modules(), pinned_layer.modules()): # iterator over all modules in the network
            # print("{} <- {}".format(gpu_m, cpu_m))
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    # one_dim_param = param.view(-1)
                    pin_m._parameters[key].data.copy_(param.data, non_blocking=MEMCPY_NONBLK)
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    pin_m._buffers[key].data.copy_(buf.data, non_blocking=MEMCPY_NONBLK)

    def swap_out_to_ssd(self):
        pinned_data = self.param_swapper.get_pinned_data()
        pinned_buffer = pinned_data.pinned_buffer
        pinned_layer = pinned_data.pinned_layer

        layer_ids = [i for i in range(len(self.shared_model))]

        for layer_id in layer_ids:
            if layer_id in self.special_layer_id:
                special_pinned_data = self.param_swapper.get_special_pinned_data(layer_id)
                special_pinned_buffer = special_pinned_data.pinned_buffer
                special_pinned_layer = special_pinned_data.pinned_layer
                self.copy_shared_model_to_pinned_buffer(layer_id, special_pinned_layer)
                self.swap_out_sync(layer_id, special_pinned_buffer)
                self._delete_param_grad_buf(self.shared_model[layer_id][0])
            else:
                self.copy_shared_model_to_pinned_buffer(layer_id, pinned_layer)
                self.swap_out_sync(layer_id, pinned_buffer)
                self._delete_param_grad_buf(self.shared_model[layer_id][0])
        self.param_swapper.return_pinned_data(pinned_data)

class SwapToNVMeInBkgd(object):
    def __init__(self, shared_model_nvme, layer_id_to_layer_idx, bwd_vts, layer_num, rank, nvprof=False):
        self.rank = rank
        self.shared_model_nvme: SharedModelNVMe = shared_model_nvme
        # self.param_swapper: AsyncPartitionedParameterSwapper = param_swapper
        self.the_last_put = None
        self.layer_num = layer_num
        self.nvprof = nvprof

        self.layer_id_to_layer_idx = layer_id_to_layer_idx

        self.swap_to_nvme_at_start(bwd_vts)
        gc.collect()

        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        self.get_queue = threadsafe_data_struct.Queue()
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()

    def swap_to_nvme_at_start(self, vts):
        for vt in vts:
            self._swap_out_from_shared(vt)


    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) CPU->NVMe(W、B)".format(vt.idx, vt.show_layers())) 
            self._swap_out_from_pinned_buffer(vt)
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    def _swap_out_from_pinned_buffer(self, vt):
        for layer_id in vt.layers:
            if vt.has_data and layer_id == vt.layers[0]:
                continue
            if layer_id == self.layer_num-2:
                continue
            if layer_id == self.layer_num-3:
                continue
            if vt.has_criterion and layer_id == vt.layers[-1]:
                continue

            layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
            self.shared_model_nvme.swap_out_from_pinned_buffer_sync_2(layer_id, layer_idx)

    def _swap_out_from_shared(self, vt):
        """ update W,K of this pack """  
        for layer_id in vt.layers:
            if vt.has_data and layer_id == vt.layers[0]:
                continue
            if layer_id == self.layer_num-2:
                continue
            if layer_id == self.layer_num-3:
                continue
            if vt.has_criterion and layer_id == vt.layers[-1]:
                continue

            self.shared_model_nvme.swap_out_from_shared_memory_and_release(layer_id) # Update shared model and optim using swap-out'ed local .grad

    def iput(self, vt):
        ''' Call by upstream thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()


class SwapInCpuInBkgd(object):
    def __init__(self, syncpin_handler, shared_model_nvme, layer_id_to_layer_idx, layer_num, rank, nvprof=False):
        self.rank = rank
        self.nvprof = nvprof
        self.shared_model_nvme: SharedModelNVMe = shared_model_nvme
        self.syncpin_handler = syncpin_handler
        self.layer_num = layer_num

        # self.param_swapper: AsyncPartitionedParameterSwapper = param_swapper
        self.the_last_put = None

        self.layer_id_to_layer_idx = layer_id_to_layer_idx

        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        self.get_queue = threadsafe_data_struct.Queue()
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
    
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) NVMe->CPU(W、B)".format(vt.idx, vt.show_layers())) 
            self._swap_in(vt)
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    def _swap_in(self, vt):
        """ update W,K of this pack """  
        for layer_id in vt.layers:
            if vt.has_data and layer_id == vt.layers[0]:
                self.syncpin_handler.input_one_layer(layer_id, vt)
                continue
            if layer_id == self.layer_num-3:
                self.syncpin_handler.input_one_layer(layer_id, vt)
                continue
            if layer_id == self.layer_num-2:
                self.syncpin_handler.input_one_layer(layer_id, vt)
                continue
            if layer_id == self.layer_num-1:
                self.syncpin_handler.input_one_layer(layer_id, vt)
                continue
            # if vt.has_criterion and layer_id == vt.layers[-2]:
            #     self.syncpin_handler.input_one_layer(layer_id, vt)
            
            layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
            self.shared_model_nvme.swap_in_sync_2(layer_id, layer_idx) # Update shared model and optim using swap-out'ed local .grad
            self.shared_model_nvme.make_shared_model_point_to_pinned(layer_id, layer_idx)
    # def sync_to_shared_model(self):


    def iput(self, vt):
        ''' Call by upstream thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()

class SyncPinModelInBkgd_worker6_4_2(object):
    """ Handles synchronization to local pinned model in background thread.
        Assumption:
            0) always in FIFO ordering. put each task, and get synced task. 
        TODO: skip sync_pinned_model() if already done (FWD -> BWD)
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, local_model, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.local_model = local_model
        self.rank = rank
        self.nvprof = nvprof

    @torch.no_grad()
    def sync_pinned_model(self, layer_id, vt=None):
        """ sync W,B to local pinned model for this layer """  
        self.local_model[layer_id].set_pinned_data()
        pinned_data = self.local_model[layer_id].get_pinned_data()
        pinned_layer = pinned_data.pinned_layer
        self.shared_optimizer[layer_id].sync_pinned_model(pinned_layer)

from time import perf_counter as pc
class SyncPinModelInBkgd_worker7(object):
    def __init__(self, shared_optimizer, local_model, shared_model_nvme, rank, nvprof=False):
        self.shared_optimizer: SharedOptimCPU_worker7 = shared_optimizer
        self.local_model = local_model
        self.shared_model_nvme: SharedModelNVMe_worker7 = shared_model_nvme
        self.rank = rank
        self.nvprof = nvprof

    @torch.no_grad()
    def sync_pinned_model(self, layer_id, vt=None):
        self.local_model[layer_id].set_pinned_data()
        pinned_data = self.local_model[layer_id].get_pinned_data()
        pinned_tensor = pinned_data.pinned_buffer
        self.shared_model_nvme.swap_in_sync(layer_id, pinned_tensor)
        if vt.type == 'BWD':
            # start_time = pc()
            self.shared_optimizer[layer_id].make_shared_model_point_to_pinned(pinned_data.pinned_layer)
            # end_time = pc()
