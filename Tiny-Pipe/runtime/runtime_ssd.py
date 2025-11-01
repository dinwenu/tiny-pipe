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
mp = torch.multiprocessing.get_context('spawn') # for GPU usage

def _assert_assumption(CONFIGS):
    # GPU only
    assert torch.cuda.is_available()
    # FP32 only
    torch.set_default_tensor_type('torch.FloatTensor')
    # Optimizer offload
    assert CONFIGS["opt_offld"]
    # double buffer need equal size
    if CONFIGS['mode'] == 'vPP':
        assert len(set(CONFIGS['ubatchszs_fwd'])) == 1
        assert len(set(CONFIGS['ubatchszs_bwd'])) == 1
    elif CONFIGS['mode'] == 'vDP': 
        for ubatchszs_fwd, ubatchszs_bwd in zip(CONFIGS['ubatchszs_fwd'], CONFIGS['ubatchszs_bwd']):
            assert len(set(ubatchszs_fwd)) == 1
            assert len(set(ubatchszs_bwd)) == 1
    else:
        raise ValueError
    # a single BWD task should have equal ubatchszs_fwd and ubatchszs_bwd per GPU
    if CONFIGS["pack_fwd"] == []: 
        assert CONFIGS["u_fwd"] == CONFIGS["u_bwd"]
        if CONFIGS['mode'] == 'vPP':
            assert CONFIGS['ubatchszs_fwd'] == CONFIGS['ubatchszs_bwd']
        elif CONFIGS['mode'] == 'vDP': 
            for ubatchszs_fwd, ubatchszs_bwd in zip(CONFIGS['ubatchszs_fwd'], CONFIGS['ubatchszs_bwd']):
                assert ubatchszs_fwd == ubatchszs_bwd
        else:
            raise ValueError

def worker_func(*pargs, **kwargs): # per process
    from tiny_pipe_ssd import Worker
    w = Worker(*pargs, **kwargs)
    # w.run_initial_iteration()
    w.run_training_loop()
    w.finish()

def collect_layer_param_info(model, special_layers, param_swapper):
    transformer_layer = model[1][0]
    
    transformer_layer_info = {
        'param_idx_to_numel': {},
        'param_idx_to_start_pos': {},
        'param_idx_to_shape': {},
        'layer_size': 0,
        'layer_aligned_size': 0
    }
    
    total_size = 0
    global_param_idx = 0
    for m in transformer_layer.modules():
        for key, param in m._parameters.items():
            if param is not None:
                param_size = param.numel()
                transformer_layer_info['param_idx_to_numel'][global_param_idx] = param_size
                transformer_layer_info['param_idx_to_start_pos'][global_param_idx] = total_size
                transformer_layer_info['param_idx_to_shape'][global_param_idx] = param.shape
                total_size += param_size
                global_param_idx += 1

        for key, buf in m._buffers.items():
            if buf is not None:
                buf_size = buf.numel()
                transformer_layer_info['param_idx_to_numel'][global_param_idx] = buf_size
                transformer_layer_info['param_idx_to_start_pos'][global_param_idx] = total_size
                transformer_layer_info['param_idx_to_shape'][global_param_idx] = buf.shape
                total_size += buf_size
                global_param_idx += 1

    transformer_layer_info['layer_size'] = total_size
    transformer_layer_info['layer_aligned_size'] = param_swapper._io_aligned_numel(total_size)
    
    special_layers_info = {}
    for layer_id in special_layers:
        if layer_id < len(model):
            layer = model[layer_id][0]
            layer_info = {
                'param_idx_to_numel': {},
                'param_idx_to_start_pos': {},
                'param_idx_to_shape': {},
                'layer_size': 0,
                'layer_aligned_size': 0
            }
            
            total_size = 0
            global_param_idx = 0
            for m in layer.modules():
                for key, param in m._parameters.items():
                    if param is not None:
                        param_size = param.numel()
                        layer_info['param_idx_to_numel'][global_param_idx] = param_size
                        layer_info['param_idx_to_start_pos'][global_param_idx] = total_size
                        layer_info['param_idx_to_shape'][global_param_idx] = param.shape
                        total_size += param_size
                        global_param_idx += 1
                        
                for key, buf in m._buffers.items():
                    if buf is not None:
                        buf_size = buf.numel()
                        layer_info['param_idx_to_numel'][global_param_idx] = buf_size
                        layer_info['param_idx_to_start_pos'][global_param_idx] = total_size
                        layer_info['param_idx_to_shape'][global_param_idx] = buf.shape
                        total_size += buf_size
                        global_param_idx += 1
            
            layer_info['layer_size'] = total_size
            layer_info['layer_aligned_size'] = param_swapper._io_aligned_numel(total_size)
            special_layers_info[layer_id] = layer_info
    
    return transformer_layer_info, special_layers_info

def run(args, real_dataset, create_model, create_optimizer, get_train_steps=None, get_lr_sched=None, compute_loss=None, save_model=None): # main process
    
    import seeding
    seeding.seed(args.seed, args.seed_cudnn)
    
    """ Initialize Harmony. """
    module_path = os.path.join(args.module_dir, args.module_name)
    assert os.path.exists(module_path)
    assert os.path.basename(module_path) not in ["prof", "sched"], "no base_dir in module_path"
    
    # read profiles
    from prof_data_struct import ConstMeta, TensorMeta, XMeta, TMeta, load_prof_data_struct
    prof = ODict()
    print("........args.profile_fnames: ",args.profile_fnames)
    # ['prof_XMETA', 'prof_TMETA']
    for name in args.profile_fnames:
        key = name.split("prof_")[-1]
        prof[key] = load_prof_data_struct(module_path, name + args.suffix, base_dir="my_prof", verbose=True)
    
    # read schedule
    from task_data_struct import Medium, vTask, unserialize_scheduled
    if args.schedule_dir == "":
        args.schedule_dir = module_path
    rTASKS, CONFIGS = unserialize_scheduled(args.schedule_dir, args.schedule_fname + args.suffix, base_dir="work6_sched", verbose=False)
    _assert_assumption(CONFIGS)
    
    """ Initialize data. """
    if args.synthetic_data:
        args.num_epochs = 1
        assert args.num_iters is not None
        args.num_train_steps = args.num_iters
        print('----- Training Info -----')
        print("  num epoches = %d" % args.num_epochs)
        print("  num iterations per epoch = %d" % (args.num_iters))
        print("  num optimization steps = %d" % (args.num_train_steps))

    else:
        data_loader, examples, _, _, _, _, _ = real_dataset(args, CONFIGS["D"], data_workers=0)
        print(f"........len(data_loader):{len(data_loader)}")
        if get_train_steps is not None: # "bert_thomwolf"
            args.num_train_steps = get_train_steps(args, examples, CONFIGS["D"])
        else:
            args.num_train_steps = len(data_loader) * args.num_epochs
        if args.num_iters is None:
            args.num_iters = len(data_loader) # num_minibatch
        else:
            args.num_iters = min(args.num_iters, len(data_loader))
        print('----- Training Info -----')
        print("  num epoches = %d" % args.num_epochs)
        print("  num minibatches per epoch = %d" % len(data_loader))
        print("  num iterations per epoch = %d" % (args.num_iters))
        print("  num optimization steps = %d" % (args.num_train_steps))
        del data_loader
    
    if args.nvprof:
        assert args.num_epochs == 1, "num_epochs must be 1 during nvprof"
        if args.nvprof_iter == "first":
            args.nvprof_iter = { "start" : 0, "end" : 0 }
        elif args.nvprof_iter == "last":
            args.nvprof_iter = { "start" : args.num_iters - 1, "end" : args.num_iters - 1 }
        elif args.nvprof_iter == "all":
            args.nvprof_iter = { "start" : 0, "end" : args.num_iters - 1 } 
        else:
            raise ValueError

    """ Initialize model. """
    from utils import PrintCPUMem
    pcm = PrintCPUMem()
    pcm.print("before creating model")
    model = create_model(args)
    pcm.print("model created")

    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) != 0:
            for param in vlayer.parameters():
                assert not param.is_cuda
    
    from local_model_gpu import delete_param_grad_buf
    empty_model = []
    for vlayer, _, _ in model:
        with torch.no_grad():
            vlayer_copy = deepcopy(vlayer)
        delete_param_grad_buf(vlayer_copy)
        empty_model.append(vlayer_copy)
    
    # initialize shared model on CPU  
    for vlayer, _1, _2 in model:
        print(_1,_2)
        vlayer.share_memory() # move parameter into shared memory    
    pcm.print("shared model created")

    """ Initialize optimizer. """
    optimizer = create_optimizer(args, model)
    pcm.print("optimizer created")
    
    # initialize shared optimizer on CPU
    from shared_optim_cpu import SharedOptimCPU_worker7
    shared_model = model # model is already shared
    shared_optimizer = [] # wrapper object for optimizer
    for id, ((vlayer, _, _), optim) in enumerate(zip(shared_model, optimizer)): 
        shared_optimizer.append(SharedOptimCPU_worker7(vlayer, optim, id))
    pcm.print("shared optimizer created")

    ######################################################################
    # ==== my version: initialize empty model and shared model on CPU ====
    num_layers = CONFIGS["R"]
    special_layers_id = [0, num_layers - 1, num_layers - 2, num_layers - 3]
    special_layers_in_rank = [(layer_id, shared_model[layer_id][0]) for layer_id in special_layers_id]

    _ds_config = {
        "train_batch_size": 16,
        "zero_optimization": {
            "offload_param":{
                "device": 'nvme',
                "nvme_path": "/nvme_test",
                "buffer_count": 1 # 只创建一个buffer，因为我们只保存一个
            },
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 16,
            "single_submit": False,
            "overlap_events": True,
            "thread_count": 2
        }
    }
    from deepspeed.runtime.config import DeepSpeedConfig
    ds_config = DeepSpeedConfig(_ds_config)
    model_dtype = next(shared_model[1][0].parameters()).dtype

    from param_swapper import AsyncParameterSwapper
    param_swapper_handler = AsyncParameterSwapper(ds_config, special_layers_id, 1, model_dtype, None, transformer_layer_info=None, special_layers_info=None, transformer_layer=shared_model[1][0], special_layers_in_rank=special_layers_in_rank)

    transformer_layer_info, special_layers_info = collect_layer_param_info(shared_model, special_layers_id, param_swapper_handler)

    #
    from shared_optim_cpu import SharedModelNVMe_worker7
    shared_model_nvme_handler = SharedModelNVMe_worker7(shared_model, param_swapper_handler, special_layers_id)
    shared_model_nvme_handler.swap_out_to_ssd()

    del model
    pcm.print("shared model swap to SSD and detele")

    param_swapper_handler.clear_pinned_tensors()
    #######################################################################


    """ Initialize distributed training. """ 
    gc.collect(); torch.cuda.empty_cache() 
    assert torch.cuda.memory_reserved() == 0, "fork process begins w/ alloc = {} B".format(torch.cuda.memory_reserved()) 

    processes = []
    if args.numa_bind:
        from utils import NumaBinder
        numa_binder = NumaBinder(args.numa_bind_config)
    for rank in range(CONFIGS["N"]):
        p = mp.Process(target=worker_func, 
                        args=(args, real_dataset, shared_model, shared_optimizer, empty_model, get_lr_sched, compute_loss, save_model, prof['XMETA'], prof['TMETA'], rTASKS, CONFIGS, rank, transformer_layer_info, special_layers_info),
                        name="rank%d"%rank)
        # NOTE: this moves parameter from pinned memory to shared memory
        p.start()
        processes.append(p)
        if args.numa_bind:
            numa_binder.bind(p, rank)
        
    if args.nvprof:
        from viewer.probe_cpu import ProbeCPU
        probe_cpu = ProbeCPU(pids=[p.pid for p in processes], 
                            ranks=[rank for rank in range(CONFIGS["N"])])
        probe_cpu.run(processes[0])
        print("--- rank -1: Done ---")
        print("--- all pids = (%s) ---"% " ".join("%d"%pid for pid in list([os.getpid()]+[p.pid for p in processes])) )

    for p in processes:
        p.join()
    print("--- all workers joined successfully. ---")
