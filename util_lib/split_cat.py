# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

def split_minibatch_size(minibatch_size, ubatch_size):
    """ split a minibatch size into a list of microbatch sizes """
    assert isinstance(minibatch_size, int) and isinstance(ubatch_size, int)
    assert minibatch_size >= ubatch_size
    if minibatch_size % ubatch_size == 0:
        ubatch_sizes = [ubatch_size] * int(minibatch_size/ubatch_size)
    else:
        ubatch_sizes = [ubatch_size] * int(minibatch_size/ubatch_size) \
                        + [minibatch_size%ubatch_size ]
    assert sum(ubatch_sizes) == minibatch_size
    
    return ubatch_sizes

# 将一个张量 t 按照指定的大小 split_size 在第一个维度上进行分割，并返回分割后的子张量组成的元组
def _split_tensor(t, split_size):
    assert t.ndim > 0, "scalar tensor cannot be split'ed" # dim=0 must be ubatchsize
    return torch.split(t, split_size, dim=0) # share the same underlying memory # inherit tensor's device
    # tensor will be split into equally sized chunks (if possible). 
    # Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.
    # return (t1,t2) or (t1,res) or (t1,) or (res,)

# minibatch：一个tuple，里面装着minibatch
# 将minibatch按照 ubatch_sizes 这个micro batch size列表首个元素的大小进行切分，返回切分后的microbatch列表
def split_minibatch(minibatch, ubatch_sizes):
    # print("...................................................")
    # print(ubatch_sizes)
    # >>> [8, 8, 8, 8]
    """ split a minibatch into a list of microbatches """
    assert isinstance(minibatch, tuple)
    # input_ids, input_mask, segment_ids, label_ids = minibatch
    
    ### split minibatch
    # 对minibatch这个tuple中的每个tensor（目前来看只有一个），按照 ubatch_sizes 中第一个 microbatch size 进行分割，
    # 即每个micorbatch中sample的数量均为 第一个 microbatch size 的值。返回分割后的子张量组成的元组。
    # 这个tensor t 是所有sample组成的大tensor，例如shape为 32×1024
    splits = [_split_tensor(t, ubatch_sizes[0]) for t in minibatch]
    # 确保每个micro batch的大小和micro batch的数量相同
    # print(len(splits[0])) # 4
    # print(splits[0][0].shape) # torch.Size([8, 1024])

    # split_t是一个元组，里面装着microbatch，逻辑为 microbatch 的数量要等于 ubatch_sizes (装着所有micobatchsize的list)的长度
    for split_t in splits:
        assert len(split_t) == len(ubatch_sizes)
    # print(len(ubatch_sizes)) # 4

    ### make microbatches
    ubatches = [] # [ubatch#0, ubatch#1, ...] 
    # 取出每一个microbatch，装进 ubatches 这个list中
    for i, u in enumerate(ubatch_sizes):
        # 
        ubatch = tuple(split_t[i] for split_t in splits)
        ubatches.append(ubatch)
        
        # 检查 microbatch 的第一个维度是否正确，即一个microbatch的sample数量是否等于 ubatch_sizes 对应位置的大小
        for t in ubatch:
            assert t.shape[0] == u
    
    # 返回给定minibatch切分后的 microbatch
    return ubatches

def split_DP_minibatch_size(num_gpus, minibatch_size, ubatch_size):
    """ split the global minibatch size into a list of per-GPU microbatch sizes """
    per_gpu_ubatch_sizes = []
    for n in range(num_gpus):
        # ----- find per-GPU microbatch sizes -----
        DD = int(float(minibatch_size)/num_gpus)
        if minibatch_size % num_gpus != 0: # uneven batch size across GPUs
            if n < minibatch_size % num_gpus:
                DD += 1
        ubszs = split_minibatch_size(DD, ubatch_size)
        per_gpu_ubatch_sizes.append(ubszs)
    
    return per_gpu_ubatch_sizes

def split_DP_minibatch(minibatch, per_gpu_ubatch_sizes, rank):
    """ split the global minibatch into a local batch """
    assert isinstance(minibatch, tuple)
    # input_ids, input_mask, segment_ids, label_ids = minibatch
    minibatchsize0 = sum(per_gpu_ubatch_sizes[0])
    ### split minibatch across GPUs
    splits = [_split_tensor(t, minibatchsize0) for t in minibatch]
    for split_t in splits:
        assert len(split_t) == len(per_gpu_ubatch_sizes)
    ### choose which split by rank
    batch_local = tuple(split_t[rank] for split_t in splits)
    
    return batch_local

