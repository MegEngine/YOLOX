#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

    def next(self):
        inputs, target, _, _ = next(self.loader)
        return inputs.numpy(), target.numpy()


def random_resize(data_loader, exp, epoch, rank, is_distributed):
    tensor = mge.tensor([1])

    if rank == 0:
        if epoch > exp.max_epoch - 10:
            size = exp.input_size
        else:
            size = random.randint(*exp.random_size)
            size = int(32 * size)
        tensor *= size

    if is_distributed:
        tensor = F.distributed.broadcast(tensor)
        dist.group_barrier()

    input_size = data_loader.change_input_dim(multiple=tensor.item(), random_range=None)
    return input_size
