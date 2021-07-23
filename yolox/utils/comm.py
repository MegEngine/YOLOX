#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import pickle

import megengine.distributed as dist


def gather_pyobj(obj, obj_name, target_rank_id=0, reset_after_gather=True):
    """
    gather non tensor object into target rank.

    Args:
        obj (object): object to gather, for non python-buildin object, please
            make sure that it's picklable, otherwise gather process might be stucked.
        obj_name (str): name of pyobj, used for distributed client.
        target_rank_id (int): rank of target device. default: 0.
        reset_after_gather (bool): wheather reset value in client after get value. defualt: True.

    Returns:
        A list contains all objects if on target device, else None.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [obj]

    local_rank = dist.get_rank()
    if local_rank == target_rank_id:
        obj_list = []
        for rank in range(world_size):
            if rank == target_rank_id:
                obj_list.append(obj)
            else:
                rank_data = dist.get_client().user_get(f"{obj_name}{rank}")
                obj_list.append(pickle.loads(rank_data.data))
                if reset_after_gather:
                    dist.get_client().user_set(f"{obj_name}{rank}", None)
        return obj_list
    else:
        dist.get_client().user_set(f"{obj_name}{local_rank}", pickle.dumps(obj))
