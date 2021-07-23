#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import megengine as mge
import megengine.functional as F
import megengine.module as M

__all__ = ["fuse_conv_and_bn", "fuse_model", "replace_module"]


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = M.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True,
    )

    # fused_conv.weight = bn.weight / running_var * conv.weight
    w_conv = conv.weight.reshape(conv.out_channels, -1)
    factor = (bn.weight / F.sqrt(bn.eps + bn.running_var)).reshape(-1)
    # diag_factor = diag(factor)
    fusedconv.weight = mge.Parameter(
        (factor.reshape(-1, 1) * w_conv).reshape(fusedconv.weight.shape)
    )

    # fused_conv.bias = bn.bias + (conv.bias - running_mean) * bn.weight / runing_var
    conv_bias = F.zeros(bn.running_mean.shape) if conv.bias is None else conv.bias
    fuse_bias = bn.bias + (conv_bias - bn.running_mean) * factor.reshape(1, -1, 1, 1)
    fusedconv.bias = mge.Parameter(fuse_bias)

    return fusedconv


def fuse_model(model):
    from yolox.models.network_blocks import BaseConv

    for m in model.modules():
        if type(m) is BaseConv and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


def replace_module(module, replaced_module_type, new_module_type, replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """
    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model
