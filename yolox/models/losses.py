#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import functools

import megengine.functional as F


def iou_loss(pred, target, reduction="none", loss_type="iou"):
    assert pred.shape[0] == target.shape[0]

    pred = pred.reshape(-1, 4)
    target = target.reshape(-1, 4)
    tl = F.maximum(
        (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
    )
    br = F.minimum(
        (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
    )

    area_p = pred[:, 2] * pred[:, 3]
    area_g = target[:, 2] * target[:, 3]

    mask = (tl < br).astype("int32")
    mask = mask[..., 0] * mask[..., 1]

    diff = br - tl
    area_i = diff[..., 0] * diff[..., 1] * mask
    iou = (area_i) / (area_p + area_g - area_i + 1e-16)

    if loss_type == "iou":
        loss = 1 - iou ** 2

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def smooth_l1_loss(pred, target, beta=1.0):
    r"""Smooth L1 Loss.

    .. math::

        loss_{i} =
        \begin{cases}
        0.5 (x_i - y_i)^2 / beta, & \text{if } |x_i - y_i| < beta \\
        |x_i - y_i| - 0.5 * beta, & \text{otherwise }
        \end{cases}

    Args:
        pred (Tensor): the predictions
        target (Tensor): the assigned targets with the same shape as pred
        beta (int): the parameter of smooth l1 loss.

    Returns:
        the calculated smooth l1 loss.
    """
    x = pred - target
    abs_x = F.abs(x)
    if beta < 1e-5:
        loss = abs_x
    else:
        in_loss = 0.5 * x ** 2 / beta
        out_loss = abs_x - 0.5 * beta
        loss = F.where(abs_x < beta, in_loss, out_loss)
    return loss


l1_loss = functools.partial(smooth_l1_loss, beta=0.0)


def binary_cross_entropy(pred, label, with_logits=True):
    r"""
    Computes the binary cross entropy loss (using logits by default).
    By default(``with_logitis`` is True), ``pred`` is assumed to be logits,
    class probabilities are given by sigmoid.

    Args:
        pred (Tensor): `(N, *)`, where `*` means any number of additional dimensions.
        label (Tensor): `(N, *)`, same shape as the input.
        with_logits (bool): whether to apply sigmoid first. Default: True

    Return:
        loss (Tensor): bce loss value.
    """
    if with_logits:
        # logsigmoid(pred) and logsigmoid(-pred) has common sub-expression
        # hopefully the backend would optimize this
        loss = -(label * F.logsigmoid(pred) + (1 - label) * F.logsigmoid(-pred))
    else:
        loss = -(label * F.log(pred) + (1 - label) * F.log(1 - pred))
    return loss
