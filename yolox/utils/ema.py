#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import math
from copy import deepcopy

import megengine as mge


class ModelEMA:
    """
    Model Exponential Moving Average.
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    """
    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        # NOTE: EMA don't need gradient
        self.ema = deepcopy(model)
        self.ema.eval()
        # self._ema_states = list(self.ema.parameters())
        self._ema_states = {k: v for k, v in self.ema.named_parameters()}
        self._ema_states.update({n: p for n, p in self.ema.named_buffers()})

        self._model_states = {k: v for k, v in model.named_parameters()}
        self._model_states.update({n: p for n, p in model.named_buffers()})

        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))

    def update(self):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        for k, v in self._ema_states.items():
            v._reset(v * mge.tensor(d) + mge.tensor(1 - d) * self._model_states[k])
