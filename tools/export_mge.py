#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import numpy as np

import megengine as mge
from megengine import jit

from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX export for MegEngine")
    parser.add_argument("-s", "--shape", default=640, type=int, help="input shape of image")
    parser.add_argument(
        "-f", "--exp_file", default=None, type=str, help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default="yolox-s", help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--dump_path", default="model.mge", help="path to save the dumped model")
    return parser


def dump_static_graph(model, graph_name="model.mge", input_shape=640):
    model.eval()
    model.head.decode_in_inference = False

    data = mge.Tensor(np.random.random((1, 3, input_shape, input_shape)))

    @jit.trace(capture_as_const=True)
    def pred_func(data):
        outputs = model(data)
        return outputs

    pred_func(data)
    pred_func.dump(
        graph_name,
        arg_names=["data"],
        optimize_for_inference=True,
        enable_fuse_conv_bias_nonlinearity=True,
    )


def build_and_load(model, weight_file=None):
    if weight_file is not None:
        model_weights = mge.load(weight_file)
        model.load_state_dict(model_weights, strict=False)

    model.head.decode_in_inference = False
    model.eval()
    return model


def main(args, exp):
    model = exp.get_model()
    model = build_and_load(model, exp)
    dump_static_graph(model, args.dump_path, args.shape)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(args, exp)
