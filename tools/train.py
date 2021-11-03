#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
from loguru import logger
import multiprocessing as mp

import megengine as mge
import megengine.distributed as dist

from yolox.core import Trainer
from yolox.exp import get_exp
from yolox.utils import configure_nccl


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--dtr", action="store_true", help="use dtr for training")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--start-epoch", default=None, type=int, help="start epoch of training")
    parser.add_argument(
        "--num_machine", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--sync_level", type=int, default=None, help="config sync level, use 0 to debug"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # set environment variables for distributed training
    configure_nccl()

    # enable dtr to avoid CUDA OOM
    if args.dtr:
        logger.info("enable DTR during training...")
        mge.dtr.enable()

    if args.sync_level is not None:
        # NOTE: use sync_level = 0 to debug mge error
        from megengine.core._imperative_rt.core2 import config_async_level
        logger.info("Using aysnc_level {}".format(args.sync_level))
        config_async_level(args.sync_level)

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    mp.set_start_method("spawn")
    num_gpus = mge.device.get_device_count("gpu")

    if args.devices is None:
        args.devices = num_gpus

    assert args.devices <= num_gpus

    if args.devices > 1:
        train = dist.launcher(main, n_gpus=args.devices)
        train(exp, args)
    else:
        main(exp, args)
