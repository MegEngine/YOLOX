#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

# from torch.nn.parallel import DistributedDataParallel as DDP
import megengine as mge
import megengine.distributed as dist

from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, setup_logger


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machine", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed", dest="speed", default=False, action="store_true", help="speed test only."
    )
    parser.add_argument("--origin-eval", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, num_gpu, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()

    rank = dist.get_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(
        file_name, distributed_rank=rank, filename="val_log.txt", mode="a"
    )
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test)

    model.eval()

    if not args.speed:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pkl")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = mge.load(ckpt_file, map_location="cpu")
        # load the model state dict

        model_state = ckpt
        if "model" in ckpt:
            model_state = ckpt["model"]
            # model_state = ckpt["origin_model"]
        for i in range(3):
            model_state.pop("head.grids.{}".format(i), None)
        model.load_state_dict(model_state, strict=False)
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    # start evaluate
    evaluator.evaluate(model, is_distributed, test_size=exp.test_size)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    num_gpus = dist.helper.get_device_count_by_fork("gpu")

    if args.devices is None:
        args.devices = num_gpus

    assert args.devices <= num_gpus

    if args.devices > 1:
        test = dist.launcher(main, n_gpus=args.devices)
        test(exp, num_gpus, args)
    else:
        main(exp, num_gpus, args)
