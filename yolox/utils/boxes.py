#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np
from megengine import Tensor
from typing import Optional
import megengine.functional as F

__all__ = [
    "batched_nms",
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
]


def batched_nms(
    boxes: Tensor, scores: Tensor, idxs: Tensor, iou_thresh: float, max_output: Optional[int] = None
) -> Tensor:
    r"""
    Performs non-maximum suppression (NMS) on the boxes according to
    their intersection-over-union (IoU).
    :param boxes: tensor of shape `(N, 4)`; the boxes to perform nms on;
        each box is expected to be in `(x1, y1, x2, y2)` format.
    :param iou_thresh: ``IoU`` threshold for overlapping.
    :param idxs: tensor of shape `(N,)`, the class indexs of boxes in the batch.
    :param scores: tensor of shape `(N,)`, the score of boxes.

    :return: indices of the elements that have been kept by NMS.

    Examples:
    .. testcode::
        import numpy as np
        from megengine import tensor
        x = np.zeros((100,4))
        np.random.seed(42)
        x[:,:2] = np.random.rand(100,2) * 20
        x[:,2:] = np.random.rand(100,2) * 20 + 100
        scores = tensor(np.random.rand(100))
        idxs = tensor(np.random.randint(0, 10, 100))
        inp = tensor(x)
        result = batched_nms(inp, scores, idxs, iou_thresh=0.6)
        print(result.numpy())

    Outputs:
    .. testoutput::
        [75 41 99 98 69 64 11 27 35 18]
    """
    assert (
        boxes.ndim == 2 and boxes.shape[1] == 4
    ), "the expected shape of boxes is (N, 4)"
    assert scores.ndim == 1, "the expected shape of scores is (N,)"
    assert idxs.ndim == 1, "the expected shape of idxs is (N,)"
    assert (
        boxes.shape[0] == scores.shape[0] == idxs.shape[0]
    ), "number of boxes, scores and idxs are not matched"

    idxs = idxs.detach()
    max_coordinate = boxes.max()
    offsets = idxs * (max_coordinate + 1)
    boxes = boxes + offsets.reshape(-1, 1)
    return F.vision.nms(boxes, scores, iou_thresh, max_output)


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = F.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue
        # Get score and class with highest confidence
        # class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        class_conf = F.max(image_pred[:, 5: 5 + num_classes], axis=1, keepdims=True)
        class_pred = F.argmax(image_pred[:, 5: 5 + num_classes], axis=1, keepdims=True)

        conf_mask = (image_pred[:, 4] * class_conf.reshape(-1) >= conf_thre)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = F.concat((image_pred[:, :5], class_conf, class_pred), 1)
        detections = detections[conf_mask]
        if not detections.shape[0]:
            continue

        nms_out_index = batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = F.concat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = F.maximum(F.expand_dims(bboxes_a[:, :2], axis=1), bboxes_b[:, :2])
        br = F.minimum(F.expand_dims(bboxes_a[:, 2:], axis=1), bboxes_b[:, 2:])
        area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
        area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
    else:
        tl = F.maximum(
            F.expand_dims(bboxes_a[:, :2] - bboxes_a[:, 2:] / 2, axis=1),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = F.minimum(
            F.expand_dims(bboxes_a[:, :2] + bboxes_a[:, 2:] / 2, axis=1),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = bboxes_a[:, 2] * bboxes_a[:, 3]
        area_b = bboxes_b[:, 2] * bboxes_b[:, 3]
    mask = (tl < br).astype("int32")
    mask = mask[..., 0] * mask[..., 1]
    diff = br - tl
    area_i = diff[..., 0] * diff[..., 1] * mask
    del diff
    return area_i / (F.expand_dims(area_a, axis=1) + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes
