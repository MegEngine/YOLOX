#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M

from yolox.utils import bboxes_iou

from .losses import binary_cross_entropy, iou_loss, l1_loss
from .network_blocks import BaseConv, DWConv


def meshgrid(x, y):
    """meshgrid wrapper for megengine"""
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    mesh_shape = (y.shape[0], x.shape[0])
    mesh_x = F.broadcast_to(x, mesh_shape)
    mesh_y = F.broadcast_to(y.reshape(-1, 1), mesh_shape)
    return mesh_x, mesh_y


class YOLOXHead(M.Module):
    def __init__(
        self, num_classes, width=1.0, strides=[8, 16, 32],
        in_channels=[256, 512, 1024], act="silu", depthwise=False
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): wheather apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = []
        self.reg_convs = []
        self.cls_preds = []
        self.reg_preds = []
        self.obj_preds = []
        self.stems = []
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                M.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                M.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                M.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                M.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                M.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.strides = strides
        self.grids = [np.array(0)] * len(in_channels)
        self.expanded_strides = [None] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            M.init.fill_(conv.bias, bias_value)

        for conv in self.obj_preds:
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            M.init.fill_(conv.bias, bias_value)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = F.concat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level)
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(F.full((1, grid.shape[1]), stride_this_level))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.reshape(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = (
                        F.transpose(reg_output, (0, 1, 3, 4, 2)).reshape(batch_size, -1, 4)
                    )
                    origin_preds.append(mge.Tensor(reg_output))

            else:
                output = F.concat([reg_output, F.sigmoid(obj_output), F.sigmoid(cls_output)], 1)

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs, x_shifts, y_shifts, expanded_strides,
                labels, F.concat(outputs, 1), origin_preds,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = F.concat([F.flatten(x, start_axis=2) for x in outputs], axis=2)
            outputs = F.transpose(outputs, (0, 2, 1))

            if self.decode_in_inference:
                return self.decode_outputs(outputs)
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            xv, yv = meshgrid(F.arange(wsize), F.arange(hsize))
            grid = F.stack((xv, yv), 2).reshape(1, 1, hsize, wsize, 2)
            self.grids[k] = grid

        output = output.reshape(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = (
            F.transpose(output, (0, 1, 3, 4, 2))
            .reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        )
        grid = grid.reshape(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = F.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            xv, yv = meshgrid(F.arange(wsize), F.arange(hsize))
            grid = F.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(F.full((*shape, 1), stride))

        grids = F.concat(grids, axis=1)
        strides = F.concat(strides, axis=1)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = F.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = F.expand_dims(outputs[:, :, 4], axis=-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(axis=2) > 0).sum(axis=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = F.concat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = F.concat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = F.concat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = F.concat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = F.zeros((0, self.num_classes))
                reg_target = F.zeros((0, 4))
                l1_target = F.zeros((0, 4))
                obj_target = F.zeros((total_num_anchors, 1))
                fg_mask = F.zeros(total_num_anchors).astype("bool")
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(  # noqa
                    batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                    cls_preds, bbox_preds, obj_preds, labels, imgs,
                )

                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.astype("int32"), self.num_classes
                ) * F.expand_dims(pred_ious_this_matching, axis=-1)
                obj_target = F.expand_dims(fg_mask, axis=-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        F.zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = F.concat(cls_targets, 0)
        reg_targets = F.concat(reg_targets, 0)
        obj_targets = F.concat(obj_targets, 0)
        fg_masks = F.concat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (iou_loss(bbox_preds.reshape(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (binary_cross_entropy(obj_preds.reshape(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (
            binary_cross_entropy(cls_preds.reshape(-1, self.num_classes)[fg_masks], cls_targets)
        ).sum() / num_fg

        if self.use_l1:
            l1_targets = F.concat(l1_targets, 0)
            loss_l1 = (l1_loss(origin_preds.reshape(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = mge.Tensor(0.0)

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1)

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = F.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = F.log(gt[:, 3] / stride + eps)
        return l1_target

    def get_assignments(
        self, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
        cls_preds, bbox_preds, obj_preds, labels, imgs
    ):
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        pair_wise_ious = bboxes_iou(
            gt_bboxes_per_image, bboxes_preds_per_image, False
        )

        # MGE might bring bad exper
        gt_cls_per_image = (
            F.repeat(
                F.expand_dims(
                    F.one_hot(gt_classes.astype("int32"), self.num_classes).astype("float32"),
                    axis=1,
                ),
                repeats=num_in_boxes_anchor, axis=1,
            )
        )
        pair_wise_ious_loss = -F.log(pair_wise_ious + 1e-8)

        # ditto
        cls_preds_ = F.sigmoid(
            F.repeat(F.expand_dims(cls_preds_.astype("float32"), axis=0), repeats=num_gt, axis=0)
        ) * F.sigmoid(F.repeat(F.expand_dims(obj_preds_, axis=0), repeats=num_gt, axis=0))

        pair_wise_cls_loss = binary_cross_entropy(
            F.sqrt(cls_preds_), gt_cls_per_image, with_logits=False,
        ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
            gt_matched_classes.detach(),
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg
        )

    def get_in_boxes_info(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            F.repeat(
                F.expand_dims(x_shifts_per_image + 0.5 * expanded_strides_per_image, axis=0),
                repeats=num_gt, axis=0,
            )
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = F.repeat(
            F.expand_dims(y_shifts_per_image + 0.5 * expanded_strides_per_image, axis=0),
            repeats=num_gt, axis=0,
        )

        gt_bboxes_per_image_l = F.repeat(
            F.expand_dims(gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2], axis=1),
            repeats=total_num_anchors, axis=1,
        )
        gt_bboxes_per_image_r = F.repeat(
            F.expand_dims(gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2], axis=1),
            repeats=total_num_anchors, axis=1,
        )
        gt_bboxes_per_image_t = F.repeat(
            F.expand_dims(gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3], axis=1),
            repeats=total_num_anchors, axis=1,
        )
        gt_bboxes_per_image_b = F.repeat(
            F.expand_dims(gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3], axis=1),
            repeats=total_num_anchors, axis=1,
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = F.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(axis=-1) > 0.0
        is_in_boxes_all = is_in_boxes.sum(axis=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = F.repeat(
            F.expand_dims(gt_bboxes_per_image[:, 0], axis=1),
            repeats=total_num_anchors, axis=1,
        ) - center_radius * F.expand_dims(expanded_strides_per_image, axis=0)
        gt_bboxes_per_image_r = F.repeat(
            F.expand_dims(gt_bboxes_per_image[:, 0], axis=1),
            repeats=total_num_anchors, axis=1,
        ) + center_radius * F.expand_dims(expanded_strides_per_image, axis=0)
        gt_bboxes_per_image_t = F.repeat(
            F.expand_dims(gt_bboxes_per_image[:, 1], axis=1),
            repeats=total_num_anchors, axis=1,
        ) - center_radius * F.expand_dims(expanded_strides_per_image, axis=0)
        gt_bboxes_per_image_b = F.repeat(
            F.expand_dims(gt_bboxes_per_image[:, 1], axis=1),
            repeats=total_num_anchors, axis=1,
        ) + center_radius * F.expand_dims(expanded_strides_per_image, axis=0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = F.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(axis=-1) > 0.0
        is_in_centers_all = is_in_centers.sum(axis=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor.detach(), is_in_boxes_and_center.detach()

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = F.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.shape[1])
        topk_ious, _ = F.topk(ious_in_boxes_matrix, n_candidate_k, descending=True)
        dynamic_ks = F.clip(topk_ious.sum(1).astype("int32"), lower=1)
        for gt_idx in range(num_gt):
            _, pos_idx = F.topk(cost[gt_idx], k=dynamic_ks[gt_idx], descending=False)
            matching_matrix[gt_idx, pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_argmin = F.argmin(cost[:, anchor_matching_gt > 1], axis=0)
            matching_matrix[:, anchor_matching_gt > 1] = 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum()

        # set True part to fg_mask_inboxes
        fg_mask[fg_mask] = fg_mask_inboxes

        matched_gt_inds = F.argmax(matching_matrix[:, fg_mask_inboxes], axis=0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return (
            num_fg.detach(),
            gt_matched_classes.detach(),
            pred_ious_this_matching.detach(),
            matched_gt_inds.detach(),
        )

    def state_dict(self, *args, **kwargs):
        head_state_dict = super().state_dict(*args, **kwargs)
        drop_keynames = ["grids.0", "grids.1", "grids.2"]
        for keyname in drop_keynames:
            head_state_dict.pop(keyname, None)
        return head_state_dict
