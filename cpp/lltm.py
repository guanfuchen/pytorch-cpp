import math
from torch import nn
from torch.autograd import Function
import torch
import numpy as np
import time

import lltm_cpp

torch.manual_seed(42)


class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        d_old_h, d_input, d_weights, d_bias, d_old_cell = lltm_cpp.backward(
            grad_h, grad_cell, *ctx.saved_variables)
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


def bbox_ious(boxes1_xy, boxes1_wh, boxes2_xy, boxes2_wh):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    # b1_len = boxes1.size(0)
    # b2_len = boxes2.size(0)
    #
    # b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    # b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    # b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    # b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)
    #
    # dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    # dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    # intersections = dx * dy
    #
    # areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    # areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    # unions = (areas1 + areas2.t()) - intersections
    # bbox_ious_function.apply
    ious = lltm_cpp.bbox_ious(boxes1_xy, boxes1_wh, boxes2_xy, boxes2_wh)
    # print('ious:', ious)
    # return intersections / unions
    return ious


class LLTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

        # lltm_cpp.bbox_ious()

        boxes1 = torch.FloatTensor(np.arange(24*4).reshape(24, 4))
        boxes2 = torch.FloatTensor(np.arange(24*4).reshape(24, 4))

        test_num = 1
        t_cost = 0
        for _ in range(test_num):
            t_start = time.time()
            boxes1_xy = boxes1[:, :2]
            boxes1_wh = boxes1[:, 2:4]
            boxes2_xy = boxes2[:, :2]
            boxes2_wh = boxes2[:, 2:4]
            bbox_ious(boxes1_xy, boxes1_wh, boxes2_xy, boxes2_wh)

            t_end = time.time()
            t_cost += t_end - t_start
        print('cost time avg: {} s'.format((t_cost) / test_num))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)
