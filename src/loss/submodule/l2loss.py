"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    L2Loss implementation
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, pred, gt):
        mask = (gt > self.t_valid).type_as(pred).detach()

        loss_val = F.mse_loss(pred, gt, reduction='none')

        loss_val = loss_val * mask

        num_valid = torch.sum(mask, dim=[1, 2, 3])
        loss_val = torch.sum(loss_val, dim=[1, 2, 3])

        loss = loss_val / (num_valid + 1e-8)

        loss = loss.sum()

        return loss
