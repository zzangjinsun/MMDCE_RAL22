"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    Some of useful functions are defined here.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


model_path = {
    'resnet18': 'pretrained/resnet18.pth',
    'resnet34': 'pretrained/resnet34.pth'
}


def get_resnet18(pretrained=True):
    net = torchvision.models.resnet18(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet18'])
        net.load_state_dict(state_dict)

    return net


def get_resnet34(pretrained=True):
    net = torchvision.models.resnet34(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet34'])
        net.load_state_dict(state_dict)

    return net


def conv2d_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                   relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt2d_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0,
                    output_padding=0, bn=True, relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def conv3d_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                   relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv3d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm3d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt3d_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0,
                    output_padding=0, bn=True, relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose3d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm3d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def stack_blocks(block, num, ch_in, ch_out, kernel, stride=1):
    layers = []

    layers.append(block(ch_in, ch_out, kernel, stride=stride))
    for k in range(1, num):
        layers.append(block(ch_out, ch_out, kernel))

    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, stride=1):
        super(ResBlock, self).__init__()

        assert (kernel % 2) == 1, \
            'only odd kernel is supported but kernel = {}'.format(kernel)
        pad = int((kernel - 1) / 2)

        self.conv1 = conv2d_bn_relu(ch_in, ch_out, kernel, stride=stride,
                                    padding=pad, bn=True, relu=True)
        self.conv2 = conv2d_bn_relu(ch_out, ch_out, kernel, padding=pad,
                                    bn=True, relu=False)

        self.conv_skip = None
        if ch_in != ch_out or stride != 1:
            self.conv_skip = nn.Conv2d(ch_in, ch_out, kernel_size=1,
                                       stride=stride, padding=0, bias=False)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        if self.conv_skip is not None:
            x = self.conv_skip(x)

        y = F.leaky_relu(y + x, 0.2, inplace=True)

        return y


class DCGenerator:
    def __init__(self, args):
        self.args = args

    def encode(self, disp, disp_range):
        batch, channel, height, width = disp.shape
        assert channel == 1

        disp_range = disp_range.view(batch, -1, 1, 1)
        min_disp = disp_range[:, [0], :, :]
        max_disp = disp_range[:, [-1], :, :]

        num_disp = disp_range.shape[1]

        mask_valid = (disp > 0).type_as(disp)

        disp = torch.min(torch.max(disp, min_disp), max_disp)

        diff_val = disp - disp_range[:, :-1, :, :]

        _, idx = torch.min(torch.abs(diff_val), dim=1, keepdim=True)
        delta = torch.gather(diff_val, dim=1, index=idx)

        mask_x = (delta >= 0.0).type_as(idx)
        mask_y = (delta < 0.0).type_as(idx)

        # NOTE : depth = alpha*x + (1-alpha)*y, y = x + bin
        x = idx * mask_x + (idx - 1) * mask_y
        y = idx * mask_y + (idx + 1) * mask_x

        x_val = torch.gather(disp_range.expand(-1, -1, height, width), dim=1, index=x)
        y_val = torch.gather(disp_range.expand(-1, -1, height, width), dim=1, index=y)
        bin = y_val - x_val

        alpha = 1.0 - (disp - x_val) / (bin + 1e-8)

        alpha = torch.squeeze(alpha, dim=1)
        x = torch.squeeze(x, dim=1)
        y = torch.squeeze(y, dim=1)

        dc = torch.zeros((batch, num_disp, height, width)).type_as(disp)

        for k in range(0, num_disp):
            dc[:, k, :, :] += alpha * (x == k).type_as(disp)
            dc[:, k, :, :] += (1.0 - alpha) * (y == k).type_as(disp)

        dc = torch.clamp(dc, min=0, max=1.0) * mask_valid

        dc = dc.contiguous()

        return dc

    def decode(self, dc, disp_range):
        batch, num_dc, height, width = dc.shape

        num_disp = disp_range.shape[1]

        assert num_dc == num_disp

        disp_range = disp_range.view(batch, -1, 1, 1).type_as(dc)

        disp = torch.sum(disp_range * dc, dim=1, keepdim=True)

        return disp
