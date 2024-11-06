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


import os
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


class LRFactor:
    def __init__(self, decay, gamma):
        assert len(decay) == len(gamma)

        self.decay = decay
        self.gamma = gamma

    def get_factor(self, epoch):
        for (d, g) in zip(self.decay, self.gamma):
            if epoch < d:
                return g
        return self.gamma[-1]


def convert_str_to_num(val, t):
    val = val.replace('\'', '')
    val = val.replace('\"', '')

    if t == 'int':
        val = [int(v) for v in val.split(',')]
    elif t == 'float':
        val = [float(v) for v in val.split(',')]
    else:
        raise NotImplementedError

    return val


def make_optimizer_scheduler(args, target):
    # optimizer
    if hasattr(target, 'param_groups'):
        # NOTE : lr for each group must be set by the network
        trainable = target.param_groups
    else:
        trainable = filter(lambda x: x.requires_grad, target.parameters())

    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    else:
        raise NotImplementedError

    optimizer = optimizer_class(trainable, **kwargs_optimizer)

    # scheduler
    decay = convert_str_to_num(args.decay, 'int')
    gamma = convert_str_to_num(args.gamma, 'float')

    assert len(decay) == len(gamma), 'decay and gamma must have same length'

    calculator = LRFactor(decay, gamma)
    scheduler = lrs.LambdaLR(optimizer, calculator.get_factor)

    return optimizer, scheduler


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))


def warp_image(image_src, disp_tar):
    # Note : Source image is warped to target image
    # image_src = right image of a stereo pair (source image)
    # disp_tar = disparity aligned to the left image (target disparity)

    batch, _, height, width = image_src.shape
    mask = torch.ones((batch, 1, height, width),
                      requires_grad=False).type_as(image_src)

    theta = torch.zeros((batch, 2, 3), device=image_src.device,
                        requires_grad=False)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0

    grid = F.affine_grid(theta, image_src.size(), align_corners=False)
    # Note : Disparity is subtracted in here
    #        Coordinates should be 2*disp because grid range is [-1, 1]
    grid[:, :, :, 0] = grid[:, :, :, 0] - (2.0 * disp_tar[:, 0, :, :] / width)
    if disp_tar.shape[1] == 2:
        grid[:, :, :, 1] = \
            grid[:, :, :, 1] - (2.0 * disp_tar[:, 1, :, :] / height)

    image_warped = F.grid_sample(image_src, grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)
    mask_warped = F.grid_sample(mask, grid, mode='nearest',
                                padding_mode='zeros', align_corners=False)

    return image_warped, mask_warped
