"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    Multi-Modal Depth in Changing Environments (MMDCE) dataset implementation
"""


import os
import random
import numpy as np
import json

from . import BaseDataset
from .common import *

from PIL import Image
import torch
import torchvision.transforms.functional as TF

"""
Multi-Modal Depth in Changing Environments (MMDCE) dataset json file has the following format:

{
    "train": [
        {
            "rgb1": "2020-11-07-16-57-01/rgb1/1604736626568461722.png",
            "rgb2": "2020-11-07-16-57-01/rgb2/1604736626568461722.png",
            "ir1": "2020-11-07-16-57-01/ir1/1604736626568461722.png",
            "ir2": "2020-11-07-16-57-01/ir2/1604736626568461722.png",
            "dep_rgb1": "2020-11-07-16-57-01/dep_rgb1/1604736626568461722.png",
            "dep_rgb2": "2020-11-07-16-57-01/dep_rgb2/1604736626568461722.png",
            "dep_ir1": "2020-11-07-16-57-01/dep_ir1/1604736626568461722.png",
            "dep_ir2": "2020-11-07-16-57-01/dep_ir2/1604736626568461722.png",
            "gt_dep_rgb1": "2020-11-07-16-57-01/gt_dep_rgb1_filtered/1604736626568461722.png",
            "gt_dep_rgb2": "2020-11-07-16-57-01/gt_dep_rgb2/1604736626568461722.png",
            "gt_dep_ir1": "2020-11-07-16-57-01/gt_dep_ir1_filtered/1604736626568461722.png",
            "gt_dep_ir2": "2020-11-07-16-57-01/gt_dep_ir2/1604736626568461722.png",
            "calib": "2020-11-07-16-57-01/calib.npy",
            "info": "2020-11-07-16-57-01/info.txt"
        }, ...
    ],
    "val": [
        {
            "rgb1": "2020-11-07-17-18-38/rgb1/1604737119707223295.png",
            "rgb2": "2020-11-07-17-18-38/rgb2/1604737119707223295.png",
            "ir1": "2020-11-07-17-18-38/ir1/1604737119707223295.png",
            "ir2": "2020-11-07-17-18-38/ir2/1604737119707223295.png",
            "dep_rgb1": "2020-11-07-17-18-38/dep_rgb1/1604737119707223295.png",
            "dep_rgb2": "2020-11-07-17-18-38/dep_rgb2/1604737119707223295.png",
            "dep_ir1": "2020-11-07-17-18-38/dep_ir1/1604737119707223295.png",
            "dep_ir2": "2020-11-07-17-18-38/dep_ir2/1604737119707223295.png",
            "gt_dep_rgb1": "2020-11-07-17-18-38/gt_dep_rgb1_filtered/1604737119707223295.png",
            "gt_dep_rgb2": "2020-11-07-17-18-38/gt_dep_rgb2/1604737119707223295.png",
            "gt_dep_ir1": "2020-11-07-17-18-38/gt_dep_ir1_filtered/1604737119707223295.png",
            "gt_dep_ir2": "2020-11-07-17-18-38/gt_dep_ir2/1604737119707223295.png",
            "calib": "2020-11-07-17-18-38/calib.npy",
            "info": "2020-11-07-17-18-38/info.txt"
        }, ...
    ],
    "test": [
        {
            "rgb1": "2020-10-02-17-34-35/rgb1/1601627722067240953.png",
            "rgb2": "2020-10-02-17-34-35/rgb2/1601627722067240953.png",
            "ir1": "2020-10-02-17-34-35/ir1/1601627722067240953.png",
            "ir2": "2020-10-02-17-34-35/ir2/1601627722067240953.png",
            "dep_rgb1": "2020-10-02-17-34-35/dep_rgb1/1601627722067240953.png",
            "dep_rgb2": "2020-10-02-17-34-35/dep_rgb2/1601627722067240953.png",
            "dep_ir1": "2020-10-02-17-34-35/dep_ir1/1601627722067240953.png",
            "dep_ir2": "2020-10-02-17-34-35/dep_ir2/1601627722067240953.png",
            "gt_dep_rgb1": "2020-10-02-17-34-35/gt_dep_rgb1_filtered/1601627722067240953.png",
            "gt_dep_rgb2": "2020-10-02-17-34-35/gt_dep_rgb2/1601627722067240953.png",
            "gt_dep_ir1": "2020-10-02-17-34-35/gt_dep_ir1_filtered/1601627722067240953.png",
            "gt_dep_ir2": "2020-10-02-17-34-35/gt_dep_ir2/1601627722067240953.png",
            "calib": "2020-10-02-17-34-35/calib.npy",
            "info": "2020-10-02-17-34-35/info.txt"
        }, ...
    ]
}
"""


class MMDCE(BaseDataset):
    def __init__(self, args, mode):
        super(MMDCE, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        if mode not in ['train', 'val', 'test']:
            raise NotImplementedError

        self.cost_factor = args.cost_factor
        self.num_disp = args.num_disp
        self.max_disp = self.num_disp - 1
        self.max_depth = args.max_depth

        self.th_depth = args.th_depth
        self.th_disp = args.th_disp

        self.height = args.patch_height
        self.width = args.patch_width

        self.augment = args.augment

        with open(self.args.list_data) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb_l, rgb_r, dep_rgb_l, dep_rgb_r, \
        disp_rgb_l, disp_rgb_r, \
        gt_dep_rgb_l, gt_dep_rgb_r, \
        gt_disp_rgb_l, gt_disp_rgb_r, K_rgb, B_rgb, \
        ir_l, ir_r, dep_ir_l, dep_ir_r, \
        disp_ir_l, disp_ir_r, \
        gt_dep_ir_l, gt_dep_ir_r, \
        gt_disp_ir_l, gt_disp_ir_r, K_ir, B_ir, \
        R_ir2rgb, T_ir2rgb = self._load_data(idx)

        if self.augment and self.mode == 'train':
            w_rgb, h_rgb = rgb_l.size
            w_ir, h_ir = ir_l.size

            _scale = np.random.uniform(1.0, 1.5)
            scale_rgb = np.int(h_rgb*_scale)
            scale_ir = np.int(h_ir*_scale)

            # Color jitter
            brightness = np.random.uniform(0.6, 1.4)
            contrast = np.random.uniform(0.6, 1.4)
            saturation = np.random.uniform(0.6, 1.4)

            rgb_l = TF.adjust_brightness(rgb_l, brightness)
            rgb_l = TF.adjust_contrast(rgb_l, contrast)
            rgb_l = TF.adjust_saturation(rgb_l, saturation)

            rgb_r = TF.adjust_brightness(rgb_r, brightness)
            rgb_r = TF.adjust_contrast(rgb_r, contrast)
            rgb_r = TF.adjust_saturation(rgb_r, saturation)

            ir_l = TF.adjust_brightness(ir_l, brightness)
            ir_l = TF.adjust_contrast(ir_l, contrast)
            ir_l = TF.adjust_saturation(ir_l, saturation)

            ir_r = TF.adjust_brightness(ir_r, brightness)
            ir_r = TF.adjust_contrast(ir_r, contrast)
            ir_r = TF.adjust_saturation(ir_r, saturation)

            # Resize
            rgb_l = TF.resize(rgb_l, scale_rgb, Image.BICUBIC)
            rgb_r = TF.resize(rgb_r, scale_rgb, Image.BICUBIC)

            dep_rgb_l = TF.resize(dep_rgb_l, scale_rgb, Image.NEAREST)
            dep_rgb_r = TF.resize(dep_rgb_r, scale_rgb, Image.NEAREST)
            disp_rgb_l = TF.resize(disp_rgb_l, scale_rgb, Image.NEAREST)
            disp_rgb_r = TF.resize(disp_rgb_r, scale_rgb, Image.NEAREST)

            gt_dep_rgb_l = TF.resize(gt_dep_rgb_l, scale_rgb, Image.NEAREST)
            gt_dep_rgb_r = TF.resize(gt_dep_rgb_r, scale_rgb, Image.NEAREST)
            gt_disp_rgb_l = TF.resize(gt_disp_rgb_l, scale_rgb, Image.NEAREST)
            gt_disp_rgb_r = TF.resize(gt_disp_rgb_r, scale_rgb, Image.NEAREST)

            K_rgb[0] = K_rgb[0] * _scale
            K_rgb[1] = K_rgb[1] * _scale
            K_rgb[2] = K_rgb[2] * _scale
            K_rgb[3] = K_rgb[3] * _scale

            ir_l = TF.resize(ir_l, scale_ir, Image.BICUBIC)
            ir_r = TF.resize(ir_r, scale_ir, Image.BICUBIC)

            dep_ir_l = TF.resize(dep_ir_l, scale_ir, Image.NEAREST)
            dep_ir_r = TF.resize(dep_ir_r, scale_ir, Image.NEAREST)
            disp_ir_l = TF.resize(disp_ir_l, scale_ir, Image.NEAREST)
            disp_ir_r = TF.resize(disp_ir_r, scale_ir, Image.NEAREST)

            gt_dep_ir_l = TF.resize(gt_dep_ir_l, scale_ir, Image.NEAREST)
            gt_dep_ir_r = TF.resize(gt_dep_ir_r, scale_ir, Image.NEAREST)
            gt_disp_ir_l = TF.resize(gt_disp_ir_l, scale_ir, Image.NEAREST)
            gt_disp_ir_r = TF.resize(gt_disp_ir_r, scale_ir, Image.NEAREST)

            K_ir[0] = K_ir[0] * _scale
            K_ir[1] = K_ir[1] * _scale
            K_ir[2] = K_ir[2] * _scale
            K_ir[3] = K_ir[3] * _scale

            # Crop
            w_rgb, h_rgb = rgb_l.size
            w_ir, h_ir = ir_l.size

            assert self.height <= h_rgb and self.width <= w_rgb \
                   and self.height <= h_ir and self.width <= w_ir, \
                "patch size is larger than the input size"

            h_s_rgb = random.randint(0, h_rgb - self.height)
            w_s_rgb = random.randint(0, w_rgb - self.width)

            h_s_ir = random.randint(0, h_ir - self.height)
            w_s_ir = random.randint(0, w_ir - self.width)

            rgb_l = TF.crop(rgb_l, h_s_rgb, w_s_rgb, self.height, self.width)
            rgb_r = TF.crop(rgb_r, h_s_rgb, w_s_rgb, self.height, self.width)

            dep_rgb_l = TF.crop(dep_rgb_l, h_s_rgb, w_s_rgb, self.height, self.width)
            dep_rgb_r = TF.crop(dep_rgb_r, h_s_rgb, w_s_rgb, self.height, self.width)
            disp_rgb_l = TF.crop(disp_rgb_l, h_s_rgb, w_s_rgb, self.height, self.width)
            disp_rgb_r = TF.crop(disp_rgb_r, h_s_rgb, w_s_rgb, self.height, self.width)

            gt_dep_rgb_l = TF.crop(gt_dep_rgb_l, h_s_rgb, w_s_rgb, self.height, self.width)
            gt_dep_rgb_r = TF.crop(gt_dep_rgb_r, h_s_rgb, w_s_rgb, self.height, self.width)
            gt_disp_rgb_l = TF.crop(gt_disp_rgb_l, h_s_rgb, w_s_rgb, self.height, self.width)
            gt_disp_rgb_r = TF.crop(gt_disp_rgb_r, h_s_rgb, w_s_rgb, self.height, self.width)

            K_rgb[2] = K_rgb[2] - w_s_rgb
            K_rgb[3] = K_rgb[3] - h_s_rgb

            ir_l = TF.crop(ir_l, h_s_ir, w_s_ir, self.height, self.width)
            ir_r = TF.crop(ir_r, h_s_ir, w_s_ir, self.height, self.width)

            dep_ir_l = TF.crop(dep_ir_l, h_s_ir, w_s_ir, self.height, self.width)
            dep_ir_r = TF.crop(dep_ir_r, h_s_ir, w_s_ir, self.height, self.width)
            disp_ir_l = TF.crop(disp_ir_l, h_s_ir, w_s_ir, self.height, self.width)
            disp_ir_r = TF.crop(disp_ir_r, h_s_ir, w_s_ir, self.height, self.width)

            gt_dep_ir_l = TF.crop(gt_dep_ir_l, h_s_ir, w_s_ir, self.height, self.width)
            gt_dep_ir_r = TF.crop(gt_dep_ir_r, h_s_ir, w_s_ir, self.height, self.width)
            gt_disp_ir_l = TF.crop(gt_disp_ir_l, h_s_ir, w_s_ir, self.height, self.width)
            gt_disp_ir_r = TF.crop(gt_disp_ir_r, h_s_ir, w_s_ir, self.height, self.width)

            K_ir[2] = K_ir[2] - w_s_ir
            K_ir[3] = K_ir[3] - h_s_ir

            # Convert to tensor
            rgb_l = TF.to_tensor(rgb_l)
            rgb_l = TF.normalize(rgb_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            rgb_r = TF.to_tensor(rgb_r)
            rgb_r = TF.normalize(rgb_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            dep_rgb_l = TF.to_tensor(np.array(dep_rgb_l))
            dep_rgb_l = dep_rgb_l / _scale
            dep_rgb_r = TF.to_tensor(np.array(dep_rgb_r))
            dep_rgb_r = dep_rgb_r / _scale
            gt_dep_rgb_l = TF.to_tensor(np.array(gt_dep_rgb_l))
            gt_dep_rgb_l = gt_dep_rgb_l / _scale
            gt_dep_rgb_r = TF.to_tensor(np.array(gt_dep_rgb_r))
            gt_dep_rgb_r = gt_dep_rgb_r / _scale

            disp_rgb_l = TF.to_tensor(np.array(disp_rgb_l))
            disp_rgb_l = disp_rgb_l * _scale
            disp_rgb_r = TF.to_tensor(np.array(disp_rgb_r))
            disp_rgb_r = disp_rgb_r * _scale
            gt_disp_rgb_l = TF.to_tensor(np.array(gt_disp_rgb_l))
            gt_disp_rgb_l = gt_disp_rgb_l * _scale
            gt_disp_rgb_r = TF.to_tensor(np.array(gt_disp_rgb_r))
            gt_disp_rgb_r = gt_disp_rgb_r * _scale

            ir_l = TF.to_tensor(ir_l)
            ir_l = TF.normalize(ir_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            ir_r = TF.to_tensor(ir_r)
            ir_r = TF.normalize(ir_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            dep_ir_l = TF.to_tensor(np.array(dep_ir_l))
            dep_ir_l = dep_ir_l / _scale
            dep_ir_r = TF.to_tensor(np.array(dep_ir_r))
            dep_ir_r = dep_ir_r / _scale
            gt_dep_ir_l = TF.to_tensor(np.array(gt_dep_ir_l))
            gt_dep_ir_l = gt_dep_ir_l / _scale
            gt_dep_ir_r = TF.to_tensor(np.array(gt_dep_ir_r))
            gt_dep_ir_r = gt_dep_ir_r / _scale

            disp_ir_l = TF.to_tensor(np.array(disp_ir_l))
            disp_ir_l = disp_ir_l * _scale
            disp_ir_r = TF.to_tensor(np.array(disp_ir_r))
            disp_ir_r = disp_ir_r * _scale
            gt_disp_ir_l = TF.to_tensor(np.array(gt_disp_ir_l))
            gt_disp_ir_l = gt_disp_ir_l * _scale
            gt_disp_ir_r = TF.to_tensor(np.array(gt_disp_ir_r))
            gt_disp_ir_r = gt_disp_ir_r * _scale
        elif self.mode in ['train', 'val']:
            # Crop
            w_rgb, h_rgb = rgb_l.size
            w_ir, h_ir = ir_l.size

            assert self.height <= h_rgb and self.width <= w_rgb \
                   and self.height <= h_ir and self.width <= w_ir, \
                "patch size is larger than the input size"

            h_s_rgb = random.randint(0, h_rgb - self.height)
            w_s_rgb = random.randint(0, w_rgb - self.width)

            h_s_ir = random.randint(0, h_ir - self.height)
            w_s_ir = random.randint(0, w_ir - self.width)

            rgb_l = TF.crop(rgb_l, h_s_rgb, w_s_rgb, self.height, self.width)
            rgb_r = TF.crop(rgb_r, h_s_rgb, w_s_rgb, self.height, self.width)

            dep_rgb_l = TF.crop(dep_rgb_l, h_s_rgb, w_s_rgb, self.height, self.width)
            dep_rgb_r = TF.crop(dep_rgb_r, h_s_rgb, w_s_rgb, self.height, self.width)
            disp_rgb_l = TF.crop(disp_rgb_l, h_s_rgb, w_s_rgb, self.height, self.width)
            disp_rgb_r = TF.crop(disp_rgb_r, h_s_rgb, w_s_rgb, self.height, self.width)

            gt_dep_rgb_l = TF.crop(gt_dep_rgb_l, h_s_rgb, w_s_rgb, self.height, self.width)
            gt_dep_rgb_r = TF.crop(gt_dep_rgb_r, h_s_rgb, w_s_rgb, self.height, self.width)
            gt_disp_rgb_l = TF.crop(gt_disp_rgb_l, h_s_rgb, w_s_rgb, self.height, self.width)
            gt_disp_rgb_r = TF.crop(gt_disp_rgb_r, h_s_rgb, w_s_rgb, self.height, self.width)

            K_rgb[2] = K_rgb[2] - w_s_rgb
            K_rgb[3] = K_rgb[3] - h_s_rgb

            ir_l = TF.crop(ir_l, h_s_ir, w_s_ir, self.height, self.width)
            ir_r = TF.crop(ir_r, h_s_ir, w_s_ir, self.height, self.width)

            dep_ir_l = TF.crop(dep_ir_l, h_s_ir, w_s_ir, self.height, self.width)
            dep_ir_r = TF.crop(dep_ir_r, h_s_ir, w_s_ir, self.height, self.width)
            disp_ir_l = TF.crop(disp_ir_l, h_s_ir, w_s_ir, self.height, self.width)
            disp_ir_r = TF.crop(disp_ir_r, h_s_ir, w_s_ir, self.height, self.width)

            gt_dep_ir_l = TF.crop(gt_dep_ir_l, h_s_ir, w_s_ir, self.height, self.width)
            gt_dep_ir_r = TF.crop(gt_dep_ir_r, h_s_ir, w_s_ir, self.height, self.width)
            gt_disp_ir_l = TF.crop(gt_disp_ir_l, h_s_ir, w_s_ir, self.height, self.width)
            gt_disp_ir_r = TF.crop(gt_disp_ir_r, h_s_ir, w_s_ir, self.height, self.width)

            K_ir[2] = K_ir[2] - w_s_ir
            K_ir[3] = K_ir[3] - h_s_ir

            # Convert to tensor
            rgb_l = TF.to_tensor(rgb_l)
            rgb_l = TF.normalize(rgb_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            rgb_r = TF.to_tensor(rgb_r)
            rgb_r = TF.normalize(rgb_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            dep_rgb_l = TF.to_tensor(np.array(dep_rgb_l))
            dep_rgb_r = TF.to_tensor(np.array(dep_rgb_r))
            gt_dep_rgb_l = TF.to_tensor(np.array(gt_dep_rgb_l))
            gt_dep_rgb_r = TF.to_tensor(np.array(gt_dep_rgb_r))

            disp_rgb_l = TF.to_tensor(np.array(disp_rgb_l))
            disp_rgb_r = TF.to_tensor(np.array(disp_rgb_r))
            gt_disp_rgb_l = TF.to_tensor(np.array(gt_disp_rgb_l))
            gt_disp_rgb_r = TF.to_tensor(np.array(gt_disp_rgb_r))

            ir_l = TF.to_tensor(ir_l)
            ir_l = TF.normalize(ir_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            ir_r = TF.to_tensor(ir_r)
            ir_r = TF.normalize(ir_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            dep_ir_l = TF.to_tensor(np.array(dep_ir_l))
            dep_ir_r = TF.to_tensor(np.array(dep_ir_r))
            gt_dep_ir_l = TF.to_tensor(np.array(gt_dep_ir_l))
            gt_dep_ir_r = TF.to_tensor(np.array(gt_dep_ir_r))

            disp_ir_l = TF.to_tensor(np.array(disp_ir_l))
            disp_ir_r = TF.to_tensor(np.array(disp_ir_r))
            gt_disp_ir_l = TF.to_tensor(np.array(gt_disp_ir_l))
            gt_disp_ir_r = TF.to_tensor(np.array(gt_disp_ir_r))
        else:
            # Convert to tensor
            rgb_l = TF.to_tensor(rgb_l)
            rgb_l = TF.normalize(rgb_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            rgb_r = TF.to_tensor(rgb_r)
            rgb_r = TF.normalize(rgb_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            dep_rgb_l = TF.to_tensor(np.array(dep_rgb_l))
            dep_rgb_r = TF.to_tensor(np.array(dep_rgb_r))
            gt_dep_rgb_l = TF.to_tensor(np.array(gt_dep_rgb_l))
            gt_dep_rgb_r = TF.to_tensor(np.array(gt_dep_rgb_r))

            disp_rgb_l = TF.to_tensor(np.array(disp_rgb_l))
            disp_rgb_r = TF.to_tensor(np.array(disp_rgb_r))
            gt_disp_rgb_l = TF.to_tensor(np.array(gt_disp_rgb_l))
            gt_disp_rgb_r = TF.to_tensor(np.array(gt_disp_rgb_r))

            ir_l = TF.to_tensor(ir_l)
            ir_l = TF.normalize(ir_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            ir_r = TF.to_tensor(ir_r)
            ir_r = TF.normalize(ir_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            dep_ir_l = TF.to_tensor(np.array(dep_ir_l))
            dep_ir_r = TF.to_tensor(np.array(dep_ir_r))
            gt_dep_ir_l = TF.to_tensor(np.array(gt_dep_ir_l))
            gt_dep_ir_r = TF.to_tensor(np.array(gt_dep_ir_r))

            disp_ir_l = TF.to_tensor(np.array(disp_ir_l))
            disp_ir_r = TF.to_tensor(np.array(disp_ir_r))
            gt_disp_ir_l = TF.to_tensor(np.array(gt_disp_ir_l))
            gt_disp_ir_r = TF.to_tensor(np.array(gt_disp_ir_r))

        # Calculate depth range
        dep_range_rgb, disp_range_rgb, dep_range_ir, disp_range_ir = \
            calculate_depth_range_new(
                self.num_disp, self.max_depth,
                self.th_depth, self.th_disp,
                K_rgb, B_rgb, K_ir, B_ir, R_ir2rgb, T_ir2rgb
            )

        dep_range_rgb = torch.Tensor(dep_range_rgb)
        disp_range_rgb = torch.Tensor(disp_range_rgb)
        dep_range_ir = torch.Tensor(dep_range_ir)
        disp_range_ir = torch.Tensor(disp_range_ir)

        K_rgb = torch.Tensor(K_rgb)
        B_rgb = torch.Tensor([B_rgb])
        K_ir = torch.Tensor(K_ir)
        B_ir = torch.Tensor([B_ir])

        R_ir2rgb = torch.Tensor(R_ir2rgb)
        T_ir2rgb = torch.Tensor(T_ir2rgb)

        output = {
            'rgb_l': rgb_l, 'rgb_r': rgb_r,
            'ir_l': ir_l, 'ir_r': ir_r,
            'dep_rgb_l': dep_rgb_l, 'dep_rgb_r': dep_rgb_r,
            'dep_ir_l': dep_ir_l, 'dep_ir_r': dep_ir_r,
            'disp_rgb_l': disp_rgb_l, 'disp_rgb_r': disp_rgb_r,
            'disp_ir_l': disp_ir_l, 'disp_ir_r': disp_ir_r,
            'gt_dep_rgb_l': gt_dep_rgb_l, 'gt_dep_rgb_r': gt_dep_rgb_r,
            'gt_dep_ir_l': gt_dep_ir_l, 'gt_dep_ir_r': gt_dep_ir_r,
            'gt_disp_rgb_l': gt_disp_rgb_l, 'gt_disp_rgb_r': gt_disp_rgb_r,
            'gt_disp_ir_l': gt_disp_ir_l, 'gt_disp_ir_r': gt_disp_ir_r,
            'K_rgb': K_rgb, 'B_rgb': B_rgb, 'K_ir': K_ir, 'B_ir': B_ir,
            'R_ir2rgb': R_ir2rgb, 'T_ir2rgb': T_ir2rgb,
            'dep_range_rgb': dep_range_rgb, 'disp_range_rgb': disp_range_rgb,
            'dep_range_ir': dep_range_ir, 'disp_range_ir': disp_range_ir
        }

        return output

    def _load_set(self, path_img_l, path_img_r, path_dep_l, path_dep_r,
                  path_gt_dep_l, path_gt_dep_r, K, B):
        img_l = Image.open(path_img_l)
        img_r = Image.open(path_img_r)

        # 1-channel to 3-channel
        if img_l.mode == 'L':
            img_l = img_l.convert('RGB')
        if img_r.mode == 'L':
            img_r = img_r.convert('RGB')

        # NOTE : only gt_dep_l is filtered
        dep_l = read_depth(path_dep_l)
        dep_r = read_depth(path_dep_r)
        gt_dep_l = read_depth(path_gt_dep_l)
        gt_dep_r = read_depth(path_gt_dep_r)

        # Convert depth to disparity
        disp_l = convert_depth_disp(dep_l, K, B, self.max_disp)
        disp_r = convert_depth_disp(dep_r, K, B, self.max_disp)
        gt_disp_l = convert_depth_disp(gt_dep_l, K, B, self.max_disp)
        gt_disp_r = convert_depth_disp(gt_dep_r, K, B, self.max_disp)

        dep_l = Image.fromarray(dep_l.astype('float32'), mode='F')
        dep_r = Image.fromarray(dep_r.astype('float32'), mode='F')
        disp_l = Image.fromarray(disp_l.astype('float32'), mode='F')
        disp_r = Image.fromarray(disp_r.astype('float32'), mode='F')

        gt_dep_l = Image.fromarray(gt_dep_l.astype('float32'), mode='F')
        gt_dep_r = Image.fromarray(gt_dep_r.astype('float32'), mode='F')
        gt_disp_l = Image.fromarray(gt_disp_l.astype('float32'), mode='F')
        gt_disp_r = Image.fromarray(gt_disp_r.astype('float32'), mode='F')

        assert img_l.size == img_r.size
        assert img_l.size == dep_l.size == gt_dep_l.size
        assert img_r.size == dep_r.size == gt_dep_r.size

        return img_l, img_r, dep_l, dep_r, disp_l, disp_r, \
               gt_dep_l, gt_dep_r, gt_disp_l, gt_disp_r

    def _load_data(self, idx):
        # Calibration
        path_calib = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['calib'])

        calib = np.load(path_calib, allow_pickle=True).item()

        K_ir1 = calib['K_ir1'].astype(np.double)
        T_ir1 = calib['T_ir1'].astype(np.double)
        T_ir2 = calib['T_ir2'].astype(np.double)

        K_rgb1 = calib['K_rgb1'].astype(np.double)
        T_rgb1 = calib['T_rgb1'].astype(np.double)
        T_rgb2 = calib['T_rgb2'].astype(np.double)

        # NOTE : Point in IR coordinates to Point in RGB coordinates
        R_ir2rgb = calib['R_ir2rgb'].astype(np.double)
        # mm to m
        T_ir2rgb = calib['T_ir2rgb'].astype(np.double) / 1000.0
        T_ir2rgb = np.squeeze(T_ir2rgb, axis=1)

        # Baseline (mm to m)
        B_ir = np.linalg.norm(T_ir1 - T_ir2) / 1000.0
        B_rgb = np.linalg.norm(T_rgb1 - T_rgb2) / 1000.0

        # Intrinsics of the reference (left) cameras
        K_ir = [K_ir1[0, 0], K_ir1[1, 1], K_ir1[0, 2], K_ir1[1, 2]]
        K_rgb = [K_rgb1[0, 0], K_rgb1[1, 1], K_rgb1[0, 2], K_rgb1[1, 2]]

        # RGB
        path_rgb_l = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['rgb1'])
        path_rgb_r = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['rgb2'])
        path_dep_l_rgb = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['dep_rgb1'])
        path_dep_r_rgb = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['dep_rgb2'])
        path_gt_dep_l_rgb = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['gt_dep_rgb1'])
        path_gt_dep_r_rgb = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['gt_dep_rgb2'])

        rgb_l, rgb_r, dep_rgb_l, dep_rgb_r, disp_rgb_l, disp_rgb_r, \
               gt_dep_rgb_l, gt_dep_rgb_r, gt_disp_rgb_l, gt_disp_rgb_r = \
            self._load_set(path_rgb_l, path_rgb_r, path_dep_l_rgb, path_dep_r_rgb,
                           path_gt_dep_l_rgb, path_gt_dep_r_rgb,
                           K_rgb, B_rgb)

        # IR
        path_ir_l = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['ir1'])
        path_ir_r = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['ir2'])
        path_dep_l_ir = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['dep_ir1'])
        path_dep_r_ir = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['dep_ir2'])
        path_gt_dep_l_ir = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['gt_dep_ir1'])
        path_gt_dep_r_ir = os.path.join(
            self.args.path_mmdce, self.mode, self.sample_list[idx]['gt_dep_ir2'])

        ir_l, ir_r, dep_ir_l, dep_ir_r, disp_ir_l, disp_ir_r, \
               gt_dep_ir_l, gt_dep_ir_r, gt_disp_ir_l, gt_disp_ir_r = \
            self._load_set(path_ir_l, path_ir_r, path_dep_l_ir, path_dep_r_ir,
                           path_gt_dep_l_ir, path_gt_dep_r_ir,
                           K_ir, B_ir)

        return rgb_l, rgb_r, dep_rgb_l, dep_rgb_r, \
               disp_rgb_l, disp_rgb_r, \
               gt_dep_rgb_l, gt_dep_rgb_r, \
               gt_disp_rgb_l, gt_disp_rgb_r, K_rgb, B_rgb, \
               ir_l, ir_r, dep_ir_l, dep_ir_r, \
               disp_ir_l, disp_ir_r, \
               gt_dep_ir_l, gt_dep_ir_r, \
               gt_disp_ir_l, gt_disp_ir_r, K_ir, B_ir, \
               R_ir2rgb, T_ir2rgb
