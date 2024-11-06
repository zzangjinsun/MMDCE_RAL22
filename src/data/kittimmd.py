"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    KITTI Multi-Modal Depth (KITTI MMD) dataset implementation
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
KITTI Multi-Modal Depth (KITTI MMD) dataset json file has the following format:

{
    "train": [
        {
            "img0": "train/2011_10_03_drive_0034_sync/image_00/data/0000001243.png",
            "img1": "train/2011_10_03_drive_0034_sync/image_01/data/0000001243.png",
            "img2": "train/2011_10_03_drive_0034_sync/image_02/data/0000001243.png",
            "img3": "train/2011_10_03_drive_0034_sync/image_03/data/0000001243.png",
            "dep0": "train/2011_10_03_drive_0034_sync/proj_depth/velodyne_raw/image_00/0000001243.png",
            "dep1": "train/2011_10_03_drive_0034_sync/proj_depth/velodyne_raw/image_01/0000001243.png",
            "dep2": "train/2011_10_03_drive_0034_sync/proj_depth/velodyne_raw/image_02/0000001243.png",
            "dep3": "train/2011_10_03_drive_0034_sync/proj_depth/velodyne_raw/image_03/0000001243.png",
            "gt0": "train/2011_10_03_drive_0034_sync/proj_depth/groundtruth/image_00/0000001243.png",
            "gt1": "train/2011_10_03_drive_0034_sync/proj_depth/groundtruth/image_01/0000001243.png",
            "gt2": "train/2011_10_03_drive_0034_sync/proj_depth/groundtruth/image_02/0000001243.png",
            "gt3": "train/2011_10_03_drive_0034_sync/proj_depth/groundtruth/image_03/0000001243.png",
            "K": "train/2011_10_03_drive_0034_sync/calib_cam_to_cam.txt"
        }, ...
    ],      
    "val": [
        {
            "img0": "val/2011_09_26_drive_0002_sync/image_00/data/0000000005.png",
            "img1": "val/2011_09_26_drive_0002_sync/image_01/data/0000000005.png",
            "img2": "val/2011_09_26_drive_0002_sync/image_02/data/0000000005.png",
            "img3": "val/2011_09_26_drive_0002_sync/image_03/data/0000000005.png",
            "dep0": "val/2011_09_26_drive_0002_sync/proj_depth/velodyne_raw/image_00/0000000005.png",
            "dep1": "val/2011_09_26_drive_0002_sync/proj_depth/velodyne_raw/image_01/0000000005.png",
            "dep2": "val/2011_09_26_drive_0002_sync/proj_depth/velodyne_raw/image_02/0000000005.png",
            "dep3": "val/2011_09_26_drive_0002_sync/proj_depth/velodyne_raw/image_03/0000000005.png",
            "gt0": "val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_00/0000000005.png",
            "gt1": "val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_01/0000000005.png",
            "gt2": "val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000005.png",
            "gt3": "val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_03/0000000005.png",
            "K": "val/2011_09_26_drive_0002_sync/calib_cam_to_cam.txt"
        }, ...
    ],
    "test": [
        {
            "img0": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/image_00/data/0000000005.png",
            "img1": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/image_01/data/0000000005.png",
            "img2": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/image_02/data/0000000005.png",
            "img3": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/image_03/data/0000000005.png",
            "dep0": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/proj_depth/velodyne_raw/image_00/0000000005.png",
            "dep1": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/proj_depth/velodyne_raw/image_01/0000000005.png",
            "dep2": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/proj_depth/velodyne_raw/image_02/0000000005.png",
            "dep3": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/proj_depth/velodyne_raw/image_03/0000000005.png",
            "gt0": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_00/0000000005.png",
            "gt1": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_01/0000000005.png",
            "gt2": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000005.png",
            "gt3": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_03/0000000005.png",
            "K": "depth_selection/val_multi_modal/2011_09_26_drive_0002_sync/calib_cam_to_cam.txt"
        }, ...
    ]
}
"""


class KITTIMMD(BaseDataset):
    def __init__(self, args, mode):
        super(KITTIMMD, self).__init__(args, mode)

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
        gray_l, gray_r, rgb_l, rgb_r,\
        dep_gray_l, dep_gray_r, dep_rgb_l, dep_rgb_r, \
        disp_gray_l, disp_gray_r, disp_rgb_l, disp_rgb_r,\
        gt_dep_gray_l, gt_dep_gray_r, gt_dep_rgb_l, gt_dep_rgb_r,\
        gt_disp_gray_l, gt_disp_gray_r, gt_disp_rgb_l, gt_disp_rgb_r, \
        K, B_gray, B_rgb, R_gray2rgb, T_gray2rgb = self._load_data(idx)

        K_gray = K.copy()
        K_rgb = K.copy()

        if self.augment and self.mode == 'train':
            # Top crop if needed
            if self.args.top_crop > 0:
                w_rgb, h_rgb = rgb_l.size
                rgb_l = TF.crop(rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                rgb_r = TF.crop(rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                dep_rgb_l = TF.crop(dep_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                dep_rgb_r = TF.crop(dep_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                disp_rgb_l = TF.crop(disp_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                disp_rgb_r = TF.crop(disp_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_dep_rgb_l = TF.crop(gt_dep_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_dep_rgb_r = TF.crop(gt_dep_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_disp_rgb_l = TF.crop(gt_disp_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_disp_rgb_r = TF.crop(gt_disp_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                K_rgb[3] = K_rgb[3] - self.args.top_crop

                w_gray, h_gray = gray_l.size
                gray_l = TF.crop(gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gray_r = TF.crop(gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                dep_gray_l = TF.crop(dep_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                dep_gray_r = TF.crop(dep_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                disp_gray_l = TF.crop(disp_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                disp_gray_r = TF.crop(disp_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_dep_gray_l = TF.crop(gt_dep_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_dep_gray_r = TF.crop(gt_dep_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_disp_gray_l = TF.crop(gt_disp_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_disp_gray_r = TF.crop(gt_disp_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                K_gray[3] = K_gray[3] - self.args.top_crop

            w_rgb, h_rgb = rgb_l.size
            w_gray, h_gray = gray_l.size

            _scale = np.random.uniform(1.0, 1.5)
            scale_rgb = np.int(h_rgb * _scale)
            scale_gray = np.int(h_gray * _scale)

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

            gray_l = TF.adjust_brightness(gray_l, brightness)
            gray_l = TF.adjust_contrast(gray_l, contrast)
            gray_l = TF.adjust_saturation(gray_l, saturation)

            gray_r = TF.adjust_brightness(gray_r, brightness)
            gray_r = TF.adjust_contrast(gray_r, contrast)
            gray_r = TF.adjust_saturation(gray_r, saturation)

            # Resize
            rgb_l = TF.resize(rgb_l, scale_rgb, Image.BICUBIC)
            rgb_r = TF.resize(rgb_r, scale_rgb, Image.BICUBIC)
            gray_l = TF.resize(gray_l, scale_gray, Image.BICUBIC)
            gray_r = TF.resize(gray_r, scale_gray, Image.BICUBIC)

            dep_rgb_l = TF.resize(dep_rgb_l, scale_rgb, Image.NEAREST)
            dep_rgb_r = TF.resize(dep_rgb_r, scale_rgb, Image.NEAREST)
            dep_gray_l = TF.resize(dep_gray_l, scale_gray, Image.NEAREST)
            dep_gray_r = TF.resize(dep_gray_r, scale_gray, Image.NEAREST)

            disp_rgb_l = TF.resize(disp_rgb_l, scale_rgb, Image.NEAREST)
            disp_rgb_r = TF.resize(disp_rgb_r, scale_rgb, Image.NEAREST)
            disp_gray_l = TF.resize(disp_gray_l, scale_gray, Image.NEAREST)
            disp_gray_r = TF.resize(disp_gray_r, scale_gray, Image.NEAREST)

            gt_dep_rgb_l = TF.resize(gt_dep_rgb_l, scale_rgb, Image.NEAREST)
            gt_dep_rgb_r = TF.resize(gt_dep_rgb_r, scale_rgb, Image.NEAREST)
            gt_dep_gray_l = TF.resize(gt_dep_gray_l, scale_gray, Image.NEAREST)
            gt_dep_gray_r = TF.resize(gt_dep_gray_r, scale_gray, Image.NEAREST)

            gt_disp_rgb_l = TF.resize(gt_disp_rgb_l, scale_rgb, Image.NEAREST)
            gt_disp_rgb_r = TF.resize(gt_disp_rgb_r, scale_rgb, Image.NEAREST)
            gt_disp_gray_l = TF.resize(gt_disp_gray_l, scale_gray, Image.NEAREST)
            gt_disp_gray_r = TF.resize(gt_disp_gray_r, scale_gray, Image.NEAREST)

            K_rgb[0] = K_rgb[0] * _scale
            K_rgb[1] = K_rgb[1] * _scale
            K_rgb[2] = K_rgb[2] * _scale
            K_rgb[3] = K_rgb[3] * _scale

            K_gray[0] = K_gray[0] * _scale
            K_gray[1] = K_gray[1] * _scale
            K_gray[2] = K_gray[2] * _scale
            K_gray[3] = K_gray[3] * _scale

            # Crop
            assert self.height <= h_rgb and self.width <= w_rgb \
                   and self.height <= h_gray and self.width <= w_gray, \
                   "patch size is larger than the input size"

            h_start = random.randint(0, min([h_rgb, h_gray]) - self.height)
            w_start = random.randint(0, min([w_rgb, w_gray]) - self.width)

            rgb_l = TF.crop(rgb_l, h_start, w_start, self.height, self.width)
            rgb_r = TF.crop(rgb_r, h_start, w_start, self.height, self.width)
            gray_l = TF.crop(gray_l, h_start, w_start, self.height, self.width)
            gray_r = TF.crop(gray_r, h_start, w_start, self.height, self.width)

            dep_rgb_l = TF.crop(dep_rgb_l, h_start, w_start, self.height, self.width)
            dep_rgb_r = TF.crop(dep_rgb_r, h_start, w_start, self.height, self.width)
            dep_gray_l = TF.crop(dep_gray_l, h_start, w_start, self.height, self.width)
            dep_gray_r = TF.crop(dep_gray_r, h_start, w_start, self.height, self.width)

            disp_rgb_l = TF.crop(disp_rgb_l, h_start, w_start, self.height, self.width)
            disp_rgb_r = TF.crop(disp_rgb_r, h_start, w_start, self.height, self.width)
            disp_gray_l = TF.crop(disp_gray_l, h_start, w_start, self.height, self.width)
            disp_gray_r = TF.crop(disp_gray_r, h_start, w_start, self.height, self.width)

            gt_dep_rgb_l = TF.crop(gt_dep_rgb_l, h_start, w_start, self.height, self.width)
            gt_dep_rgb_r = TF.crop(gt_dep_rgb_r, h_start, w_start, self.height, self.width)
            gt_dep_gray_l = TF.crop(gt_dep_gray_l, h_start, w_start, self.height, self.width)
            gt_dep_gray_r = TF.crop(gt_dep_gray_r, h_start, w_start, self.height, self.width)

            gt_disp_rgb_l = TF.crop(gt_disp_rgb_l, h_start, w_start, self.height, self.width)
            gt_disp_rgb_r = TF.crop(gt_disp_rgb_r, h_start, w_start, self.height, self.width)
            gt_disp_gray_l = TF.crop(gt_disp_gray_l, h_start, w_start, self.height, self.width)
            gt_disp_gray_r = TF.crop(gt_disp_gray_r, h_start, w_start, self.height, self.width)

            K_rgb[2] = K_rgb[2] - w_start
            K_rgb[3] = K_rgb[3] - h_start

            K_gray[2] = K_gray[2] - w_start
            K_gray[3] = K_gray[3] - h_start

            rgb_l = TF.to_tensor(rgb_l)
            rgb_l = TF.normalize(rgb_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            rgb_r = TF.to_tensor(rgb_r)
            rgb_r = TF.normalize(rgb_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            gray_l = TF.to_tensor(gray_l)
            gray_l = TF.normalize(gray_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            gray_r = TF.to_tensor(gray_r)
            gray_r = TF.normalize(gray_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            dep_rgb_l = TF.to_tensor(np.array(dep_rgb_l))
            dep_rgb_l = dep_rgb_l / _scale
            dep_rgb_r = TF.to_tensor(np.array(dep_rgb_r))
            dep_rgb_r = dep_rgb_r / _scale

            dep_gray_l = TF.to_tensor(np.array(dep_gray_l))
            dep_gray_l = dep_gray_l / _scale
            dep_gray_r = TF.to_tensor(np.array(dep_gray_r))
            dep_gray_r = dep_gray_r / _scale

            disp_rgb_l = TF.to_tensor(np.array(disp_rgb_l))
            disp_rgb_l = disp_rgb_l * _scale
            disp_rgb_r = TF.to_tensor(np.array(disp_rgb_r))
            disp_rgb_r = disp_rgb_r * _scale

            disp_gray_l = TF.to_tensor(np.array(disp_gray_l))
            disp_gray_l = disp_gray_l * _scale
            disp_gray_r = TF.to_tensor(np.array(disp_gray_r))
            disp_gray_r = disp_gray_r * _scale

            gt_dep_rgb_l = TF.to_tensor(np.array(gt_dep_rgb_l))
            gt_dep_rgb_l = gt_dep_rgb_l / _scale
            gt_dep_rgb_r = TF.to_tensor(np.array(gt_dep_rgb_r))
            gt_dep_rgb_r = gt_dep_rgb_r / _scale

            gt_dep_gray_l = TF.to_tensor(np.array(gt_dep_gray_l))
            gt_dep_gray_l = gt_dep_gray_l / _scale
            gt_dep_gray_r = TF.to_tensor(np.array(gt_dep_gray_r))
            gt_dep_gray_r = gt_dep_gray_r / _scale

            gt_disp_rgb_l = TF.to_tensor(np.array(gt_disp_rgb_l))
            gt_disp_rgb_l = gt_disp_rgb_l * _scale
            gt_disp_rgb_r = TF.to_tensor(np.array(gt_disp_rgb_r))
            gt_disp_rgb_r = gt_disp_rgb_r * _scale

            gt_disp_gray_l = TF.to_tensor(np.array(gt_disp_gray_l))
            gt_disp_gray_l = gt_disp_gray_l * _scale
            gt_disp_gray_r = TF.to_tensor(np.array(gt_disp_gray_r))
            gt_disp_gray_r = gt_disp_gray_r * _scale
        elif self.mode in ['train', 'val']:
            # Top crop if needed
            if self.args.top_crop > 0:
                w_rgb, h_rgb = rgb_l.size
                rgb_l = TF.crop(rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                rgb_r = TF.crop(rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                dep_rgb_l = TF.crop(dep_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                dep_rgb_r = TF.crop(dep_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                disp_rgb_l = TF.crop(disp_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                disp_rgb_r = TF.crop(disp_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_dep_rgb_l = TF.crop(gt_dep_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_dep_rgb_r = TF.crop(gt_dep_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_disp_rgb_l = TF.crop(gt_disp_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_disp_rgb_r = TF.crop(gt_disp_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                K_rgb[3] = K_rgb[3] - self.args.top_crop

                w_gray, h_gray = gray_l.size
                gray_l = TF.crop(gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gray_r = TF.crop(gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                dep_gray_l = TF.crop(dep_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                dep_gray_r = TF.crop(dep_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                disp_gray_l = TF.crop(disp_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                disp_gray_r = TF.crop(disp_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_dep_gray_l = TF.crop(gt_dep_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_dep_gray_r = TF.crop(gt_dep_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_disp_gray_l = TF.crop(gt_disp_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_disp_gray_r = TF.crop(gt_disp_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                K_gray[3] = K_gray[3] - self.args.top_crop

            w_rgb, h_rgb = rgb_l.size
            w_gray, h_gray = gray_l.size

            # Crop
            assert self.height <= h_rgb and self.width <= w_rgb \
                   and self.height <= h_gray and self.width <= w_gray, \
                   "patch size is larger than the input size"

            h_start = random.randint(0, min([h_rgb, h_gray]) - self.height)
            w_start = random.randint(0, min([w_rgb, w_gray]) - self.width)

            rgb_l = TF.crop(rgb_l, h_start, w_start, self.height, self.width)
            rgb_r = TF.crop(rgb_r, h_start, w_start, self.height, self.width)
            gray_l = TF.crop(gray_l, h_start, w_start, self.height, self.width)
            gray_r = TF.crop(gray_r, h_start, w_start, self.height, self.width)

            dep_rgb_l = TF.crop(dep_rgb_l, h_start, w_start, self.height, self.width)
            dep_rgb_r = TF.crop(dep_rgb_r, h_start, w_start, self.height, self.width)
            dep_gray_l = TF.crop(dep_gray_l, h_start, w_start, self.height, self.width)
            dep_gray_r = TF.crop(dep_gray_r, h_start, w_start, self.height, self.width)

            disp_rgb_l = TF.crop(disp_rgb_l, h_start, w_start, self.height, self.width)
            disp_rgb_r = TF.crop(disp_rgb_r, h_start, w_start, self.height, self.width)
            disp_gray_l = TF.crop(disp_gray_l, h_start, w_start, self.height, self.width)
            disp_gray_r = TF.crop(disp_gray_r, h_start, w_start, self.height, self.width)

            gt_dep_rgb_l = TF.crop(gt_dep_rgb_l, h_start, w_start, self.height, self.width)
            gt_dep_rgb_r = TF.crop(gt_dep_rgb_r, h_start, w_start, self.height, self.width)
            gt_dep_gray_l = TF.crop(gt_dep_gray_l, h_start, w_start, self.height, self.width)
            gt_dep_gray_r = TF.crop(gt_dep_gray_r, h_start, w_start, self.height, self.width)

            gt_disp_rgb_l = TF.crop(gt_disp_rgb_l, h_start, w_start, self.height, self.width)
            gt_disp_rgb_r = TF.crop(gt_disp_rgb_r, h_start, w_start, self.height, self.width)
            gt_disp_gray_l = TF.crop(gt_disp_gray_l, h_start, w_start, self.height, self.width)
            gt_disp_gray_r = TF.crop(gt_disp_gray_r, h_start, w_start, self.height, self.width)

            K_rgb[2] = K_rgb[2] - w_start
            K_rgb[3] = K_rgb[3] - h_start

            K_gray[2] = K_gray[2] - w_start
            K_gray[3] = K_gray[3] - h_start

            rgb_l = TF.to_tensor(rgb_l)
            rgb_l = TF.normalize(rgb_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            rgb_r = TF.to_tensor(rgb_r)
            rgb_r = TF.normalize(rgb_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            gray_l = TF.to_tensor(gray_l)
            gray_l = TF.normalize(gray_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            gray_r = TF.to_tensor(gray_r)
            gray_r = TF.normalize(gray_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            dep_rgb_l = TF.to_tensor(np.array(dep_rgb_l))
            dep_rgb_r = TF.to_tensor(np.array(dep_rgb_r))

            dep_gray_l = TF.to_tensor(np.array(dep_gray_l))
            dep_gray_r = TF.to_tensor(np.array(dep_gray_r))

            disp_rgb_l = TF.to_tensor(np.array(disp_rgb_l))
            disp_rgb_r = TF.to_tensor(np.array(disp_rgb_r))

            disp_gray_l = TF.to_tensor(np.array(disp_gray_l))
            disp_gray_r = TF.to_tensor(np.array(disp_gray_r))

            gt_dep_rgb_l = TF.to_tensor(np.array(gt_dep_rgb_l))
            gt_dep_rgb_r = TF.to_tensor(np.array(gt_dep_rgb_r))

            gt_dep_gray_l = TF.to_tensor(np.array(gt_dep_gray_l))
            gt_dep_gray_r = TF.to_tensor(np.array(gt_dep_gray_r))

            gt_disp_rgb_l = TF.to_tensor(np.array(gt_disp_rgb_l))
            gt_disp_rgb_r = TF.to_tensor(np.array(gt_disp_rgb_r))

            gt_disp_gray_l = TF.to_tensor(np.array(gt_disp_gray_l))
            gt_disp_gray_r = TF.to_tensor(np.array(gt_disp_gray_r))
        else:
            # Top crop if needed
            if self.args.top_crop > 0 and self.args.test_crop:
                w_rgb, h_rgb = rgb_l.size
                rgb_l = TF.crop(rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                rgb_r = TF.crop(rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                dep_rgb_l = TF.crop(dep_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                dep_rgb_r = TF.crop(dep_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                disp_rgb_l = TF.crop(disp_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                disp_rgb_r = TF.crop(disp_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_dep_rgb_l = TF.crop(gt_dep_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_dep_rgb_r = TF.crop(gt_dep_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_disp_rgb_l = TF.crop(gt_disp_rgb_l, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                gt_disp_rgb_r = TF.crop(gt_disp_rgb_r, self.args.top_crop, 0, h_rgb - self.args.top_crop, w_rgb)
                K_rgb[3] = K_rgb[3] - self.args.top_crop

                w_gray, h_gray = gray_l.size
                gray_l = TF.crop(gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gray_r = TF.crop(gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                dep_gray_l = TF.crop(dep_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                dep_gray_r = TF.crop(dep_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                disp_gray_l = TF.crop(disp_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                disp_gray_r = TF.crop(disp_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_dep_gray_l = TF.crop(gt_dep_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_dep_gray_r = TF.crop(gt_dep_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_disp_gray_l = TF.crop(gt_disp_gray_l, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                gt_disp_gray_r = TF.crop(gt_disp_gray_r, self.args.top_crop, 0, h_gray - self.args.top_crop, w_gray)
                K_gray[3] = K_gray[3] - self.args.top_crop

            rgb_l = TF.to_tensor(rgb_l)
            rgb_l = TF.normalize(rgb_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            rgb_r = TF.to_tensor(rgb_r)
            rgb_r = TF.normalize(rgb_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            gray_l = TF.to_tensor(gray_l)
            gray_l = TF.normalize(gray_l, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            gray_r = TF.to_tensor(gray_r)
            gray_r = TF.normalize(gray_r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            dep_rgb_l = TF.to_tensor(np.array(dep_rgb_l))
            dep_rgb_r = TF.to_tensor(np.array(dep_rgb_r))

            dep_gray_l = TF.to_tensor(np.array(dep_gray_l))
            dep_gray_r = TF.to_tensor(np.array(dep_gray_r))

            disp_rgb_l = TF.to_tensor(np.array(disp_rgb_l))
            disp_rgb_r = TF.to_tensor(np.array(disp_rgb_r))

            disp_gray_l = TF.to_tensor(np.array(disp_gray_l))
            disp_gray_r = TF.to_tensor(np.array(disp_gray_r))

            gt_dep_rgb_l = TF.to_tensor(np.array(gt_dep_rgb_l))
            gt_dep_rgb_r = TF.to_tensor(np.array(gt_dep_rgb_r))

            gt_dep_gray_l = TF.to_tensor(np.array(gt_dep_gray_l))
            gt_dep_gray_r = TF.to_tensor(np.array(gt_dep_gray_r))

            gt_disp_rgb_l = TF.to_tensor(np.array(gt_disp_rgb_l))
            gt_disp_rgb_r = TF.to_tensor(np.array(gt_disp_rgb_r))

            gt_disp_gray_l = TF.to_tensor(np.array(gt_disp_gray_l))
            gt_disp_gray_r = TF.to_tensor(np.array(gt_disp_gray_r))

        # Calculate depth range
        dep_range_rgb, disp_range_rgb, dep_range_gray, disp_range_gray = \
            calculate_depth_range_new(
                self.num_disp, self.max_depth,
                self.th_depth, self.th_disp,
                K_rgb, B_rgb, K_gray, B_gray, R_gray2rgb, T_gray2rgb
            )

        dep_range_rgb = torch.Tensor(dep_range_rgb)
        disp_range_rgb = torch.Tensor(disp_range_rgb)
        dep_range_gray = torch.Tensor(dep_range_gray)
        disp_range_gray = torch.Tensor(disp_range_gray)

        K_rgb = torch.Tensor(K_rgb)
        B_rgb = torch.Tensor([B_rgb])
        K_gray = torch.Tensor(K_gray)
        B_gray = torch.Tensor([B_gray])

        R_gray2rgb = torch.Tensor(R_gray2rgb)
        T_gray2rgb = torch.Tensor(T_gray2rgb)

        output = {
            'rgb_l': rgb_l, 'rgb_r': rgb_r,
            'ir_l': gray_l, 'ir_r': gray_r,
            'dep_rgb_l': dep_rgb_l, 'dep_rgb_r': dep_rgb_r,
            'dep_ir_l': dep_gray_l, 'dep_ir_r': dep_gray_r,
            'disp_rgb_l': disp_rgb_l, 'disp_rgb_r': disp_rgb_r,
            'disp_ir_l': disp_gray_l, 'disp_ir_r': disp_gray_r,
            'gt_dep_rgb_l': gt_dep_rgb_l, 'gt_dep_rgb_r': gt_dep_rgb_r,
            'gt_dep_ir_l': gt_dep_gray_l, 'gt_dep_ir_r': gt_dep_gray_r,
            'gt_disp_rgb_l': gt_disp_rgb_l, 'gt_disp_rgb_r': gt_disp_rgb_r,
            'gt_disp_ir_l': gt_disp_gray_l, 'gt_disp_ir_r': gt_disp_gray_r,
            'K_rgb': K_rgb, 'B_rgb': B_rgb, 'K_ir': K_gray, 'B_ir': B_gray,
            'R_ir2rgb': R_gray2rgb, 'T_ir2rgb': T_gray2rgb,
            'dep_range_rgb': dep_range_rgb, 'disp_range_rgb': disp_range_rgb,
            'dep_range_ir': dep_range_gray, 'disp_range_ir': disp_range_gray
        }

        return output

    def _load_data(self, idx):
        # Cam 0 and 1 : Grayscale stereo pair
        # Cam 2 and 3 : RGB stereo pair
        path_img0 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['img0'])
        path_img1 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['img1'])
        path_img2 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['img2'])
        path_img3 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['img3'])

        path_dep0 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['dep0'])
        path_dep1 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['dep1'])
        path_dep2 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['dep2'])
        path_dep3 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['dep3'])

        path_gt0 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['gt0'])
        path_gt1 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['gt1'])
        path_gt2 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['gt2'])
        path_gt3 = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['gt3'])

        path_calib = os.path.join(self.args.path_kittimmd, self.sample_list[idx]['K'])

        dep0 = read_depth(path_dep0)
        dep1 = read_depth(path_dep1)
        dep2 = read_depth(path_dep2)
        dep3 = read_depth(path_dep3)

        gt0 = read_depth(path_gt0)
        gt1 = read_depth(path_gt1)
        gt2 = read_depth(path_gt2)
        gt3 = read_depth(path_gt3)

        calib = read_calib_file(path_calib)

        # Reference :
        # https://github.com/utiasSTARS/pykitti/blob/master/pykitti/raw.py
        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(calib['P_rect_00'], (3, 4))
        P_rect_10 = np.reshape(calib['P_rect_01'], (3, 4))
        P_rect_20 = np.reshape(calib['P_rect_02'], (3, 4))
        P_rect_30 = np.reshape(calib['P_rect_03'], (3, 4))

        # Create 4x4 matrices from the rectifying rotation matrices
        R_rect_00 = np.eye(4)
        R_rect_00[0:3, 0:3] = np.reshape(calib['R_rect_00'], (3, 3))
        R_rect_10 = np.eye(4)
        R_rect_10[0:3, 0:3] = np.reshape(calib['R_rect_01'], (3, 3))
        R_rect_20 = np.eye(4)
        R_rect_20[0:3, 0:3] = np.reshape(calib['R_rect_02'], (3, 3))
        R_rect_30 = np.eye(4)
        R_rect_30[0:3, 0:3] = np.reshape(calib['R_rect_03'], (3, 3))

        # Compute the rectified extrinsics from cam0 to camN
        T0 = np.eye(4)
        T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        T_cam2_cam0 = T2 @ np.linalg.inv(T0)
        T_cam1_cam0 = T1 @ np.linalg.inv(T0)
        T_cam3_cam2 = T3 @ np.linalg.inv(T2)

        # Compute the camera intrinsics
        K_cam0 = P_rect_00[0:3, 0:3]

        B_01 = np.linalg.norm(T_cam1_cam0[:3, 3])  # gray baseline
        B_23 = np.linalg.norm(T_cam3_cam2[:3, 3])   # rgb baseline

        R_02 = T_cam2_cam0[:3, :3]
        T_02 = T_cam2_cam0[:3, 3]

        # Note : After rectification, intrinsics of cameras are the same
        K = [K_cam0[0, 0], K_cam0[1, 1], K_cam0[0, 2], K_cam0[1, 2]]

        # Convert depth to disparity
        disp0 = convert_depth_disp(dep0, K, B_01, self.max_disp)
        disp1 = convert_depth_disp(dep1, K, B_01, self.max_disp)

        disp2 = convert_depth_disp(dep2, K, B_23, self.max_disp)
        disp3 = convert_depth_disp(dep3, K, B_23, self.max_disp)

        gt_disp0 = convert_depth_disp(gt0, K, B_01, self.max_disp)
        gt_disp1 = convert_depth_disp(gt1, K, B_01, self.max_disp)

        gt_disp2 = convert_depth_disp(gt2, K, B_23, self.max_disp)
        gt_disp3 = convert_depth_disp(gt3, K, B_23, self.max_disp)

        img0 = Image.open(path_img0)
        img1 = Image.open(path_img1)
        img2 = Image.open(path_img2)
        img3 = Image.open(path_img3)

        # 1-channel to 3-channel
        if img0.mode == 'L':
            img0 = img0.convert('RGB')
        if img1.mode == 'L':
            img1 = img1.convert('RGB')

        dep0 = Image.fromarray(dep0.astype('float32'), mode='F')
        dep1 = Image.fromarray(dep1.astype('float32'), mode='F')
        dep2 = Image.fromarray(dep2.astype('float32'), mode='F')
        dep3 = Image.fromarray(dep3.astype('float32'), mode='F')

        disp0 = Image.fromarray(disp0.astype('float32'), mode='F')
        disp1 = Image.fromarray(disp1.astype('float32'), mode='F')
        disp2 = Image.fromarray(disp2.astype('float32'), mode='F')
        disp3 = Image.fromarray(disp3.astype('float32'), mode='F')

        gt0 = Image.fromarray(gt0.astype('float32'), mode='F')
        gt1 = Image.fromarray(gt1.astype('float32'), mode='F')
        gt2 = Image.fromarray(gt2.astype('float32'), mode='F')
        gt3 = Image.fromarray(gt3.astype('float32'), mode='F')

        gt_disp0 = Image.fromarray(gt_disp0.astype('float32'), mode='F')
        gt_disp1 = Image.fromarray(gt_disp1.astype('float32'), mode='F')
        gt_disp2 = Image.fromarray(gt_disp2.astype('float32'), mode='F')
        gt_disp3 = Image.fromarray(gt_disp3.astype('float32'), mode='F')

        assert img0.size == img1.size == img2.size == img3.size
        assert img0.size == dep0.size == gt0.size
        assert img1.size == dep1.size == gt1.size
        assert img2.size == dep2.size == gt2.size
        assert img3.size == dep3.size == gt3.size

        return img0, img1, img2, img3, \
               dep0, dep1, dep2, dep3, \
               disp0, disp1, disp2, disp3, \
               gt0, gt1, gt2, gt3, \
               gt_disp0, gt_disp1, gt_disp2, gt_disp3, \
               K, B_01, B_23, R_02, T_02
