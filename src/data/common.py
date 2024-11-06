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
import numpy as np
from PIL import Image


def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth


# Reference : https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def convert_depth_disp(val, K, B, max_val):
    mask_invalid = val == 0

    val_out = K[0] * B / (val + 1e-8)
    val_out[mask_invalid] = 0.0
    val_out[val_out > max_val] = max_val

    return val_out


def calculate_depth_range(num_disp, max_depth, cost_factor, interval_depth,
                          K_rgb, B_rgb, K_gray, B_gray, R_gray2rgb, T_gray2rgb):
    # K = [fx, fy, cx, cy]
    assert len(K_rgb) == 4 and len(K_gray) == 4
    assert interval_depth >= 0

    # At cost level scale, this interval becomes 0.5 disparity
    interval = 0.5 * cost_factor

    # Get depth and disparity range
    min_disp = 0
    max_disp = num_disp - 1

    # RGB is the reference
    min_depth = K_rgb[0] * B_rgb / max_disp

    disp_range = np.arange(min_disp, max_disp, interval)

    # RGB is the reference
    dep_range = K_rgb[0] * B_rgb / (disp_range + 1e-8)
    dep_range[dep_range > max_depth] = max_depth
    dep_range[dep_range < min_depth] = min_depth

    dep_range_valid = [max_depth]

    for k, v in enumerate(dep_range):
        if dep_range_valid[-1] - v < interval_depth:
            continue
        dep_range_valid.append(v)

    if dep_range_valid[-1] > min_depth:
        dep_range_valid.append(min_depth)

    # Interpolate from valid samples
    num_disp_cost = int(np.ceil(len(dep_range_valid) / 4.0) * 4.0)

    x_sample = np.linspace(0, 1, len(dep_range_valid))
    x_interp = np.linspace(0, 1, num_disp_cost)

    dep_range_rgb = np.interp(x_interp, x_sample, dep_range_valid)

    disp_range_rgb = K_rgb[0] * B_rgb / (dep_range_rgb + 1e-8)
    # disp_range_rgb[disp_range_rgb < min_disp] = min_disp
    # disp_range_rgb[disp_range_rgb > max_disp] = max_disp

    # Rotation angle
    rot_angle_cos = (np.trace(R_gray2rgb) - 1.0) / 2.0
    err_ratio = (1 - rot_angle_cos)

    # Depth error due to rotation should be negligible
    assert err_ratio < 1e-3, 'Depth error due to rotation is not negligible'

    depth_diff_rgb2gray = -T_gray2rgb[2]

    # Compensate for depth difference
    dep_range_gray = dep_range_rgb + depth_diff_rgb2gray

    disp_range_gray = K_gray[0] * B_gray / (dep_range_gray + 1e-8)

    return dep_range_rgb, disp_range_rgb, dep_range_gray, disp_range_gray


def calculate_depth_range_new(num_disp, max_depth, th_depth, th_disp,
                          K_rgb, B_rgb, K_gray, B_gray, R_gray2rgb, T_gray2rgb):
    # K = [fx, fy, cx, cy]
    assert len(K_rgb) == 4 and len(K_gray) == 4
    assert th_depth >= 0
    assert th_disp >= 0

    # Get depth and disparity range
    max_disp = num_disp - 1

    # RGB is the reference
    min_depth = K_rgb[0] * B_rgb / max_disp

    # RGB is the reference
    dep_range = [max_depth]

    while dep_range[-1] > min_depth:
        disp_tmp = K_rgb[0] * B_rgb / dep_range[-1]
        dep_alpha = dep_range[-1] - th_depth
        disp_alpha = K_rgb[0] * B_rgb / dep_alpha

        disp_diff = disp_alpha - disp_tmp

        if disp_diff < th_disp:
            disp_alpha = disp_tmp + th_disp
            dep_alpha = K_rgb[0] * B_rgb / disp_alpha

        if dep_alpha <= min_depth:
            dep_alpha = min_depth

        dep_range.append(dep_alpha)

    # Interpolate from valid samples
    num_disp_cost = int(np.ceil(len(dep_range) / 4.0) * 4.0)

    x_sample = np.linspace(0, 1, len(dep_range))
    x_interp = np.linspace(0, 1, num_disp_cost)

    dep_range_rgb = np.interp(x_interp, x_sample, dep_range)

    disp_range_rgb = K_rgb[0] * B_rgb / (dep_range_rgb + 1e-8)

    # Rotation angle
    rot_angle_cos = (np.trace(R_gray2rgb) - 1.0) / 2.0
    err_ratio = (1 - rot_angle_cos)

    # Depth error due to rotation should be negligible
    assert err_ratio < 1e-3, 'Depth error due to rotation is not negligible'

    depth_diff_rgb2gray = -T_gray2rgb[2]

    # Compensate for depth difference
    dep_range_gray = dep_range_rgb + depth_diff_rgb2gray

    disp_range_gray = K_gray[0] * B_gray / (dep_range_gray + 1e-8)

    return dep_range_rgb, disp_range_rgb, dep_range_gray, disp_range_gray
