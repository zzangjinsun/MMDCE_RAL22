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


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


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


def project_velo_to_image(velo, P, image_size):
    velo = np.dot(P, velo)
    velo[0:2, :] = velo[0:2, :] / velo[2, :]

    # image_size = (width, height)
    width, height = image_size

    x = np.round(velo[0, :]).astype(int)
    y = np.round(velo[1, :]).astype(int)

    idx_valid = (x >= 0) & (y >= 0) & (x <= width - 1) & (y <= height - 1)
    velo = velo[:, idx_valid]
    x = x[idx_valid]
    y = y[idx_valid]

    # Generate sparse depth image
    depth = np.zeros((height, width))
    depth[y, x] = velo[2, :]

    # Choose closer one for duplicated indices
    idx = y * width + x
    unique, counts = np.unique(idx, return_counts=True)
    idx_dup = unique[counts > 1]
    for i in idx_dup:
        p = np.where(idx == i)[0]
        xx = x[p[0]]
        yy = y[p[0]]
        idx_min = np.argmin(velo[2, p])
        depth[yy, xx] = velo[2, p[idx_min]]

    depth[depth < 0.0] = 0.0

    return depth


def generate_depth_image(xx, yy, dep, image_size):
    # image_size = (width, height)
    width, height = image_size

    x = np.round(xx).astype(int)
    y = np.round(yy).astype(int)

    idx_valid = (x >= 0) & (y >= 0) & (x <= width - 1) & (y <= height - 1)
    dep = dep[idx_valid]
    x = x[idx_valid]
    y = y[idx_valid]

    # Generate sparse depth image
    depth = np.zeros((height, width))
    depth[y, x] = dep

    # Choose closer one for duplicated indices
    idx = y * width + x
    unique, counts = np.unique(idx, return_counts=True)
    idx_dup = unique[counts > 1]
    for i in idx_dup:
        p = np.where(idx == i)[0]
        xx = x[p[0]]
        yy = y[p[0]]
        idx_min = np.argmin(dep[p])
        depth[yy, xx] = dep[p[idx_min]]

    depth[depth < 0.0] = 0.0

    return depth
