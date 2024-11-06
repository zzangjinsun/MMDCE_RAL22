"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    This script generates a JSON file for the KITTI Multi-Modal Depth (KITTI MMD) dataset.
"""


import os
import json
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser(
    description="KITTI Multi-Modal Depth (KITTI MMD) dataset jason generator")

parser.add_argument('--path_kittimmd', type=str, required=True,
                    help="Path to the KITTI Multi-Modal Depth (KITTI MMD) dataset")

parser.add_argument('--path_st_mapping', type=str, required=False,
                    default='kitti_stereo_train_mapping.txt',
                    help="Path to the KITTI Stereo 2015 dataset mapping")
parser.add_argument('--path_out', type=str, required=False,
                    default='../list_data', help="Output path")
parser.add_argument('--name', type=str, required=False,
                    default='kitti_mmd.json', help="Output file name")
parser.add_argument('--num_train', type=int, required=False, default=int(1e10),
                    help="Maximum number of training pairs")
parser.add_argument('--num_val', type=int, required=False, default=int(1e10),
                    help="Maximum number of validation pairs")
parser.add_argument('--num_test', type=int, required=False, default=int(1e10),
                    help="Maximum number of testing pairs")
parser.add_argument('--shuffle', default=True, action='store_true',
                    help="Shuffle train data")
parser.add_argument('--no_shuffle', action='store_false', dest='shuffle',
                    help="No shuffle")
parser.add_argument('--seed', type=int, required=False, default=7240,
                    help="Random seed point")

args = parser.parse_args()

np.random.seed(args.seed)


# NOTE: some images have inconsistent sizes between images and depths.
# TODO: check whether this is the problem of the dataset or preprocessing
list_skip_img = [
    'train/2011_09_30_drive_0028_sync/image_00/data/0000004579.png'
]


def generate_json():
    os.makedirs(args.path_out, exist_ok=True)

    # KITTI Stereo 2015 sequences
    f_mapping = open(args.path_st_mapping, 'r')

    list_seq_st = []
    for k in range(0, 200):
        map_line = f_mapping.readline()
        map_info = map_line.strip()

        if len(map_info) == 0:
            continue

        map_info = map_info.split()

        seq = map_info[1]

        if seq not in list_seq_st:
            list_seq_st.append(seq)

    print('Sequences in KITTI Stereo 2015 :')
    for seq in list_seq_st:
        print(seq)

    # Train, Val, and Test sets
    dict_json = {}
    for split in ['train', 'val', 'depth_selection/val_multi_modal']:
        list_seq = os.listdir(args.path_kittimmd + '/' + split)
        list_seq.sort()
    
        list_pairs = []
        for seq in list_seq:
            # Skip overlapping sequences only for train split to avoid validation data leakage
            if split == 'train' and seq in list_seq_st:
                print('Skipping overlapping sequence with Stereo 2015 : {}'.format(seq))
                continue

            list_image = os.listdir(args.path_kittimmd + '/' + split + '/'
                                    + seq + '/image_02/data')
            list_image.sort()
    
            cnt_seq = 0
            for img in list_image:
                # Generate file paths
                img0 = split + '/' + seq + '/image_00/data/' + img

                if img0 in list_skip_img:
                    print('Skip invalid image : {}'.format(img0))
                    continue

                img1 = split + '/' + seq + '/image_01/data/' + img
                img2 = split + '/' + seq + '/image_02/data/' + img
                img3 = split + '/' + seq + '/image_03/data/' + img

                dep0 = split + '/' + seq + '/proj_depth/velodyne_raw/image_00/' + img
                dep1 = split + '/' + seq + '/proj_depth/velodyne_raw/image_01/' + img
                dep2 = split + '/' + seq + '/proj_depth/velodyne_raw/image_02/' + img
                dep3 = split + '/' + seq + '/proj_depth/velodyne_raw/image_03/' + img

                gt0 = split + '/' + seq + '/proj_depth/groundtruth/image_00/' + img
                gt1 = split + '/' + seq + '/proj_depth/groundtruth/image_01/' + img
                gt2 = split + '/' + seq + '/proj_depth/groundtruth/image_02/' + img
                gt3 = split + '/' + seq + '/proj_depth/groundtruth/image_03/' + img
    
                calib_cam_to_cam = split + '/' + seq + '/calib_cam_to_cam.txt'
    
                dict_sample = {
                    'img0': img0, 'img1': img1, 'img2': img2, 'img3': img3,
                    'dep0': dep0, 'dep1': dep1, 'dep2': dep2, 'dep3': dep3,
                    'gt0': gt0, 'gt1': gt1, 'gt2': gt2, 'gt3': gt3,
                    'K': calib_cam_to_cam
                }
    
                flag_valid = True
                for val in dict_sample.values():
                    flag_valid &= os.path.exists(args.path_kittimmd + '/' + val)
                    if not flag_valid:
                        break
    
                if not flag_valid:
                    continue
    
                list_pairs.append(dict_sample)
                cnt_seq += 1
    
            print("{} / {} : {} samples".format(split, seq, cnt_seq))

        if split == 'depth_selection/val_multi_modal':
            dict_json['test'] = list_pairs
        else:
            dict_json[split] = list_pairs

    if args.shuffle:
        random.shuffle(dict_json['train'])

    # Cut if maximum is set
    for s in [('train', args.num_train), ('val', args.num_val),
              ('test', args.num_test)]:
        print("{} split : Total {} samples".format(s[0], len(dict_json[s[0]])))

        if len(dict_json[s[0]]) > s[1]:
            num_orig = len(dict_json[s[0]])
            dict_json[s[0]] = dict_json[s[0]][0:s[1]]
            print("{} split : {} -> {}".format(s[0], num_orig,
                                               len(dict_json[s[0]])))

    f = open(args.path_out + '/' + args.name, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


if __name__ == '__main__':
    print('\nArguments')
    for arg in vars(args):
        print('{:<15} : {}'.format(arg, getattr(args, arg)))
    print('')

    generate_json()
