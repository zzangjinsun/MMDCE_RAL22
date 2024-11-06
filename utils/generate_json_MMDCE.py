"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    This script generates a JSON file for the Multi-Modal Depth in Changing Environments (MMDCE) dataset.
"""


import os
import json
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser(
    description="Multi-Modal Depth in Changing Environments (MMDCE) dataset jason generator")

parser.add_argument('--path_mmdce', type=str, required=True,
                    help="Path to the Multi-Modal Depth in Changing Environments (MMDCE) dataset")

parser.add_argument('--path_out', type=str, required=False,
                    default='../list_data', help="Output path")
parser.add_argument('--name', type=str, required=False,
                    default='mmdce.json', help="Output file name")
parser.add_argument('--num_train', type=int, required=False, default=int(1e10),
                    help="Maximum number of training pairs")
parser.add_argument('--num_val', type=int, required=False, default=int(1e10),
                    help="Maximum number of validation pairs")
parser.add_argument('--num_test', type=int, required=False, default=int(1e10),
                    help="Maximum number of testing pairs")
parser.add_argument('--shuffle', default=True, action='store_true', help="Shuffle train data")
parser.add_argument('--no_shuffle', action='store_false', dest='shuffle',
                    help="No shuffle")
parser.add_argument('--seed', type=int, required=False, default=7240,
                    help="Random seed point")

args = parser.parse_args()

np.random.seed(args.seed)


def generate_json():
    os.makedirs(args.path_out, exist_ok=True)

    dict_json = {}

    for split in ['train', 'val', 'test']:
        list_seq = os.listdir(args.path_mmdce + '/' + split)
        list_seq.sort()

        list_pairs = []
        for seq in list_seq:
            list_image = os.listdir(args.path_mmdce + '/' + split + '/' + seq + '/ir1')
            list_image.sort()

            cnt_seq = 0
            for img in list_image:
                rgb1 = seq + '/rgb1/' + img
                rgb2 = seq + '/rgb2/' + img
                ir1 = seq + '/ir1/' + img
                ir2 = seq + '/ir2/' + img

                dep_rgb1 = seq + '/dep_rgb1/' + img
                dep_rgb2 = seq + '/dep_rgb2/' + img
                dep_ir1 = seq + '/dep_ir1/' + img
                dep_ir2 = seq + '/dep_ir2/' + img

                # NOTE: Only reference GT depth maps are filtered (i.e., left cameras)
                # gt_dep_rgb1 = seq + '/gt_dep_rgb1/' + img
                gt_dep_rgb1 = seq + '/gt_dep_rgb1_filtered/' + img
                gt_dep_rgb2 = seq + '/gt_dep_rgb2/' + img
                # gt_dep_ir1 = seq + '/gt_dep_ir1/' + img
                gt_dep_ir1 = seq + '/gt_dep_ir1_filtered/' + img
                gt_dep_ir2 = seq + '/gt_dep_ir2/' + img

                calib = seq + '/calib.npy'
                info = seq + '/info.txt'

                dict_sample = {
                    'rgb1': rgb1, 'rgb2': rgb2, 'ir1': ir1, 'ir2': ir2,
                    'dep_rgb1': dep_rgb1, 'dep_rgb2': dep_rgb2,
                    'dep_ir1': dep_ir1, 'dep_ir2': dep_ir2,
                    'gt_dep_rgb1': gt_dep_rgb1, 'gt_dep_rgb2': gt_dep_rgb2,
                    'gt_dep_ir1': gt_dep_ir1, 'gt_dep_ir2': gt_dep_ir2,
                    'calib': calib,
                    'info': info
                }

                flag_valid = True
                for val in dict_sample.values():
                    flag_valid &= os.path.exists(
                        args.path_mmdce + '/' + split + '/' + val)
                    if not flag_valid:
                        break

                if not flag_valid:
                    continue

                list_pairs.append(dict_sample)
                cnt_seq += 1

            print("{} / {} : {} samples".format(split, seq, cnt_seq))

        dict_json[split] = list_pairs

        print('{} : {} samples in total'.format(split, len(list_pairs)))

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
