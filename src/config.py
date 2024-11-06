"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    All of the parameters are defined here.
"""

import time
import argparse


parser = argparse.ArgumentParser(
    description='Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments'
)


# Dataset
parser.add_argument('--path_kittimmd',
                    type=str,
                    default='/home/viplab/Dataset/KITTIMMD',
                    help='path to the KITTI Multi-Modal Depth (KITTI MMD) dataset')
parser.add_argument('--path_mmdce',
                    type=str,
                    default='/home/viplab/Dataset/MMDCE/day',
                    # default='/home/viplab/Dataset/MMDCE/night',
                    help='path to the Multi-Modal Depth in Changing Environments (MMDCE) dataset')
parser.add_argument('--dataset',
                    type=str,
                    default='KITTIMMD',
                    choices=('KITTIMMD', 'MMDCE'),
                    help='dataset name')
parser.add_argument('--list_data',
                    type=str,
                    default='../list_data/kitti_mmd.json',
                    help='path to json file')
parser.add_argument('--patch_height',
                    type=int,
                    default=240,
                    help='height of a patch to crop')
parser.add_argument('--patch_width',
                    type=int,
                    default=912,
                    help='width of a patch to crop')
parser.add_argument('--top_crop',
                    type=int,
                    default=0,
                    help='top crop size for KITTI dataset')
parser.add_argument('--test_crop',
                    action='store_true',
                    help='crop for test')
parser.add_argument('--method',
                    type=str,
                    default='RGB',
                    # default='IR',
                    # default='LIDAR',
                    # default='RGB-LIDAR',
                    # default='RGB-IR-LIDAR',
                    help='modality fusion method')


# Hardware
parser.add_argument('--seed',
                    type=int,
                    default=7240,
                    help='random seed point')
parser.add_argument('--gpus',
                    type=str,
                    default="0,1",
                    help='visible GPUs')
parser.add_argument('--port',
                    type=str,
                    default='29500',
                    help='multiprocessing port')
parser.add_argument('--num_threads',
                    type=int,
                    default=4,
                    help='number of threads')
parser.add_argument('--no_multiprocessing',
                    action='store_true',
                    default=False,
                    help='do not use multiprocessing')


# Network
parser.add_argument('--model_name',
                    type=str,
                    default='MMDNet',
                    choices=('MMDNet'),
                    help='model name')
parser.add_argument('--network',
                    type=str,
                    default='resnet18',
                    choices=('resnet18', 'resnet34'),
                    help='network name')
parser.add_argument('--from_scratch',
                    action='store_true',
                    default=False,
                    help='train from scratch')
parser.add_argument('--cost_factor',
                    type=float,
                    default=3.0,
                    help='cost volume scale factor')
parser.add_argument('--aggregation_kernel',
                    type=int,
                    default=3,
                    help='cost aggregation kernel size')
parser.add_argument('--num_disp',
                    type=int,
                    default=192,
                    help='number of disparities')
parser.add_argument('--max_depth',
                    type=float,
                    default=90.0,
                    help='maximum depth')
parser.add_argument('--th_depth',
                    type=float,
                    default=0.5,
                    help='valid depth interval')
parser.add_argument('--th_disp',
                    type=float,
                    default=1.5,
                    help='valid disp interval')
parser.add_argument('--individual_supervision',
                    action='store_true',
                    default=False,
                    help='supervise individual outputs')
parser.add_argument('--individual_supervision_weight',
                    type=float,
                    default=0.1,
                    help='individual supervision weight')


# Training
parser.add_argument('--loss',
                    type=str,
                    default='1.0*L1',
                    # default='1.0*L2',
                    # default='1.0*L1+1.0*L2',
                    help='loss function configuration')
parser.add_argument('--pretrain',
                    type=str,
                    default=None,
                    help='ckpt path')
parser.add_argument('--resume',
                    action='store_true',
                    help='resume training')
parser.add_argument('--test_only',
                    action='store_true',
                    help='test only flag')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    help='number of epochs to train')
parser.add_argument('--batch_size',
                    type=int,
                    default=8,
                    help='input batch size for training')
parser.add_argument('--augment',
                    action='store_true',
                    help='data augmentation')


# Summary
parser.add_argument('--num_summary',
                    type=int,
                    default=4,
                    help='maximum number of summary images to save')


# Optimizer
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='learning rate')
parser.add_argument('--decay',
                    type=str,
                    default='5,8,10',
                    help='learning rate decay schedule')
parser.add_argument('--gamma',
                    type=str,
                    default='1.0,0.2,0.04',
                    help='learning rate multiplicative factors')
parser.add_argument('--optimizer',
                    default='ADAM',
                    choices=('SGD', 'ADAM'),
                    help='optimizer to use (SGD | ADAM)')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas',
                    type=tuple,
                    default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon',
                    type=float,
                    default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0,
                    help='weight decay')
parser.add_argument('--warm_up',
                    action='store_true',
                    default=True,
                    help='do lr warm up during the 1st epoch')
parser.add_argument('--no_warm_up',
                    action='store_false',
                    dest='warm_up',
                    help='no lr warm up')

# Logs
parser.add_argument('--save',
                    type=str,
                    default='trial',
                    help='file name to save')
parser.add_argument('--save_full',
                    action='store_true',
                    default=False,
                    help='save optimizer, scheduler and amp in checkpoints (large memory)')
parser.add_argument('--save_image',
                    action='store_true',
                    default=False,
                    help='save images for test')
parser.add_argument('--save_result_only',
                    action='store_true',
                    default=False,
                    help='save result images only with submission format')
parser.add_argument('--save_best',
                    type=str,
                    default='none',
                    choices=('none', 'min', 'max'),
                    help='save the best ckpt based on the first metric (min/max for the minimum/maximum metric)')


args = parser.parse_args()

args.num_gpus = len(args.gpus.split(','))

current_time = time.strftime('%y%m%d_%H%M%S_')
save_dir = '../experiments/' + current_time + args.save
args.save_dir = save_dir
