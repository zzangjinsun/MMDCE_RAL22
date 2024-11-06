"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    This script copies stereo RGB, stereo grayscale, poses and calibrations
    from the KITTI Raw dataset to the KITTI Depth Completion dataset to
    generate the KITTI Multi-Modal Depth (KITTI MMD) dataset.
"""


import os
import sys
import shutil
import argparse

from common import *
from pykitti.raw import raw as kitti_raw


parser = argparse.ArgumentParser(
    description="KITTI Multi-Modal Depth (KITTI MMD) dataset preparer")

parser.add_argument('--path_dc', type=str, required=True,
                    help="Path to the Depth completion dataset")
parser.add_argument('--path_raw', type=str, required=True,
                    help="Path to the Raw dataset")

args = parser.parse_args()


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


# Reorganization functions
def prepare_reorganization():
    # Check directory existence
    check_dir_existence(args.path_dc)
    check_dir_existence(args.path_raw)

    print("Preparation of reorganization is done.")


def reorganize_train_val():
    # Train and validation splits
    for split in ['train', 'val']:
        path_dc = args.path_dc + '/' + split

        check_dir_existence(path_dc)

        # Get the list of sequences
        list_seq = os.listdir(path_dc)
        list_seq.sort()

        for seq in list_seq:
            path_raw_seq_src = args.path_raw + '/' + seq[0:10] + '/' + seq

            path_seq_dst = path_dc + '/' + seq

            try:
                # Copy stereo images and poses
                print("Copying raw dataset : {} -> {}".format(
                    path_raw_seq_src + '/image_00', path_seq_dst + '/image_00'))
                shutil.copytree(path_raw_seq_src + '/image_00',
                                path_seq_dst + '/image_00', dirs_exist_ok=True)

                print("Copying raw dataset : {} -> {}".format(
                    path_raw_seq_src + '/image_01', path_seq_dst + '/image_01'))
                shutil.copytree(path_raw_seq_src + '/image_01',
                                path_seq_dst + '/image_01', dirs_exist_ok=True)

                print("Copying raw dataset : {} -> {}".format(
                    path_raw_seq_src + '/image_02', path_seq_dst + '/image_02'))
                shutil.copytree(path_raw_seq_src + '/image_02',
                                path_seq_dst + '/image_02', dirs_exist_ok=True)

                print("Copying raw dataset : {} -> {}".format(
                    path_raw_seq_src + '/image_03', path_seq_dst + '/image_03'))
                shutil.copytree(path_raw_seq_src + '/image_03',
                                path_seq_dst + '/image_03', dirs_exist_ok=True)

                # Copy calibrations
                for calib in ['calib_cam_to_cam.txt',
                              'calib_imu_to_velo.txt',
                              'calib_velo_to_cam.txt']:
                    shutil.copy2(args.path_raw + '/' + seq[0:10] + '/'
                                 + calib, path_seq_dst + '/' + calib)
            except OSError:
                print("Failed to copy files for {}".format(seq))
                sys.exit(-1)

        print("Reorganization for {} split finished".format(split))


def generate_depth():
    for split in ['train', 'val']:
        path_dc = args.path_dc + '/' + split

        check_dir_existence(path_dc)

        # Get the list of sequences
        list_seq = os.listdir(path_dc)
        list_seq.sort()

        cnt = 0
        for seq in list_seq:
            path_dst_seq = path_dc + '/' + seq

            list_image = os.listdir(
                path_dst_seq + '/proj_depth/velodyne_raw/image_02')
            list_image.sort()

            for img in list_image:
                frame_number = int(img[:-4])

                data = kitti_raw(args.path_raw, seq[:10], seq[17:21],
                                 frames=[frame_number])

                T_velo_to_cam0 = data.calib.T_cam0_velo
                P_rect_cam0 = np.eye(4, 4)
                P_rect_cam0[0:3, 0:3] = data.calib.K_cam0
                P_velo_to_cam0 = P_rect_cam0.dot(T_velo_to_cam0)

                T_velo_to_cam1 = data.calib.T_cam1_velo
                P_rect_cam1 = np.eye(4, 4)
                P_rect_cam1[0:3, 0:3] = data.calib.K_cam1
                P_velo_to_cam1 = P_rect_cam1.dot(T_velo_to_cam1)

                T_velo_to_cam2 = data.calib.T_cam2_velo
                P_rect_cam2 = np.eye(4, 4)
                P_rect_cam2[0:3, 0:3] = data.calib.K_cam2
                P_velo_to_cam2 = P_rect_cam2.dot(T_velo_to_cam2)

                T_velo_to_cam3 = data.calib.T_cam3_velo
                P_rect_cam3 = np.eye(4, 4)
                P_rect_cam3[0:3, 0:3] = data.calib.K_cam3
                P_velo_to_cam3 = P_rect_cam3.dot(T_velo_to_cam3)

                # [Forward; Left; Up; Reflectance]
                velo = data.get_velo(0).transpose()
                velo = velo[:, velo[0, :] >= 0]
                velo[3, :] = 1.0

                depth0 = project_velo_to_image(
                    velo, P_velo_to_cam0, data.get_cam0(0).size)
                depth1 = project_velo_to_image(
                    velo, P_velo_to_cam1, data.get_cam1(0).size)
                depth2 = project_velo_to_image(
                    velo, P_velo_to_cam2, data.get_cam2(0).size)
                depth3 = project_velo_to_image(
                    velo, P_velo_to_cam3, data.get_cam3(0).size)

                # Change to depth completion format
                depth0 = (depth0 * 256.0).astype(np.uint16)
                depth1 = (depth1 * 256.0).astype(np.uint16)

                depth0 = Image.fromarray(depth0)
                depth1 = Image.fromarray(depth1)

                # Make subdirectories
                os.makedirs(path_dst_seq
                            + '/proj_depth/velodyne_raw/image_00',
                            exist_ok=True)
                os.makedirs(path_dst_seq
                            + '/proj_depth/velodyne_raw/image_01',
                            exist_ok=True)

                path_depth0_dst = \
                    path_dst_seq + '/proj_depth/velodyne_raw/image_00/{:010d}.png'.format(frame_number)
                path_depth1_dst = \
                    path_dst_seq + '/proj_depth/velodyne_raw/image_01/{:010d}.png'.format(frame_number)
                try:
                    depth0.save(path_depth0_dst)
                    depth1.save(path_depth1_dst)
                except OSError:
                    print('Failed to save : {}'.format(seq + '/' + img))
                    sys.exit(-1)

                cnt += 1
                print('Saved : {} - {}'.format(seq, img))

        print("Depth generation for {} : {} images".format(split, cnt))


def reorganize_test():
    # Selected validation datasets
    path_dc_test = args.path_dc + '/depth_selection/val_selection_cropped'
    path_dc_val = args.path_dc + '/val'
    path_dc_out = args.path_dc + '/depth_selection/val_multi_modal'

    list_img = os.listdir(path_dc_test + '/image')
    list_img.sort()

    for img in list_img:
        # Name format : 2011_09_26_drive_0002_sync_image_0000000005_image_02.png
        seq = img[0:26]

        path_dst = path_dc_out + '/' + seq

        os.makedirs(path_dst, exist_ok=True)

        idx = img[33:43]

        try:
            # Copy calibrations
            for calib in ['calib_cam_to_cam.txt',
                          'calib_imu_to_velo.txt',
                          'calib_velo_to_cam.txt']:
                path_cal_src = '{}/{}/{}'.format(path_dc_val, seq, calib)
                path_cal_dst = '{}/{}'.format(path_dst, calib)

                if not os.path.exists(path_cal_dst):
                    shutil.copy2(path_cal_src, path_cal_dst)

            # Copy images and depths
            for cam in range(0, 4):
                os.makedirs(
                    '{}/image_{:02d}/data'.format(path_dst, cam), exist_ok=True
                )
                os.makedirs(
                    '{}/proj_depth/velodyne_raw/image_{:02d}'.format(path_dst, cam),
                    exist_ok=True
                )
                os.makedirs(
                    '{}/proj_depth/groundtruth/image_{:02d}'.format(path_dst, cam),
                    exist_ok=True
                )

                path_img_src = '{}/{}/image_{:02d}/data/{}.png'.format(
                    path_dc_val, seq, cam, idx
                )
                path_img_dst = '{}/image_{:02d}/data/{}.png'.format(
                    path_dst, cam, idx
                )

                shutil.copy2(path_img_src, path_img_dst)

                path_dep_src = '{}/{}/proj_depth/velodyne_raw/image_{:02d}/{}.png'.format(
                    path_dc_val, seq, cam, idx
                )
                path_dep_dst = '{}/proj_depth/velodyne_raw/image_{:02d}/{}.png'.format(
                    path_dst, cam, idx
                )

                shutil.copy2(path_dep_src, path_dep_dst)

                if cam in [2, 3]:
                    path_gt_src = '{}/{}/proj_depth/groundtruth/image_{:02d}/{}.png'.format(
                        path_dc_val, seq, cam, idx
                    )
                    path_gt_dst = '{}/proj_depth/groundtruth/image_{:02d}/{}.png'.format(
                        path_dst, cam, idx
                    )

                    shutil.copy2(path_gt_src, path_gt_dst)
        except OSError:
                print("Failed to copy files for {}".format(seq))
                sys.exit(-1)

        print('Copying test set : val/{}/{} -> val_multi_modal/{}/{}'.format(
            seq, idx, seq, idx
        ))

    print("Reorganization for test split finished ({} images)".format(len(list_img)))


def generate_gt_data():
    for split in ['train', 'val', 'depth_selection/val_multi_modal']:
        path_split = args.path_dc + '/' + split

        # Get the list of sequences
        list_seq = os.listdir(path_split)
        list_seq.sort()

        for seq in list_seq:
            calib_c2c = read_calib_file(path_split + '/' + seq + '/calib_cam_to_cam.txt')

            # Read calibrations
            P_rect_00 = calib_c2c['P_rect_00'].reshape((3, 4))
            P_rect_01 = calib_c2c['P_rect_01'].reshape((3, 4))
            P_rect_02 = calib_c2c['P_rect_02'].reshape((3, 4))
            P_rect_03 = calib_c2c['P_rect_03'].reshape((3, 4))

            P_rect_00 = np.concatenate((P_rect_00, np.array([[0, 0, 0, 1.0]])), axis=0)
            P_rect_01 = np.concatenate((P_rect_01, np.array([[0, 0, 0, 1.0]])), axis=0)
            P_rect_02 = np.concatenate((P_rect_02, np.array([[0, 0, 0, 1.0]])), axis=0)
            P_rect_03 = np.concatenate((P_rect_03, np.array([[0, 0, 0, 1.0]])), axis=0)

            K = P_rect_00
            K_inv = np.linalg.inv(K)

            # Between cameras
            T_rect_00 = K_inv @ P_rect_00
            T_rect_01 = K_inv @ P_rect_01
            T_rect_02 = K_inv @ P_rect_02
            T_rect_03 = K_inv @ P_rect_03

            # 3D points from cam2 to cam0
            T_rect_20 = np.linalg.inv(T_rect_02)

            # From velo-cam to cam
            R_rect_00 = np.eye(4)
            R_rect_01 = np.eye(4)
            R_rect_02 = np.eye(4)
            R_rect_03 = np.eye(4)

            R_rect_00[:3, :3] = calib_c2c['R_rect_00'].reshape((3, 3))
            R_rect_01[:3, :3] = calib_c2c['R_rect_01'].reshape((3, 3))
            R_rect_02[:3, :3] = calib_c2c['R_rect_02'].reshape((3, 3))
            R_rect_03[:3, :3] = calib_c2c['R_rect_03'].reshape((3, 3))

            list_img = os.listdir(path_split + '/' + seq + '/image_02/data')
            list_img.sort()

            os.makedirs(path_split + '/' + seq + '/proj_depth/groundtruth/image_00/', exist_ok=True)
            os.makedirs(path_split + '/' + seq + '/proj_depth/groundtruth/image_01/', exist_ok=True)

            for img in list_img:
                path_dep = path_split + '/' + seq + '/proj_depth/groundtruth/image_02/' + img

                if not os.path.exists(path_dep):
                    continue

                # Read depth
                dep2 = read_depth(path_dep)

                H, W = dep2.shape

                xx, yy = np.meshgrid(range(0, W), range(0, H))

                xx = xx.reshape((1, H*W))
                yy = yy.reshape((1, H*W))
                dep2 = dep2.reshape((1, H*W))

                flag = dep2 > 0.0

                xx = xx[flag]
                yy = yy[flag]
                zz = np.ones_like(xx)
                dep2 = dep2[flag]

                # Get 3D points from cam2
                pts2d = np.stack((xx, yy, zz, zz), axis=0)
                pts3d_cam2 = K_inv @ pts2d
                pts3d_cam2[:3, :] = dep2 * pts3d_cam2[:3, :]

                # Project to cam0 and cam1
                pts3d_cam0 = T_rect_20 @ pts3d_cam2
                pts3d_cam1 = T_rect_01 @ pts3d_cam0

                dep0 = pts3d_cam0[2, :]
                dep1 = pts3d_cam1[2, :]

                # Project to image plane
                pts2d_cam0 = K @ pts3d_cam0
                pts2d_cam1 = K @ pts3d_cam1

                xx0 = pts2d_cam0[0, :] / pts2d_cam0[2, :]
                yy0 = pts2d_cam0[1, :] / pts2d_cam0[2, :]

                xx1 = pts2d_cam1[0, :] / pts2d_cam1[2, :]
                yy1 = pts2d_cam1[1, :] / pts2d_cam1[2, :]

                dep0_img = generate_depth_image(xx0, yy0, dep0, (W, H))
                dep1_img = generate_depth_image(xx1, yy1, dep1, (W, H))

                # Change to depth completion format
                dep0_img = (dep0_img * 256.0).astype(np.uint16)
                dep1_img = (dep1_img * 256.0).astype(np.uint16)

                dep0_img = Image.fromarray(dep0_img)
                dep1_img = Image.fromarray(dep1_img)

                path_dep0_img = path_split + '/' + seq + '/proj_depth/groundtruth/image_00/' + img
                path_dep1_img = path_split + '/' + seq + '/proj_depth/groundtruth/image_01/' + img

                dep0_img.save(path_dep0_img)
                dep1_img.save(path_dep1_img)

                print('{} : Processed - {}'.format(split, seq + '/proj_depth/groundtruth/image_02/' + img))

        print('{} : Finished'.format(split))


if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':',  getattr(args, arg))
    print('')

    # Check dataset path
    prepare_reorganization()

    # Copy data from raw to DC
    reorganize_train_val()

    # Generate depths for cam0 and cam1
    generate_depth()

    # Copy data from val to test
    # NOTE : test is subset of val
    reorganize_test()

    # Generate semi-dense gt for test
    generate_gt_data()
