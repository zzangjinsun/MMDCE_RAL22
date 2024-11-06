"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    MMDNetSummary implementation
"""


from . import BaseSummary
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

cm = plt.get_cmap('plasma')
cm._init()

cm_w = cm.__copy__()
cm_w._lut[0, :] = [1, 1, 1, 1]

fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 15)


class MMDNetSummary(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):
        assert mode in ['train', 'val', 'test'], \
            "mode should be one of ['train', 'val', 'test'] but got {}".format(mode)

        super(MMDNetSummary, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.method = args.method

        self.max_depth = self.args.max_depth
        self.num_disp = self.args.num_disp
        self.max_disp = self.num_disp - 1

        self.individual_supervision = args.individual_supervision

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None

        # ImageNet normalization
        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def _process_image(self, img, unnorm=False, max_val=1.0):
        B, C, H, W = img.shape

        img = img.detach()

        if unnorm:
            img = img * self.img_std.type_as(img) + self.img_mean.type_as(img)

        img = img.data.cpu().numpy()

        if B > self.args.num_summary:
            B = self.args.num_summary

            img = img[0:B, :, :, :]

        img = np.clip(img, a_min=0, a_max=max_val)

        return img

    def _horizontal_image(self, img, to_color=False, max_val=1.0, cmap=cm, name=None):
        B, C, H, W = img.shape

        list_img = []

        for b in range(0, B):
            if to_color:
                img_tmp = 255.0 * img[b, 0, :, :] / max_val
                img_tmp = cmap(img_tmp.astype('uint8'))
                img_tmp = np.transpose(img_tmp[:, :, :3], (2, 0, 1))
            else:
                img_tmp = img[b, :, :, :]

            list_img.append(img_tmp)

        img_concat = np.concatenate(list_img, axis=2)

        if name is not None:
            img_tmp = np.transpose(img_concat, [1, 2, 0])
            img_tmp = Image.fromarray((255.0 * img_tmp).astype('uint8'), mode='RGB')
            img_draw = ImageDraw.Draw(img_tmp)
            img_draw.text((2, 2), name, font=fnt, fill=(0, 255, 0))
            img_tmp = np.asarray(img_tmp).astype('float') / 255.0
            img_concat = np.transpose(img_tmp, [2, 0, 1])

        return img_concat

    def _to_image(self, img, to_color=False, max_val=1.0, cmap=cm):
        B, C, H, W = img.shape

        assert B == 1

        if to_color:
            img_tmp = 255.0 * img[0, 0, :, :] / max_val
            img_tmp = cmap(img_tmp.astype('uint8'))
            img_tmp = 255.0 * img_tmp[:, :, :3]
        else:
            img_tmp = 255.0 * img[0, :, :, :]
            img_tmp = np.transpose(img_tmp, [1, 2, 0])

        img_tmp = Image.fromarray(img_tmp.astype('uint8'), 'RGB')

        return img_tmp

    def update(self, global_step, sample, output):
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:04d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.5f}  ".format(name, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:04d} | {}\n'.format(global_step, msg))
            f_metric.close()

        rgb_l = self._process_image(sample['rgb_l'], True, 1.0)
        rgb_r = self._process_image(sample['rgb_r'], True, 1.0)
        dep_rgb_l = self._process_image(sample['dep_rgb_l'], False, self.max_depth)
        disp_rgb_l = self._process_image(sample['disp_rgb_l'], False, self.max_disp)
        gt_dep_rgb_l = self._process_image(sample['gt_dep_rgb_l'], False, self.max_depth)
        gt_disp_rgb_l = self._process_image(sample['gt_disp_rgb_l'], False, self.max_disp)

        if 'IR' in self.method:
            ir_l = self._process_image(sample['ir_l'], True, 1.0)
            ir_r = self._process_image(sample['ir_r'], True, 1.0)
            dep_ir_l = self._process_image(sample['dep_ir_l'], False, self.max_depth)
            disp_ir_l = self._process_image(sample['disp_ir_l'], False, self.max_disp)
            gt_dep_ir_l = self._process_image(sample['gt_dep_ir_l'], False, self.max_depth)
            gt_disp_ir_l = self._process_image(sample['gt_disp_ir_l'], False, self.max_disp)

        if self.individual_supervision and 'RGB' in self.method:
            pred_dep_rgb = self._process_image(output['pred_dep3_rgb'], False, self.max_depth)
            pred_disp_rgb = self._process_image(output['pred_disp3_rgb'], False, self.max_disp)

        if self.individual_supervision and 'IR' in self.method:
            pred_dep_ir = self._process_image(output['pred_dep3_ir'], False, self.max_depth)
            pred_disp_ir = self._process_image(output['pred_disp3_ir'], False, self.max_disp)

        if self.individual_supervision and 'LIDAR' in self.method:
            pred_dep_lidar = self._process_image(output['pred_dep3_lidar'], False, self.max_depth)
            pred_disp_lidar = self._process_image(output['pred_disp3_lidar'], False, self.max_disp)

        pred_dep = self._process_image(output['pred_dep3'], False, self.max_depth)
        pred_disp = self._process_image(output['pred_disp3'], False, self.max_disp)

        list_img = []
        list_img_ir = []

        list_img.append(self._horizontal_image(rgb_l, False, name='L'))
        list_img.append(self._horizontal_image(rgb_r, False, name='R'))
        list_img.append(self._horizontal_image(dep_rgb_l, True, self.max_depth, cmap=cm_w, name='Dep L'))
        list_img.append(self._horizontal_image(gt_dep_rgb_l, True, self.max_depth, cmap=cm_w, name='GT L'))

        if 'IR' in self.method:
            list_img_ir.append(self._horizontal_image(ir_l, False, name='L'))
            list_img_ir.append(self._horizontal_image(ir_r, False, name='R'))
            list_img_ir.append(self._horizontal_image(dep_ir_l, True, self.max_depth, cmap=cm_w, name='Dep L'))
            list_img_ir.append(self._horizontal_image(gt_dep_ir_l, True, self.max_depth, cmap=cm_w, name='GT L'))

        list_img.append(self._horizontal_image(pred_dep, True, self.max_depth, name='Pred L'))

        if self.individual_supervision and 'RGB' in self.method:
            list_img.append(self._horizontal_image(pred_dep_rgb, True, self.max_depth, name='Pred RGB'))

        if self.individual_supervision and 'IR' in self.method:
            list_img_ir.append(self._horizontal_image(pred_dep_ir, True, self.max_depth, name='Pred IR'))

        if self.individual_supervision and 'LIDAR' in self.method:
            list_img.append(self._horizontal_image(pred_dep_lidar, True, self.max_depth, name='Pred LIDAR'))

        img_total = np.concatenate(list_img, axis=1)
        img_total = torch.from_numpy(img_total)

        self.add_image(self.mode + '/rgb', img_total, global_step)

        if 'IR' in self.method:
            img_total_ir = np.concatenate(list_img_ir, axis=1)
            img_total_ir = torch.from_numpy(img_total_ir)

            self.add_image(self.mode + '/ir', img_total_ir, global_step)

        self.flush()

        # Reset
        self.loss = []
        self.metric = []

    def save(self, epoch, idx, sample, output):
        with torch.no_grad():
            if self.args.save_result_only:
                self.path_output = '{}/{}/epoch{:04d}/{:08d}'.format(self.log_dir, self.mode, epoch, idx)
                os.makedirs(self.path_output, exist_ok=True)

                pred_dep = output['pred_dep3'].detach()
                pred_disp = output['pred_disp3'].detach()

                pred_dep = torch.clamp(pred_dep, min=0, max=self.max_depth)
                pred_disp = torch.clamp(pred_disp, min=0, max=self.max_disp)

                pred_dep = pred_dep[0, 0, :, :].data.cpu().numpy()
                pred_disp = pred_disp[0, 0, :, :].data.cpu().numpy()

                pred_dep = (pred_dep * 256.0).astype(np.uint16)
                pred_disp = (pred_disp * 256.0).astype(np.uint16)

                pred_dep = Image.fromarray(pred_dep)
                pred_disp = Image.fromarray(pred_disp)

                path_save_pred_dep = '{}/pred_dep.png'.format(self.path_output)
                path_save_pred_disp = '{}/pred_disp.png'.format(self.path_output)

                pred_dep.save(path_save_pred_dep)
                pred_disp.save(path_save_pred_disp)
            else:
                self.path_output = '{}/{}/epoch{:04d}/{:08d}'.format(self.log_dir, self.mode, epoch, idx)
                os.makedirs(self.path_output, exist_ok=True)

                rgb_l = self._process_image(sample['rgb_l'], True, 1.0)
                rgb_r = self._process_image(sample['rgb_r'], True, 1.0)
                dep_rgb_l = self._process_image(sample['dep_rgb_l'], False, self.max_depth)
                gt_dep_rgb_l = self._process_image(sample['gt_dep_rgb_l'], False, self.max_depth)

                rgb_l = self._to_image(rgb_l, False)
                rgb_r = self._to_image(rgb_r, False)
                dep_rgb_l = self._to_image(dep_rgb_l, True, self.max_depth, cmap=cm_w)
                gt_dep_rgb_l = self._to_image(gt_dep_rgb_l, True, self.max_depth, cmap=cm_w)

                path_save_rgb_l = '{}/01_1_rgb_l.png'.format(self.path_output)
                path_save_rgb_r = '{}/01_2_rgb_r.png'.format(self.path_output)
                path_save_dep_rgb_l = '{}/01_3_dep_rgb_l.png'.format(self.path_output)
                path_save_gt_dep_rgb_l = '{}/01_4_gt_dep_rgb_l.png'.format(self.path_output)

                rgb_l.save(path_save_rgb_l)
                rgb_r.save(path_save_rgb_r)
                dep_rgb_l.save(path_save_dep_rgb_l)
                gt_dep_rgb_l.save(path_save_gt_dep_rgb_l)

                if 'IR' in self.method:
                    ir_l = self._process_image(sample['ir_l'], True, 1.0)
                    ir_r = self._process_image(sample['ir_r'], True, 1.0)
                    dep_ir_l = self._process_image(sample['dep_ir_l'], False, self.max_depth)
                    gt_dep_ir_l = self._process_image(sample['gt_dep_ir_l'], False, self.max_depth)

                    ir_l = self._to_image(ir_l, False)
                    ir_r = self._to_image(ir_r, False)
                    dep_ir_l = self._to_image(dep_ir_l, True, self.max_depth, cmap=cm_w)
                    gt_dep_ir_l = self._to_image(gt_dep_ir_l, True, self.max_depth, cmap=cm_w)

                    path_save_ir_l = '{}/02_1_ir_l.png'.format(self.path_output)
                    path_save_ir_r = '{}/02_2_ir_r.png'.format(self.path_output)
                    path_save_dep_ir_l = '{}/02_3_dep_ir_l.png'.format(self.path_output)
                    path_save_gt_dep_ir_l = '{}/02_4_gt_dep_ir_l.png'.format(self.path_output)

                    ir_l.save(path_save_ir_l)
                    ir_r.save(path_save_ir_r)
                    dep_ir_l.save(path_save_dep_ir_l)
                    gt_dep_ir_l.save(path_save_gt_dep_ir_l)

                if self.individual_supervision and 'RGB' in self.method:
                    pred_dep_rgb = self._process_image(output['pred_dep3_rgb'], False, self.max_depth)
                    pred_dep_rgb = self._to_image(pred_dep_rgb, True, self.max_depth)
                    path_save_pred_dep_rgb = '{}/03_1_pred_dep_rgb.png'.format(self.path_output)

                    pred_dep_rgb.save(path_save_pred_dep_rgb)

                if self.individual_supervision and 'IR' in self.method:
                    pred_dep_ir = self._process_image(output['pred_dep3_ir'], False, self.max_depth)
                    pred_dep_ir = self._to_image(pred_dep_ir, True, self.max_depth)
                    path_save_pred_dep_ir = '{}/04_1_pred_dep_ir.png'.format(self.path_output)

                    pred_dep_ir.save(path_save_pred_dep_ir)

                if self.individual_supervision and 'LIDAR' in self.method:
                    pred_dep_lidar = self._process_image(output['pred_dep3_lidar'], False, self.max_depth)
                    pred_dep_lidar = self._to_image(pred_dep_lidar, True, self.max_depth)
                    path_save_pred_dep_lidar = '{}/05_1_pred_dep_lidar.png'.format(self.path_output)

                    pred_dep_lidar.save(path_save_pred_dep_lidar)

                pred_dep = self._process_image(output['pred_dep3'], False, self.max_depth)
                pred_dep = self._to_image(pred_dep, True, self.max_depth)
                path_save_pred_dep = '{}/06_1_pred_dep.png'.format(self.path_output)

                pred_dep.save(path_save_pred_dep)
