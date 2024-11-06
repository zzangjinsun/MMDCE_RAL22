"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    MMDNetLoss implementation
"""


from . import BaseLoss
import torch
import sys
sys.path.insert(0, '..')
from utility import warp_image


class MMDNetLoss(BaseLoss):
    def __init__(self, args):
        super(MMDNetLoss, self).__init__(args)

        self.loss_name = []
        self.method = args.method
        self.individual_supervision = args.individual_supervision
        self.w_ind_sup = args.individual_supervision_weight

        self.num_disp = args.num_disp
        self.max_disp = self.num_disp - 1
        self.max_depth = args.max_depth

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output):
        loss_val = []

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            # Reference is RGB
            rgb_l = sample['rgb_l']
            rgb_r = sample['rgb_r']
            gt_dep_rgb_l = sample['gt_dep_rgb_l']
            gt_disp_rgb_l = sample['gt_disp_rgb_l']

            loss_tmp = torch.zeros(1).type_as(rgb_l).squeeze()

            if self.individual_supervision and 'RGB' in self.method:
                pred_dep1_rgb = output['pred_dep1_rgb']
                pred_dep2_rgb = output['pred_dep2_rgb']
                pred_dep3_rgb = output['pred_dep3_rgb']

                pred_dep1_rgb = torch.clamp(pred_dep1_rgb, min=0, max=self.max_depth)
                pred_dep2_rgb = torch.clamp(pred_dep2_rgb, min=0, max=self.max_depth)
                pred_dep3_rgb = torch.clamp(pred_dep3_rgb, min=0, max=self.max_depth)

                pred_disp1_rgb = output['pred_disp1_rgb']
                pred_disp2_rgb = output['pred_disp2_rgb']
                pred_disp3_rgb = output['pred_disp3_rgb']

                pred_disp1_rgb = torch.clamp(pred_disp1_rgb, min=0, max=self.max_disp)
                pred_disp2_rgb = torch.clamp(pred_disp2_rgb, min=0, max=self.max_disp)
                pred_disp3_rgb = torch.clamp(pred_disp3_rgb, min=0, max=self.max_disp)

                cost_out_rgb = output['cost_out_rgb']
                dc_gt_rgb = output['dc_gt_rgb']

            if self.individual_supervision and 'IR' in self.method:
                ir_l = sample['ir_l']
                ir_r = sample['ir_r']
                gt_dep_ir_l = sample['gt_dep_ir_l']
                gt_disp_ir_l = sample['gt_disp_ir_l']

                pred_dep1_ir = output['pred_dep1_ir']
                pred_dep2_ir = output['pred_dep2_ir']
                pred_dep3_ir = output['pred_dep3_ir']

                pred_dep1_ir = torch.clamp(pred_dep1_ir, min=0, max=self.max_depth)
                pred_dep2_ir = torch.clamp(pred_dep2_ir, min=0, max=self.max_depth)
                pred_dep3_ir = torch.clamp(pred_dep3_ir, min=0, max=self.max_depth)

                pred_disp1_ir = output['pred_disp1_ir']
                pred_disp2_ir = output['pred_disp2_ir']
                pred_disp3_ir = output['pred_disp3_ir']

                pred_disp1_ir = torch.clamp(pred_disp1_ir, min=0, max=self.max_disp)
                pred_disp2_ir = torch.clamp(pred_disp2_ir, min=0, max=self.max_disp)
                pred_disp3_ir = torch.clamp(pred_disp3_ir, min=0, max=self.max_disp)

                cost_out_ir = output['cost_out_ir']
                dc_gt_ir = output['dc_gt_ir']

            if self.individual_supervision and 'LIDAR' in self.method:
                pred_dep1_lidar = output['pred_dep1_lidar']
                pred_dep2_lidar = output['pred_dep2_lidar']
                pred_dep3_lidar = output['pred_dep3_lidar']

                pred_dep1_lidar = torch.clamp(pred_dep1_lidar, min=0, max=self.max_depth)
                pred_dep2_lidar = torch.clamp(pred_dep2_lidar, min=0, max=self.max_depth)
                pred_dep3_lidar = torch.clamp(pred_dep3_lidar, min=0, max=self.max_depth)

                pred_disp1_lidar = output['pred_disp1_lidar']
                pred_disp2_lidar = output['pred_disp2_lidar']
                pred_disp3_lidar = output['pred_disp3_lidar']

                pred_disp1_lidar = torch.clamp(pred_disp1_lidar, min=0, max=self.max_disp)
                pred_disp2_lidar = torch.clamp(pred_disp2_lidar, min=0, max=self.max_disp)
                pred_disp3_lidar = torch.clamp(pred_disp3_lidar, min=0, max=self.max_disp)

                cost_out_lidar = output['cost_out_lidar']
                dc_gt_lidar = output['dc_gt_lidar']

            # Final output
            pred_dep1 = output['pred_dep1']
            pred_dep2 = output['pred_dep2']
            pred_dep3 = output['pred_dep3']

            pred_dep1 = torch.clamp(pred_dep1, min=0, max=self.max_depth)
            pred_dep2 = torch.clamp(pred_dep2, min=0, max=self.max_depth)
            pred_dep3 = torch.clamp(pred_dep3, min=0, max=self.max_depth)

            pred_disp1 = output['pred_disp1']
            pred_disp2 = output['pred_disp2']
            pred_disp3 = output['pred_disp3']

            pred_disp1 = torch.clamp(pred_disp1, min=0, max=self.max_disp)
            pred_disp2 = torch.clamp(pred_disp2, min=0, max=self.max_disp)
            pred_disp3 = torch.clamp(pred_disp3, min=0, max=self.max_disp)

            cost_out = output['cost_out']
            dc_gt = output['dc_gt']

            if loss_type in ['L1', 'L2']:
                if self.individual_supervision and 'RGB' in self.method:
                    loss_tmp += self.w_ind_sup * (0.5 * loss_func(pred_dep1_rgb, gt_dep_rgb_l)
                                 + 0.7 * loss_func(pred_dep2_rgb, gt_dep_rgb_l)
                                 + 1.0 * loss_func(pred_dep3_rgb, gt_dep_rgb_l)) / 2.2

                if self.individual_supervision and 'IR' in self.method:
                    loss_tmp += self.w_ind_sup * (0.5 * loss_func(pred_dep1_ir, gt_dep_ir_l)
                                 + 0.7 * loss_func(pred_dep2_ir, gt_dep_ir_l)
                                 + 1.0 * loss_func(pred_dep3_ir, gt_dep_ir_l)) / 2.2

                if self.individual_supervision and 'LIDAR' in self.method:
                    loss_tmp += self.w_ind_sup * (0.5 * loss_func(pred_dep1_lidar, gt_dep_rgb_l)
                                 + 0.7 * loss_func(pred_dep2_lidar, gt_dep_rgb_l)
                                 + 1.0 * loss_func(pred_dep3_lidar, gt_dep_rgb_l)) / 2.2

                loss_tmp += (0.5 * loss_func(pred_dep1, gt_dep_rgb_l)
                             + 0.7 * loss_func(pred_dep2, gt_dep_rgb_l)
                             + 1.0 * loss_func(pred_dep3, gt_dep_rgb_l)) / 2.2
            else:
                raise NotImplementedError

            loss_tmp = loss['weight']*loss_tmp
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val
