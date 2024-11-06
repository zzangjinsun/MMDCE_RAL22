"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    MMDNetMetric implementation
"""


import torch
from . import BaseMetric


class MMDNetMetric(BaseMetric):
    def __init__(self, args):
        super(MMDNetMetric, self).__init__(args)

        self.args = args
        self.t_valid = 0.0001
        self.method = self.args.method

        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE'
        ]

    def evaluate(self, sample, output, mode):
        with torch.no_grad():
            # RGB is the reference
            gt = sample['gt_dep_rgb_l'].detach()
            gt_disp = sample['gt_disp_rgb_l'].detach()

            pred = output['pred_dep3'].detach()
            pred_disp = output['pred_disp3'].detach()

            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # For numerical stability
            mask = gt > self.t_valid
            num_valid = mask.sum()

            pred = pred[mask]
            gt = gt[mask]

            pred_inv = pred_inv[mask]
            gt_inv = gt_inv[mask]

            pred_inv[pred <= self.t_valid] = 0.0
            gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            result = [rmse, mae, irmse, imae]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result
