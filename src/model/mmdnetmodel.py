"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    MMDNetModel implementation
"""


from .common import *
import numpy as np

from .AANet.refinement import StereoDRNetRefinement


class ImageFeatNet(nn.Module):
    def __init__(self, args):
        super(ImageFeatNet, self).__init__()

        self.args = args
        self.method = args.method

        self.num_disp = args.num_disp
        self.max_disp = self.num_disp - 1
        self.max_depth = args.max_depth

        # Network
        if args.network == 'resnet18':
            layers = [2, 2, 2]
            ch = [64, 128, 256]
        elif args.network == 'resnet34':
            layers = [3, 4, 6]
            ch = [128, 256, 512]
        else:
            raise NotImplementedError

        # 1/3
        self.conv1_img = conv2d_bn_relu(3, 64, kernel=7, stride=3, padding=3,
                                            bn=True, relu=True)

        self.layer1 = stack_blocks(ResBlock, layers[0], 64, ch[0],
                                   kernel=3, stride=1)

        # 1/6
        self.layer2 = stack_blocks(ResBlock, layers[1], ch[0], ch[1],
                                   kernel=3, stride=2)
        # 1/12
        self.layer3 = stack_blocks(ResBlock, layers[2], ch[1], ch[2],
                                   kernel=3, stride=2)

        # Lateral convs
        self.lconv1 = nn.Conv2d(ch[0], 128, kernel_size=1)
        self.lconv2 = nn.Conv2d(ch[1], 128, kernel_size=1)
        self.lconv3 = nn.Conv2d(ch[2], 128, kernel_size=1)

        # FPN convs
        self.fpn1 = conv2d_bn_relu(128, 128, kernel=3, stride=1, padding=1,
                                   bn=True, relu=True)
        self.fpn2 = conv2d_bn_relu(128, 128, kernel=3, stride=1, padding=1,
                                   bn=True, relu=True)
        self.fpn3 = conv2d_bn_relu(128, 128, kernel=3, stride=1, padding=1,
                                   bn=True, relu=True)

    def forward(self, img):
        f_img = self.conv1_img(img)

        y1 = self.layer1(f_img)

        # Encoding
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)

        # Lateral convs
        l1 = self.lconv1(y1)
        l2 = self.lconv2(y2)
        l3 = self.lconv3(y3)

        # Top-Down fusion
        l2 = l2 + F.interpolate(l3, scale_factor=2, mode='nearest')
        l1 = l1 + F.interpolate(l2, scale_factor=2, mode='nearest')

        # FPN convs
        out1 = self.fpn1(l1)
        out2 = self.fpn2(l2)
        out3 = self.fpn3(l3)

        # 1/3, 1/6, 1/12
        return [out1, out2, out3]


class CostVolumeDispRange(nn.Module):
    def __init__(self, args):
        super(CostVolumeDispRange, self).__init__()

        self.args = args

        self.theta = torch.from_numpy(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])).view(1, 2, 3)

    def _warp_feature(self, feat, dx):
        B, C, H, W = feat.shape

        theta = torch.repeat_interleave(self.theta.type_as(feat), B, dim=0)

        grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)

        # Calculate warped x coordinates
        grid[:, :, :, 0] = grid[:, :, :, 0] - 2.0 * dx.view(-1, 1, 1) / W

        feat_w = F.grid_sample(feat, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=False)

        return feat_w

    def _construct_cost(self, f_left, f_right, disp_range):
        b, c, h, w = f_left.size()
        _, num_disp = disp_range.shape

        cost_volume = f_left.new_zeros(b, num_disp, h, w)

        for i in range(num_disp):
            dx = disp_range[:, i]

            f_right_w = self._warp_feature(f_right, dx)

            cost_volume[:, i, :, :] = (f_left * f_right_w).mean(dim=1)

        cost_volume = cost_volume.contiguous()

        return cost_volume

    def forward(self, list_f_left, list_f_right, disp_range):
        assert len(list_f_left) == len(list_f_right)

        num_scales = len(list_f_left)

        # 1/3, 1/6, 1/12
        cost_volume_pyramid = []

        for s in range(num_scales):
            step = 2 ** s
            scale = 3.0 * step
            disp_range_tmp = disp_range[:, ::step] / scale

            cost_volume = self._construct_cost(
                list_f_left[s], list_f_right[s], disp_range_tmp
            )
            cost_volume_pyramid.append(cost_volume)

        return cost_volume_pyramid


class IntraAggregationLayer(nn.Module):
    def __init__(self, kernel=3):
        super(IntraAggregationLayer, self).__init__()

        assert (kernel - 1) % 2 == 0, 'Only odd kernel is supported'

        pad = int(((kernel - 1) / 2))

        self.conv1 = conv3d_bn_relu(1, 1, kernel=kernel, stride=1, padding=pad,
                                    bn=True, relu=True)
        self.conv2 = conv3d_bn_relu(1, 1, kernel=kernel, stride=1, padding=pad,
                                    bn=True, relu=False)

    def forward(self, x):
        assert x.dim() == 4

        x = x.unsqueeze(dim=1)

        y = self.conv1(x)
        y = self.conv2(y)
        y = y + x

        y = y.squeeze(dim=1)

        return y


class InterAggregationLayer(nn.Module):
    def __init__(self, scale_factor, kernel=3):
        super(InterAggregationLayer, self).__init__()

        assert scale_factor in [1/4, 1/2, 2, 4]

        assert (kernel - 1) % 2 == 0, 'Only odd kernel is supported'

        pad = int(((kernel - 1) / 2))

        self.scale_factor = scale_factor

        if scale_factor == 1/4:
            conv1 = conv3d_bn_relu(1, 1, kernel=kernel, stride=2, padding=pad,
                                   bn=True, relu=True)
            conv2 = conv3d_bn_relu(1, 1, kernel=kernel, stride=2, padding=pad,
                                   bn=True, relu=False)
        elif scale_factor == 1/2:
            conv1 = conv3d_bn_relu(1, 1, kernel=kernel, stride=2, padding=pad,
                                   bn=True, relu=True)
            conv2 = conv3d_bn_relu(1, 1, kernel=kernel, stride=1, padding=pad,
                                   bn=True, relu=False)
        elif scale_factor == 2:
            conv1 = convt3d_bn_relu(1, 1, kernel=kernel, stride=2, padding=pad,
                                    output_padding=1, bn=True, relu=True)
            conv2 = conv3d_bn_relu(1, 1, kernel=kernel, stride=1, padding=pad,
                                   bn=True, relu=False)
        elif scale_factor == 4:
            conv1 = convt3d_bn_relu(1, 1, kernel=kernel, stride=2, padding=pad,
                                    output_padding=1, bn=True, relu=True)
            conv2 = convt3d_bn_relu(1, 1, kernel=kernel, stride=2, padding=pad,
                                    output_padding=1, bn=True, relu=False)
        else:
            raise NotImplementedError

        self.conv = nn.Sequential(conv1, conv2)

    def forward(self, x):
        assert x.dim() == 4

        x = x.unsqueeze(dim=1)

        y = self.conv(x)

        y = y.squeeze(dim=1)

        return y


class AggregationNet(nn.Module):
    def __init__(self, args):
        super(AggregationNet, self).__init__()

        self.args = args

        self.num_disp = args.num_disp
        self.max_disp = self.num_disp - 1
        self.max_depth = args.max_depth

        kernel = args.aggregation_kernel

        # 1/3, 1/6, 1/12
        self.intra1 = IntraAggregationLayer(kernel=kernel)
        self.intra2 = IntraAggregationLayer(kernel=kernel)
        self.intra3 = IntraAggregationLayer(kernel=kernel)

        self.inter21 = InterAggregationLayer(2, kernel=kernel)
        self.inter31 = InterAggregationLayer(4, kernel=kernel)

    def forward(self, cost):
        # 1/3, 1/6, 1/12
        cost1 = self.intra1(cost[0])
        cost2 = self.intra2(cost[1])
        cost3 = self.intra3(cost[2])

        cost1_agg = cost1 + self.inter21(cost2) + self.inter31(cost3)

        # 1/3
        return cost1_agg


class RefinementNet(nn.Module):
    def __init__(self, args):
        super(RefinementNet, self).__init__()

        self.args = args

        self.num_disp = args.num_disp
        self.max_disp = self.num_disp - 1
        self.max_depth = args.max_depth

        self.ref1 = StereoDRNetRefinement()
        self.ref2 = StereoDRNetRefinement()

    def forward(self, img_l, img_r, disp):
        img_l_tmp = F.interpolate(img_l, scale_factor=0.5, mode='bilinear',
                                  align_corners=False,
                                  recompute_scale_factor=True)
        img_r_tmp = F.interpolate(img_r, scale_factor=0.5, mode='bilinear',
                                  align_corners=False,
                                  recompute_scale_factor=True)

        disp1 = self.ref1(disp, img_l_tmp, img_r_tmp)
        disp2 = self.ref2(disp1, img_l, img_r)

        # 1/2, 1/1
        return [disp1, disp2]


class MMDNetModel(nn.Module):
    def __init__(self, args):
        super(MMDNetModel, self).__init__()

        self.args = args
        self.method = args.method

        list_method = self.method.split('-')

        for modality in list_method:
            assert modality in ['RGB', 'IR', 'LIDAR'], \
                'Unrecognized modality : {}'.format(modality)

        self.num_disp = args.num_disp
        self.max_disp = self.num_disp - 1
        self.max_depth = args.max_depth

        self.num_scales = 3

        if 'RGB' in self.method or 'IR' in self.method:
            if 'RGB' in self.method:
                self.rgb_net = ImageFeatNet(args)
            if 'IR' in self.method:
                self.ir_net = ImageFeatNet(args)

            self.cost_constructor = CostVolumeDispRange(args)

        self.dc_gen = DCGenerator(args)

        self.aggregation_net = AggregationNet(args)
        self.refinement_net = RefinementNet(args)

    def _preprocess(self, img, factor=12):
        _, C, H, W = img.shape

        h_pad = 0
        w_pad = 0

        # Pad if necessary
        if (H % factor) != 0:
            H_new = factor * np.ceil(H / factor)
            h_pad = int(H_new - H)
        if (W % factor) != 0:
            W_new = factor * np.ceil(W / factor)
            w_pad = int(W_new - W)

        if h_pad != 0 or w_pad != 0:
            # Pad : [Left, Right, Top, Bottom]
            img = F.pad(img, [0, w_pad, h_pad, 0])

        return img, h_pad, w_pad

    def _postprocess(self, img, h_pad, w_pad):
        if h_pad != 0:
            img = img[:, :, h_pad:, :]

        if w_pad != 0:
            img = img[:, :, :, :-w_pad]

        return img

    def _depth_disp_convert(self, val, calib, baseline):
        B = val.shape[0]

        f = calib[:, 0].view(B, 1, 1, 1)
        b = baseline.view(B, 1, 1, 1)

        out = f * b / (val + 1e-8)

        return out

    def _forward_stereo_cost(self, img_l, img_r, disp_range, mode):
        if mode == 'RGB':
            feat_l = self.rgb_net(img_l)
            feat_r = self.rgb_net(img_r)
        elif mode == 'IR':
            feat_l = self.ir_net(img_l)
            feat_r = self.ir_net(img_r)
        else:
            raise NotImplementedError

        cost = self.cost_constructor(feat_l, feat_r, disp_range)

        return cost, feat_l, feat_r

    def _forward_aggregation_and_estimation(self, img_l, img_r, cost,
                                            disp_range, K, B):
        batch, _, H, W = img_l.shape

        num_disp = disp_range.shape[1]

        # 1/3
        aggregation = self.aggregation_net(cost)

        _, ca, ha, wa = aggregation.shape

        if ca != num_disp:
            aggregation = F.interpolate(
                aggregation.unsqueeze(dim=1), (num_disp, ha, wa),
                mode='trilinear', align_corners=False).squeeze(dim=1)

        coeff = F.softmax(aggregation, dim=1)

        # NOTE : Scale and order should be correct
        disp_range = disp_range.view(batch, -1, 1, 1)

        disp0 = torch.sum(disp_range * coeff / 3.0,
                          dim=1, keepdim=True)

        # 1/3
        disparity_pyramid = [disp0]

        # 1/3, 1/2, 1/1
        disparity_pyramid += self.refinement_net(img_l, img_r, disp0)

        pred_disp1 = 3.0 * F.interpolate(disparity_pyramid[0], (H, W),
                                         mode='bilinear', align_corners=False)
        pred_disp2 = 2.0 * F.interpolate(disparity_pyramid[1], (H, W),
                                         mode='bilinear', align_corners=False)
        pred_disp3 = disparity_pyramid[2]

        pred_disp1 = torch.clamp(pred_disp1, min=0, max=self.max_disp)
        pred_disp2 = torch.clamp(pred_disp2, min=0, max=self.max_disp)
        pred_disp3 = torch.clamp(pred_disp3, min=0, max=self.max_disp)

        pred_dep1 = self._depth_disp_convert(pred_disp1, K, B)
        pred_dep2 = self._depth_disp_convert(pred_disp2, K, B)
        pred_dep3 = self._depth_disp_convert(pred_disp3, K, B)

        pred_dep1 = torch.clamp(pred_dep1, min=0, max=self.max_depth)
        pred_dep2 = torch.clamp(pred_dep2, min=0, max=self.max_depth)
        pred_dep3 = torch.clamp(pred_dep3, min=0, max=self.max_depth)

        # Upsample unnormalized cost for output
        cost_out = F.interpolate(aggregation, (H, W), mode='bilinear', align_corners=False)

        return pred_dep1, pred_dep2, pred_dep3,\
               pred_disp1, pred_disp2, pred_disp3, cost_out

    def _forward_cost_conf(self, cost_rgb, cost_ir, cost_lidar, net_cost):
        B, C, H, W = cost_rgb.shape

        conf_rgb_tmp = torch.zeros((B, 1, H, W)).type_as(cost_rgb).detach()
        conf_ir_tmp = torch.zeros((B, 1, H, W)).type_as(cost_ir).detach()
        conf_lidar_tmp = torch.zeros((B, 1, H, W)).type_as(cost_lidar).detach()

        list_conf_valid = []
        list_name_valid = []
        if 'RGB' in self.method:
            conf_rgb_tmp = net_cost(cost_rgb)
            list_conf_valid.append(conf_rgb_tmp)
            list_name_valid.append('RGB')
        if 'IR' in self.method:
            conf_ir_tmp = net_cost(cost_ir)
            list_conf_valid.append(conf_ir_tmp)
            list_name_valid.append('IR')
        if 'LIDAR' in self.method:
            conf_lidar_tmp = net_cost(cost_lidar)
            list_conf_valid.append(conf_lidar_tmp)
            list_name_valid.append('LIDAR')

        # Softmax
        conf_tmp = torch.cat(list_conf_valid, dim=1)
        conf_tmp = F.softmax(conf_tmp, dim=1)
        conf_tmp_sm = torch.chunk(conf_tmp, len(list_conf_valid), dim=1)

        for idx, val in enumerate(conf_tmp_sm):
            if list_name_valid[idx] == 'RGB':
                conf_rgb_tmp = val
            if list_name_valid[idx] == 'IR':
                conf_ir_tmp = val
            if list_name_valid[idx] == 'LIDAR':
                conf_lidar_tmp = val

        return conf_rgb_tmp, conf_ir_tmp, conf_lidar_tmp

    def forward(self, sample):
        output = {}

        # RGB is the reference
        rgb_l = sample['rgb_l']
        rgb_r = sample['rgb_r']
        disp_range_rgb = sample['disp_range_rgb']
        dep_range_rgb = sample['dep_range_rgb']
        K_rgb = sample['K_rgb']
        B_rgb = sample['B_rgb']

        rgb_l, h_rgb, w_rgb = self._preprocess(rgb_l, 12)
        rgb_r, _, _ = self._preprocess(rgb_r, 12)

        batch, _, H_rgb, W_rgb = rgb_l.shape

        if h_rgb > 0:
            K_rgb[:, 3] = K_rgb[:, 3] + h_rgb

        if 'RGB' in self.method:
            cost_rgb, feat_rgb_l, feat_rgb_r = \
                self._forward_stereo_cost(
                    rgb_l, rgb_r, disp_range_rgb, 'RGB'
                )

            if self.args.individual_supervision:
                pred_dep1_rgb, pred_dep2_rgb, pred_dep3_rgb, \
                pred_disp1_rgb, pred_disp2_rgb, pred_disp3_rgb, cost_out_rgb = \
                    self._forward_aggregation_and_estimation(
                        rgb_l, rgb_r, cost_rgb, disp_range_rgb, K_rgb, B_rgb
                    )

                # Remove paddings
                pred_disp1_rgb = self._postprocess(pred_disp1_rgb, h_rgb, w_rgb)
                pred_disp2_rgb = self._postprocess(pred_disp2_rgb, h_rgb, w_rgb)
                pred_disp3_rgb = self._postprocess(pred_disp3_rgb, h_rgb, w_rgb)

                pred_dep1_rgb = self._postprocess(pred_dep1_rgb, h_rgb, w_rgb)
                pred_dep2_rgb = self._postprocess(pred_dep2_rgb, h_rgb, w_rgb)
                pred_dep3_rgb = self._postprocess(pred_dep3_rgb, h_rgb, w_rgb)

                cost_out_rgb = self._postprocess(cost_out_rgb, h_rgb, w_rgb)

                dc_gt_rgb = self.dc_gen.encode(sample['gt_disp_rgb_l'], disp_range_rgb)

                output['pred_disp1_rgb'] = pred_disp1_rgb
                output['pred_disp2_rgb'] = pred_disp2_rgb
                output['pred_disp3_rgb'] = pred_disp3_rgb
                output['pred_dep1_rgb'] = pred_dep1_rgb
                output['pred_dep2_rgb'] = pred_dep2_rgb
                output['pred_dep3_rgb'] = pred_dep3_rgb
                output['cost_out_rgb'] = cost_out_rgb
                output['dc_gt_rgb'] = dc_gt_rgb.detach()

        if 'IR' in self.method:
            ir_l = sample['ir_l']
            ir_r = sample['ir_r']
            disp_range_ir = sample['disp_range_ir']
            K_ir = sample['K_ir']
            B_ir = sample['B_ir']

            ir_l, h_ir, w_ir = self._preprocess(ir_l, 12)
            ir_r, _, _ = self._preprocess(ir_r, 12)

            _, _, H_ir, W_ir = ir_l.shape

            if h_ir > 0:
                K_ir[:, 3] = K_ir[:, 3] + h_ir

            cost_ir, feat_ir_l, feat_ir_r = \
                self._forward_stereo_cost(
                    ir_l, ir_r, disp_range_ir, 'IR'
                )

            if self.args.individual_supervision:
                pred_dep1_ir, pred_dep2_ir, pred_dep3_ir, \
                pred_disp1_ir, pred_disp2_ir, pred_disp3_ir, cost_out_ir = \
                    self._forward_aggregation_and_estimation(
                        ir_l, ir_r, cost_ir, disp_range_ir, K_ir, B_ir
                    )

                # Remove paddings
                pred_disp1_ir = self._postprocess(pred_disp1_ir, h_ir, w_ir)
                pred_disp2_ir = self._postprocess(pred_disp2_ir, h_ir, w_ir)
                pred_disp3_ir = self._postprocess(pred_disp3_ir, h_ir, w_ir)

                pred_dep1_ir = self._postprocess(pred_dep1_ir, h_ir, w_ir)
                pred_dep2_ir = self._postprocess(pred_dep2_ir, h_ir, w_ir)
                pred_dep3_ir = self._postprocess(pred_dep3_ir, h_ir, w_ir)

                cost_out_ir = self._postprocess(cost_out_ir, h_rgb, w_rgb)

                dc_gt_ir = self.dc_gen.encode(sample['gt_disp_ir_l'], disp_range_ir)

                output['pred_disp1_ir'] = pred_disp1_ir
                output['pred_disp2_ir'] = pred_disp2_ir
                output['pred_disp3_ir'] = pred_disp3_ir
                output['pred_dep1_ir'] = pred_dep1_ir
                output['pred_dep2_ir'] = pred_dep2_ir
                output['pred_dep3_ir'] = pred_dep3_ir
                output['cost_out_ir'] = cost_out_ir
                output['dc_gt_ir'] = dc_gt_ir.detach()

        if 'LIDAR' in self.method:
            disp = sample['disp_rgb_l']

            disp, h_lidar, w_lidar = self._preprocess(disp, 12)

            _, _, H_lidar, W_lidar = disp.shape

            # 1/3, 1/6, 1/12
            disp_3 = F.max_pool2d(disp, kernel_size=3, stride=3)
            disp_6 = F.max_pool2d(disp, kernel_size=6, stride=6)
            disp_12 = F.max_pool2d(disp, kernel_size=12, stride=12)

            dc_3 = self.dc_gen.encode(disp_3, disp_range_rgb)
            dc_6 = self.dc_gen.encode(disp_6, disp_range_rgb[:, ::2])
            dc_12 = self.dc_gen.encode(disp_12, disp_range_rgb[:, ::4])

            cost_lidar = [dc_3, dc_6, dc_12]

            if self.args.individual_supervision:
                pred_dep1_lidar, pred_dep2_lidar, pred_dep3_lidar, \
                pred_disp1_lidar, pred_disp2_lidar, pred_disp3_lidar, cost_out_lidar = \
                    self._forward_aggregation_and_estimation(
                        rgb_l, rgb_r, cost_lidar, disp_range_rgb, K_rgb, B_rgb
                    )

                # Remove paddings
                pred_disp1_lidar = self._postprocess(pred_disp1_lidar, h_lidar, w_lidar)
                pred_disp2_lidar = self._postprocess(pred_disp2_lidar, h_lidar, w_lidar)
                pred_disp3_lidar = self._postprocess(pred_disp3_lidar, h_lidar, w_lidar)

                pred_dep1_lidar = self._postprocess(pred_dep1_lidar, h_lidar, w_lidar)
                pred_dep2_lidar = self._postprocess(pred_dep2_lidar, h_lidar, w_lidar)
                pred_dep3_lidar = self._postprocess(pred_dep3_lidar, h_lidar, w_lidar)

                cost_out_lidar = self._postprocess(cost_out_lidar, h_lidar, w_lidar)

                dc_gt_lidar = self.dc_gen.encode(sample['gt_disp_rgb_l'], disp_range_rgb)

                output['pred_disp1_lidar'] = pred_disp1_lidar
                output['pred_disp2_lidar'] = pred_disp2_lidar
                output['pred_disp3_lidar'] = pred_disp3_lidar
                output['pred_dep1_lidar'] = pred_dep1_lidar
                output['pred_dep2_lidar'] = pred_dep2_lidar
                output['pred_dep3_lidar'] = pred_dep3_lidar
                output['cost_out_lidar'] = cost_out_lidar
                output['dc_gt_lidar'] = dc_gt_lidar.detach()

        # Prepare cost fusion3
        if 'RGB' not in self.method:
            # Dummy cost volume if not exist
            num_dep = disp_range_rgb.shape[1]

            cost0_rgb = torch.zeros((batch, num_dep, H_rgb // 3, W_rgb // 3)).type_as(rgb_l)
            cost1_rgb = torch.zeros((batch, num_dep // 2, H_rgb // 6, W_rgb // 6)).type_as(rgb_l)
            cost2_rgb = torch.zeros((batch, num_dep // 4, H_rgb // 12, W_rgb // 12)).type_as(rgb_l)

            cost_rgb = [cost0_rgb, cost1_rgb, cost2_rgb]
            mask_rgb = [torch.zeros_like(cost0_rgb).detach(),
                        torch.zeros_like(cost1_rgb).detach(),
                        torch.zeros_like(cost2_rgb).detach()]
        else:
            mask_rgb = [torch.ones_like(cost_rgb[0]).detach(),
                        torch.ones_like(cost_rgb[1]).detach(),
                        torch.ones_like(cost_rgb[2]).detach()]

        # Warp IR cost to RGB coordinates
        if 'IR' not in self.method:
            # Dummy cost volume if not exist
            num_dep = disp_range_rgb.shape[1]

            # NOTE : assume warped cost whose size is the same as RGB size
            cost0_ir = torch.zeros((batch, num_dep, H_rgb // 3, W_rgb // 3)).type_as(rgb_l)
            cost1_ir = torch.zeros((batch, num_dep // 2, H_rgb // 6, W_rgb // 6)).type_as(rgb_l)
            cost2_ir = torch.zeros((batch, num_dep // 4, H_rgb // 12, W_rgb // 12)).type_as(rgb_l)

            cost_ir_warped = [cost0_ir, cost1_ir, cost2_ir]
            mask_ir_warped = [torch.zeros_like(cost0_ir).detach(),
                              torch.zeros_like(cost1_ir).detach(),
                              torch.zeros_like(cost2_ir).detach()]
        else:
            assert len(cost_rgb) == len(cost_ir)

            R_ir2rgb = sample['R_ir2rgb'].type_as(rgb_l)
            T_ir2rgb = sample['T_ir2rgb'].type_as(rgb_l)

            R_rgb2ir = torch.inverse(R_ir2rgb)

            fx_rgb = K_rgb[:, 0].view(-1, 1, 1)
            fy_rgb = K_rgb[:, 1].view(-1, 1, 1)
            cx_rgb = K_rgb[:, 2].view(-1, 1, 1)
            cy_rgb = K_rgb[:, 3].view(-1, 1, 1)

            fx_ir = K_ir[:, 0].view(-1, 1, 1)
            fy_ir = K_ir[:, 1].view(-1, 1, 1)
            cx_ir = K_ir[:, 2].view(-1, 1, 1)
            cy_ir = K_ir[:, 3].view(-1, 1, 1)

            cost_ir_warped = []
            mask_ir_warped = []

            # Warp IR cost and fuse to RGB cost (RGB is the reference)
            for s in range(0, self.num_scales):
                cost_rgb_tmp = cost_rgb[s]
                cost_ir_tmp = cost_ir[s]

                cost_ir_warped_tmp = torch.zeros_like(cost_rgb_tmp)
                mask_ir_warped_tmp = torch.zeros_like(cost_rgb_tmp)

                step = 2 ** s

                dep_range_rgb_tmp = dep_range_rgb[:, ::step]

                num_dep = dep_range_rgb_tmp.shape[1]

                _, C_rgb_tmp, H_rgb_tmp, W_rgb_tmp = cost_rgb_tmp.shape
                _, C_ir_tmp, H_ir_tmp, W_ir_tmp = cost_ir_tmp.shape

                assert num_dep == C_rgb_tmp and num_dep == C_ir_tmp

                # NOTE : coordinates are calculated in the original scale
                yy, xx = torch.meshgrid(
                    torch.linspace(0, H_rgb - 1, H_rgb_tmp),
                    torch.linspace(0, W_rgb - 1, W_rgb_tmp),
                    indexing='ij'
                )
                xx = xx.type_as(cost_rgb_tmp).view(-1, 1, H_rgb_tmp*W_rgb_tmp)
                yy = yy.type_as(cost_rgb_tmp).view(-1, 1, H_rgb_tmp*W_rgb_tmp)

                for d in range(0, num_dep):
                    dep_val_rgb_tmp = dep_range_rgb_tmp[:, d].view(-1, 1, 1)

                    # 3D in RGB coordinates
                    X_rgb = dep_val_rgb_tmp * (xx - cx_rgb) / fx_rgb
                    Y_rgb = dep_val_rgb_tmp * (yy - cy_rgb) / fy_rgb
                    Z_rgb = dep_val_rgb_tmp * torch.ones_like(X_rgb)

                    P_rgb = torch.cat((X_rgb, Y_rgb, Z_rgb), dim=1)

                    # 3D in IR coordinates
                    P_ir = torch.bmm(R_rgb2ir, P_rgb - T_ir2rgb.unsqueeze(dim=2))

                    X_ir, Y_ir, Z_ir = \
                        torch.split(P_ir, split_size_or_sections=1, dim=1)

                    x_ir = fx_ir * X_ir / (Z_ir + 1e-8) + cx_ir
                    y_ir = fy_ir * Y_ir / (Z_ir + 1e-8) + cy_ir

                    grid_x = (2.0 * x_ir) / (W_ir - 1.0) - 1.0
                    grid_y = (2.0 * y_ir) / (H_ir - 1.0) - 1.0

                    # NOTE : size fits to RGB cost
                    grid_ir = torch.cat(
                        (grid_x.view(-1, H_rgb_tmp, W_rgb_tmp, 1),
                         grid_y.view(-1, H_rgb_tmp, W_rgb_tmp, 1)),
                        dim=3
                    )

                    cost_ir_warped_tmp[:, [d], :, :] = F.grid_sample(
                        cost_ir_tmp[:, [d], :, :], grid_ir, mode='bilinear',
                        padding_mode='zeros', align_corners=False
                    )

                    mask_ir_warped_tmp[:, [d], :, :] = F.grid_sample(
                        torch.ones_like(cost_ir_tmp[:, [d], :, :]), grid_ir,
                        mode='nearest', padding_mode='zeros', align_corners=False
                    )

                cost_ir_warped.append(cost_ir_warped_tmp.contiguous())
                mask_ir_warped.append(mask_ir_warped_tmp.contiguous().detach())

        if 'LIDAR' not in self.method:
            # Dummy cost volume if not exist
            num_dep = disp_range_rgb.shape[1]

            cost0_lidar = torch.zeros((batch, num_dep, H_rgb // 3, W_rgb // 3)).type_as(rgb_l)
            cost1_lidar = torch.zeros((batch, num_dep // 2, H_rgb // 6, W_rgb // 6)).type_as(rgb_l)
            cost2_lidar = torch.zeros((batch, num_dep // 4, H_rgb // 12, W_rgb // 12)).type_as(rgb_l)

            cost_lidar = [cost0_lidar, cost1_lidar, cost2_lidar]
            mask_lidar = [torch.zeros_like(cost0_lidar).detach(),
                          torch.zeros_like(cost1_lidar).detach(),
                          torch.zeros_like(cost2_lidar).detach()]
        else:
            mask_lidar = [(disp_3 > 0.0).type_as(cost_lidar[0]).detach(),
                          (disp_6 > 0.0).type_as(cost_lidar[1]).detach(),
                          (disp_12 > 0.0).type_as(cost_lidar[2]).detach()]

        # Multi-Modal Cost Fusion
        cost_fused = []

        for s in range(0, 3):
            # Direct fusion
            mask_fused_tmp = (mask_rgb[s] + mask_ir_warped[s] + mask_lidar[s] + 1e-8)
            cost_fused_tmp = (cost_rgb[s] * mask_rgb[s]
                              + cost_ir_warped[s] * mask_ir_warped[s]
                              + cost_lidar[s] * mask_lidar[s]) / mask_fused_tmp

            cost_fused.append(cost_fused_tmp)

        pred_dep1, pred_dep2, pred_dep3, \
        pred_disp1, pred_disp2, pred_disp3, cost_out = \
            self._forward_aggregation_and_estimation(
                rgb_l, rgb_r, cost_fused, disp_range_rgb, K_rgb, B_rgb
            )

        # Remove paddings
        pred_disp1 = self._postprocess(pred_disp1, h_rgb, w_rgb)
        pred_disp2 = self._postprocess(pred_disp2, h_rgb, w_rgb)
        pred_disp3 = self._postprocess(pred_disp3, h_rgb, w_rgb)

        pred_dep1 = self._postprocess(pred_dep1, h_rgb, w_rgb)
        pred_dep2 = self._postprocess(pred_dep2, h_rgb, w_rgb)
        pred_dep3 = self._postprocess(pred_dep3, h_rgb, w_rgb)

        cost_out = self._postprocess(cost_out, h_rgb, w_rgb)

        dc_gt = self.dc_gen.encode(sample['gt_disp_rgb_l'], disp_range_rgb)

        output['pred_disp1'] = pred_disp1
        output['pred_disp2'] = pred_disp2
        output['pred_disp3'] = pred_disp3
        output['pred_dep1'] = pred_dep1
        output['pred_dep2'] = pred_dep2
        output['pred_dep3'] = pred_dep3
        output['cost_out'] = cost_out
        output['dc_gt'] = dc_gt.detach()

        return output
