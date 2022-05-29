import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, DepthwiseSeparableConvModule
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox2distance, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox, distance2kps, kps2distance,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)

                
from ..builder import HEADS, build_loss
from ..utils.yunet_layer import ConvDPUnit

import os
import math

@HEADS.register_module()
class WWHead(nn.Module):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        loss_qfl (dict): Config of Quality Focal Loss (QFL).
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                num_classes,
                in_channels,   
                stacked_convs_num,
                feat_channels,
                strides,
                use_kps=False,
                kps_num=5,
                loss_kps=None,
                loss_cls=None,
                loss_bbox=None,
                train_cfg=None,
                test_cfg=None):
        super(WWHead, self).__init__()
        self.stacked_convs_num = stacked_convs_num
        self.extra_flops = 0.0
        self.use_kps = use_kps
        self.NK = kps_num
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        if use_kps:
            self.out_channels = self.num_classes + self.NK * 2 + 4
        else:
            self.out_channels = self.num_classes + 4
        self.strides = strides
        self.strides_num = len(self.strides)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)


        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_kps = build_loss(loss_kps)
        self._init_layers()


        self.train_step = 0
        self.pos_count = {}
        self.gtgroup_count = {}

    def _init_layers(self):
        """Initialize layers of the head."""
        self.stacked_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()

        for i in range(self.strides_num):
            stack_conv = [ConvDPUnit(self.in_channels, self.feat_channels)]
            for _ in range(self.stacked_convs_num - 1):
                stack_conv.append(ConvDPUnit(self.feat_channels, self.feat_channels))    
            self.stacked_convs.append(nn.Sequential(*stack_conv))
            self.cls_convs.append(ConvDPUnit(self.feat_channels, self.out_channels, False))
        self.init_weights()

    def init_weights(self):
        for m in self.stacked_convs.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        bias_cls = -4.595
        for m in self.cls_convs.modules():   
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(bias_cls)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        outs = []
        for i in range(self.strides_num):
            x = self.stacked_convs[i](feats[i])
            x = self.cls_convs[i](x)
            outs.append(x)
        head_data=torch.cat([x.flatten(start_dim=2) for x in outs], dim=-1).permute(0, 2, 1)
        if self.use_kps:
            output = head_data.split_with_sizes(split_sizes=(4, self.NK * 2, self.num_classes), dim=-1)
        else:
            output = head_data.split_with_sizes(split_sizes=(4, self.num_classes), dim=-1)
        return output

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_keypointss=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypointss, img_metas)      
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)

    @torch.no_grad()
    def target_assign_single_img(
        self, cls_preds, bbox_decoded, kps_preds, center_priors, gt_bboxes, gt_kps, gt_labels
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors (Tensor): All priors of one image, a 2D-Tensor with
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            bbox_decoded (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = center_priors.size(0)
        num_gts = gt_labels.size(0)

        bbox_targets = torch.zeros_like(bbox_decoded)
        bbox_weights = torch.zeros_like(bbox_decoded)
        kps_targets = torch.zeros_like(kps_preds)
        kps_weights = torch.zeros_like(kps_preds)
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)
        # No target
        if num_gts == 0:
            return labels, label_scores, bbox_targets, kps_targets, 0

        assign_result = self.assigner.assign(
            cls_preds.sigmoid(), center_priors, bbox_decoded, gt_bboxes, gt_labels
        )
        sampling_result = self.sampler.sample(
            assign_result, center_priors, gt_bboxes
        )
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]
        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds, :] = 1.0
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            if self.use_kps:
                kps_targets[pos_inds, :] = gt_kps[pos_assigned_gt_inds,:,:2].reshape( (-1, self.NK*2) )
                kps_weights[pos_inds, :] = torch.mean(gt_kps[pos_assigned_gt_inds, :, 2], dim=1, keepdims=True)
        return (
            labels,
            label_scores,
            bbox_targets,
            bbox_weights,
            kps_targets,
            kps_weights,
            num_pos_per_img,
        )
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             bbox_preds,
             kps_preds,
             cls_preds,
             gt_bboxes,
             gt_labels,
             gt_keypointss,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_preds (list[Tensor]): Cls and quality scores for each image with
                shape (Batch, N, num_class) (e.g. num_class=1 in face detection).
                N = \sum(H_i x W_i), where H_i, W_i is resolution of each scale of
                feature map.
            bbox_preds (list[Tensor]): Box regression for each image with
                shape [Batch, N, 4] in [l_offset, t_offset, r_offset, b_offset] / stride_i
                format.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # featmap_sizes = [featmap.size()[-2:] for featmap in cls_preds]
        # assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_preds.device
        batch_size = cls_preds.shape[0]
        input_height, input_width = img_metas[0]['batch_input_shape']
        featmap_sizes = [
            (math.floor(input_height / stride), math.floor(input_width / stride))
            for stride in self.strides
        ]

        # get grid cells of batch image [batch, N, 4]
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1).detach()
        bboxes_decoded = distance2bbox(center_priors[..., :2], bbox_preds.exp() * center_priors[..., 2, None])
        batch_assign_res = multi_apply(
            self.target_assign_single_img,
            cls_preds.detach(),
            bboxes_decoded.detach(),
            kps_preds.detach(),
            center_priors,
            gt_bboxes,
            gt_keypointss,
            gt_labels,
        )



        (labels, label_scores, bbox_targets, bbox_weights, kps_targets, kps_weights, num_pos_per_img) = batch_assign_res
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos_per_img), dtype=torch.float, device=device)).item(), 1.0
        )
        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)

        bboxes_decoded = bboxes_decoded.reshape(-1, 4)
        center_priors = center_priors.reshape(-1, 4)
        loss_qfl = self.loss_cls(
            cls_preds, (labels, label_scores), avg_factor=num_total_samples
        )

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            loss_bbox = self.loss_bbox(
                bboxes_decoded[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )
            if self.use_kps:
                kps_preds = kps_preds.reshape(-1, self.NK * 2)
                kps_targets = torch.cat(kps_targets, dim=0)
                kps_weights = torch.cat(kps_weights, dim=0)
                pos_kps_weights = kps_weights.max(dim=1)[0][pos_inds] * weight_targets
                pos_kps_targets = kps_targets[pos_inds]
                pos_kps_targets_encoded = \
                    kps2distance(center_priors[pos_inds, :2], pos_kps_targets) / \
                        center_priors[pos_inds, 2, None]
                loss_kps = self.loss_kps(
                    kps_preds[pos_inds],
                    pos_kps_targets_encoded,
                    weight=pos_kps_weights.reshape(-1, 1),
                    avg_factor=bbox_avg_factor)
        else:
            loss_bbox = bbox_preds.sum() * 0.
            if self.use_kps:
                loss_kps = kps_preds.sum() * 0.

        losses_dict = dict(loss_cls=loss_qfl, loss_bbox=loss_bbox)
        if self.use_kps:
            losses_dict['loss_kps'] = loss_kps

        return losses_dict

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'kps_preds'))
    def get_bboxes(self,
                   bbox_preds,
                   kps_preds,
                   cls_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_preds) == len(bbox_preds)
        # num_levels = len(cls_preds)
        batch_size = cls_preds.shape[0]

        device = cls_preds[0].device
        featmap_sizes = [cls_preds[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_preds[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                has shape (num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for a single
                scale level with shape (4*(n+1), H, W), n is max value of
                integral set.
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): Bbox predictions in shape (N, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (N,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, stride, anchors in zip(
                cls_scores, bbox_preds, self.anchor_generator.strides,
                mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0)
            if self.use_dfl:
                bbox_pred = self.integral(bbox_pred) * stride[0]
            else:
                bbox_pred = bbox_pred.reshape( (-1,4) ) * stride[0]

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(
                self.anchor_center(anchors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
