# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from addict import Dict
from mmdet.models import build_detector

from ..builder import MODELS, build_scidecoder
from .scidet_base import BaseSCIDetector


@MODELS.register_module()
class SCISELSA(BaseSCIDetector):
    """Sequence Level Semantics Aggregation for Video Object Detection.

    This video object detector is the implementation of `SELSA
    <https://arxiv.org/abs/1907.06390>`_.
    """

    def __init__(self,
                 detector,
                 scidecoder=None,
                 pretrains=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SCISELSA, self).__init__(init_cfg)
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            detector_pretrain = pretrains.get('detector', None)
            if detector_pretrain:
                detector.init_cfg = dict(
                    type='Pretrained', checkpoint=detector_pretrain)
            else:
                detector.init_cfg = None
        self.detector = build_detector(detector)
        if scidecoder:
            self.scidecoder = build_scidecoder(scidecoder)
        assert hasattr(self.detector, 'roi_head'), \
            'selsa video detector only supports two stage detector'
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def _data_debundle(self, data):
        """
        rearange bundled data
        split data into separate lists according to their first column item (frame index)
        return [lists]
        """
        index = data[:, 0]
        data = [data[index == i][:,1:] for i in index.unique()]
        return data

    def forward_train(self, frames, sci_mask, coded_meas, **kwargs):
        """
        Args:
        sci_mask (Tensor): sci encoding masks, (1, N, C, H, W)
        coded_meas (Tensor): coded measurements, (1, C, H, W)
        frames: original annotated video frames contain the following keys:
            img (Tensor): of shape (1, N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[list[dicts]]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [frame_id, tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box with
                shape (num_gts, 2) in [frame_id, class_id] format.

            gt_instance_ids (None | list[Tensor]): specify the instance id for
                each ground truth bbox.

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals (None | Tensor) : override rpn proposals with custom
                proposals. Use when `with_rpn` is False.
            ref_img_metas (list[list[dict]]): The first list only has one
                element. The second list contains reference image information
                dict where each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # ---------------------------------------
        # augments arrange
        assert len(coded_meas) == 1, \
            'sci detection only supports 1 batch size per gpu for now.'
        sci_mask = sci_mask[0]
        coded_meas = coded_meas[0]
        # ---------------------------------------

        # sci decoder
        all_scidec, meas_re = self.scidecoder(coded_meas, sci_mask)

        # ---------------------------------------
        # data assign
        img = all_scidec  # img: (N,C,H,W)
        img_metas = frames['img_metas'][0]
        gt_bboxes = self._data_debundle(frames['gt_bboxes'][0])
        gt_labels = self._data_debundle(frames['gt_labels'][0])
        gt_labels = [_label.squeeze(1).long() for _label in gt_labels]
        gt_bboxes_ignore = None
        gt_masks = None,
        proposals = None
        ref_proposals = None
        # use meas_re as ref img
        if meas_re is not None:
            ref_img = meas_re.unsqueeze(0)  # ref_img: (1,C,H,W)
            ref_img_metas = frames['img_metas'][0][0].copy()
            ref_img_metas['filename'] = ''
            _name_split = ref_img_metas['ori_filename'].split('/')
            ref_img_metas['ori_filename'] = 'meas_'+'-'.join(_name_split)
            ref_img_metas = [ref_img_metas]
        # ---------------------------------------

        all_imgs = torch.cat((img, ref_img), dim=0)
        all_x = self.detector.extract_feat(all_imgs)
        x = []  # -> [[10, 512, 34, 60]]
        ref_x = []  # -> [[1, 512, 34, 60]]
        for i in range(len(all_x)):  # split key feature and ref feature
            x.append(all_x[i][0:10])
            ref_x.append(all_x[i][[10]])

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            # rpn_losses = {'loss_rpn_cls':[v], 'loss_rpn_bbox':[v]}
            # proposal_list = [10*[600,5]]
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            # ref_proposal_list = [[300,5]]
            ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                ref_x, ref_img_metas)
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        roi_losses = self.detector.roi_head.forward_train(
            x, ref_x, img_metas, proposal_list, ref_proposals_list, gt_bboxes,
            gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        return losses

    def extract_feats(self, img, img_metas, ref_img, ref_img_metas):
        """Extract features for `img` during testing.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.

            img_metas (list[dicts]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_img (Tensor | None): of shape (1, M, C, H, W) encoding input
                reference images. Typically these should be mean centered and
                std scaled. N denotes the number of reference images. There
                may be no reference images in some cases.

            ref_img_metas (list[list[dicts]] | None): The first list only has
                one element. The second list contains image information dict
                where each dict has: 'img_shape', 'scale_factor', 'flip', and
                may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`. There
                may be no reference images in some cases.

        Returns:
            tuple(x, img_metas, ref_x, ref_img_metas): x is the multi level
                feature maps of `img`, ref_x is the multi level feature maps
                of `ref_img`.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        num_left_ref_imgs = img_metas[0].get('num_left_ref_imgs', -1)
        frame_stride = img_metas[0].get('frame_stride', -1)

        # test with adaptive stride
        if frame_stride < 1:
            if frame_id == 0:
                self.memo = Dict()
                self.memo.img_metas = ref_img_metas[0]
                ref_x = self.detector.extract_feat(ref_img[0])
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.memo.feats = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])

            x = self.detector.extract_feat(img)
            ref_x = self.memo.feats.copy()
            for i in range(len(x)): # zzh: for original code, len(x)=1, but for scidet = 10
                ref_x[i] = torch.cat((ref_x[i], x[i]), dim=0) # -> ref_x: [[11,512,34,60]]
            ref_img_metas = self.memo.img_metas.copy()
            ref_img_metas.extend(img_metas)
        # test with fixed stride
        else:
            if frame_id == 0:
                self.memo = Dict()
                self.memo.img_metas = ref_img_metas[0]
                ref_x = self.detector.extract_feat(ref_img[0])
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.memo.feats = []
                # the features of img is same as ref_x[i][[num_left_ref_imgs]]
                x = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])
                    x.append(ref_x[i][[num_left_ref_imgs]])
            elif frame_id % frame_stride == 0:
                assert ref_img is not None
                x = []
                ref_x = self.detector.extract_feat(ref_img[0])
                for i in range(len(ref_x)):
                    self.memo.feats[i] = torch.cat(
                        (self.memo.feats[i], ref_x[i]), dim=0)[1:]
                    x.append(self.memo.feats[i][[num_left_ref_imgs]])
                self.memo.img_metas.extend(ref_img_metas[0])
                self.memo.img_metas = self.memo.img_metas[1:]
            else:
                assert ref_img is None
                x = self.detector.extract_feat(img)

            ref_x = self.memo.feats.copy()
            for i in range(len(x)):
                ref_x[i][num_left_ref_imgs] = x[i]
            ref_img_metas = self.memo.img_metas.copy()
            ref_img_metas[num_left_ref_imgs] = img_metas[0]

        return x, img_metas, ref_x, ref_img_metas

    def simple_test(self, frames, sci_mask, coded_meas,
                    proposals=None,
                    ref_proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
        sci_mask (Tensor): sci encoding masks, (1, N, C, H, W)
        coded_meas (Tensor): coded measurements (1, C, H, W)
        frames: original annotated video frames contain the following keys:
            img (Tensor): of shape (1, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.

            img_metas (list[dicts]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

        proposals (None | Tensor): Override rpn proposals with custom
                proposals. Use when `with_rpn` is False. Defaults to None.

        rescale (bool): If False, then returned bboxes and masks will fit
                the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The detection results.
        """

       # ---------------------------------------
        # augments arrange
        assert len(coded_meas) == 1, \
            'sci detection only supports 1 batch size per gpu for now.'
        sci_mask = sci_mask[0]
        coded_meas = coded_meas[0]
        # ---------------------------------------

        # sci decoder
        all_scidec, meas_re = self.scidecoder(coded_meas, sci_mask)

        # ---------------------------------------
        # data assign
        img = all_scidec  # img: (N,C,H,W)
        img_metas = frames['img_metas'][0]
        # gt_bboxes = self._data_debundle(frames['gt_bboxes'][0])
        # gt_labels = self._data_debundle(frames['gt_labels'][0])
        # gt_labels = [_label.squeeze(1).long() for _label in gt_labels]
        # gt_bboxes_ignore = None
        # gt_masks = None,
        proposals = None
        ref_proposals = None
        # use meas_re as ref img
        if meas_re is not None:
            ref_img = meas_re.unsqueeze(0)  # ref_img: (1,C,H,W)
            ref_img_metas = frames['img_metas'][0][0].copy()
            ref_img_metas['filename'] = ''
            ref_img_metas['ori_filename'] =''
            # _name_split = ref_img_metas['ori_filename'].split('/')
            # ref_img_metas['ori_filename'] = 'meas_'+'-'.join(_name_split)
            ref_img_metas = [ref_img_metas]
        # ---------------------------------------

        
        #----------------------------------------
        # zzh: original extracting test features
        # x, img_metas, ref_x, ref_img_metas = self.extract_feats(
        #     img, img_metas, ref_img, ref_img_metas)
        #-----------------------------------------
        # scidet: (like extracting train features)
        all_imgs = torch.cat((img, ref_img), dim=0)
        all_x = self.detector.extract_feat(all_imgs)
        x = []  # -> [[10, 512, 34, 60]]
        ref_x = []  # -> [[1, 512, 34, 60]]
        for i in range(len(all_x)):  # split key feature and ref feature
            x.append(all_x[i][0:10])
            ref_x.append(all_x[i][[10]])
        #----------------------------------------
        

        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(
                x, img_metas)
            ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                ref_x, ref_img_metas)
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals



        outs = self.detector.roi_head.simple_test(
            x,
            ref_x,
            proposal_list,
            ref_proposals_list,
            img_metas,
            rescale=rescale)

        results = dict()
        results['det_bboxes'] = outs # outs[0]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
