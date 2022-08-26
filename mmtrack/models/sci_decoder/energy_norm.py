# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule
from ..builder import SCIDECODERS
import torch
# import mmcv
# import numpy as np


@SCIDECODERS.register_module()
class EnergyNorm(BaseModule):
    """SCI decoding: decode sci coded measurements with encoding masks

    Args:
        # norm4det (dict | {'mean': list, 'std': list, 'to_rgb': bool<false>): 
        #       normalize SCI decoding results for the following detetion 
        # init_cfg (dict or list[dict], optional): Initialization config dict.
        #     Defaults to None.
    """

    def __init__(self,
                 norm4det=None,
                 init_cfg=None):
        super(EnergyNorm, self).__init__(init_cfg)
        if norm4det:
            self.norm4det_flag = True
            self.mean = torch.tensor(norm4det['mean'], dtype=torch.float64)
            self.std = torch.tensor(norm4det['std'], dtype=torch.float64)
            self.to_rgb = norm4det['to_rgb']

    def _imnormalize(self, img, mean, std, to_rgb=True):
        """ 
        normalize images with given `mean` and `std`
        Args:
            imgs (tensor): Image to be normalized.
            mean (tensor): The mean to be used for normalize.
            std (tensor): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.
        """
        assert img.dtype != torch.uint8, 'float or double is required'
        mean = mean.to(img.device).double()
        stdinv = 1 / std.to(img.device).double()
        if to_rgb and len(img.shape) == 3 and img.shape[2] == 3:
            img = img[:, :, [2,1,0]] # bgr2rgb
        torch.subtract(img, mean, out=img)  # inplace
        torch.multiply(img, stdinv, out=img)  # inplace
        return img

    def _norm4det(self, imgs, mean, std, to_rgb=True):
        """ 
        normalize images with given `mean` and `std`
        Args:
            imgs (tensor[N,C,H,W]): Image to be normalized.
            mean (tensor): The mean to be used for normalize.
            std (tensor): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.
        """
        assert imgs.shape[1] == len(
            mean), 'channel  of `imgs` should be in accordance with normalization config'

        for k, img in enumerate(imgs):
            img = img.permute((1, 2, 0)).float()
            img = self._imnormalize(
                img, mean=mean, std=std, to_rgb=to_rgb)
            imgs[k, ...] = img.permute((2, 0, 1))

        return imgs

    def _energy_norm(self, coded_meas, sci_mask):
        """
        energy normalization based SCI decoder 

        orig_pred = meas./(mask_sum)*sci_mask

        Args:
            coded_meas (tensor): coded measurements
            sci_mask (tensor[H*W*Cr]): sci encoding masks
        """
        # measurement energy normalization
        mask_sum = torch.sum(sci_mask, dim=0)
        # replace 0 to avoid NaN problem when dividing
        mask_sum[mask_sum == 0] = 1
        meas_re = torch.div(coded_meas, mask_sum)

        # sci decoding res
        sci_dec = sci_mask.mul(meas_re)
        # # cat meas_re to sci_dec
        # sci_dec = torch.cat((sci_dec, meas_re.unsqueeze(0)), dim=0)

        return sci_dec, meas_re

    def forward(self, coded_meas, sci_mask):
        """
        energy normalization based SCI decoder 

        orig_pred = meas./(mask_sum)*sci_mask

        Args:
            coded_meas (tensor): coded measurements
            sci_mask (tensor[H*W*Cr]): sci encoding masks
        """
        sci_dec, meas_re = self._energy_norm(coded_meas, sci_mask)
        if self.norm4det_flag:
            sci_dec = self._norm4det(sci_dec, self.mean, self.std, self.to_rgb)
            meas_re = self._norm4det(meas_re.unsqueeze(
                0), self.mean, self.std, self.to_rgb).squeeze()

        return sci_dec, meas_re
