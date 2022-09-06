# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule
from ..builder import PREDET
import torch


@PREDET.register_module()
class SeqNorm4Det(BaseModule):
    """sequential normalization for detection: normalize video frames for the following detection

    Args:
        # norm_cfg (dict {'mean': list, 'std': list, 'to_rgb'}  | None)
        #     normalization configs. Defaults to None.
        # init_cfg (dict or list[dict], optional): Initialization config dict.
        #     Defaults to None.
    """

    def __init__(self,
                 norm_cfg=None,
                 init_cfg=None):
        super(SeqNorm4Det, self).__init__(init_cfg)
        if norm_cfg:
            self.norm_flag = True
            self.mean = torch.tensor(norm_cfg['mean'], dtype=torch.float64)
            self.std = torch.tensor(norm_cfg['std'], dtype=torch.float64)
            self.to_rgb = norm_cfg['to_rgb']

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
            img = img[:, :, [2, 1, 0]]  # bgr2rgb
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

    def forward(self, imgs):
        """
        Args:
            imgs (tensor[N*C*H*W] | tensor[C*H*W]): video frames to be normalized
        """
        if self.norm_flag:
            if len(imgs.shape) == 3:
                imgs_norm = self._norm4det(
                    imgs.unsqueeze(0), self.mean, self.std, self.to_rgb)
                imgs_norm = imgs_norm.squeeze(0)
            else:
                imgs_norm = self._norm4det(
                    imgs, self.mean, self.std, self.to_rgb)
                
        return imgs_norm
