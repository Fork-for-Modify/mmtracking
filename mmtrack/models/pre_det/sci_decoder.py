# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule
from ..builder import PREDET
import torch
# import mmcv
# import numpy as np


@PREDET.register_module()
class EnergyNormDec(BaseModule):
    """energy normalization based SCI decoding: decode sci coded measurements with encoding masks

    Args:
        # init_cfg (dict or list[dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 init_cfg=None):
        super(EnergyNormDec, self).__init__(init_cfg)

    def _energy_norm(self, coded_meas, sci_mask):
        """
        energy normalization based SCI decoder 

        orig_pred = meas./(mask_sum)*sci_mask

        Args:
            coded_meas (tensor): coded measurements
            sci_mask (tensor[Cr*C*H*W]): sci encoding masks
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
            sci_mask (tensor[Cr*C*H*W]): sci encoding masks
        """
        sci_dec, meas_re = self._energy_norm(coded_meas, sci_mask)

        return sci_dec, meas_re
