# -----------------------
# snapshot compressive image processing
# -----------------------
import math
from this import d
import cv2
from cv2 import add
import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class SCIEncoding(object):
    """SCI encoding

    encoding video frames with sampling masks


    Args:
        fixed_mask (bool): If True, use loaded fixed masks; if false, use randomly generated masks. Defaults to True.
        mask_path (str): mask path for using fixed mask
        noise_sigma (float,0-1): guassian noise ratio for measurement (i.e std assuming the pixel to be 0-1)
        mask_shape (H*W*Cr): mask shape
        norm2one (bool): normalize original frames to 1 before encoding (/255)

    """

    def __init__(self, fixed_mask=True, mask_path=None, noise_sigma=0, norm2one=True):
        super(SCIEncoding, self).__init__()
        self.fixed_mask = fixed_mask
        self.norm2one = norm2one
        self.noise_std = noise_sigma if norm2one else noise_sigma*255
        if self.fixed_mask:
            if mask_path == 'all_one':
                # all_one mask, i.e. frame sum, generate in __call__
                self.sci_mask = None
                self.mask_type = mask_path
            else:
                # load mask from .npy file
                self.sci_mask = np.load(mask_path)
                self.mask_shape = self.sci_mask.shape

    def __call__(self, results):
        # get encoding mask
        frame_shape = [*results[0]['img'].shape, len(results)]
        if self.fixed_mask:
            if self.sci_mask is None and self.mask_type == 'all_one':
                sci_mask = np.ones((frame_shape), dtype=np.float32)
            else:
                sci_mask = self.sci_mask
                assert frame_shape == self.mask_shape, \
                    f'frame shape {frame_shape} should be equal to mask shape {self.mask_shape}'
        else:
            sci_mask = np.random.randint(
                0, 2, size=frame_shape).astype(np.float32)

        # calc coded measurement
        coded_meas = np.zeros_like(sci_mask[..., 0])
        for i, _results in enumerate(results):
            if self.norm2one:
                _results['img'] = _results['img'].astype(np.float32)/255
            coded_meas += _results['img']*sci_mask[..., i]

        # add gaussian noise
        if self.noise_std != 0:
            coded_meas += self.noise_std * \
                np.random.randn(*coded_meas.shape).astype(coded_meas.dtype)

        # debug
        # cv2.imwrite('./_debug_img.png', np.uint8(results[0]['img']))
        # cv2.imwrite('./_debug_mask.png',
        #             np.uint8(255*sci_mask[..., 0]))
        # cv2.imwrite('./_debug_meas.png', np.uint8(coded_meas/10))

        return {'frames': results, 'sci_mask': sci_mask, 'coded_meas': coded_meas}
