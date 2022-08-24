# -----------------------
# snapshot compressive image processing
# -----------------------
import math
import cv2
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
        mask_shape (H*W*Cr): mask shape
        norm2one (bool): normalize original frames to one before encoding (/255)

    """

    def __init__(self, fixed_mask=True, mask_path=None, norm2one=True):
        super(SCIEncoding, self).__init__()
        self.fixed_mask = fixed_mask
        self.norm2one = norm2one
        if self.fixed_mask:
            # load mask from .npy file
            self.sci_mask = np.load(mask_path)
            self.mask_shape = self.sci_mask.shape
        # else:
        #     self.sci_mask = None
        #     self.mask_shape = mask_shape

    def __call__(self, results):
        # get encoding mask
        if self.fixed_mask:
            sci_mask = self.sci_mask
        else:
            mask_shape = [*results[0]['img'].shape, len(results)]
            sci_mask = np.random.randint(
                0, 2, size=mask_shape).astype(np.float32)

        # calc coded measurement
        coded_meas = np.zeros_like(sci_mask[..., 0])
        for i, _results in enumerate(results):
            if self.norm2one:
               _results['img'] = _results['img'].astype(np.float32)/255
            coded_meas += _results['img']*sci_mask[..., i]

        # debug
        cv2.imwrite('./_debug_img.png', np.uint8(results[0]['img']))
        cv2.imwrite('./_debug_mask.png',
                    np.uint8(255*sci_mask[..., 0]))
        cv2.imwrite('./_debug_meas.png', np.uint8(coded_meas/10))

        return {'frames': results, 'sci_mask': sci_mask, 'coded_meas': coded_meas}
