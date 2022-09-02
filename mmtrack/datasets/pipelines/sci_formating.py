import warnings

import cv2
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class SCIDataCollect(object):
    """Collect data from the loader relevant to the SCI detection task.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str]): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('filename',
            'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'frame_id', 'is_video_data').
    """

    def __init__(self,
                 keys,
                 meta_keys=None,
                 default_meta_keys=('filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor',
                                    'flip', 'flip_direction', 'img_norm_cfg',
                                    'frame_id', 'is_video_data'),
                 default_meta_key_values = None):
        self.keys = keys
        self.meta_keys = default_meta_keys
        self.default_meta_key_values = default_meta_key_values
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def __call__(self, sci_results):
        """Call function to collect keys in results.

        The keys in ``meta_keys`` and ``default_meta_keys`` will be converted
        to :obj:mmcv.DataContainer.

        Args:
            results (list[dict] | dict): List of dict or dict which contains
                the data to collect.

        Returns:
            list[dict] | dict: List of dict or dict that contains the
            following keys:

            - keys in ``self.keys``
            - ``img_metas``
        """
        results = sci_results['frames']
        assert not isinstance(
            results, dict), '$frames in SCIEncoding results should be a list of dict'
        outs = []
        for _results in results:
            _results = self._add_default_meta_keys(
                _results, default_meta_key_values=self.default_meta_key_values)
            _results = self._collect_meta_keys(_results)
            outs.append(_results)

        sci_results['frames'] = outs
        return sci_results

    def _collect_meta_keys(self, results):
        """Collect `self.keys` and `self.meta_keys` from `results` (dict)."""
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            elif key in results['img_info']:
                img_meta[key] = results['img_info'][key]
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def _add_default_meta_keys(self, results, default_meta_key_values=None):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.
            default_meta_values: default values for default_meta_keys

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        if default_meta_key_values is None:
            default_meta_key_values = {}
        results.setdefault(
            'pad_shape', default_meta_key_values.get('pad_shape', img.shape))
        results.setdefault(
            'scale_factor', default_meta_key_values.get('scale_factor', 1.0))
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault('img_norm_cfg', default_meta_key_values.get('img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False)))
        return results


@PIPELINES.register_module()
class SCIDataArrange(object):
    """Rearrange the sci data into three types: 
    - frames: original frames including `img_meta`, `img` (H*W*C*Cr), `gt_bboxes`, `gt_labels`, etc
    - sci_mask: sci encoding mask
    - coded_meas: coded measurement
    
    """

    # def __init__(self):  # , num_key_frames=1
    #     pass
    #     # self.num_key_frames = num_key_frames

    def concat_one_mode_results(self, results):
        """Concatenate the results of the same mode."""
        out = dict()
        for i, result in enumerate(results):
            if 'img' in result:
                img = result['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if i == 0:
                    result['img'] = np.expand_dims(img, -1)
                else:
                    out['img'] = np.concatenate(
                        (out['img'], np.expand_dims(img, -1)), axis=-1)
            for key in ['img_metas', 'gt_masks']:
                if key in result:
                    if i == 0:
                        result[key] = [result[key]]
                    else:
                        out[key].append(result[key])
            for key in [
                    'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                    'gt_instance_ids'
            ]:
                if key not in result:
                    continue
                value = result[key]
                #--- orig
                if value.ndim == 1:
                    value = value[:, None]
                N = value.shape[0]
                value = np.concatenate((np.full(
                    (N, 1), i, dtype=np.float32), value),
                    axis=1)
                if i == 0:
                    result[key] = value
                else:
                    out[key] = np.concatenate((out[key], value), axis=0)
                #--- sci
                # if i == 0:
                #     result[key] = [value]
                # else:
                #     out[key].append(value)
                #--- end
            if 'gt_semantic_seg' in result:
                if i == 0:
                    result['gt_semantic_seg'] = result['gt_semantic_seg'][...,
                                                                          None,
                                                                          None]
                else:
                    out['gt_semantic_seg'] = np.concatenate(
                        (out['gt_semantic_seg'],
                         result['gt_semantic_seg'][..., None, None]),
                        axis=-1)

            if 'padding_mask' in result:
                if i == 0:
                    result['padding_mask'] = np.expand_dims(
                        result['padding_mask'], 0)
                else:
                    out['padding_mask'] = np.concatenate(
                        (out['padding_mask'],
                         np.expand_dims(result['padding_mask'], 0)),
                        axis=0)

            if i == 0:
                out = result
        return out

    def __call__(self, sci_results):
        """Call function.

        Args:
            results (list[dict]): list of dict that contain keys such as 'img',
                'img_metas', 'gt_masks','proposals', 'gt_bboxes',
                'gt_bboxes_ignore', 'gt_labels','gt_semantic_seg',
                'gt_instance_ids', 'padding_mask'.

        Returns:
            list[dict]: The first dict of outputs concats the dicts of 'key'
                information. The second dict of outputs concats the dicts of
                'reference' information.
        """
        assert (isinstance(sci_results, dict) and len(
            sci_results) == 3), 'sci_results must be a dict contain 3 items (frames, sci_mask, coded_meas)'

        frames = sci_results['frames']
        # rearrage `frames` from separate dicts to a whole tensor
        sci_results['frames'] = self.concat_one_mode_results(frames)

        return sci_results


@PIPELINES.register_module()
class SCIFormatBundle(object):
    """ SCI data formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "img_metas", "proposals", "gt_bboxes", "gt_instance_ids",
    "gt_match_indices", "gt_bboxes_ignore", "gt_labels", "gt_masks",
    "gt_semantic_seg" and 'padding_mask'.
    These fields are formatted as follows.

    - img: (1) transpose, (2) to tensor, (3) to DataContainer (stack=True)
    - img_metas: (1) to DataContainer (cpu_only=True)
    - proposals: (1) to tensor, (2) to DataContainer
    - gt_bboxes: (1) to tensor, (2) to DataContainer
    - gt_instance_ids: (1) to tensor, (2) to DataContainer
    - gt_match_indices: (1) to tensor, (2) to DataContainer
    - gt_bboxes_ignore: (1) to tensor, (2) to DataContainer
    - gt_labels: (1) to tensor, (2) to DataContainer
    - gt_masks: (1) to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1) unsqueeze dim-0 (2) to tensor, \
                       (3) to DataContainer (stack=True)
    - padding_mask: (1) to tensor, (2) to DataContainer

    Args:
        ref_prefix (str): The prefix of key added to the second dict of input
            list. Defaults to 'ref'.
    """

    # def __init__(self):  # , ref_prefix='ref'
    #     pass
    #     # self.ref_prefix = ref_prefix

    def __call__(self, sci_results):
        """Transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
            SCI bundle.
        """
        frames = sci_results['frames']

        # format `frames`
        if 'img' in frames:
            img = frames['img']
            if len(img.shape) == 3:  # gray images
                img = np.ascontiguousarray(img.transpose(2, 0, 1))  # N*H*W
            else:
                img = np.ascontiguousarray(
                    img.transpose(3, 2, 0, 1))  # N*C*H*W
            frames['img'] = DC(to_tensor(img), stack=True)
        if 'padding_mask' in frames:
            frames['padding_mask'] = DC(
                to_tensor(frames['padding_mask'].copy()), stack=True)
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_instance_ids', 'gt_match_indices'
        ]:
            if key not in frames:
                continue
            frames[key] = DC(to_tensor(frames[key]))
        for key in ['img_metas', 'gt_masks']:
            if key in frames:
                frames[key] = DC(frames[key], cpu_only=True)
        if 'gt_semantic_seg' in frames:
            semantic_seg = frames['gt_semantic_seg']
            if len(semantic_seg.shape) == 2:
                semantic_seg = semantic_seg[None, ...]
            else:
                semantic_seg = np.ascontiguousarray(
                    semantic_seg.transpose(3, 2, 0, 1))
            frames['gt_semantic_seg'] = DC(
                to_tensor(frames['gt_semantic_seg']), stack=True)

        # format `sci_mask` according to `frames`
        if 'sci_mask' in sci_results:
            sci_mask = sci_results['sci_mask']
            if len(sci_mask.shape) == 3:
                sci_mask = np.ascontiguousarray(
                    sci_mask.transpose(2, 0, 1))  # Cr*H*W
            else:
                sci_mask = np.ascontiguousarray(
                    sci_mask.transpose(3, 2, 0, 1))  # Cr*C*H*W
        else:
            raise KeyError('Missing `sci_mask` in `sci_results`')
        
        # format `coded_meas` to C*H*W
        coded_meas = np.transpose(sci_results['coded_meas'], (2, 0, 1))

        # collect formatted results
        sci_results['frames'] = frames
        sci_results['sci_mask'] = DC(to_tensor(sci_mask), stack=True)
        sci_results['coded_meas'] = DC(to_tensor(coded_meas), stack=True)

        return sci_results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class SCIMultiImagesToTensor(object):
    """Multi images to tensor.

    1. Transpose and convert image/multi-images to Tensor.
    2. Add keys and corresponding values into the outputs.

    """

    # def __init__(self, ref_prefix='ref'):
    #     self.ref_prefix = ref_prefix

    def __call__(self, sci_results):
        """Multi images in `sci_results` to tensor.

        Transpose and convert image/multi-images to Tensor.

        Args:
            results (dict): dict.

        Returns:
            dict: Each key in the first dict of `results` remains unchanged.
            Each key in the second dict of `results` adds `self.ref_prefix`
            as prefix.
        """
        frames = sci_results['frames']

        # format frames
        frames = self.images_to_tensor(frames)

        # format `sci_mask` according to `frames`
        if 'sci_mask' in sci_results:
            sci_mask = sci_results['sci_mask']
            if len(sci_mask.shape) == 3:
                sci_mask = np.ascontiguousarray(
                    sci_mask.transpose(2, 0, 1))  # Cr*H*W
            else:
                sci_mask = np.ascontiguousarray(
                    sci_mask.transpose(3, 2, 0, 1))  # Cr*C*H*W
        else:
            raise KeyError('Missing `sci_mask` in `sci_results`')

        # format `coded_meas` to C*H*W
        coded_meas = np.transpose(sci_results['coded_meas'], (2, 0, 1))

        # collect formatted results
        sci_results['frames'] = frames
        sci_results['sci_mask'] = to_tensor(sci_mask)
        sci_results['coded_meas'] = to_tensor(coded_meas)

        return sci_results

    def images_to_tensor(self, results):
        """Transpose and convert images/multi-images to Tensor."""
        if 'img' in results:
            img = results['img']
            if len(img.shape) == 3:
                # (H, W, 3) to (3, H, W)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            else:
                # (H, W, 3, N) to (N, 3, H, W)
                img = np.ascontiguousarray(img.transpose(3, 2, 0, 1))
            results['img'] = to_tensor(img)
        if 'proposals' in results:
            results['proposals'] = to_tensor(results['proposals'])
        if 'img_metas' in results:
            results['img_metas'] = DC(results['img_metas'], cpu_only=True)
        return results

# the same as formatting.py, omitted
# @PIPELINES.register_module()
# class ToList(object):
#     """Use list to warp each value of the input dict.

#     Args:
#         results (dict): Result dict contains the data to convert.

#     Returns:
#         dict: Updated result dict contains the data to convert.
#     """

#     def __call__(self, results):
#         out = {}
#         for k, v in results.items():
#             out[k] = [v]
#         return out
