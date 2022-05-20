# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmdet.datasets.pipelines import LoadAnnotations as MMDet_LoadAnnotations

from mmtrack.core import results2outs
from mmtrack.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadTrackAnnotations(MMDet_LoadAnnotations):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,

                # Used in tracking.
                # Id of instances.
                'instance_id': 100,

                # Used in instance/panoptic segmentation. The segmentation mask
                # of the instance or the information of segments.
                # 1. If list[list[float]], it represents a list of polygons,
                # one for each connected component of the object. Each
                # list[float] is one simple polygon in the format of
                # [x1, y1, ..., xn, yn] (n≥3). The Xs and Ys are absolute
                # coordinates in unit of pixels.
                # 2. If dict, it represents the per-pixel segmentation mask in
                # COCO’s compressed RLE format. The dict should have keys
                # “size” and “counts”.  Can be loaded by pycocotools
                'mask': list[list[float]] or dict,

                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': np.ndarray(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
        }

    Required Keys:

    - height
    - width
    - instances

      - bbox (optional)
      - bbox_label
      - instance_id (optional)
      - mask (optional)
      - ignore_flag

    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int32)
    - gt_instances_id (np.int32)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (np.bool)

    Args:
        with_instance_id (bool): Whether to parse and load the instance id
            annotation. Defaults to False.
    """

    def __init__(self, with_instance_id: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.with_instance_id = with_instance_id

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        The only difference is that we record the type of `gt_ignore_flags`
        as np.int32.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results['instances']:
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_bboxes'] = np.array(
            gt_bboxes, dtype=np.float32).reshape(-1, 4)
        # The only difference is that we record the type of `gt_ignore_flags`
        # as np.int32.
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=np.bool)

        if self.denorm_bbox:
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                results['gt_bboxes'][:, 0::2] *= results['width']
                results['gt_bboxes'][:, 1::2] *= results['height']

    def _load_instances_id(self, results: dict) -> None:
        """Private function to load instances id annotations.

        Args:
            results (dict): Result dict from :obj :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded instances id annotations.
        """
        gt_instances_id = []
        for instance in results['instances']:
            gt_instances_id.append(instance['instance_id'])
        results['gt_instances_id'] = np.array(gt_instances_id, dtype=np.int32)

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label, instances id
            and semantic segmentation and keypoints annotations.
        """
        results = super().transform(results)
        if self.with_instance_id:
            self._load_instances_id(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_instance_id={self.with_instance_id}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadDetections(object):
    """Load public detections from MOT benchmark.

    Args:
        results (dict): Result dict from :obj:`mmtrack.CocoVideoDataset`.
    """

    def __call__(self, results):
        outs_det = results2outs(bbox_results=results['detections'])
        bboxes = outs_det['bboxes']
        labels = outs_det['labels']

        results['public_bboxes'] = bboxes[:, :4]
        if bboxes.shape[1] > 4:
            results['public_scores'] = bboxes[:, -1]
        results['public_labels'] = labels
        results['bbox_fields'].append('public_bboxes')
        return results
