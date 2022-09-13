# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import DATASETS
from mmdet.datasets.api_wrappers import COCO

from .coco_sci_dataset import CocoSCIDataset
from .parsers import CocoVID


@DATASETS.register_module()
class UADETRACSCIDataset(CocoSCIDataset):
    """ UA-DETRAC based SCI dataset for SCI video object detection."""

    CLASSES = ('car', 'van', 'bus', 'others')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def load_annotations(self, ann_file):
    #     """Load annotations from COCO/COCOVID style annotation file.

    #     Args:
    #         ann_file (str): Path of annotation file.

    #     Returns:
    #         list[dict]: Annotation information from COCO/COCOVID api.
    #     """
    #     if self.load_as_video:
    #         data_infos = self.load_video_anns(ann_file)
    #     else:
    #         data_infos = self.load_image_anns(ann_file)
    #     return data_infos

    # def load_image_anns(self, ann_file):
    #     """Load annotations from COCO style annotation file.

    #     Args:
    #         ann_file (str): Path of annotation file.

    #     Returns:
    #         list[dict]: Annotation information from COCO api.
    #     """
    #     self.coco = COCO(ann_file)
    #     self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
    #     self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    #     all_img_ids = self.coco.get_img_ids()
    #     self.img_ids = []
    #     data_infos = []
    #     for img_id in all_img_ids:
    #         info = self.coco.load_imgs([img_id])[0]
    #         info['filename'] = info['file_name']
    #         self.img_ids.append(img_id)
    #         data_infos.append(info)
    #     return data_infos
