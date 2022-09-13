'''
Info: /hdd/0/zzh/project/SCIDet/mmlab/mmtracking/tools/coco_viz.py
Author: Zhihong Zhang <z_zhi_hong@163.com>
Created: 2022-09-13 20:39:02
-----
Last Modified: 2022-09-13 22:04:44
Modified By: Zhihong Zhang <z_zhi_hong@163.com>
-----
Copyright (c) 2022 Zhihong Zhang
-----
HISTORY:
Date         			By          	Comments
----------------------	------------	---------------------------------------------------------
'''


import cv2
import numpy as np
import os
from tqdm import tqdm
import os.path as osp
from pycocotools.coco import COCO


def coco_bbox_viz(img_ids, json_path, img_path, save_dir=None, color=None, line_width=2, font_scale=0.5):
    """
    coco bbox visualization

    Args:
        img_ids (list[ints]): image ids to be visualized
        json_path (str): json annotation path
        img_path (str): image dataset dir
        save_dir (str, optional): visualization result saving dir. Defaults to None, i.e. show rather than save.
        color (list[3*(3,)], optional): bbox/text colors for different catagories. Defaults to None, i.e. use random colors.
        line_width (int, optional): bbox/text line width Defaults to 2.
        font_scale (float, optional): text font scale. Defaults to 0.5.
    """
    # instantiate COCO object
    coco = COCO(json_path)

    # get class cat_id2cat_name maps
    CLASSES = coco.dataset['categories']
    cls_map = {cls['id']: cls['name'] for cls in CLASSES}

    # cat_id2cat_color map
    if color is None:
        color_map = {cls['id']: np.random.randint(
            0, 256, (3,)) for cls in CLASSES}
    else:
        color_map = {cls['id']: color[k] for k, cls in enumerate(CLASSES)}

    # get all image infos
    list_imgIds = coco.getImgIds()
    imgs = coco.loadImgs(ids=img_ids)

    # draw anno
    for img in tqdm(imgs):
        image = cv2.imread(img_path + img['file_name'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_name = img['file_name']
        image_id = img['id']
        img_annIds = coco.getAnnIds(imgIds=[image_id])
        img_anns = coco.loadAnns(ids=img_annIds)
        for i in range(len(img_annIds)):
            x, y, w, h = img_anns[i - 1]['bbox']  # bbox
            cat_id = img_anns[i - 1]['category_id']  # category
            image = cv2.rectangle(image, (int(x), int(y)), (int(
                x + w), int(y + h)), color_map[cat_id].tolist(), line_width)
            image = cv2.putText(
                image, cls_map[cat_id], (x-5, y-5),  cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_map[cat_id].tolist(), line_width//2)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if save_dir:
            save_path = osp.join(save_dir, image_name)
            if not osp.exists(save_path.rsplit(os.sep, 1)[0]):
                os.makedirs(save_path.rsplit(os.sep, 1)[0])
            cv2.imwrite(save_path, image)
        else:
            cv2.imshow(image_name, image)


if __name__ == "__main__":

    img_ids = list(range(1, 5900))
    save_dir = '/hdd/0/zzh/project/SCIDet/mmlab/mmtracking/output/tmp/demo'
    json_path = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style_NewSplit/annotations/uadetrac_vid_val.json'
    img_path = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style_NewSplit/Data/VID/'
    coco_bbox_viz(img_ids, json_path, img_path, save_dir=save_dir)
