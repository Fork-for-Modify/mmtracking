# ================================
# Convert UA-DETRAC dataset to coco style
# 1. DETRAC_xmlParser.py: convert UA-DETRAC xml annotation files to VOC style xml annotation files
# 2. separate train/test/val (both image files and annotation files)
# 3. uadetrac_lists_gen.py: generate frame samping List txt files for them
# 4. uadetrac2coco_vid.py: generate coco style annotation
# Note:
# - some video contains frames without annotatation.
# ================================

import os
import os.path as osp
import xml.etree.ElementTree as ET
from collections import defaultdict

import mmcv
from tqdm import tqdm


def parse_txt_list(txt_file_path):
    """Parse the txt file of UA-DETRAC VID dataset."""
    img_list = mmcv.list_from_file(txt_file_path)
    train_infos = defaultdict(list)
    for info in img_list:
        info = info.split(' ')
        if info[0] not in train_infos:
            train_infos[info[0]] = dict(
                vid_frame_ids=[int(info[2]) - 1], num_frames=int(info[-1]))  # frame_id starts from 0
        else:
            train_infos[info[0]]['vid_frame_ids'].append(
                int(info[2]) - 1)  # frame_id starts from 0
    return train_infos


def convert_vid(CLASSES, CLASSES_ENCODES, xml_dir, txt_path, save_dir, mode='train'):
    """Convert UA-DETRAC VID dataset in COCO style.

    Args:
        CLASSES (tuple): The classes contained in the dataset.
        CLASSES_ENCODES (tuple): The encodes of the classes.
        xml_dir (str): The dir of UA-DETRAC VID dataset's image-wise xml annotations.
        txt_path (str): The path of UA-DETRAC VID dataset's frame_ids txt file
        save_dir (str): The path to save converted coco style annotations.
        mode (str): Convert train dataset or validation dataset. Options are
            'train', 'val', 'test'. 
    """
    assert mode in ['train', 'val', 'test', 'val_small']

    # category
    categories = []
    for k, v in enumerate(CLASSES, 1):
        categories.append(
            dict(id=k, name=v, encode_name=CLASSES_ENCODES[k - 1]))
    VID = defaultdict(list)
    VID['categories'] = categories

    cats_id_maps = {}
    for k, v in enumerate(CLASSES_ENCODES, 1):
        cats_id_maps[v] = k

    records = dict(
        vid_id=1,
        img_id=1,
        ann_id=1,
        global_instance_id=1,
        num_vid_frame_ids=0,
        num_no_objects=0)
    obj_num_classes = dict()
    vid_infos = parse_txt_list(txt_path)
    for vid_info in tqdm(vid_infos):
        instance_id_maps = dict()
        vid_frame_ids = vid_infos[vid_info].get('vid_frame_ids', [])
        records['num_vid_frame_ids'] += len(vid_frame_ids)
        video = dict(
            id=records['vid_id'],
            name=vid_info)
        VID['videos'].append(video)
        # num_frames = vid_infos[vid_info]['num_frames']
        for frame_id in vid_frame_ids:
            is_vid_train_frame = True if mode == 'train' else False  # train frame flag
            img_prefix = osp.join(vid_info, 'img%05d' % (frame_id+1))
            xml_name = osp.join(xml_dir, f'{img_prefix}.xml')
            # parse XML annotation file
            tree = ET.parse(xml_name)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            image = dict(
                file_name=img_prefix.rsplit(
                    '/', 1)[0] + '/' + root.find('filename').text,
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'],
                is_vid_train_frame=is_vid_train_frame)
            VID['images'].append(image)
            if root.findall('object') == []:
                print(xml_name, 'has no objects.')
                records['num_no_objects'] += 1
                records['img_id'] += 1
                continue
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in cats_id_maps:
                    continue
                category_id = cats_id_maps[name]
                bnd_box = obj.find('bndbox')
                x1, y1, x2, y2 = [
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)
                ]
                w = x2 - x1
                h = y2 - y1
                track_id = obj.find('trackid').text
                if track_id in instance_id_maps:
                    instance_id = instance_id_maps[track_id]
                else:
                    instance_id = records['global_instance_id']
                    records['global_instance_id'] += 1
                    instance_id_maps[track_id] = instance_id
                # occluded = obj.find('occluded').text
                speed = float(obj.find('speed').text)
                truncation_ratio = float(obj.find('truncation_ratio').text)
                ann = dict(
                    id=records['ann_id'],
                    video_id=records['vid_id'],
                    image_id=records['img_id'],
                    category_id=category_id,
                    instance_id=instance_id,
                    bbox=[x1, y1, w, h],
                    area=w * h,
                    # iscrowd=False,
                    # occluded=occluded,
                    speed=speed,
                    truncation_ratio=truncation_ratio)
                if category_id not in obj_num_classes:
                    obj_num_classes[category_id] = 1
                else:
                    obj_num_classes[category_id] += 1
                VID['annotations'].append(ann)
                records['ann_id'] += 1
            records['img_id'] += 1
        records['vid_id'] += 1
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(VID, osp.join(save_dir, f'uadetrac_vid_{mode}.json'))
    print(f'-----UA-DETRAC VID {mode}------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["img_id"]- 1} images')
    # print(
    #     f'{records["num_vid_frame_ids"]} train frames for video detection')
    print(f'{records["num_no_objects"]} images have no objects')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------')
    for i in range(1, len(CLASSES) + 1):
        print(f'Class {i} {CLASSES[i - 1]} has {obj_num_classes[i]} objects.')


def main():

    # params
    mode = 'val_small'  # 'train', 'test', 'val', 'val_small'
    xml_dir = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style/annotations_xml/VID/'
    output_dir = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style/annotations/'
    txt_path = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style/Lists/VID_'+mode+'_frames.txt'

    # info
    CLASSES = ('car', 'van', 'bus', 'others')
    CLASSES_ENCODES = ('car', 'van', 'bus', 'others')

    convert_vid(CLASSES, CLASSES_ENCODES, xml_dir,
                txt_path, output_dir, mode=mode)


if __name__ == '__main__':
    main()
