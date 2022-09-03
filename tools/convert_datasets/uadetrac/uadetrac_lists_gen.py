
# ================================
# sample frames and generate frame id List (.txt files)
# ================================

from logging import root
import os
# --------------
# VID_train_15frames.txt
# --------------
interval = 1  # sampling interval
prefix = 'train/'
# root_dir: annotations_xml dir (contains image-wise annotations converted by DETRAC_xmlParser.py)
root_dir = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style/annotations_xml/VID/train/'
dst_path = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style/Lists/VID_train_frames.txt'

video_dirs = sorted(os.listdir(root_dir))  # get video dir video_dirmes
with open(dst_path, 'w') as f:
    for video_dir in video_dirs:

        video_full_dir = "".join([root_dir, video_dir, os.sep])
        frames = sorted(os.listdir(video_full_dir))
        frame_ids = [int(frame[3:8])
                     for frame in frames]  # frames e.g.: img00001.xml
        frame_num = len(frame_ids)

        head_str = prefix + video_dir + ' 1 '
        tail_str = ' ' + str(frame_num) + '\n'

        for k in range(0, frame_num, interval):
            # frame_id starts from 1
            info = ''.join([head_str, str(frame_ids[k]), tail_str])
            f.write(info)
print('Finished, save file to: '+dst_path)
