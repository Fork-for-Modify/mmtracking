import os
import os.path as osp
from tqdm import tqdm

# param
root_dir = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style/Data/VID/test/'
dst_dir = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style/Data/VID/test_small/'
extract_num = 200

# proc
subdirs = sorted(os.listdir(root_dir))

for subdir in tqdm(subdirs):
    os.makedirs(osp.join(dst_dir, subdir), exist_ok=True)
    files = sorted(os.listdir(osp.join(root_dir, subdir)))
    for i in range(extract_num):
        os.system(
            f'cp {osp.join(root_dir, subdir,files[i])} {osp.join(dst_dir, subdir)}')
