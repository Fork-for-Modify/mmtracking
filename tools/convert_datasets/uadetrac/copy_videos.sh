# copy videos to new dir
root_dir='/hdd/0/zzh/dataset/UA_DETRAC/coco_style/Data/VID/test/'
dst_dir='/hdd/0/zzh/dataset/UA_DETRAC/coco_style/Data/VID/test_dkm/'
videos='MVI_39051 MVI_39211 MVI_39271 MVI_39311 MVI_39511 MVI_40701 MVI_40711 MVI_40714 MVI_40742'

for vid in $videos
do
cp -r $root_dir$vid $dst_dir
done