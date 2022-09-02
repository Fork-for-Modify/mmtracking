# Tutorial for Open-MMLab
### 教程网址
1. MMDetection: https://mmdetection.readthedocs.io/zh_CN/latest/
    - 配置文件的详细介绍 https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/config.html?highlight=norm_cfg
2. MMTracking: https://mmtracking.readthedocs.io/zh_CN/latest/

### 训练已有的 VID 模型
1. 数据集转换为CocoVID格式：https://github.com/open-mmlab/mmtracking/tree/master/tools/convert_datasets/

### 训练自己的 VID 模型

1. 自定义数据集
https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/customize_dataset.html

2. 自定义 VID 模型
https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/customize_vid_model.html
e.g.
https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/config_vid.html

3. 准备配置文件
https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/config.html

4. 训练新模型
单GPU
`python tools/scidet_train.py ${CONFIG_FILE} [optional arguments]`
e.g. TemporalROI
`CUDA_VISIBLE_DEVICES=1 PORT=29501 python tools/train.py ./configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_uadetracvid_zzh.py --work-dir ./output/tmp/train/ `

e.g. SCIDet
`CUDA_VISIBLE_DEVICES=6 PORT=29501 python tools/scidet_train.py ./configs/scidet/scidet_troi_uadetracsci.py --work-dir ./output/dev/train/  `


多GPU
`CUDA_VISIBLE_DEVICES=6,3 PORT=29502 ./tools/dist_scidet_train.sh ${config_file} ${gpu_num} --work-dir ${work_dir}`
e.g.
`CUDA_VISIBLE_DEVICES=6,3 PORT=29502 ./tools/dist_scidet_train.sh ./configs/scidet/scidet_troi_uadetracsci.py 2 --work-dir ./output/dev/train/ `

常用配置
 - SCIFCOS: ./configs/scidet/sci_fcos_uadetracsci.py
 - SCISELSA(troi): ./configs/scidet/scidet_troi_uadetracsci.py
 - temporal ROI: ./configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_zzh.py

5. 测试和评估
下载预训练模型
单GPU
python ./tools/scidet_test.py \
    ${config_file_path} \
    --checkpoint ${checkpoint_path} \
    --work-dir ${eval metrics res dir}
    --out ${bbox result .pkl path} \
    --eval bbox \
    --format-only \ #  
    --show \ # show the resutls
    --show-dir ${save_res_dir}

e.g. SCIDet
python ./tools/scidet_test.py ./configs/scidet/scidet_troi_uadetracsci.py --checkpoint ./record/temporalROI_train1/latest.pth --work-dir ./output/tmp/test/ --out ./output/tmp/test/res.pkl --eval bbox --gpu-id 1 --show-dir ./output/tmp/test/res_imgs/

e.g. TemporalROI
python tools/test.py ./configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_uadetracvid_zzh.py --checkpoint ./model_zoo/temporalROI/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth --out ./output/tmp/test/res.pkl --eval bbox --gpu-id 1 --show-dir ./output/tmp/test/res_imgs/


多GPU
e.g.
使用 8 GPUs 测试 DFF，并且评估 bbox mAP
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh ./configs/scidet/scidet_troi_uadetracsci.py 2 --checkpoint ./output/dev/train1/epoch_229.pth --out /output/tmp/test/res.pkl 

6. demo
python ./tools/scidet_demo.py \
    ${config_file_path} \
    --input ${VIDEO_FILE} \
    --checkpoint ${checkpoint_path} \
    --output ${OUTPUT} \
    --show

e.g. SCIDet
python ./tools/scidet_demo.py ./configs/scidet/scidet_troi_uadetracsci.py --device cuda:1 --input ./data/uadetrac_40201_200/VID/val/40201/ --checkpoint ./record/temporalROI_train1/latest.pth --output ./output/tmp/test/ 

e.g. TemporalROI
python ./demo/demo_vid.py ./configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_uadetracvid_zzh.py --device cuda:1 --input ./data/uadetrac_40201_200/VID/val/40201/ --checkpoint ./model_zoo/temporalROI/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth --output ./output/tmp/test/ 