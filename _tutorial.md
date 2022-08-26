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
`python tools/train.py ${CONFIG_FILE} [optional arguments]`
e.g.
`CUDA_VISIBLE_DEVICES=6 PORT=29501 python tools/train.py ./configs/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_zzh.py --work-dir ./output/test/ `

`CUDA_VISIBLE_DEVICES=0 PORT=29501 python tools/train.py ./configs/vid/scidet/sci_fcos_faster_rcnn_r50_dc5_1x_imagenetvid.py --work-dir ./output/test/ `

多GPU
`python tools/train.py ${NEW_CONFIG_FILE}`
e.g.
`CUDA_VISIBLE_DEVICES=6,7 PORT=29501 ./tools/dist_train.sh ./configs/vid/dff/dff_faster_rcnn_r50_dc5_1x_imagenetvid_zzh.py 2 --work-dir ./output/train/ `

常用配置
 - SCIFCOS: ./configs/scidet/sci_fcos_uadetracsci.py
 - SCISELSA: ./configs/scidet/sci_troi_uadetracsci.py
 - temporal ROI: ./configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_zzh.py

5. 测试和推理
`python tools/test.py ${NEW_CONFIG_FILE} ${TRAINED_MODEL} --eval bbox track `

