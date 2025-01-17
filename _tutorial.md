# Tutorial for Open-MMLab
## 0. 教程网址
1. MMDetection: https://mmdetection.readthedocs.io/zh_CN/latest/
    - 配置文件的详细介绍 https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/config.html?highlight=norm_cfg
    
2. MMTracking: https://mmtracking.readthedocs.io/zh_CN/latest/


## 1. 安装注事项
因为安装的目的是为了修改代码，写自己的模型，所以安装的时候要“从源码构建并安装”，具体的，

对于MMDetection
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

对于MMTracking

```
git clone https://github.com/open-mmlab/mmtracking.git
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## 2. 训练自己的模型

1. 自定义数据集
https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/customize_dataset.html

2. 自定义 VID 模型
https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/customize_vid_model.html
e.g.
https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/config_vid.html

3. 准备配置文件
https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/config.html

4. 训练新模型

### 单GPU

`python tools/scidet_train.py ${CONFIG_FILE} [optional arguments]`

- e.g. SCIDet

    - `CUDA_VISIBLE_DEVICES=5 PORT=29501 python tools/scidet_train.py ./configs/scidet/scidet_troi_uadetracsci.py --work-dir ./output/tmp/train/  `

- e.g. TemporalROI

    - `CUDA_VISIBLE_DEVICES=1 PORT=29501 python tools/train.py ./configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_uadetracvid_zzh.py --work-dir ./output/tmp/train/ `

### 多GPU

`CUDA_VISIBLE_DEVICES=0,1 PORT=29502 ./tools/dist_scidet_train.sh ${config_file} ${gpu_num} --work-dir ${work_dir}`

- e.g. SCIDet
    - `CUDA_VISIBLE_DEVICES=0,1 PORT=29502 ./tools/dist_scidet_train.sh ./configs/scidet/scidet_troi_uadetracsci.py 2 --work-dir ./output/dev/train/ `

### 常用配置
 - SCIFCOS: ./configs/scidet/sci_fcos_uadetracsci.py
 - SCISELSA(troi): ./configs/scidet/scidet_troi_uadetracsci.py
 - temporal ROI: ./configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_zzh.py

## 3. 测试和评估模型
下载预训练模型
### 单GPU

```
python ./tools/scidet_test.py \
    ${config_file_path} \
    --checkpoint ${checkpoint_path} \
    --work-dir ${eval metrics res dir}
    --out ${bbox result .pkl path} \
    --eval bbox \
    --format-only \ # Format the results without evaluation 
    --show \ # show the resutls
    --show-dir ${save_res_dir}
    --no-evaluate # no evaluation
```

- e.g. SCIDet
    - `python ./tools/scidet_test.py ./configs/scidet/scidet_troi_uadetracsci.py --checkpoint ./output/dev/train/latest.pth --work-dir ./output/tmp/test/ --out ./output/tmp/test/scidet_res_test.pkl --eval bbox --gpu-id 5 --show-dir ./output/tmp/test/res_imgs/`

- e.g. TemporalROI
    - `python tools/test.py ./configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_uadetracvid_zzh.py --checkpoint ./model_zoo/temporalROI/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth --out ./output/tmp/test/res.pkl --eval bbox --gpu-id 1 --show-dir ./output/tmp/test/res_imgs/`


### 多GPU
```
python ./tools/dist_scidet_test.sh \
    ${config_file_path} \
    --checkpoint ${checkpoint_path} \
    --out ${bbox result .pkl path} \
    --eval bbox \
    --format-only \ # Format the results without evaluation
    --no-evaluate # no evaluation
```

- e.g. SCIDet
    - `CUDA_VISIBLE_DEVICES=0,7 PORT=29506 ./tools/dist_scidet_test.sh ./configs/scidet/scidet_troi_uadetracsci.py 2 --checkpoint ./output/dev/train/latest.pth --out ./output/tmp/test/scidet_res.pkl --eval bbox`


- e.g. TemporalROI
    - `CUDA_VISIBLE_DEVICES=0,1 PORT=29506 ./tools/dist_test.sh ./configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_uadetracvid_zzh.py 2 --checkpoint ./model_zoo/temporalROI/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth --out ./output/tmp/test/res.pkl --eval bbox`

## 4. 模型 Demo/Inference
```
python ./tools/scidet_demo.py \
    ${config_file_path} \
    --input ${VIDEO_FILE} \
    --checkpoint ${checkpoint_path} \
    --output ${OUTPUT} \
    --show
```

- e.g. SCIDet
    - `python ./tools/scidet_demo.py ./configs/scidet/scidet_troi_uadetracsci.py --device cuda:7 --input ./data/uadetrac_40201_200/VID/val/40201/ --checkpoint ./output/dev/train/latest.pth --output ./output/tmp/demo/`

- e.g. TemporalROI
    - `python ./demo/demo_vid.py ./configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_uadetracvid_zzh.py --device cuda:1 --input ./data/uadetrac_40201_200/VID/val/40201/ --checkpoint ./model_zoo/temporalROI/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth --output ./output/tmp/test/ `