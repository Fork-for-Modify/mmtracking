# dataset settings
dataset_type = 'UADETRACVIDDataset'
data_root = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style/'  # zzh:数据集根目录 | 'data/ILSVRC/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/uadetrac_vid_train.json',
        img_prefix=data_root + 'Data/VID',
        ref_img_sampler=dict(
            num_ref_imgs=2,
            frame_range=9,
            filter_key_img=True,
            method='bilateral_uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/uadetrac_vid_val.json',
        img_prefix=data_root + 'Data/VID',
        ref_img_sampler=dict(
            num_ref_imgs=30,
            frame_range=[-15, 15],
            stride=1,
            method='test_with_fix_stride'),
        pipeline=test_pipeline,
        test_mode=False),  # zzh: should be false for val
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/uadetrac_vid_val.json', # zzh: test/val
        img_prefix=data_root + 'Data/VID',
        ref_img_sampler=dict(
            num_ref_imgs=30,
            frame_range=[-15, 15],
            stride=1,
            method='test_with_fix_stride'),
        pipeline=test_pipeline,
        test_mode=True))
