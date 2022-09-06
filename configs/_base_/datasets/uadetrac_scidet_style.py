# dataset settings
dataset_type = 'UADETRACSCIDataset'
data_root = '/hdd/0/zzh/dataset/UA_DETRAC/coco_style/'  # root dir for dataset
Cr = 10  # zzh: compressive ratio
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    # dict(type='SeqCvtColor', src_color='bgr', dst_color='gray'),
    # dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqPad', size_divisor=16),
    dict(type='SCIEncoding', fixed_mask=False, mask_path=None, norm2one=False),
    dict(
        type='SCIDataCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']
        # default_meta_key_values=dict(img_norm_cfg=img_norm_cfg)
    ),
    dict(type='SCIDataArrange'),
    dict(type='SCIFormatBundle')
]

test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    # dict(type='SeqCvtColor', src_color='bgr', dst_color='gray'),
    # dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    # dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqPad', size_divisor=16),
    dict(type='SCIEncoding', fixed_mask=False, mask_path=None, norm2one=False),
    dict(
        type='SCIDataCollect',
        keys=['img']),
    dict(type='SCIDataArrange'),
    dict(type='SCIMultiImagesToTensor')
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/uadetrac_vid_train.json',
        img_prefix=data_root + 'Data/VID',
        key_img_sampler=dict(interval=5),
        ref_img_sampler=dict(
            num_ref_imgs=Cr,  # = Cr
            method='right'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/uadetrac_vid_val_small.json',  # val/val_small
        img_prefix=data_root + 'Data/VID',
        key_img_sampler=dict(interval=1),
        ref_img_sampler=dict(
            num_ref_imgs=Cr,  # = Cr
            method='right'),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/uadetrac_vid_test.json',
        img_prefix=data_root + 'Data/VID',
        key_img_sampler=dict(interval=Cr),
        ref_img_sampler=dict(
            num_ref_imgs=Cr,  # = Cr
            method='right'),
        pipeline=test_pipeline,
        test_mode=True))
