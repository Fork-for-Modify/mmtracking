# dataset settings
dataset_type = 'UADETRACSCIDataset'
data_root = '/hdd/0/dkm/mmtracking_lite/data/ILSVRC1/'  # zzh:数据集根目录
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    # dict(type='SeqCvtColor', src_color='bgr', dst_color='gray'),
    # dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqPad', size_divisor=16),
    dict(type='SCIEncoding', fixed_mask=False, mask_path=None, norm2one=True),
    dict(
        type='SCIDataCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids'],
        default_meta_key_values=dict(img_norm_cfg=img_norm_cfg)),
    dict(type='SCIDataArrange'),
    dict(type='SCIFormatBundle')
]

test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    # dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    # dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqPad', size_divisor=16),
    dict(type='SCIEncoding', fixed_mask=False, mask_path=None),
    dict(
        type='SCIDataCollect',
        keys=['img']),
    dict(type='SCIDataArrange'),
    dict(type='SCIFormatBundle')
    # dict(type='MultiImagesToTensor', ref_prefix='ref'),
    # dict(type='ToList')
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/imagenet_vid_train.json',
            img_prefix=data_root + 'Data/VID',
            ref_img_sampler=dict(
                num_ref_imgs=10,  # = Cr
                frame_range=5,  # omit
                filter_key_img=False,  # omit
                method='bilateral'),
            pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'Data/VID',
        ref_img_sampler=dict(
            num_ref_imgs=10,  # = Cr
            frame_range=5,  # omit
            filter_key_img=False,  # omit
            method='bilateral'),
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'Data/VID',
        ref_img_sampler=dict(
            num_ref_imgs=10,  # = Cr
            frame_range=5,  # omit
            filter_key_img=False,  # omit
            method='bilateral'),
        pipeline=test_pipeline,
        test_mode=True))
