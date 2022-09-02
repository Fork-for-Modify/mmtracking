
_base_ = [
    '../_base_/datasets/uadetrac_scidet_style.py',
    '../_base_/default_runtime.py'
]

#-------------------
# import setting
#-------------------
# custom_imports = dict(
#     imports=['mmtrack.models.necks.my_fpn.py'],
#     allow_failed_imports=False)


#-------------------
# model setting
#-------------------
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)  # img_norm_cfg for [0,255] image
# img_norm_cfg_1 = dict(
#     mean=[103.530/255, 116.280/255, 123.675/255], std=[1.0, 1.0, 1.0], to_rgb=False)  # img_norm_cfg for [0,1] image


detector = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=4,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

model = dict(
    type='SCIFCOS',
    detector=detector,
    scidecoder=dict(type='EnergyNorm', norm4det=img_norm_cfg))


#-------------------
# dataset settings
#-------------------

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
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids'],
        default_meta_key_values=dict(img_norm_cfg=img_norm_cfg)),
    # dict(
    #     type='SCIDataCollect',
    #     keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='SCIDataArrange'),
    dict(type='SCIFormatBundle')
]

test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    # dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    # dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqPad', size_divisor=16),
    dict(type='SCIEncoding', fixed_mask=False, mask_path=None, norm2one=False),
    dict(
        type='SCIDataCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='SCIDataArrange'),
    dict(type='SCIFormatBundle')
    # dict(type='MultiImagesToTensor', ref_prefix='ref'),
    # dict(type='ToList')
]

# update pipeline setting
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

#-------------------
# optimizer settings
#-------------------
# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

#-------------------
# runtime settings
#-------------------
total_epochs = 7
evaluation = dict(metric=['bbox'], interval=7)
