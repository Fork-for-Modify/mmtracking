
_base_ = [
    '../_base_/models/faster_rcnn_r50_dc5.py',
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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    type='SCISELSA',
    detector=dict(
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='TemporalRoIAlign',
                num_most_similar_points=2,
                num_temporal_attention_blocks=4,
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='SelsaBBoxHead',
                num_shared_fcs=3,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)))),
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
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 5])

#-------------------
# runtime settings
#-------------------
total_epochs = 100
evaluation = dict(metric=['bbox'], interval=7)
