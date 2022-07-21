_base_ = [
    './dff_faster_rcnn_r50_dc5_1x_imagenetvid.py'
]
# model = dict(
#     type='DFF',
#     detector=dict(
#         train_cfg=dict(
#             rpn_proposal=dict(max_per_img=1000),
#             rcnn=dict(sampler=dict(num=512)))),
#     motion=dict(
#         type='FlowNetSimple',
#         img_scale_factor=0.5,
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint=  # noqa: E251
#             'https://download.openmmlab.com/mmtracking/pretrained_weights/flownet_simple.pth'  # noqa: E501
#         )),
#     train_cfg=None,
#     test_cfg=dict(key_frame_interval=10))

# # optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[2, 5])
# # runtime settings
# total_epochs = 7
# evaluation = dict(metric=['bbox'], interval=7)

# dataset settings
data_root = '/hdd/0/dkm/mmtracking_lite/data/ILSVRC1/'  # zzh:数据集根目录
dataset_type = 'ImagenetVIDDataset'
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
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/imagenet_vid_train.json',
            img_prefix=data_root + 'Data/VID',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=9,
                filter_key_img=False,
                method='uniform'),
            pipeline=train_pipeline)
    ],
    val=dict(
        ann_file=data_root + 'annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'Data/VID'),
    test=dict(
        ann_file=data_root + 'annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'Data/VID'))
