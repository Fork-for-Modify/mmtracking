# zzh: NotImp
_base_ = [
    '../../_base_/models/faster_rcnn_r50_dc5.py',
    '../../_base_/datasets/uadetrac_vid_scidet_style.py',
    '../../_base_/default_runtime.py'
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
model = dict(
    type='SCI_FCOS',
    detector=dict(
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_head=dict(
                type='SelsaBBoxHead',
                num_shared_fcs=2,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)))))


#-------------------
# dataset settings
#-------------------


#-------------------
# optimizer settings
#-------------------
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 5])

#-------------------
# runtime settings
#-------------------
total_epochs = 7
evaluation = dict(metric=['bbox'], interval=7)
