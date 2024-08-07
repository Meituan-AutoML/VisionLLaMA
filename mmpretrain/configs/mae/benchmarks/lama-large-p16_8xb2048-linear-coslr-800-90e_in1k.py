_base_ = [
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
train_dataloader = dict(batch_size=2048, drop_last=True)
val_dataloader = dict(drop_last=False)
test_dataloader = dict(drop_last=False)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionLLaMA',
        arch='large',
        img_size=224,
        patch_size=16,
        frozen_stages=24,
        out_type='avg_featmap',
        final_norm=False,
        init_cfg=dict(type='Pretrained',
                      checkpoint='work_dirs/mae_lama-large-p16_8xb512-amp-coslr-800e_in1k/epoch_800.pth',
                      prefix='backbone.')),
    neck=dict(type='ClsBatchNormNeck', input_features=1024),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.01)]))

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='LARS', lr=6.4, weight_decay=0.0, momentum=0.9))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        by_epoch=True,
        begin=10,
        end=90,
        eta_min=0.0,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=90)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=10))

randomness = dict(seed=0, diff_rank_seed=True)
