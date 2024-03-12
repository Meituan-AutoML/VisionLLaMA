_base_ = [
    '../_base_/models/upernet_mae.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)


model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='mmpretrain.VisionLLaMA',
        arch='base',
        img_size=512,
        patch_size=16,
        drop_path_rate=0.1,
        out_type='featmap',
        final_norm=False,
        out_indices=[3, 5, 7, 11],
        init_cfg=dict(type='Pretrained',
                      checkpoint='../mmpretrain/work_dirs/mae_lama-base-p16_8xb512-amp-coslr-800e_in1k/epoch_800.pth',
                      prefix='backbone.')),
    neck=dict(embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[768, 768, 768, 768], num_classes=150, channels=768),
    auxiliary_head=dict(in_channels=768, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
    constructor='LayerDecayOptimizerConstructor')

# optim_wrapper = dict(
#     optimizer=dict(
#         type='AdamW', lr=2e-3, weight_decay=0.05, betas=(0.9, 0.999)),
#     constructor='LearningRateDecayOptimWrapperConstructor',
#     paramwise_cfg=dict(
#         layer_decay_rate=0.65,
#         custom_keys={
#             '.ln': dict(decay_mult=0.0),
#             '.bias': dict(decay_mult=0.0),
#             '.cls_token': dict(decay_mult=0.0),
#             '.pos_embed': dict(decay_mult=0.0)
#         }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
