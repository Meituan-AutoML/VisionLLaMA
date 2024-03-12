_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    './lsj-100e_coco-instance.py',
]

custom_imports = dict(imports=['projects.ViTDet.vitdet'])

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='LLaMADETv1',
        arch='base',
        img_size=1024,
        patch_size=16,
        drop_path_rate=0.1,
        out_type='featmap',
        final_norm=False,
        out_indices=[-1],
        window_size=14,
        window_block_indexes=[
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        avg=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../mmpretrain/work_dirs/mae_lama-base-p16_8xb512-amp-coslr-800e_in1k/epoch_800.pth',
            prefix='backbone.'
        )),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))


optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='MyLearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
    },
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ))

custom_hooks = [dict(type='Fp16CompresssionHook')]
