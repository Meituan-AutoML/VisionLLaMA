_base_ = './fpn_poolformer_s12_8xb4-40k_ade20k-512x512.py'
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m48_3rdparty_32xb128_in1k_20220414-9378f3eb.pth'  # noqa

# model settings
model = dict(
    backbone=dict(
        arch='m48',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384, 768]))
