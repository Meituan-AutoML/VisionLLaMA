_base_ = './fpn_poolformer_s12_8xb4-40k_ade20k-512x512.py'
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s24_3rdparty_32xb128_in1k_20220414-d7055904.pth'  # noqa
# model settings
model = dict(
    backbone=dict(
        arch='s24',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')))
