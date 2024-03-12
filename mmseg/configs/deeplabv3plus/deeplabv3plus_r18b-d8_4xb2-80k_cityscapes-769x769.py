_base_ = './deeplabv3plus_r50-d8_4xb2-40k_cityscapes-769x769.py'
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
