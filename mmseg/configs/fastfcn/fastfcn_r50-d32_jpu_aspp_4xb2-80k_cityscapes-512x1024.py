# model settings
_base_ = './fastfcn_r50-d32_jpu_psp_4xb2-80k_cityscapes-512x1024.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        _delete_=True,
        type='ASPPHead',
        in_channels=2048,
        in_index=2,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
