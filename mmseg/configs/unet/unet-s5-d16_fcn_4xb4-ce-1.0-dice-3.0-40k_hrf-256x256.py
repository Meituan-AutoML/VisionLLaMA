_base_ = './unet-s5-d16_fcn_4xb4-40k_hrf-256x256.py'
model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ]))
