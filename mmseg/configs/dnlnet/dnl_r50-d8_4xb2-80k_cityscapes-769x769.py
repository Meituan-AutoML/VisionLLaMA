_base_ = [
    '../_base_/models/dnl_r50-d8.py',
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (769, 769)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(theta=dict(wd_mult=0.), phi=dict(wd_mult=0.))))
