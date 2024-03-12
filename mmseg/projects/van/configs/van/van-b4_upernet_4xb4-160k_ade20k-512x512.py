_base_ = './van-b2_upernet_4xb2-160k_ade20k-512x512.py'
ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b4_3rdparty_20230522-1d71c077.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[3, 6, 40, 3],
        init_cfg=dict(type='Pretrained', checkpoint=ckpt_path),
        drop_path_rate=0.4))

# By default, models are trained on 4 GPUs with 4 images per GPU
train_dataloader = dict(batch_size=4)
