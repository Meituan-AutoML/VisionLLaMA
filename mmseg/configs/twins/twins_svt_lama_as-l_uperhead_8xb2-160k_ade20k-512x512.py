_base_ = ['./twins_svt-s_uperhead_8xb2-160k_ade20k-512x512.py']

checkpoint = '../deit/cpt/pllama_wols_large_patch16_224_300/best_checkpoint.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='PyramidVisionLLaMA',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        patch_size=4, embed_dim=1024, depth=12, num_heads=[4, 8, 16, 32],
        embed_dims=[128, 256, 512, 1024], depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
        auto_scale=True
    ),
    decode_head=dict(in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(in_channels=512),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

resume = True



