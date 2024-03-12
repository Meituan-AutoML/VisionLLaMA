_base_ = ['./twins_svt-s_uperhead_8xb2-160k_ade20k-512x512.py']

checkpoint = '../deit/cpt/pllama_wols_small_patch16_224_300/best_checkpoint.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='PyramidVisionLLaMA',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        patch_size=4, embed_dim=512, depth=12, num_heads=[2, 4, 8, 16],
        embed_dims=[64, 128, 256, 512], depths=[2, 2, 10, 4],
        drop_path_rate=0.2,
        auto_scale=True
    ),
    decode_head=dict(in_channels=[64, 128, 256, 512]),
    auxiliary_head=dict(in_channels=256),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

resume = True



