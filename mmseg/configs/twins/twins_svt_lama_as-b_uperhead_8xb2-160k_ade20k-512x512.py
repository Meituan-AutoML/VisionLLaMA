_base_ = ['./twins_svt-s_uperhead_8xb2-160k_ade20k-512x512.py']

checkpoint = '../deit/cpt/pllama_wols_base_patch16_224_300/best_checkpoint.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='PyramidVisionLLaMA',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        patch_size=4, embed_dim=768, depth=12, num_heads=[3, 6, 12, 24],
        embed_dims=[96, 192, 384, 768], depths=[2, 2, 18, 2],
        drop_path_rate=0.2,
        auto_scale=True
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768]),
    auxiliary_head=dict(in_channels=384),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

resume = True



