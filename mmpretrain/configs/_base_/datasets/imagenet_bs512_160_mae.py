# dataset settings
dataset_type = 'ImageNet'
data_root = '/workdir/ILSVRC2012/' # 'data/imagenet/'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True, #pillow RGB
    non_blocking=True
)

train_pipeline = [
    # dict(type='LoadImageFromFile', imdecode_backend='pillow', channel_order='rgb'),
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=160,
        crop_ratio_range=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=512,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        pipeline=train_pipeline))
