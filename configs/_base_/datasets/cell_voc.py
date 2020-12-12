# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/cell/'
classes = ['226', '50', '224', '39', '323', '74', '27', '174', '66', '61', '285', '281', '209', '63', '324', '59', '173', '34', '28', '211', '210', '49', '225', '172', '60', '19', '46', '43', '57', '52', '93', '35', '91']
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'ImageSets/Main/train.txt',
            img_prefix=data_root,
            pipeline=train_pipeline,
            classes=classes)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes))
evaluation = dict(interval=1, metric='mAP')
