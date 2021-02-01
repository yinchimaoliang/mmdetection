# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/cell/complete-1-26/'
classes = ['巨杆状核粒细胞', '原单核细胞', '巨晚幼红细胞', '多核浆细胞', '中性杆状核粒细胞', '巨中性分叶核粒细胞', '中幼酸', '中性晚幼粒细胞', '嗜酸性粒细胞', '幼稚淋巴细胞', '嗜酸性分叶核粒细胞', '幼稚浆细胞', '裸核型巨核细', '嗜酸杆', '原始浆细胞', '嗜碱', '嗜酸性杆', '嗜碱晚', '晚幼酸', '小巨核细胞', '成熟淋巴细胞', '嗜酸性分叶', '嗜酸晚', '嗜酸中', '中性中幼红细胞', '中幼粒细胞', '嗜酸性晚幼粒', '幼稚单核细胞', '巨幼样杆', '巨中性中幼粒细胞', '幼淋巴细胞', '分叶核粒细胞', '成熟单核细胞', '幼单核细胞', '巨中性杆状核粒细胞', '巨幼样杆状核', '嗜酸性中幼粒细胞', '早幼红细胞', '杆状核粒细胞', '巨幼样晚幼', '双核浆细胞', '巨中幼红细胞', '吞噬细胞', '巨中性杆状核幼粒细胞', '内皮细胞', '分裂相', '中性杆状核细胞', '含H-J小体的晚幼红细胞', '巨晚幼粒细胞', '巨原始红细胞', '退化细胞', '中性分叶核粒细胞', '淋巴细胞', '巨晚幼红细胞伴核畸形', '嗜酸分叶', '原始红细胞', '晚幼', '中性杆状核幼粒细胞', '异常早幼粒', '粒细胞核畸形', '中幼红细胞', '异常早幼粒细胞', '巨中性晚幼粒细胞', '含H-J小体晚幼红细胞', '中性中幼粒细胞', '原始淋巴细胞', '巨中性分叶核幼粒细胞', '成熟浆细胞', '嗜酸性晚幼粒细胞', '原始粒细胞', '原始单核细胞', '中幼分叶核粒细胞', '幼稚巨核细胞', '异型淋巴细胞', '晚幼粒细胞', '分叶核', '晚幼红细胞', '嗜酸性中幼粒', '巨早幼红细胞', '组织嗜碱细胞', '巨晚幼红细胞伴豪焦小体', '杆状核', '巨中幼粒细胞', '原始巨核细胞', '嗜酸性杆状核粒细胞', '网状细胞', '巨杆状核', '浆细胞', '嗜酸分叶核', '早幼粒细胞', '嗜碱性粒细胞']
img_norm_cfg = dict(
    mean=[216.29945567, 185.00888325, 197.98572283], std=[16.8434652,  53.26599443, 37.6515256], to_rgb=True)
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
    samples_per_gpu=6,
    workers_per_gpu=6,
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
        samples_per_gpu=12,
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes))
evaluation = dict(interval=1, metric='mAP')
