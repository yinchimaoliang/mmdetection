# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/butterfly/'
img_norm_cfg = dict(
    mean=[77.00059091, 108.81410419, 101.77048004], std=[60.49471351, 57.49471009, 58.85109194], to_rgb=True)
classes=['蟾福蛱蝶', '素弄蝶', '线灰蝶', '无斑珂弄蝶', '老豹蛱蝶', '突角小粉蝶', '尖翅翠蛱蝶', '柑橘凤蝶', '箭纹绢粉蝶', '碧凤蝶', '亮灰蝶', '银斑豹蛱蝶', '宽边黄粉蝶',
               '灿福蛱蝶', '蛇目褐蚬蝶', '巴黎翠凤蝶', '黄钩蛱蝶', '翠袖锯眼蝶', '红基美凤蝶', '虬眉带蛱蝶', '黄环蛱蝶', '翠蓝眼蛱蝶', '隐纹谷弄蝶', '蓝点紫斑蝶', '大紫琉璃灰蝶',
               '古北拟酒眼蝶', '绿豹蛱蝶', '西门珍眼蝶', '伊诺小豹蛱蝶', '网蛱蝶', '阿芬眼蝶', '波太玄灰蝶', '红灰蝶', '雅弄蝶', '花弄蝶', '美眼蛱蝶', '银豹蛱蝶',
               '牧女珍眼蝶', '柳紫闪蛱蝶', '婀灰蝶', '扬眉线蛱蝶', '绢斑蝶', '箭纹云粉蝶', '中环蛱蝶', '青海红珠灰蝶', '大卫粉蝶', '蓝凤蝶', '曲斑珠蛱蝶', '金裳凤蝶',
               '边纹黛眼蝶', '链环蛱蝶', '荨麻蛱蝶', '黎明豆粉蝶', '秀蛱蝶', '艳灰蝶', '依帕绢蝶', '白眼蝶', '白钩蛱蝶', '青凤蝶', '云粉蝶', '珍蛱蝶', '直纹稻弄蝶',
               '拟稻眉眼蝶', '蓝灰蝶', '大翅绢粉蝶', '玄珠带蛱蝶', '珍珠绢蝶', '曲纹紫灰蝶', '琉璃蛱蝶', '山豆粉蝶', '云豹蛱蝶', '菜粉蝶', '小红蛱蝶', '橙黄豆粉蝶', '绢蛱蝶',
               '朴喙蝶', '玄灰蝶', '虎斑蝶', '玉带凤蝶', '钩翅眼蛱蝶', '侏粉蝶', '小黄斑弄蝶', '咖灰蝶', '柱菲蛱蝶', '密纹飒弄蝶', '镉黄迁粉蝶', '斐豹蛱蝶', '四川绢蝶',
               '黑网蛱蝶', '维纳斯眼灰蝶', '绢粉蝶', '红襟粉蝶', '锦瑟蛱蝶', '菩萨酒眼蝶']

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True),
    #     keymap={
    #         'img': 'image',
    #         'gt_bboxes': 'bboxes'
    #     },
    #     update_pad_shape=False,
    #     skip_img_without_anno=True),
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
        type='ClassBalancedDataset',
        dataset=dict(
            type='RepeatDataset',
            times=3,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root + 'ImageSets/Main/trainval.txt',
                img_prefix=data_root,
                pipeline=train_pipeline,
                classes=classes
                ),

        ),
        oversample_thr=0.01,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root + 'TestData',
        pipeline=test_pipeline,
        test_mode=True,
        classes=['蟾福蛱蝶', '素弄蝶', '线灰蝶', '无斑珂弄蝶', '老豹蛱蝶', '突角小粉蝶', '尖翅翠蛱蝶', '柑橘凤蝶', '箭纹绢粉蝶', '碧凤蝶', '亮灰蝶', '银斑豹蛱蝶',
                 '宽边黄粉蝶',
                 '灿福蛱蝶', '蛇目褐蚬蝶', '巴黎翠凤蝶', '黄钩蛱蝶', '翠袖锯眼蝶', '红基美凤蝶', '虬眉带蛱蝶', '黄环蛱蝶', '翠蓝眼蛱蝶', '隐纹谷弄蝶', '蓝点紫斑蝶',
                 '大紫琉璃灰蝶',
                 '古北拟酒眼蝶', '绿豹蛱蝶', '西门珍眼蝶', '伊诺小豹蛱蝶', '网蛱蝶', '阿芬眼蝶', '波太玄灰蝶', '红灰蝶', '雅弄蝶', '花弄蝶', '美眼蛱蝶', '银豹蛱蝶',
                 '牧女珍眼蝶', '柳紫闪蛱蝶', '婀灰蝶', '扬眉线蛱蝶', '绢斑蝶', '箭纹云粉蝶', '中环蛱蝶', '青海红珠灰蝶', '大卫粉蝶', '蓝凤蝶', '曲斑珠蛱蝶', '金裳凤蝶',
                 '边纹黛眼蝶', '链环蛱蝶', '荨麻蛱蝶', '黎明豆粉蝶', '秀蛱蝶', '艳灰蝶', '依帕绢蝶', '白眼蝶', '白钩蛱蝶', '青凤蝶', '云粉蝶', '珍蛱蝶', '直纹稻弄蝶',
                 '拟稻眉眼蝶', '蓝灰蝶', '大翅绢粉蝶', '玄珠带蛱蝶', '珍珠绢蝶', '曲纹紫灰蝶', '琉璃蛱蝶', '山豆粉蝶', '云豹蛱蝶', '菜粉蝶', '小红蛱蝶', '橙黄豆粉蝶',
                 '绢蛱蝶',
                 '朴喙蝶', '玄灰蝶', '虎斑蝶', '玉带凤蝶', '钩翅眼蛱蝶', '侏粉蝶', '小黄斑弄蝶', '咖灰蝶', '柱菲蛱蝶', '密纹飒弄蝶', '镉黄迁粉蝶', '斐豹蛱蝶', '四川绢蝶',
                 '黑网蛱蝶', '维纳斯眼灰蝶', '绢粉蝶', '红襟粉蝶', '锦瑟蛱蝶', '菩萨酒眼蝶']
    ))
evaluation = dict(interval=1, metric='mAP')
