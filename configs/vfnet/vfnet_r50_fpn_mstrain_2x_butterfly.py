_base_ = './vfnet_r50_fpn_1x_butterfly.py'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
