_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/butterfly_voc.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=0.01),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            num_classes=94,
            )))

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)

