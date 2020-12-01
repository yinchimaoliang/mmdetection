_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/butterfly_voc.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(

    roi_head=dict(
        bbox_head=dict(
            num_classes=94,
            )))

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)

