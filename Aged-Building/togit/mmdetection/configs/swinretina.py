_base_ = './swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py'

# dataset_type = 'MyDataset'
dataset_type = 'CocoDataset'
classes = ('crack_Good', 'crack_Normal', 'crack_Faulty', 
            'peel_Good', 'peel_Normal', 'peel_Faulty', 
            'rebar_Good', 'rebar_Normal', 'rebar_Faulty', 
            'ground_Good', 'ground_Normal', 'ground_Faulty', 
            'finish_Good', 'finish_Normal', 'finish_Faulty', 
            'window_Good', 'window_Normal', 'window_Faulty', 
            'living_Good', 'living_Normal', 'living_Faulty')

train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),#, with_mask=True),
            dict(type='Resize', img_scale=(1080, 1440), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=.0),
            # dict(type='RandomShift'),
            # dict(type='PhotoMetricDistortion'),
            # dict(type='CutOut', n_holes=2, cutout_ratio=(0.6,0.4)),
            # dict(type='RandomAffine'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'])
        ]
test_pipeline =[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512,512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]
# 1. dataset settings
data = dict(
    samples_per_gpu = 8,
    workers_per_gpu = 0,
    train = dict(
        # type='MultiImageMixDataset',
        classes = classes,
        ann_file = '../aug/trainnew2.json',
        img_prefix = '../aug/',
        pipeline=train_pipeline
    ),
    val=dict(
        # type='MultiImageMixDataset',
        classes=classes,
        ann_file = '../data/valnew2.json',
        img_prefix = '../data/val2017',
        pipeline = test_pipeline
    ),
    test = dict(
        # type='MultiImageMixDataset',
        classes=classes,
        ann_file = '../data/valnew2.json',
        img_prefix = '../data/val2017',
        pipeline = test_pipeline
    )
)

# 2. model settings

model = dict(
        bbox_head=
            dict(
                type = 'RetinaHead',
                num_classes=21,
        ))        

# fp16 = dict(loss_scale=512.)

runner = dict(type='EpochBasedRunner', max_epochs=48)
resume_from = '../sr1217/epoch_36.pth'