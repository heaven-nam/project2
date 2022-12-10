_base_ = './swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'

# dataset_type = 'MyDataset'
dataset_type = 'CocoDataset'
classes = ('crack_Good', 'crack_Normal', 'crack_Faulty', 'peel_Good', 'peel_Normal', 'peel_Faulty', 'rebar_Good', 'rebar_Normal', 'rebar_Faulty', 'ground_Good', 'ground_Normal', 'ground_Faulty', 'finish_Good', 'finish_Normal', 'finishi_Faulty', 'window_Good', 'window_Normal', 'window_Faulty', 'living_Good', 'living_Normal', 'living_Faulty')

# 1. dataset settings
data = dict(
    samples_per_gpu = 4,
    workers_per_gpu = 0,
    train = dict(
        # type=dataset_type,
        classes = classes,
        ann_file = '/mnt/f/b_data/annotations/trainnew2.json',
        img_prefix = '/mnt/f/b_data/train2017',
    ),
    val=dict(
        # type=dataset_type,
        classes=classes,
        ann_file = '/mnt/f/b_data/annotations/valnew2.json',
        img_prefix = '/mnt/f/b_data/val2017',
    ),
    test = dict(
        # type = dataset_type,
        classes=classes,
        ann_file = '/mnt/f/b_data/annotations/valnew2.json',
        img_prefix = '/mnt/f/b_data/val2017',
    )
)

# 2. model settings

model = dict(
    roi_head = dict(
        bbox_head=
            dict(
                type = 'Shared2FCBBoxHead',
                num_classes=21,
        ),
        mask_head = dict(
            type='FCNMaskHead',
            num_classes=21
        )
    ))