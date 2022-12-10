_base_ = './retinanet/retinanet_r50_fpn_1x_coco.py'

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
        ann_file = '/mnt/f/b_data/annotations/trainnew.json',
        img_prefix = '/mnt/f/b_data/train2017'
    ),
    val=dict(
        # type=dataset_type,
        classes=classes,
        ann_file = '/mnt/f/b_data/annotations/valnew.json',
        img_prefix = '/mnt/f/b_data/val2017'
    ),
    test = dict(
        # type = dataset_type,
        classes=classes,
        ann_file = '/mnt/f/b_data/annotations/valnew.json',
        img_prefix = '/mnt/f/b_data/val2017'
    )
)

# 2. model settings

model = dict(
        bbox_head=
            dict(
                type = 'RetinaHead',
                num_classes=21,
        ),
    )

resume_from = './work_dirs/custom2/epoch_11.pth'

