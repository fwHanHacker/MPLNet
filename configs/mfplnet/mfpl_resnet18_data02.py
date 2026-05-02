net = dict(type='Detector', )

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 5
sample_y = range(530, 0, -5)

heads = dict(type='MFPLHead',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=108)

iou_loss_weight = 2.
cls_loss_weight = 6.
xyt_loss_weight = 0.5
seg_loss_weight = 1.0

work_dirs = "work_dirs/mfpl/r18_data02"

neck = dict(type='FPN',
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=True)

test_parameters = dict(conf_threshold=0.30, nms_thres=45, nms_topk=max_lanes)

epochs = 100
batch_size = 16

optimizer = dict(type='AdamW', lr=1.0e-3)
total_iter = (3616 // batch_size + 1) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 3
save_ep = epochs

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])

ori_img_w = 360
ori_img_h = 540
img_h = 360
img_w = 540
cut_height = 0

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = 'mfplnet/data/data02'
dataset_type = 'Data02'
test_json_file = 'mfplnet/data/data02/seg_label/test.json'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='train',
    processes=train_process,
),
val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='val',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

workers = 0
log_interval = 100
num_classes = 7 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = True