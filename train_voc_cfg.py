import numpy as np
from PIL import Image

import os.path as osp
from tqdm import tqdm

import mmcv
import mmengine
import matplotlib.pyplot as plt

data_root = '/data/home/scv9243/run/mmsegmentation/data/VOCdevkit/VOC2012'
img_dir = 'JPEGImages'
ann_dir = 'SegmentationClass'

classes = ('background', 'object')
palette = [[128, 128, 128], [151, 189, 8]]


# classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
#          'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#          'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
#          'sofa', 'train', 'tvmonitor'),
# palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
#          [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
#          [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
#          [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
#          [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
#          [0, 64, 128]]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class VOCBackgroundDataset(BaseSegDataset):
  METAINFO = dict(classes=classes, palette=palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)

from mmengine import Config
cfg = Config.fromfile('/data/home/scv9243/run/mmsegmentation/configs/upernet/upernet_r50_4xb2-40k_cityscapes-512x1024.py')

cfg.norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
cfg.crop_size = (512, 1024)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.pretrained = '/data/home/scv9243/run/mmsegmentation/checkpoint/resnest50_d2-7497a55b.pth'
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
# cfg.model.decode_head.num_classes = 21
# cfg.model.auxiliary_head.num_classes = 21

# 修改数据集的 type 和 root
cfg.dataset_type = 'PascalVOCDataset'
cfg.data_root = data_root

cfg.train_dataloader.batch_size = 8

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.dataset.ann_file = '/data/home/scv9243/run/mmsegmentation/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = '/data/home/scv9243/run/mmsegmentation/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

cfg.test_dataloader = cfg.val_dataloader


# 载入预训练模型权重
cfg.load_from = '/data/home/scv9243/run/mmsegmentation/checkpoint/upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth'

# 工作目录
cfg.work_dir = '/data/home/scv9243/run/mmsegmentation/work_dirs/upernet'

# 训练迭代次数
cfg.train_cfg.max_iters = 4000
# 评估模型间隔
cfg.train_cfg.val_interval = 400
# 日志记录间隔
cfg.default_hooks.logger.interval = 100
# 模型权重保存间隔
cfg.default_hooks.checkpoint.interval = 400

# 随机数种子
cfg['randomness'] = dict(seed=0)
cfg.dump('upernet_voc.py')

from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# register all modules in mmseg into the registries
# do not init the default scope here because it will be init in the runner
register_all_modules(init_default_scope=False)
runner = Runner.from_cfg(cfg)

runner.train()
