# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 下午9:49
# @Author  : zxq
# @File    : test_tmp.py
# @Software: PyCharm
import mmcv

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, inference_detector, show_result_pyplot
from tools.train_tmp import CustomerTrain

customer_train = CustomerTrain()
cfg = customer_train.cfg

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

img = mmcv.imread('../data/kitti_tiny/training/image_2/000068.jpeg')

model.cfg = cfg
result = inference_detector(model, img)
show_result_pyplot(model, img, result)