# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 下午9:20
# @Author  : zxq
# @File    : kitti_tiny_dataset.py
# @Software: PyCharm

import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class KittiTinyDataset(CustomDataset):
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file)  # 获取所有的图片名

        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:  # 根据图片名，找到对应的标注文件，解析并获取标注信息
            filename = f'{self.img_prefix}/{image_id}.jpeg'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]

            data_info = dict(filename=f'{image_id}.jpeg', width=width, height=height)

            # load annotations
            label_prefix = self.img_prefix.replace('image_2', 'label_2')
            lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))  # 一张图片的多个标注信息

            content = [line.strip().split(' ') for line in lines]  # 每个item是一个标注
            bbox_names = [x[0] for x in content]
            bboxes = [[float(info) for info in x[4:8]] for x in content]  # x是一个标注信息， 索引[4, 8)是bbox， 把所有的bbox转成float

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])  # 通过Pedestrian获取对应的label=1, 保存到gt_labels
                    gt_bboxes.append(bbox)  # Pedestrian相对应的bbox，要保存到gt_bboxes
                else:
                    gt_labels_ignore.append(-1)  # 如果文件里的标注类别不在声明的类别CLASSES里面，则忽略这些标注，并记录下来
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),  # 目的是list转<np.ndarray> (n, 4)
                labels=np.array(gt_labels, dtype=np.long),  # 目的是list转<np.ndarray> (n, )
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)  # 完善当前图片的其他信息
            data_infos.append(data_info)  # 添加一张图片的完整信息

        return data_infos