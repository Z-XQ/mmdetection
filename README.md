官方的README主要讲的是有些什么模型，模型的性能如何。这里介绍的是源码和如何定制我们自己的训练策略。

# 1. 最重要的机制-注册机制
注册机制，是预先把类注册到一个字典中，在任何需要的时候，实例化这个类即可，这样如果有新的功能，不需要修改旧有的代码，比如我有个新的
数据集KittyDataset类，只需要注册，然后通过配置，从注册器里面找到对应的类，实例化这个类就可以用起来了。
## Registry
通过register_module方法，注册类。所有的类将会保存在Registry对象的属性self._module_dict中，key是类名或者指定的名字，value是类本身。

先把所有以修饰器方式@x.register_module()或者普通函数方式x.register_module(module=SomeClass)，注册完这些类，然后才会通过build_from_cfg
建立对象。
## build_from_cfg(cfg, registry, default_args=None)
通过cfg字典里面具体的配置，从registry里面找到对应的预先注册好的类，并实例化这个类，返回类对象。default_args用来修改cfg里面的配置。
cfg有个key是type，value是类名或者注册时指定的其他名字，比如cfg['type']='KittiTinyDataset', 
则会从registry中找到对应的类KittiTinyDataset，然后实例化并返回这个类对象。

# 2. Dataset
自定义数据集比较简单的有两种方式，一种是converting the data into the format of existing datasets like COCO, VOC, etc.
一种是need to read annotations of each image and convert them into middle format MMDetection accept，下面介绍第二种方式
## 2.1 继承CustomDataset
自定义的Dataset需要继承CustomDataset类，然后复写方法load_annotations，该方法的作用是读取并解析标注信息，返回一个list, 该list格式是：

    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

示例：
```python
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
```     
## 2.2 建立
需要使用的时候，通过：dataset = build_dataset(cfg_dict)，实例化KittiTinyDataset，其中cfg_dict实例化所需要的参数，内容如下：
```
    train=dict(
    type='KittiTinyDataset',
    ann_file='train.txt',  # 如果没有指定data_root，则ann_file就是标注文件的完整路径，其中train.txt保存的是所有图片名
    img_prefix='training/image_2',  # 如果没有指定data_root，则img_prefix就是原始图片的完整路径
    pipeline=train_pipeline,
    data_root='../data/kitti_tiny/'),
    
    也可以是（即把data_root去掉）：
    train=dict(
    type='KittiTinyDataset',
    ann_file='../data/kitti_tiny/train.txt',  
    img_prefix='../data/kitti_tiny/training/image_2',  
    pipeline=train_pipeline),
```
具体过程：
build_dataset会调用build_from_cfg(cfg_dict, DATASETS, default_args)，其中DATASETS是
注册器DATASETS = Registry('dataset'), 刚实例化时这个注册器的两个属性成员self._name = 'dataset'
self._module_dict = dict()是为空，而KittiTinyDataset类会以修饰器的方式@DATASETS.register_module()
注册到了DATASETS注册器中。build_from_cfg在前面的注册机制已经讲过了。

# 3. model
建立的过程和dataset是一样的，以FasterRCNN为例，通过
model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)实例化model，
其中，cfg.model是实例化FasterRCNN需要的参数, train_cfg是训练阶段用到的参数，test_cfg是测试时用到的参数。
cfg.model内容：
```python
model = dict(
    type='FasterRCNN',
    pretrained=None,
    backbone=...,
    neck=...,
    rpn_head=...,
    roi_head=...)
```
build_detector会同样调用build_from_cfg(cfg, registry, default_args)，其中default_args包含了train_cfg和
test_cfg，实例化FasterRCNN：
```python
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
```
可以看到实例化FasterRCNN需要这7个参数，准确的说两阶段的检测器TwoStageDetector都需要这7个参数，
其中5个参数通过cfg.model传递，另外2个，train_cfg和test_cfg需要单独传递。

FasterRCNN -> TwoStageDetector -> BaseDetector, BaseDetector是抽象类，子类TwoStageDetector必须要实现4个方法：
```python
@abstractmethod
def extract_feat(self, imgs):
    """Extract features from images."""
    pass

@abstractmethod
def forward_train(self, imgs, img_metas, **kwargs):
    """
    Args:
        img (list[Tensor]): List of tensors of shape (1, C, H, W).
            Typically these should be mean centered and std scaled.
        img_metas (list[dict]): List of image info dict where each dict
            has: 'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys, see
            :class:`mmdet.datasets.pipelines.Collect`.
        kwargs (keyword arguments): Specific to concrete implementation.
    """
    pass

@abstractmethod
def aug_test(self, imgs, img_metas, **kwargs):
    """Test function with test time augmentation."""
    pass

@abstractmethod
def simple_test(self, img, img_metas, **kwargs):
    pass
```
