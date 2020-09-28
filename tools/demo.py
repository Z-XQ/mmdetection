from mmdet.apis import init_detector
from mmdet.apis import inference_detector, show_result_pyplot


if __name__ == '__main__':
    # Choose to use a config and initialize the detector
    config = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = '../weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # initialize the detector
    model = init_detector(config, checkpoint, device='cuda:0')

    img = '../data/car.jpg'
    result = inference_detector(model, img)

    show_result_pyplot(model, img, result, score_thr=0.3)

