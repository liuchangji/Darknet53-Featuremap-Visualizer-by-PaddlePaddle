import json

import paddle
import paddle.fluid as fluid
from darknet53 import YOLOv3
from utils.NMS import multiclass_nms
from utils.draw_bounding_box import draw_results
from paddle.fluid.dygraph.base import to_variable
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

from config.config_example import get_cfg

cfg = get_cfg()

ANCHORS = cfg.ANCHOR
ANCHOR_MASKS = cfg.ANCHOR_MASKS
VALID_THRESH = cfg.VALID_THRESH
NMS_TOPK = cfg.NMS_TOPK
NMS_POSK = cfg.NMS_POSK
NMS_THRESH = cfg.NMS_THRESH

NUM_CLASSES = cfg.NUM_CLASSES

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# 读取单张测试图片

if __name__ == '__main__':
    from Data_loader import single_test_data_loader
    image_name = '/home/tuxiang/theCode/paddlepaddlestudy/YOLOfromZERO/test'
    params_file_path = './yolo_coco_epoch49.pdparams'
    with fluid.dygraph.guard():
        model = YOLOv3(num_classes=NUM_CLASSES, is_train=False)
        model_state_dict, _ = fluid.load_dygraph(params_file_path)
        model.load_dict(model_state_dict)
        model.eval()

        total_results = []
        test_loader = single_test_data_loader(image_name, mode='test')
        for i, data in enumerate(test_loader()):

            img_name, img_data, img_scale_data = data
            img = to_variable(img_data)
            img_scale = to_variable(img_scale_data)

            outputs = model.forward(img)
            bboxes, scores = model.get_pred(outputs,
                                            im_shape=img_scale,
                                            anchors=ANCHORS,
                                            anchor_masks=ANCHOR_MASKS,
                                            valid_thresh=VALID_THRESH)

            bboxes_data = bboxes.numpy()
            scores_data = scores.numpy()
            results = multiclass_nms(bboxes_data, scores_data,
                                     score_thresh=VALID_THRESH,
                                     nms_thresh=NMS_THRESH,
                                     pre_nms_topk=NMS_TOPK,
                                     pos_nms_topk=NMS_POSK)

    result = results[0]
    draw_results(result, os.path.join(image_name,(str(img_name[0])+".jpg")), draw_thresh=0.4)
    plt.show()