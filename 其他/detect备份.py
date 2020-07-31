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
    image_name = '/home/tuxiang/theCode/paddlepaddlestudy/YOLOfromZERO/2007_007836.jpg'
    params_file_path = './yolo_coco_epoch9.pdparams'
    import matplotlib.image as imgplt
    with fluid.dygraph.guard():
        model = YOLOv3(num_classes=NUM_CLASSES, is_train=False)
        model_state_dict, _ = fluid.load_dygraph(params_file_path)
        model.load_dict(model_state_dict)
        model.eval()
        img_data = imgplt.imread(image_name)

        mean = [0.471, 0.448, 0.408]
        std = [0.234, 0.239, 0.242]
        mean = np.array(mean).reshape((1, 1, -1))
        std = np.array(std).reshape((1, 1, -1))
        img_data = (img_data / 255.0 - mean) / std
        h, w, c = img_data.shape


        img_data = img_data.astype('float32').transpose((2, 0, 1))

        img = np.zeros((1, 3, h, w)).astype('float32')
        img[0,:,:,:] = img_data
        # img = np.ones()
        # total_results = []
        # test_loader = single_test_data_loader(image_name, mode='test')
        # for i, data in enumerate(test_loader()):
        img = to_variable(img)



        img_scale = 640

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
    draw_results(result, image_name, draw_thresh=0.1)
    plt.show()