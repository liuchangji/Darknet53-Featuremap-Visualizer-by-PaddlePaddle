# 开启端到端训练

from visualdl import LogWriter
from Data_loader import multithread_loader

from objectness import get_objectness_label

import paddle.fluid as fluid

from darknet53 import DarkNet53_conv_body, YoloDetectionBlock, YOLOv3

from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from paddle.fluid.dygraph.base import to_variable

from utils.txtythtw_2_xyxy import get_yolo_box_xxyy
from utils.get_iou_above_thresh_inds_function import get_iou_above_thresh_inds, label_objectness_ignore

from utils.loss import get_loss

import paddle
import numpy as np
import time

from config.config_example import get_cfg

cfg = get_cfg()

NUM_CLASSES = cfg.NUM_CLASSES

LEARNING_RATE = 0.001 #学习率必须为浮点数

TRAINDIR = cfg.TRAINDIR

BATCHSIZE = cfg.BATCHSIZE
MAX_EPOCH = cfg.MAX_EPOCH

IGNORE_THRESH = cfg.IGNORE_THRESH

def get_lr(base_lr=0.0001, lr_decay=0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    return learning_rate


if __name__ == "__main__":

    log_writer = LogWriter("./log")  # 训练参数保存

    ANCHORS = cfg.ANCHOR

    ANCHOR_MASKS = cfg.ANCHOR_MASKS

    with fluid.dygraph.guard():
        params_dict,opt11111 = fluid.load_dygraph("/home/tuxiang/theCode/paddlepaddlestudy/YOLOfromZERO/yolo_coco_epoch49")
        model = YOLOv3(num_classes=NUM_CLASSES, is_train=True)  # 创建模型
        model.load_dict(params_dict)
        #learning_rate = get_lr()
        opt = fluid.optimizer.AdadeltaOptimizer(learning_rate=LEARNING_RATE, parameter_list=model.parameters())
        # opt = fluid.optimizer.Momentum(
        #     learning_rate=learning_rate,
        #     momentum=0.9,
        #     regularization=fluid.regularizer.L2Decay(0.0005),
        #     parameter_list=model.parameters())  # 创建优化器
        train_loader = multithread_loader(TRAINDIR, batch_size=BATCHSIZE, mode='train')  # 创建训练数据读取器
        #valid_loader = multithread_loader(VALIDDIR, batch_size=BATCHSIZE, mode='valid')  # 创建验证数据读取器

        for epoch in range(MAX_EPOCH):

            for i, data in enumerate(train_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = to_variable(gt_scores)
                img = to_variable(img)
                gt_boxes = to_variable(gt_boxes)
                gt_labels = to_variable(gt_labels)
                outputs = model(img)  # 前向传播，输出[P0, P1, P2]
                loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors=ANCHORS,
                                      anchor_masks=ANCHOR_MASKS,
                                      ignore_thresh=IGNORE_THRESH,
                                      use_label_smooth=False)  # 计算损失函数

                loss.backward()  # 反向传播计算梯度
                opt.minimize(loss)  # 更新参数
                model.clear_gradients()
                # if i % 100 == 0:
                #     log_writer.add_scalar(tag='acc', step=i, value=loss.numpy())
                if i % 1 == 0:
                    timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    print('{}[TRAIN]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))

            # save params of model
            if (epoch % 20 == 0) or (epoch == MAX_EPOCH - 1):
                fluid.save_dygraph(model.state_dict(), 'yolo_coco_epoch{}'.format(epoch))
                #fluid.save_dygraph(opt.state_dict(), 'yolo_epoch{}'.format(epoch))

            # 每个epoch结束之后在验证集上进行测试
            # model.eval()
            # for i, data in enumerate(valid_loader()):
            #     img, gt_boxes, gt_labels, img_scale = data
            #     gt_scores = np.ones(gt_labels.shape).astype('float32')
            #     gt_scores = to_variable(gt_scores)
            #     img = to_variable(img)
            #     gt_boxes = to_variable(gt_boxes)
            #     gt_labels = to_variable(gt_labels)
            #     outputs = model(img)
            #     loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
            #                           anchors=ANCHORS,
            #                           anchor_masks=ANCHOR_MASKS,
            #                           ignore_thresh=IGNORE_THRESH,
            #                           use_label_smooth=False)
            #     if i % 1 == 0:
            #         timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            #         print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
            model.train()

    # reader = multithread_loader(TRAINDIR, batch_size=2, mode='train')
    # img, gt_boxes, gt_labels, im_shape = next(reader())
    # # 计算出锚框对应的标签
    # label_objectness, label_location, label_classification, scale_location = get_objectness_label(img,
    #                                                                                               gt_boxes, gt_labels,
    #                                                                                               iou_threshold=0.7,
    #                                                                                               anchors=[116, 90, 156,
    #                                                                                                        198, 373,
    #                                                                                                        326],
    #                                                                                               num_classes=NUM_CLASSES,
    #                                                                                               downsample=32)
    # num_filters = NUM_ANCHORS * (NUM_CLASSES + 5)
    # with fluid.dygraph.guard():
    #     backbone = DarkNet53_conv_body(is_test=False)  # 主干网络
    #     detection = YoloDetectionBlock(ch_in=1024, ch_out=512, is_test=False)  # 检测头，如果没有多尺度融合的话也就是一个普通的特征提取网络
    #     conv2d_pred = Conv2D(num_channels=1024, num_filters=num_filters, filter_size=1)
    #
    #     x = to_variable(img)
    #     print(img.shape)
    #     C0, C1, C2 = backbone(x)
    #     route, tip = detection(C0)
    #     P0 = conv2d_pred(tip)
    #
    #     # anchors包含了预先设定好的锚框尺寸
    #     # downsample是特征图P0的步幅
    #     pred_boxes = get_yolo_box_xxyy(P0.numpy(), anchors, num_classes=NUM_CLASSES, downsample=32)
    #     iou_above_thresh_indices = get_iou_above_thresh_inds(pred_boxes, gt_boxes, iou_threshold=IGNORE_THRESH)
    #     label_objectness = label_objectness_ignore(label_objectness, iou_above_thresh_indices)
    #     print(label_objectness.shape)
    #
    #     label_objectness = to_variable(label_objectness)
    #     label_location = to_variable(label_location)
    #     label_classification = to_variable(label_classification)
    #     scales = to_variable(scale_location)
    #     label_objectness.stop_gradient = True
    #     label_location.stop_gradient = True
    #     label_classification.stop_gradient = True
    #     scales.stop_gradient = True
    #
    #     total_loss = get_loss(P0, label_objectness, label_location, label_classification, scales,
    #                           num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES)
    #     total_loss_data = total_loss.numpy()
    #     print("total loss = %f" % total_loss_data)
