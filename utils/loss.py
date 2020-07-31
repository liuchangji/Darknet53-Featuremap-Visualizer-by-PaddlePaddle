import paddle.fluid as fluid


def get_loss(self, outputs, gtbox, gtlabel, gtscore=None,
             anchors=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
             anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
             ignore_thresh=0.7,
             use_label_smooth=False):
    """
    使用fluid.layers.yolov3_loss，直接计算损失函数，过程更简洁，速度也更快
    """
    self.losses = []
    downsample = 32
    for i, out in enumerate(outputs):  # 对三个层级分别求损失函数
        anchor_mask_i = anchor_masks[i]
        loss = fluid.layers.yolov3_loss(
            x=out,  # out是P0, P1, P2中的一个
            gt_box=gtbox,  # 真实框坐标
            gt_label=gtlabel,  # 真实框类别
            gt_score=gtscore,  # 真实框得分，使用mixup训练技巧时需要，不使用该技巧时直接设置为1，形状与gtlabel相同
            anchors=anchors,  # 锚框尺寸，包含[w0, h0, w1, h1, ..., w8, h8]共9个锚框的尺寸
            anchor_mask=anchor_mask_i,  # 筛选锚框的mask，例如anchor_mask_i=[3, 4, 5]，将anchors中第3、4、5个锚框挑选出来给该层级使用
            class_num=self.num_classes,  # 分类类别数
            ignore_thresh=ignore_thresh,  # 当预测框与真实框IoU > ignore_thresh，标注objectness = -1
            downsample_ratio=downsample,  # 特征图相对于原图缩小的倍数，例如P0是32， P1是16，P2是8
            use_label_smooth=False)  # 使用label_smooth训练技巧时会用到，这里没用此技巧，直接设置为False
        self.losses.append(fluid.layers.reduce_mean(loss))  # reduce_mean对每张图片求和
        downsample = downsample // 2  # 下一级特征图的缩放倍数会减半
    return sum(self.losses)  # 对每个层级求和

def get_loss_old(output, label_objectness, label_location, label_classification, scales, num_anchors=3, num_classes=7):
    # 将output从[N, C, H, W]变形为[N, NUM_ANCHORS, NUM_CLASSES + 5, H, W]
    reshaped_output = fluid.layers.reshape(output, [-1, num_anchors, num_classes + 5, output.shape[2], output.shape[3]])

    # 从output中取出跟objectness相关的预测值，ignore_index=-1 忽略-1
    pred_objectness = reshaped_output[:, :, 4, :, :]
    loss_objectness = fluid.layers.sigmoid_cross_entropy_with_logits(pred_objectness, label_objectness, ignore_index=-1)
    ## 对第1，2，3维求和
    #loss_objectness = fluid.layers.reduce_sum(loss_objectness, dim=[1,2,3], keep_dim=False)

    # pos_samples 只有在正样本的地方取值为1.，其它地方取值全为0.
    pos_objectness = label_objectness > 0
    pos_samples = fluid.layers.cast(pos_objectness, 'float32')
    pos_samples.stop_gradient=True

    #从output中取出所有跟位置相关的预测值
    tx = reshaped_output[:, :, 0, :, :]
    ty = reshaped_output[:, :, 1, :, :]
    tw = reshaped_output[:, :, 2, :, :]
    th = reshaped_output[:, :, 3, :, :]

    # 从label_location中取出各个位置坐标的标签
    dx_label = label_location[:, :, 0, :, :]
    dy_label = label_location[:, :, 1, :, :]
    tw_label = label_location[:, :, 2, :, :]
    th_label = label_location[:, :, 3, :, :]
    # 构建损失函数
    loss_location_x = fluid.layers.sigmoid_cross_entropy_with_logits(tx, dx_label)
    loss_location_y = fluid.layers.sigmoid_cross_entropy_with_logits(ty, dy_label)
    loss_location_w = fluid.layers.abs(tw - tw_label)
    loss_location_h = fluid.layers.abs(th - th_label)

    # 计算总的位置损失函数
    loss_location = loss_location_x + loss_location_y + loss_location_h + loss_location_w

    # 乘以scales
    loss_location = loss_location * scales
    # 只计算正样本的位置损失函数
    loss_location = loss_location * pos_samples

    #从ooutput取出所有跟物体类别相关的像素点
    pred_classification = reshaped_output[:, :, 5:5+num_classes, :, :]
    # 计算分类相关的损失函数
    loss_classification = fluid.layers.sigmoid_cross_entropy_with_logits(pred_classification, label_classification)
    # 将第2维求和
    loss_classification = fluid.layers.reduce_sum(loss_classification, dim=2, keep_dim=False)
    # 只计算objectness为正的样本的分类损失函数
    loss_classification = loss_classification * pos_samples
    total_loss = loss_objectness + loss_location + loss_classification
    # 对所有预测框的loss进行求和
    total_loss = fluid.layers.reduce_sum(total_loss, dim=[1,2,3], keep_dim=False)
    # 对所有样本求平均
    total_loss = fluid.layers.reduce_mean(total_loss)

    return total_loss


if __name__ == "__main__":
    # 计算损失函数

    # 读取数据
    reader = multithread_loader('/home/aistudio/work/insects/train', batch_size=2, mode='train')
    img, gt_boxes, gt_labels, im_shape = next(reader())
    # 计算出锚框对应的标签
    label_objectness, label_location, label_classification, scale_location = get_objectness_label(img,
                                                                                                  gt_boxes, gt_labels,
                                                                                                  iou_threshold=0.7,
                                                                                                  anchors=[116, 90, 156,
                                                                                                           198, 373,
                                                                                                           326],
                                                                                                  num_classes=7,
                                                                                                  downsample=32)
    NUM_ANCHORS = 3
    NUM_CLASSES = 7
    num_filters = NUM_ANCHORS * (NUM_CLASSES + 5)
    with fluid.dygraph.guard():
        backbone = DarkNet53_conv_body(is_test=False)
        detection = YoloDetectionBlock(ch_in=1024, ch_out=512, is_test=False)
        conv2d_pred = Conv2D(num_channels=1024, num_filters=num_filters, filter_size=1)

        x = to_variable(img)
        C0, C1, C2 = backbone(x)
        route, tip = detection(C0)
        P0 = conv2d_pred(tip)
        # anchors包含了预先设定好的锚框尺寸
        anchors = [116, 90, 156, 198, 373, 326]
        # downsample是特征图P0的步幅
        pred_boxes = get_yolo_box_xxyy(P0.numpy(), anchors, num_classes=7, downsample=32)
        iou_above_thresh_indices = get_iou_above_thresh_inds(pred_boxes, gt_boxes, iou_threshold=0.7)
        label_objectness = label_objectness_ignore(label_objectness, iou_above_thresh_indices)

        label_objectness = to_variable(label_objectness)
        label_location = to_variable(label_location)
        label_classification = to_variable(label_classification)
        scales = to_variable(scale_location)
        label_objectness.stop_gradient = True
        label_location.stop_gradient = True
        label_classification.stop_gradient = True
        scales.stop_gradient = True

        total_loss = get_loss(P0, label_objectness, label_location, label_classification, scales,
                              num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES)
        total_loss_data = total_loss.numpy()
        print(total_loss_data)