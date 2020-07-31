# darknet53的主干网络与预测框关联程序


import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from paddle.fluid.dygraph.base import to_variable
import matplotlib.pyplot as plt
from config.config_example import get_cfg
import cv2
import numpy as np

cfg = get_cfg()
ANCHORS = cfg.ANCHOR
ANCHORS_MASKS = cfg.ANCHOR_MASKS
VALID_THRESH = cfg.VALID_THRESH
IGNORE_THRESH = cfg.IGNORE_THRESH  # 当预测框与真实框IoU > ignore_thresh，标注objectness = -1
FEATURE_MAP_VISUALIZE = cfg.FEATURE_MAP_VISUALIZE
# YOLO-V3骨干网络结构Darknet53的实现代码

# 上面这段示例代码，指定输入数据的形状是(1,3,640,640)，
# 则3个层级的输出特征图的形状分别是
# C0(1,1024,20,20)
# C1(1,1024,40,40)
# C2(1,1024,80,80)

# Backbone需要建立输出特征图与预测框之间的关联

DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class ConvBNLayer(fluid.dygraph.Layer):
    """
    卷积 + 批归一化，BN层之后激活函数默认用leaky_relu
    """

    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act="leaky",
                 is_test=True):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            param_attr=None,
            bias_attr=False,
            act=None)

        self.batch_norm = BatchNorm(
            num_channels=ch_out,
            is_test=is_test,
            param_attr=None,
            bias_attr=None)
        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out


class DownSample(fluid.dygraph.Layer):
    """
    下采样，图片尺寸减半，具体实现方式是使用stirde=2的卷积
    """

    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=2,
                 padding=1,
                 is_test=True):
        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            is_test=is_test)
        self.ch_out = ch_out

    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out


class BasicBlock(fluid.dygraph.Layer):
    """
    基本残差块的定义，输入x经过两层卷积，然后接第二层卷积的输出和输入x相加
    """

    def __init__(self, ch_in, ch_out, is_test=True):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test
        )
        self.conv2 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
        )

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = fluid.layers.elementwise_add(x=inputs, y=conv2, act=None)
        return out


class LayerWarp(fluid.dygraph.Layer):
    """
    添加多层残差块，组成Darknet53网络的一个层级
    """

    def __init__(self, ch_in, ch_out, count, is_test=True):
        super(LayerWarp, self).__init__()

        self.basicblock0 = BasicBlock(ch_in,
                                      ch_out,
                                      is_test=is_test)
        self.res_out_list = []
        for i in range(1, count):
            res_out = self.add_sublayer("basic_block_%d" % (i),  # 使用add_sublayer添加子层
                                        BasicBlock(ch_out * 2,
                                                   ch_out,
                                                   is_test=is_test))
            self.res_out_list.append(res_out)

    def forward(self, inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y


class DarkNet53_conv_body(fluid.dygraph.Layer):
    def __init__(self,

                 is_test=True):
        super(DarkNet53_conv_body, self).__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        # 第一层卷积
        self.conv0 = ConvBNLayer(
            ch_in=3,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test)

        # 下采样，使用stride=2的卷积来实现
        self.downsample0 = DownSample(
            ch_in=32,
            ch_out=32 * 2,
            is_test=is_test)

        # 添加各个层级的实现
        self.darknet53_conv_block_list = []
        self.downsample_list = []
        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer(
                "stage_%d" % (i),
                LayerWarp(32 * (2 ** (i + 1)),
                          32 * (2 ** i),
                          stage,
                          is_test=is_test))
            self.darknet53_conv_block_list.append(conv_block)
        # 两个层级之间使用DownSample将尺寸减半
        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                "stage_%d_downsample" % i,
                DownSample(ch_in=32 * (2 ** (i + 1)),
                           ch_out=32 * (2 ** (i + 2)),
                           is_test=is_test))
            self.downsample_list.append(downsample)

    def forward(self, inputs):
        out = self.conv0(inputs)
        # print("conv1:",out.numpy())
        out = self.downsample0(out)
        # print("dy:",out.numpy())
        blocks = []
        if FEATURE_MAP_VISUALIZE == True:
            WINDOWSNAME = 'f'
            cv2.namedWindow(WINDOWSNAME,cv2.WINDOW_NORMAL)
            for i, conv_block_i in enumerate(self.darknet53_conv_block_list):  # 依次将各个层级作用在输入上面
                out = conv_block_i(out)
                if True:
                    # 卷积层可视化
                    out_for_visu = out.numpy()[0,:,:,:]
                    shape = out_for_visu.shape
                    featrue_sum = np.zeros(shape)
                    feature_num = shape[0]
                    print(feature_num)
                    for _ in range(feature_num):
                        featrue_sum = feature_num + out_for_visu[_,:,:]
                        feature_map_for_visulize = normalization(out_for_visu[_,:,:])*255
                        feature_map_for_visulize = np.array(feature_map_for_visulize, dtype='uint8')
                        feature_map_for_visulize = cv2.applyColorMap(feature_map_for_visulize, cv2.COLORMAP_JET)
                        cv2.imshow(WINDOWSNAME,feature_map_for_visulize)
                        # videoWrite.write(feature_map_for_visulize)
                        waittime = int(512/(i+1)/(i+1))
                        cv2.waitKey(waittime)
                    # plt.imshow(featrue_sum)
                    # plt.show()
                blocks.append(out)

                if i < len(self.stages) - 1:
                    out = self.downsample_list[i](out)
        else:
            for i, conv_block_i in enumerate(self.darknet53_conv_block_list):  # 依次将各个层级作用在输入上面
                out = conv_block_i(out)
                blocks.append(out)
                if i < len(self.stages) - 1:
                    out = self.downsample_list[i](out)
        return blocks[-1:-4:-1]  # 将C0, C1, C2作为返回值,多尺度融合


class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale, actual_shape=out_shape)
        return out


class YoloDetectionBlock(fluid.dygraph.Layer):
    # define YOLO-V3 detection head
    # 使用多层卷积和BN提取特征
    # 建立输出特征图与预测框之间的关联
    # 从骨干网络输出特征图C0得到跟预测相关的特征图P0

    # 如果不使用多尺度融合，检测头体现不出作用，只是简单的特征提取

    def __init__(self, ch_in, ch_out, is_test=True):
        super(YoloDetectionBlock, self).__init__()

        assert ch_out % 2 == 0, \
            "channel {} cannot be divided by 2".format(ch_out)

        self.conv0 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test
        )
        self.conv1 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
        )
        self.conv2 = ConvBNLayer(
            ch_in=ch_out * 2,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test
        )
        self.conv3 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
        )
        self.route = ConvBNLayer(
            ch_in=ch_out * 2,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test
        )
        self.tip = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
        )

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


class YOLOv3(fluid.dygraph.Layer):
    def __init__(self, num_classes, is_train=True):
        super(YOLOv3, self).__init__()

        self.is_train = is_train
        self.num_classes = num_classes
        # 提取图像特征的骨干代码
        self.block = DarkNet53_conv_body(
            is_test=not self.is_train)
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []
        # 生成3个层级的特征图P0, P1, P2
        for i in range(3):
            # 添加从ci生成ri和ti的模块
            yolo_block = self.add_sublayer(
                "yolo_detecton_block_%d" % (i),
                YoloDetectionBlock(
                    ch_in=512 // (2 ** i) * 2 if i == 0 else 512 // (2 ** i) * 2 + 512 // (2 ** i),
                    ch_out=512 // (2 ** i),
                    is_test=not self.is_train))
            self.yolo_blocks.append(yolo_block)

            num_filters = 3 * (self.num_classes + 5)  # 每个层级特征有3个anchor 共有3个尺度特征

            # 添加从ti生成pi的模块，这是一个Conv2D操作，输出通道数为3 * (num_classes + 5)
            block_out = self.add_sublayer(
                "block_out_%d" % (i),
                Conv2D(num_channels=512 // (2 ** i) * 2,
                       num_filters=num_filters,
                       filter_size=1,
                       stride=1,
                       padding=0,
                       act=None,
                       param_attr=None,
                       bias_attr=None))
            self.block_outputs.append(block_out)
            if i < 2:
                # 对ri进行卷积
                route = self.add_sublayer("route2_%d" % i,
                                          ConvBNLayer(ch_in=512 // (2 ** i),
                                                      ch_out=256 // (2 ** i),
                                                      filter_size=1,
                                                      stride=1,
                                                      padding=0,
                                                      is_test=(not self.is_train)))
                self.route_blocks_2.append(route)
            # 将ri放大以便跟c_{i+1}保持同样的尺寸
            self.upsample = Upsample()

    def forward(self, inputs):
        outputs = []
        blocks = self.block(inputs)  # 骨干网络特征提取
        for i, block in enumerate(blocks):
            if i > 0:
                # 将r_{i-1}经过卷积和上采样之后得到特征图，与这一级的ci进行拼接
                block = fluid.layers.concat(input=[route, block], axis=1)
            # 从ci生成ti和ri
            route, tip = self.yolo_blocks[i](block)
            # 从ti生成pi
            block_out = self.block_outputs[i](tip)
            # 将pi放入列表
            outputs.append(block_out)
            # 输出维度等于3 * (self.num_classes + 5)
            # 输出尺寸=输入图片的尺寸 /8 /16 /32 和输入尺寸有关
            if i < 2:
                # 对ri进行卷积调整通道数
                route = self.route_blocks_2[i](route)
                # 对ri进行放大，使其尺寸和c_{i+1}保持一致
                route = self.upsample(route)

        return outputs

    def get_loss(self, outputs, gtbox, gtlabel, gtscore=None,
                 anchors=ANCHORS,
                 anchor_masks=ANCHORS_MASKS,
                 ignore_thresh=IGNORE_THRESH,  # 当预测框与真实框IoU > ignore_thresh，标注objectness = -1
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

    def get_pred(self,
                 outputs,
                 im_shape=None,
                 anchors=ANCHORS,
                 anchor_masks=ANCHORS_MASKS,
                 valid_thresh=VALID_THRESH):  # valid-thresh置信度
        downsample = 32
        total_boxes = []
        total_scores = []
        for i, out in enumerate(outputs):
            anchor_mask = anchor_masks[i]
            anchors_this_level = []
            for m in anchor_mask:
                anchors_this_level.append(anchors[2 * m])
                anchors_this_level.append(anchors[2 * m + 1])

            boxes, scores = fluid.layers.yolo_box(
                x=out,
                img_size=im_shape,
                anchors=anchors_this_level,
                class_num=self.num_classes,
                conf_thresh=valid_thresh,
                downsample_ratio=downsample,
                name="yolo_box" + str(i))
            total_boxes.append(boxes)
            total_scores.append(
                fluid.layers.transpose(
                    scores, perm=[0, 2, 1]))
            downsample = downsample // 2

        yolo_boxes = fluid.layers.concat(total_boxes, axis=1)
        yolo_scores = fluid.layers.concat(total_scores, axis=2)
        return yolo_boxes, yolo_scores


if __name__ == "__main__":
    import numpy as np

    # with fluid.dygraph.guard():
    #     backbone = DarkNet53_conv_body(is_test=False)
    #     x = np.random.randn(1, 3, 640, 640).astype('float32')
    #     x = to_variable(x)
    #     C0, C1, C2 = backbone(x)
    #

    NUM_ANCHORS = 3  # 公式中的K
    NUM_CLASSES = 20
    num_filters = NUM_ANCHORS * (NUM_CLASSES + 5)
    with fluid.dygraph.guard():
        backbone = DarkNet53_conv_body(is_test=False)
        detection = YoloDetectionBlock(ch_in=1024, ch_out=512, is_test=False)
        conv2d_pred = Conv2D(num_channels=1024, num_filters=num_filters, filter_size=1)

        x = np.random.randn(1, 3, 640, 640).astype('float32')
        x = to_variable(x)
        C0, C1, C2 = backbone(x)
        print(C0.shape, C1.shape, C2.shape)
        route, tip = detection(C0)
        P0 = conv2d_pred(tip)
        print("P0 shape =" + str(P0.shape))  # [1, 75, 20, 20] 75 = K * （C+5） K为anchor数量，C为类别，20×20是输出尺度，代表了
        # 计算预测框是否包含物体的概率
        reshaped_p0 = fluid.layers.reshape(P0, [-1, NUM_ANCHORS, NUM_CLASSES + 5, P0.shape[2], P0.shape[3]])
        pred_objectness = reshaped_p0[:, :, 4, :, :]  # resize取出每个anchor对应的objectness
        pred_objectness_probability = fluid.layers.sigmoid(pred_objectness)
        print(pred_objectness.shape, pred_objectness_probability.shape)
        # 计算预测框位置坐标
        pred_location = reshaped_p0[:, :, 0:4, :, :]  #
        print(pred_location.shape)
        # 网络输出值是(tx,ty,th,tw)，还需要将其转化为(x1,y1,x2,y2)这种形式的坐标表示。Paddle里面有专门的API fluid.layers.yolo_box直接计算出结果
        from utils.txtythtw_2_xyxy import get_yolo_box_xxyy

        # anchors包含了预先设定好的锚框尺寸
        anchors = [116, 90, 156, 198, 373, 326]
        # downsample是特征图P0的步幅
        pred_boxes = get_yolo_box_xxyy(P0.numpy(), anchors, num_classes=20, downsample=32)  # 由输出特征图P0计算预测框位置坐标
        print(pred_boxes.shape)


        pred_classification = reshaped_p0[:, :, 5:5 + NUM_CLASSES, :, :]
        pred_classification_probability = fluid.layers.sigmoid(pred_classification)
        print(pred_classification.shape)
