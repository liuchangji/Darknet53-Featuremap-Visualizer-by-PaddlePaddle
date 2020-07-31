import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
import math
from config.config_example import get_cfg

cfg = get_cfg()

INSECT_NAMES = cfg.DATASETS_NAMES


def draw_rectangle(currentAxis, bbox, edgecolor = 'r', facecolor = 'b', fill=False, linestyle='-'):
    # currentAxis，坐标轴，通过plt.gca()获取
    # bbox，边界框，包含四个数值的list， [x1, y1, x2, y2]
    # edgecolor，边框线条颜色
    # facecolor，填充颜色
    # fill, 是否填充
    # linestype，边框线型
    # patches.Rectangle需要传入左上角坐标、矩形区域的宽度、高度等参数
    rect=patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1, linewidth=2,
                           edgecolor=edgecolor,facecolor=facecolor,fill=fill, linestyle=linestyle)
    currentAxis.add_patch(rect)

# 定义绘制预测结果的函数
def draw_results(result, filename, draw_thresh):
    plt.figure(figsize=(10, 10))
    im = imread(filename)
    plt.imshow(im)
    currentAxis=plt.gca()
    colors = ['r']
    for item in result:
        box = item[2:6]
        label = int(item[0])
        name = INSECT_NAMES[label]
        if item[1] > draw_thresh:
            draw_rectangle(currentAxis, box, edgecolor = colors[0])
            plt.text(box[0], box[1], name, fontsize=24, color=colors[0])


if __name__ == "__main__":


    plt.figure()

    filename = '../test.jpg'
    im = imread(filename)
    plt.imshow(im)

    currentAxis=plt.gca()

    # 预测框位置
    boxes = np.array([[4.21716537e+01, 1.28230896e+02, 2.26547668e+02, 6.00434631e+02],
           [3.18562988e+02, 1.23168472e+02, 4.79000000e+02, 6.05688416e+02],
           [2.62704697e+01, 1.39430557e+02, 2.20587097e+02, 6.38959656e+02],
           [4.24965363e+01, 1.42706665e+02, 2.25955185e+02, 6.35671204e+02],
           [2.37462646e+02, 1.35731537e+02, 4.79000000e+02, 6.31451294e+02],
           [3.19390472e+02, 1.29295090e+02, 4.79000000e+02, 6.33003845e+02],
           [3.28933838e+02, 1.22736115e+02, 4.79000000e+02, 6.39000000e+02],
           [4.44292603e+01, 1.70438187e+02, 2.26841858e+02, 6.39000000e+02],
           [2.17988785e+02, 3.02472412e+02, 4.06062927e+02, 6.29106628e+02],
           [2.00241089e+02, 3.23755096e+02, 3.96929321e+02, 6.36386108e+02],
           [2.14310303e+02, 3.23443665e+02, 4.06732849e+02, 6.35775269e+02]])

    # 预测框得分
    scores = np.array([0.5247661 , 0.51759845, 0.86075854, 0.9910175 , 0.39170712,
           0.9297706 , 0.5115228 , 0.270992  , 0.19087596, 0.64201415, 0.879036])

    # 画出所有预测框
    for box in boxes:
        draw_rectangle(currentAxis, box)
    plt.show()