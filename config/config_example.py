# ----------------------------------------------------------------------------------------


#                        功能：  样例config文件


# -----------------------------------------------------------------------------------------
# 说明：
#       你可以把你自己的config文件存起来，设置好后使用get_config.py加载配置文件
#       每个数据集都要有一个自己的config文件
# ---------------------------------
# 数据集格式：VOC   - - 目前暂时支持VOC  ^-^
#     数据按照以下格式存放:
#         DATASETS/
#           ├── train
#             ├── images
#                   ├── *.jpg
#               ├── annotations
#                   ├── *.xml
#           ├── val
#             ├── images
#                   ├── *.jpg
#               ├── annotations
#                   ├── *.xml
#           ├── test
#               ├── images
#
#      如果你使用其他格式的图片，到Data_loader.py中修改img_file = os.path.join(datadir, 'images', fid + '.jpg') jpg为其他格式


# -----------------------------------------------------------------------------------------
# 配置文件信息：
#             2020/7/23 给VOC数据训练用的

#                2008_006503
#                   2011_002810
#                           2011_002920
# -----------------------------------------------------------------------------------------
TRAINDIR = '/home/tuxiang/liuchangji/datasets/PascalVOC2012/VOC2012_train_val'  # 训练集路径
TESTDIR = './insects/test/'  # 测试及路径
VALIDDIR = '/home/tuxiang/liuchangji/datasets/PascalVOC2012/VOC2012_train_val'  # 验证集路径
# TRAINDIR = './insects/train'  # 训练集路径
# TESTDIR = './insects/test/'  # 测试及路径
# VALIDDIR = './insects/val/'  # 验证集路径

ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
VALID_THRESH = 0.1 #其实就是置信度了，fluid.layers.yolo_box中的一个参数，检测框的置信度得分阈值。置信度得分低于阈值的框应该被忽略，这并不是非极大值抑制
NMS_TOPK = 400
NMS_POSK = 100
NMS_THRESH = 0.45
TEST_SIZE = 640 #检测时图像的缩放尺寸，TEST_SIZE * TEST_SIZE 这个数字必须能被32整除
IGNORE_THRESH = 0.4 # 当预测框与真实框IoU > ignore_thresh，标注objectness = -1
NUM_ANCHORS = 3
# VOC 20 个类
DATASETS_NAMES = ['aeroplane', 'bicycle', 'bird',
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor']

# DATASETS_NAMES = ['Boerner', 'Leconte', 'Linnaeus',
#                   'acuminatus', 'armandi', 'coleoptera', 'linnaeus']

NUM_CLASSES = len(DATASETS_NAMES) #类别个数

FEATURE_MAP_VISUALIZE = True









#训练-----------------------------------
BATCHSIZE = 8
MAX_EPOCH = 50


# 暂时只支持VOC格式
# 文件






class config():
    def __init__(self,ANCHOR,
                 ANCHOR_MASKS,
                 VALID_THRESH,
                 NMS_TOPK,
                 NMS_POSK,
                 NMS_THRESH,
                 IGNORE_THRESH,
                 NUM_ANCHORS,
                 DATASETS_NAMES,
                 NUM_CLASSES,
                 BATCHSIZE,
                 MAX_EPOCH,
                 TRAINDIR,
                 TESTDIR,
                 VALIDDIR,
                 TEST_SIZE,
                 FEATURE_MAP_VISUALIZE
                ):
        self.ANCHOR = ANCHOR
        self.ANCHOR_MASKS = ANCHOR_MASKS
        self.VALID_THRESH = VALID_THRESH
        self.NMS_TOPK = NMS_TOPK
        self.NMS_POSK = NMS_POSK
        self.NMS_THRESH = NMS_THRESH
        self.IGNORE_THRESH = IGNORE_THRESH
        self.NUM_ANCHORS = NUM_ANCHORS
        self.DATASETS_NAMES = DATASETS_NAMES
        self.NUM_CLASSES = NUM_CLASSES
        self.BATCHSIZE = BATCHSIZE
        self.MAX_EPOCH = MAX_EPOCH
        self.TRAINDIR = TRAINDIR
        self.TESTDIR = TESTDIR
        self.VALIDDIR = VALIDDIR
        self.TEST_SIZE = TEST_SIZE
        self.FEATURE_MAP_VISUALIZE = FEATURE_MAP_VISUALIZE
        self.CONFIG_NAME = "样例配置文件1" #可以给配置文件起个名字
        print("加载配置文件:"+str(self.CONFIG_NAME))



def get_cfg():
    cfg = config(ANCHOR=ANCHORS,
                 ANCHOR_MASKS=ANCHOR_MASKS,
                 VALID_THRESH=VALID_THRESH,
                 NMS_TOPK=NMS_TOPK,
                 NMS_POSK=NMS_POSK,
                 NMS_THRESH=NMS_THRESH,
                 IGNORE_THRESH=IGNORE_THRESH,
                 NUM_ANCHORS=NUM_ANCHORS,
                 DATASETS_NAMES=DATASETS_NAMES,
                 NUM_CLASSES=NUM_CLASSES,
                 MAX_EPOCH=MAX_EPOCH,
                 TRAINDIR=TRAINDIR,
                 TESTDIR=TESTDIR,
                 VALIDDIR=VALIDDIR,
                 BATCHSIZE=BATCHSIZE,
                 TEST_SIZE=TEST_SIZE,
                 FEATURE_MAP_VISUALIZE = FEATURE_MAP_VISUALIZE)
    return cfg


if __name__ == "__main__":
    cfg = get_cfg()


