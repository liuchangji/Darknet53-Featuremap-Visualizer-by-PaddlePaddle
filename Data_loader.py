# 数据读取与预处理模块,最后是批量加速读取模块
import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from PIL import Image, ImageEnhance
import random
from box_utils import multi_box_iou_xywh,box_crop
import cv2
import matplotlib.pyplot as plt
import functools
import paddle
from utils.data_augment import image_augment

from config.config_example import get_cfg

cfg = get_cfg()
DATASETS_NAMES = cfg.DATASETS_NAMES
TEST_SIZE = cfg.TEST_SIZE


def get_datasets_names():
    """
    return a dict, as following,
        {'Boerner': 0,
         'Leconte': 1,
         'Linnaeus': 2,
         'acuminatus': 3,
         'armandi': 4,
         'coleoptera': 5,
         'linnaeus': 6
        }
    将类别名称映射为数字，存在字典里
    """
    datasets_category2id = {}
    for i, item in enumerate(DATASETS_NAMES):
        datasets_category2id[item] = i
    #print(datasets_category2id)
    return datasets_category2id


def get_annotations(cname2cid, datadir):
    """
    :param cname2cid:一个字典，对应了名字和序号
    :param datadir:没什么好说的，路径
    :return:   list[字典]，一个图片就是一个字典
    voc_rec = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
            }
    """
    filenames = os.listdir(os.path.join(datadir, 'annotations'))#首先读取标注的xml文件，用于获取他的名字
    #filenames.sort()
    records = []
    ct = 0
    for fname in filenames:


        # fpath = xml所在文件架
        # img_file = 图片文件
        fid = fname.split('.')[0]#以.为分隔取出文件的名字，剥离出文件名
        fpath = os.path.join(datadir, 'annotations', fname)#标注文件
        img_file = os.path.join(datadir, 'images', fid + '.jpg')#图片格式
        tree = ET.parse(fpath)

        if tree.find('id') is None:
            im_id = np.array([ct])
        else:
            im_id = np.array([int(tree.find('id').text)])

        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class = np.zeros((len(objs),), dtype=np.int32)
        is_crowd = np.zeros((len(objs),), dtype=np.int32)
        difficult = np.zeros((len(objs),), dtype=np.int32)
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i] = cname2cid[cname]
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            # 这里使用xywh格式来表示目标物体真实框
            gt_bbox[i] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1 + 1., y2 - y1 + 1.]
            is_crowd[i] = 0
            difficult[i] = _difficult

        voc_rec = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
        }
        if len(objs) != 0:
            records.append(voc_rec)
        ct += 1
    return records


## 数据读取
# 前面已经将图片的所有描述信息保存在records中了
# 其中的每一个元素包含了一张图片的描述
# 下面的程序展示了如何根据records里面的描述读取图片及标注。

def get_bbox(gt_bbox, gt_class):
    # 对于一般的检测任务来说，一张图片上往往会有多个目标物体
    # 设置参数MAX_NUM = 50， 即一张图片最多取50个真实框；如果真实
    # 框的数目少于50个，则将不足部分的gt_bbox, gt_class和gt_score的各项数值全设置为0
    #

    # --------------------
    #   当batch_size =! 1 时，如果不将每个样本的矩阵尺度相同化，就没有办法把多个不同尺寸的矩阵组合成一个batch
    #
    #
    # --------------------
    MAX_NUM = 30
    gt_bbox2 = np.zeros((MAX_NUM, 4))
    gt_class2 = np.zeros((MAX_NUM,))
    len_ = len(gt_bbox)
    for i in range(len_):
        if i >= MAX_NUM:
            return gt_bbox2, gt_class2
        gt_bbox2[i, :] = gt_bbox[i, :]
        gt_class2[i] = gt_class[i]

    return gt_bbox2, gt_class2


def get_img_data_from_file(record):
    """
    record is a dict as following,
      record = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
            }
    """
    im_file = record['im_file']
    h = record['h']
    w = record['w']
    is_crowd = record['is_crowd']
    gt_class = record['gt_class']
    gt_bbox = record['gt_bbox']
    difficult = record['difficult']

    img = cv2.imread(im_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # check if h and w in record equals that read from img
    assert img.shape[0] == int(h), \
        "image height of {} inconsistent in record({}) and img file({})".format(
            im_file, h, img.shape[0])

    assert img.shape[1] == int(w), \
        "image width of {} inconsistent in record({}) and img file({})".format(
            im_file, w, img.shape[1])

    gt_boxes, gt_labels = get_bbox(gt_bbox, gt_class) #抑制目标数量，有BUG。。。已修复

    # gt_bbox 用相对值
    gt_boxes[:, 0] = gt_boxes[:, 0] / float(w)
    gt_boxes[:, 1] = gt_boxes[:, 1] / float(h)
    gt_boxes[:, 2] = gt_boxes[:, 2] / float(w)
    gt_boxes[:, 3] = gt_boxes[:, 3] / float(h)

    return img, gt_boxes, gt_labels, (h, w)





#将上面的过程整理成一个函数get_img_data
def get_img_data(record, size):
    """
    :param record: record = records[i]
    :param size:缩放图像，这个缩放必须能被32整除
    :return:
    """
    img, gt_boxes, gt_labels, scales = get_img_data_from_file(record) #这步中将box转化为了相对值
    img, gt_boxes, gt_labels = image_augment(img, gt_boxes, gt_labels, size) #这步中包含了图像的缩放

    #这里得到的img数据数值需要调整，需要除以255，并且减去均值和方差，再将维度从[H, W, C]调整为[C, H, W]
    mean = [0.471, 0.448, 0.408]
    std = [0.234, 0.239, 0.242]
    mean = np.array(mean).reshape((1, 1, -1))
    std = np.array(std).reshape((1, 1, -1))
    img = (img / 255.0 - mean) / std
    img = img.astype('float32').transpose((2, 0, 1))
    return img, gt_boxes, gt_labels, scales



##批量数据读取与加速
# 获取一个批次内样本随机缩放的尺寸
def get_img_size(mode):
    """
    :param mode: 当训练和验证的时候，会开启自动缩放训练
                 当mode = test时，就固定了尺寸
    :return:
    """
    if (mode == 'train') or (mode == 'valid'):
        inds = np.array([0,1,2,3,4,5,6,7,8,9,10])
        ii = np.random.choice(inds)
        img_size = 320 + ii * 32
    else:
        img_size = TEST_SIZE
    return img_size

# 将 list形式的batch数据 转化成多个array构成的tuple
def make_array(batch_data):
    img_array = np.array([item[0] for item in batch_data], dtype = 'float32')
    gt_box_array = np.array([item[1] for item in batch_data], dtype = 'float32')
    gt_labels_array = np.array([item[2] for item in batch_data], dtype = 'int32')
    img_scale = np.array([item[3] for item in batch_data], dtype='int32')
    return img_array, gt_box_array, gt_labels_array, img_scale


#由于数据预处理耗时较长，可能会成为网络训练速度的瓶颈，所以需要对预处理部分进行优化。通过使用飞桨提供的API paddle.reader.xmap_readers可以开启多线程读取数据，具体实现代码如下。
# 使用paddle.reader.xmap_readers实现多线程读取数据

def multithread_loader(datadir, batch_size, mode='train'):
    cname2cid = get_datasets_names()
    records = get_annotations(cname2cid, datadir)

    def reader():
        # if mode == 'train':
        #     np.random.shuffle(records)
        img_size = get_img_size(mode) #生成一个随机会被32整除的尺寸
        batch_data = []
        for record in records:
            batch_data.append((record, img_size))
            if len(batch_data) == batch_size:
                yield batch_data
                batch_data = []
                img_size = get_img_size(mode)
        if len(batch_data) > 0:
            yield batch_data

    def get_data(samples):
        batch_data = []
        for sample in samples:
            record = sample[0]
            img_size = sample[1]
            img, gt_bbox, gt_labels, im_shape = get_img_data(record, size=img_size) #这步中对图像进行了缩放
            #print(record["im_file"])
            batch_data.append((img, gt_bbox, gt_labels, im_shape))
        return make_array(batch_data)

    mapper = functools.partial(get_data, )

    return paddle.reader.xmap_readers(mapper, reader, 8, 10)

def make_test_array(batch_data):
    img_name_array = np.array([item[0] for item in batch_data])
    img_data_array = np.array([item[1] for item in batch_data], dtype = 'float32')
    img_scale_array = np.array([item[2] for item in batch_data], dtype='int32')
    return img_name_array, img_data_array, img_scale_array

# 读取单张测试图片
def single_test_data_loader(datadir, batch_size=1, test_image_size=TEST_SIZE, mode='test'):
    """
    加载测试用的图片，测试数据没有groundtruth标签
    """
    image_names = os.listdir(datadir)
    def reader():
        batch_data = []
        img_size = test_image_size
        for image_name in image_names:
            file_path = os.path.join(datadir, image_name)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H = img.shape[0]
            W = img.shape[1]
            img = cv2.resize(img, (img_size, img_size))
            mean = [0.471, 0.448, 0.408]
            std = [0.234, 0.239, 0.242]

            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (img / 255.0 - mean) / std
            out_img = out_img.astype('float32').transpose((2, 0, 1))
            img = out_img #np.transpose(out_img, (2,0,1))
            im_shape = [H, W]

            batch_data.append((image_name.split('.')[0], img, im_shape))
            if len(batch_data) == batch_size:
                yield make_test_array(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            yield make_test_array(batch_data)

    return reader

if __name__ == "__main__":
    from config.config_example import get_cfg
    cfg = get_cfg()

    TRAINDIR = cfg.TRAINDIR
    TESTDIR = cfg.TESTDIR
    VALIDDIR = cfg.VALIDDIR
    #单张读取
    # cname2cid = get_datasets_names()
    # records = get_annotations(cname2cid, TRAINDIR)
    # print("数量"+str(len(records)))
    # print("抽取一个看看")
    # record = records[1]
    # print(record)
    #
    #
    # img, gt_boxes, gt_labels, scales = get_img_data(record, size=480)
    # d = data_loader( '/home/tuxiang/liuchangji/数据集/PascalVOC2012/VOC2012_train_val', batch_size=2, mode='train')
    # img, gt_boxes, gt_labels, im_shape = next(d())



    # print(img.shape, gt_boxes.shape, gt_labels.shape, im_shape.shape)
    # d = multithread_loader(TRAINDIR, batch_size=1, mode='train')
    # img, gt_boxes, gt_labels, im_shape = next(d())
    d = data_loader(TRAINDIR, batch_size=2, mode='train')
    img, gt_boxes, gt_labels, im_shape = next(d())

    print('asd')