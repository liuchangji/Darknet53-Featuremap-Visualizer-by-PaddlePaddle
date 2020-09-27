# FearuteMap Visualizer by PaddlePaddle
这是一个飞桨写的Yolov3,他可以检测可以训练

## 运行环境
Paddlepaddle=2.0.0a
opencv-python
numpy
基本就这些了，packages.txt中里有一大堆我装的库，很多都用不上

## 使用方法

在/config/config_example.py文件中，修改FEATURE_MAP_VISUALIZE = True
将图片放到test文件夹

下载权重文件（可以给我留言留下你的邮箱，当然你也可以自己从头训练，我是在VOC上训练了几天）

运行detect.py

我在B站上上传了一个可视化的效果，可以看看https://www.bilibili.com/video/BV15i4y1G7KY
## 训练
运行train.py

目前只写了VOC格式的dataloader
## 简单看看
<div align=center><img src="https://github.com/liuchangji/Darknet53_Featuremap_Visualizer_by_PaddlePaddle/blob/master/test/2008_000048.jpg"/></div>

![aaa](https://github.com/liuchangji/Darknet53_Featuremap_Visualizer_by_PaddlePaddle/blob/master/%E5%85%B6%E4%BB%96/Screenshot%20from%202020-09-27%2016-33-37.png)

![bbb](https://github.com/liuchangji/Darknet53_Featuremap_Visualizer_by_PaddlePaddle/blob/master/%E5%85%B6%E4%BB%96/Screenshot%20from%202020-09-27%2016-33-57.png)
