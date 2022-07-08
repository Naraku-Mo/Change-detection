# GoogleNet & alexnet-tree

## 介绍
GoogelNet与[alexnet](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)与决策树复合模型。Alexnet实现降维，决策树实现分类。

## 主要依赖

- pytorch
- numpy
- scikit-learn

## 原理

1. 数据压缩
神经网络具有降维的作用。利用这个提点，可把高维数据（图像）压缩成低维数据；
这在图像识别中非常有用，因为图像是高维数据，一般每个像素算一个维度，作为样本的属性。这回带来“维数灾难”。

这部分通过训练Alexnet网络实现；
训练后的Alexnet应用于样本，就可以实现降维，将图像的像素矩阵变换成11维向量。换言之，11个数字确定一张图；当然会牺牲精确性。


2. 决策树分类
将压缩后的11维向量可以看做输入样本的属性（features）；和对应类别构成决策树的学习样本；
训练决策树；计算测试误差

可以设置max_depth参数，调整决策树的学习性能。

3. 预测
将训练后的决策树应用于新样本；应用前依然用训练好的Alexnet进行降维


## 代码
训练主程序：`train.py`

如果Alexnet和决策树模型被保存，那么预测部分可由`predict.py`独立完成

## 使用
- 准备数据：训练数据存放于train文件夹下；自动分离出25%的测试数据；待预测图像存放于predict文件夹下

- 训练：运行`train.py` (训练后为自动保存模型)

- 预测：运行`predict.py`

## 配置

请查看`config.py`；主要包含alexnet训练参数和决策树的参数。

一般无需改动。



