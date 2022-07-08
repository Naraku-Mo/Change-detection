#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from model import AlexNet
from torch.utils.data import DataLoader
from dataset import flowerDataset

from utils import *
from config import *


import joblib
dtc = joblib.load(tree_path) 

net = AlexNet(num_classes=N_FEATURES)
net.load_state_dict(torch.load(alexnet_path))
net.eval()


# 分类预测
pred_data = flowerDataset(data_dir=pred_dir, transform=transform)
pred_loader = DataLoader(dataset=pred_data)
X, Y = preprocess(pred_loader, net)
Ypred = dtc.predict(X)
print('分类预测报告：')
for img_label, y in zip(pred_data.data_info, Ypred):
    img, label = img_label
    print(f"图像'{img}'预测为 {y}（真实类别为{'变化' if label==-1 else label}）")
    print('** Finished Test **')
    # print(f'''
    # 测试报告
    # ============
    # 预测样本数：正样本：{Xtrain.shape[0]}个；负样本：{Xtrain.shape[0]}个
    # 预测正确样本数：{Xtest.shape[0]}个
    # 测试准确率：{score:.4}
    # ''')