#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)


class flowerDataset(Dataset):
    """继承Dataset
    重写__init__, __getitem__
    """

    def __init__(self, data_dir, transform=None):
        """
        分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理，默认不进行预处理
        """
        # data_info存储所有图片路径和标签（元组的列表），在DataLoader中通过index读取样本
        self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        # 打开图片，默认为PIL，需要转成RGB
        img = Image.open(path_img).convert('RGB')
        # 如果预处理的条件不为空，应该进行预处理操作
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    # 自定义方法，用于返回所有图片的路径以及标签
    def get_img_info(self, data_dir):
        if isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)
        data_info = []
        for sub_dir in data_dir.iterdir():
            if sub_dir.is_dir():
                for img in sub_dir.iterdir():
                    if img.suffix == '.jpg':
                        label = int(sub_dir.name) if sub_dir.name.isdigit() else -1
                        data_info.append((img, label))
        self.data_info = data_info
