#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from torchvision import transforms


def preprocess(loader, net=None):
    """
    load the data, then transform them by alexnet
    
    Arguments:
        loader {Loader} -- the loader of the data
    
    Returns:
        tuple of features-label data
    """
    if net is None:
        raise Exception('Supply AlexNet please!')
        # from model import AlexNet
        # import torch
        # net = AlexNet(num_classes=11)
        # weight_path = "AlexNet.pth"
        # net.load_state_dict(torch.load(weight_path))
        # net.eval()
    X = None
    Y = np.zeros(0, dtype=np.int8)
    for data in loader:
        img, label = data
        out = net(img)
        if X is None:
            X = out.detach().numpy()
        else:
            X = np.vstack((X, out.detach().numpy()))
        Y = np.hstack((Y, label.detach().numpy()))
    return np.squeeze(X), np.squeeze(Y)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
