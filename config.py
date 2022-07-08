#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configuration of Alexnet and Decission Tree
"""

# configuration of alexnet
weight_path = "AlexNet.pth"  # path to the weigths
alexnet_path = 'AlexNet.pth'  # path to the net
N_FEATURES = 11

# params of training
MAX_EPOCH = 50
BATCH_SIZE = 10
LR = 0.0001
device = 'gpu'

# configuration of decission tree
MAX_DEPTH = 5
tree_path = 'dt.pkl'

# data path
## the path of training data
train_dir = "train"
valid_dir = train_dir
## valid_dir=pathlib.Path("valid")

## the path of data for prediction
pred_dir = 'predict'
