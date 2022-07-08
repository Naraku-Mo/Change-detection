import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from model import AlexNet
from GoogleNet import GoogleNet
from config import *
from dataset import flowerDataset
from utils import *

writer = SummaryWriter(comment='GoogleNet')

# build MyDataset
train_data = flowerDataset(data_dir=train_dir, transform=transform)
valid_data = flowerDataset(data_dir=valid_dir, transform=transform)

# build DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# AlexNet model and training
# net = AlexNet(num_classes=N_FEATURES, init_weights=True)

# googlenet model
net = GoogleNet(num_class=N_FEATURES)

input_data = Variable(torch.rand(16, 3, 224, 224))
with writer:
    writer.add_graph(net, (input_data,))

net.to(device)
loss_function = nn.CrossEntropyLoss()
# inspect the params with net.parameters()
optimizer = optim.Adam(net.parameters(), lr=LR)

best_acc = 0
val_num = len(valid_data)
train_num = len(train_loader)

for epoch in range(MAX_EPOCH):
    start = time.time()
    # train
    net.train()
    running_loss = 0

    for step, data in enumerate(train_loader):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / train_num
        a = int(np.round(rate * 50))
        niter = epoch * len(train_loader) + step

        # print(f"train loss: {rate * 100:.1f}%[{'#' *a}{'.' *(50 -a)}]{loss:.3f}")
        print(f"train loss: {rate * 100:.1f}%,{loss:.3f}")
    writer.add_scalar('Train/Loss', running_loss, epoch)
    # validate
    net.eval()
    acc = 0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in valid_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), alexnet_path)
        print(f'[epoch {epoch + 1}]train loss: {running_loss / step:.3f}, test accuracy: {val_accurate:.3f}\n')
        writer.add_scalar('Test/Accu', val_accurate, epoch)
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        finish = time.time()
        time_elapsed = finish - start
        print('本次训练耗时 {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

print(f'** Finished Training **')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

Xtrain, Ytrain = preprocess(train_loader, net)
# Xtest, Ytest = preprocess(valid_loader, net)
# 分割样本
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, test_size=0.25)

# Build decision tree model
dtc = DecisionTreeClassifier(criterion="entropy", max_depth=MAX_DEPTH)
dtc = dtc.fit(Xtrain, Ytrain)
score = dtc.score(Xtest, Ytest)

print(f'''
测试报告
============
训练样本数：{Xtrain.shape[0]}
训练准确度：{dtc.score(Xtrain, Ytrain):.4}
测试样本数：{Xtest.shape[0]}
测试准确度：{score:.4}
''')

import joblib

joblib.dump(dtc, tree_path)
writer.close()

# if __name__ == '__main__':
#     # 分类预测
#
#     pred_data = flowerDataset(data_dir=pred_dir, transform=transform)
#     pred_loader = DataLoader(dataset=pred_data, batch_size=BATCH_SIZE)
#     X, Y = preprocess(pred_loader, net)
#     Ypred = dtc.predict(X)
#     print('分类预测报告：')
#     for img_label, y in zip(pred_data.data_info, Ypred):
#         img, label = img_label
#         print(f"图像'{img}'预测为 {y} （真实类别为{'<未知>' if label==-1 else label}）"
