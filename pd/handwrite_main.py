# 加载飞桨和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Linear
import os
import numpy as np

import json
import gzip
import net_base as nb


def evaluation(model, datasets):
    model.eval()

    acc_set = list()
    for batch_id, data in enumerate(datasets()):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        pred = model(images)  # 获取预测值
        acc = paddle.metric.accuracy(input=pred, label=labels)
        # t = acc.numpy()
        # f = float(acc)
        acc_set.append(float(acc))
        # acc_set.extend(acc.numpy())
        # print(type(acc))
        # acc_set.extend(float(acc))

    # #计算多个batch的准确率
    acc_val_mean = np.array(acc_set).mean()
    return acc_val_mean


# 仅修改计算损失的函数，从均方误差（常用于回归问题）到交叉熵误差（常用于分类问题）
def train(model, bs=20, es=10, model_name='mnist.pdparams'):
    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
    # 声明数据加载函数，使用训练模式，MnistDataset构建的迭代器每次迭代只返回batch=1的数据
    train_dataset = nb.MnistDataset(mode='train')
    # 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
    # DataLoader 返回的是一个批次数据迭代器，并且是异步的；
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    val_dataset = nb.MnistDataset(mode='valid')
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=bs, drop_last=True)
    model.train()
    # 调用加载数据的函数
    # train_loader = load_data('train')
    # val_loader = load_data('valid')
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # EPOCH_NUM = 100
    for epoch_id in range(es):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            # 前向计算的过程
            predicts = model(images)

            # 计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.item()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()
        acc_train_mean = evaluation(model, train_loader)
        acc_val_mean = evaluation(model, val_loader)
        print('train_acc: {}, val acc: {}'.format(acc_train_mean, acc_val_mean))
    # 保存模型参数
    paddle.save(model.state_dict(), model_name)


# 声明网络结构
model = nb.ResNet(layers=50, class_dim=10)
bs = 120
es = 2
model_name = 'mnist_%s_%s.pdparams' % (bs, es)
print(model_name)
train(model, bs=bs, es=es, model_name=model_name)
