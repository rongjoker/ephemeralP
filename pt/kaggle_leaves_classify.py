import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import d2l_torch as d2l
import cnn_base as base
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import time
from matplotlib import pyplot as plt
import math
import torch.nn.functional as F

data_dir = '../data/kaggle_leaves/classify-leaves/'

test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
# train_data = pd.read_csv('./data/train.csv')
# test_data = pd.read_csv('./data/test.csv')

train_images = train_data.iloc[:, 0].values
pred_images = test_data.iloc[:, 0].values
train_labels = pd.get_dummies(train_data.iloc[:, 1]).values.argmax(1)
train_labels_header = pd.get_dummies(train_data.iloc[:, 1]).columns.values

batch_size = 128
learning_rate = 1e-4


# num_classes = 176


def get_train_dataset():
    # # 创建划分好的训练集和测试集
    h_flip = transforms.RandomHorizontalFlip(p=0.5)
    v_flip = transforms.RandomVerticalFlip(p=0.5)
    shape_aug = transforms.RandomResizedCrop((224, 224), scale=(0.1, 1), ratio=(0.5, 2))
    brightness_aug = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)
    train_augs = transforms.Compose([h_flip, v_flip])  # 图像增广
    train_data_trans = transforms.Compose([transforms.Resize(224),
                                           train_augs,
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_data_trans = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data_split = ImageFolder(os.path.join(data_dir, 'train_image'),
                                   transform=train_data_trans, target_transform=None)
    test_data_split = ImageFolder(os.path.join(data_dir, 'test_image'),
                                  transform=test_data_trans, target_transform=None)
    # 将ImageFolder的映射关系存到csv
    # id_code = pd.DataFrame(list(train_data.class_to_idx.items()),
    #                        columns=['label', 'id'])
    # id_code.to_csv(os.path.join(data_dir, 'id_code.csv'), index=False)
    train_dataloader = DataLoader(train_data_split, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data_split, batch_size=batch_size, shuffle=True)
    # for i, (features, labels) in enumerate(train_dataloader):
    #     print(features)
    #     print(labels)
    #     break

    return train_dataloader, test_dataloader


class LeavesTrainDataSet(d2l.DataModule):
    """The Fashion-MNIST dataset.

    Defined in :numref:`sec_fashion_mnist`"""

    def __init__(self, train_dl, test_dl):
        super().__init__()
        self.save_hyperparameters()
        self.train_loader = train_dl
        self.val_loader = test_dl

    def get_dataloader(self, train):
        """Defined in :numref:`sec_fashion_mnist`"""
        return self.train_loader if train else self.val_loader


class CLASSIFY_LEAVES(torch.utils.data.Dataset):
    def __init__(self, root, images, labels, transform):
        super(CLASSIFY_LEAVES, self).__init__()
        self.root = root
        self.images = images
        if labels is None:
            self.labels = None
        else:
            self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.root + self.images[index]
        image = Image.open(image_path)
        image = self.transform(image)
        if self.labels is None:
            return image
        label = torch.tensor(self.labels[index])
        return image, label

    def __len__(self):
        return self.images.shape[0]


def load_data(images, labels, batch_size, train):
    aug = []
    if (train):
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomVerticalFlip(),
               transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
               transforms.ToTensor()]
    else:
        aug = [transforms.ToTensor()]
    transform = transforms.Compose(aug)
    dataset = CLASSIFY_LEAVES(data_dir, images, labels, transform=transform)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, shuffle=train)


# pred_images = test_data.iloc[:, 0].values


# 微调
def get_net():
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet50(pretrained=True)  # 使用了resnet50的预训练模型
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(),  # 在后面输出层中，又增加了几层，目的是维度降到类别数
                                            nn.Linear(512, 256), nn.ReLU(),
                                            nn.Linear(256, 176))  # 树叶分类有176类
    finetune_net = finetune_net.to('cuda:0')  # 将模型送入gpu
    for param in finetune_net.features.parameters():  # 固定住预训练模型中的参数，只调节我们新加的几个层的参数
        param.requires_grad = False
    return finetune_net


def resnet_train(epoch=10, batch_size=128):
    # 10 epoch acc: tensor(0.9375, device='cuda:0')
    base.ResNet18().layer_summary((1, 3, 224, 224))
    model = base.ResNet18(lr=learning_rate, num_classes=176)
    trainer = d2l.Trainer(max_epochs=epoch, num_gpus=1)
    train_dataloader, test_dataloader = get_train_dataset()
    data = LeavesTrainDataSet(train_dataloader, test_dataloader)
    print('type', type(data.train_dataloader()))
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model, data)
    torch.save(model.state_dict(), 'model/leaves_resnet_local.pth')


def predict(pred_iter):
    # model = base.ResNet18(lr=learning_rate, num_classes=176)
    # model = torchvision.models.resnet18(num_classes=176)
    model = get_net()
    model.load_state_dict(torch.load('model/leaves_resnet_jupyter_baba_100.pth'))
    model = model.to(torch.device('cuda:0'))
    model.eval()
    prediction = []
    for index, X in enumerate(pred_iter):
        X = X.to('cuda:0')
        prediction.extend(train_labels_header[model(X).argmax(1).cpu()])
    test_data['label'] = prediction
    # test_data.to_csv('./data/submission.csv', index=None)
    test_data.to_csv(os.path.join(data_dir, 'submission.csv'), index=None)


# resnet_train(epoch=1, batch_size=128)

predict(pred_iter=load_data(pred_images, None, 128, train=False))
