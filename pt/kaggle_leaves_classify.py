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
pred_iter = load_data(pred_images, None, 256, train=False)
print(pred_iter)


def predict(pred_iter):
    learning_rate = 1e-4
    model = base.ResNet18(lr=learning_rate, num_classes=176)
    model.load_state_dict(torch.load('model/leaves_resnet_jupyter.pth'))
    model = model.to(torch.device('cuda:0'))
    model.eval()
    prediction = []
    for index, X in enumerate(pred_iter):
        X = X.to('cuda:0')
        prediction.extend(train_labels_header[model(X).argmax(1).cpu()])
    test_data['label'] = prediction
    # test_data.to_csv('./data/submission.csv', index=None)
    test_data.to_csv(os.path.join(data_dir, 'submission.csv'), index=None)


predict(pred_iter)
