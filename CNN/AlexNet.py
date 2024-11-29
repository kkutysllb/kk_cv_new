#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-11-28 20:44
# @Desc   : AlexNet图片分类
# --------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from kk_libraries.kk_functions import get_device, kk_ImageClassifierTrainer
from kk_libraries.kk_dataprocess import kk_load_data
from kk_libraries.kk_constants import text_labels_cifar10, mean, std


class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
             nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 定义数据预处理
def kk_data_transform():
    return {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)]),
        'valid': transforms.Compose([transforms.Resize(256), 
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean, std)])
    }


class Config(object):
    def __init__(self):
        self.num_epochs = 50
        self.lr = 0.001
        self.device = get_device()
        self.patience = 500
        self.save_path = os.path.join(parent_dir, "models", "AlexNet")
        self.logs_path = os.path.join(parent_dir, "logs", "AlexNet")
        self.plot_titles = "AlexNet"
        self.class_list = text_labels_cifar10

    def __call__(self):
        return self.num_epochs, self.lr, self.device, self.patience, self.save_path, self.logs_path, self.plot_titles, self.class_list

if __name__ == "__main__":
    # 数据加载
    data_path = os.path.join(parent_dir, "data/CIFAR10")
    train_loader,test_loader = kk_load_data(data_path, batch_size=512, DataSets=CIFAR10, transform=kk_data_transform())

    config = Config()
    # 模型定义
    model = AlexNet(in_channels=3, num_classes=10)
   
   # 定义损失函数和优化器C
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # 模型训练
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler)
    trainer.train_iter(train_loader, test_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 模型测试
    trainer.test(test_loader)