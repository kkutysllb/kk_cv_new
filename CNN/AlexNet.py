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
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from kk_libraries.kk_functions import get_device, kk_ImageClassifierTrainer
from kk_libraries.kk_dataprocess import kk_load_data, kk_loader_train, kk_loader_test, kk_get_data_mean_stdv2
from kk_libraries.kk_constants import text_labels_fashion_mnist, text_labels_mini_imagenet100
mean = [0.5, ]
std = [0.5, ]


class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),  # 修改kernel_size和stride
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 修改pool大小和stride
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # 修改kernel_size
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 修改pool大小
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
            nn.MaxPool2d(kernel_size=3, stride=2)  # 修改pool大小
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),  # 修改输入维度为256*6*6
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        self._initialize_weights()
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    
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
        'train': transforms.Compose([transforms.RandomResizedCrop(256),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)]),
        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean, std)])
    }
    
def kk_train_data_transform():
    """数据预处理"""
    mean, _, std = kk_get_data_mean_stdv2(os.path.join(root_dir, "data", "mini_imagenet100"))
    return transforms.Compose([transforms.RandomResizedCrop(256),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomRotation(15),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])

def kk_test_data_transform():
    """数据预处理"""
    mean, _, std = kk_get_data_mean_stdv2(os.path.join(root_dir, "data", "mini_imagenet100"))
    return transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])


class Config(object):
    def __init__(self):
        self.num_epochs = 500
        self.lr = 0.01
        self.device = "cuda:0"
        self.patience = 300
        self.save_path = os.path.join(root_dir, "models", "AlexNet")
        self.logs_path = os.path.join(root_dir, "logs", "AlexNet")
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        self.plot_titles = "AlexNet"
        self.batch_size = 256
        self.class_list = text_labels_mini_imagenet100
        self.dataset_name = "mini_imagenet100"

    def __call__(self):
        return self.num_epochs, self.lr, self.device, self.patience, self.save_path, self.logs_path, self.plot_titles, self.class_list

if __name__ == "__main__":
    config = Config()
    # 数据加载
    # data_path = os.path.join(root_dir, "data", "FashionMNIST")
    # train_loader,test_loader = kk_load_data(data_path, batch_size=config.batch_size, DataSets=FashionMNIST, transform=kk_data_transform(), num_works=4)
    train_loader, valid_loader = kk_loader_train(os.path.join(root_dir, "data", "mini_imagenet100", "train"), config.batch_size, transform=kk_train_data_transform())
    test_loader = kk_loader_test(os.path.join(root_dir, "data", "mini_imagenet100", "val"), config.batch_size, transform=kk_test_data_transform())
    
    # 模型定义
    model = AlexNet(in_channels=3, num_classes=100)
   
   # 定义损失函数和优化器C
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,  min_lr=3e-6)

    # 模型训练
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler=None)
    trainer.train_iter(train_loader, valid_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 模型测试
    trainer.test(test_loader)