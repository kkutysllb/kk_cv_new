#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-11-29 15:15
# @Desc   : LeNet5_KAN  图像分类
# --------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import FashionMNIST

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from kk_libraries.kk_functions import get_device, kk_ImageClassifierTrainer
from kk_libraries.kk_dataprocess import kk_load_data
from kk_libraries.kk_constants import text_labels_fashion_mnist


class KnowledgeModule(nn.Module):
    """知识增强模块"""
    def __init__(self, in_channels, reduction=16):
        super(KnowledgeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LeNet_5_KAN(nn.Module):
    """LeNet-5的KAN版本"""
    def __init__(self, in_channels, num_classes):
        super(LeNet_5_KAN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),  # 将Sigmoid改为ReLU以提高性能
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.k1 = KnowledgeModule(6)  # 第一个知识增强模块
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.k2 = KnowledgeModule(16)  # 第二个知识增强模块
        
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(inplace=True)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.k1(x)  # 应用第一个知识增强
        x = self.conv2(x)
        x = self.k2(x)  # 应用第二个知识增强
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x
    
    
class Config(object):
    def __init__(self):
        self.num_epochs = 200
        self.lr = 0.1
        self.patience = 500
        self.batch_size = 512
        self.device = get_device()
        self.save_path = os.path.join(parent_dir, "models", "LeNet5_KAN")
        self.logs_path = os.path.join(parent_dir, "logs", "LeNet5_KAN")
        self.plot_titles = "LeNet5_KAN"
        self.class_list = text_labels_fashion_mnist
        self.dataset_name = "FashionMNIST"
        
    def __call__(self):
        return self.num_epochs, self.lr, self.patience, self.batch_size, self.device, self.save_path, self.logs_path, self.plot_titles, self.class_list
    

def kk_data_transform():
    return {
        'train': transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))]),
        'valid': transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    }

if __name__ == "__main__":
    config = Config()
    
    # 数据加载
    train_loader, test_loader = kk_load_data(os.path.join(parent_dir, "data", "FashionMNIST"), batch_size=config.batch_size, DataSets=FashionMNIST, transform=kk_data_transform())
    
    # 模型定义
    model = LeNet_5_KAN(in_channels=1, num_classes=10)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.35, patience=70, min_lr=1e-6)
    
    # 模型训练  
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler)
    trainer.train_iter(train_loader, test_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 模型测试
    trainer.test(test_loader)