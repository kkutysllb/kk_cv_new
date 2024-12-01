#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-11-29 15:15
# @Desc   : AlexNet_KAN 图像分类
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


class KnowledgeAttention(nn.Module):
    def __init__(self, in_channels):
        super(KnowledgeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AlexNet_KAN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(AlexNet_KAN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ka1 = KnowledgeAttention(96)

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ka2 = KnowledgeAttention(256)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ka3 = KnowledgeAttention(384)

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ka4 = KnowledgeAttention(384)

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ka5 = KnowledgeAttention(256)

        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 4096),
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
        x = self.ka1(x)
        
        x = self.conv2(x)
        x = self.ka2(x)
        
        x = self.conv3(x)
        x = self.ka3(x)
        
        x = self.conv4(x)
        x = self.ka4(x)
        
        x = self.conv5(x)
        x = self.ka5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 定义数据预处理
def kk_data_transform():
    return {
        'train': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]),
        'valid': transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(mean, std)])
    }


class Config(object):
    def __init__(self):
        self.num_epochs = 200
        self.lr = 0.001
        self.device = get_device()
        self.patience = 500
        self.save_path = os.path.join(parent_dir, "models", "AlexNet_KAN")
        self.logs_path = os.path.join(parent_dir, "logs", "AlexNet_KAN")
        self.plot_titles = "AlexNet_KAN"
        self.class_list = text_labels_cifar10
        self.dataset_name = "CIFAR10"

    def __call__(self):
        return self.num_epochs, self.lr, self.device, self.patience, self.save_path, self.logs_path, self.plot_titles, self.class_list, self.dataset_name

if __name__ == "__main__":
    # 数据加载
    data_path = os.path.join(parent_dir, "data/CIFAR10")
    train_loader,test_loader = kk_load_data(data_path, batch_size=512, DataSets=CIFAR10, transform=kk_data_transform())

    config = Config()
    # 模型定义
    model = AlexNet_KAN(in_channels=3, num_classes=10)
   
   # 定义损失函数和优化器C
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.35, patience=70, min_lr=1e-6)

    # 模型训练
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler=scheduler)
    trainer.train_iter(train_loader, test_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 模型测试
    trainer.test(test_loader)