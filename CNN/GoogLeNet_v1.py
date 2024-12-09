#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-02 14:36
# @Desc   : GoogLeNet_v1 模型图像分类
# --------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from kk_libraries.kk_functions import get_device, kk_ImageClassifierTrainer
from kk_libraries.kk_dataprocess import kk_load_data
from kk_libraries.kk_constants import text_labels_cifar10, text_labels_fashion_mnist

mean=[0.5,]
std=[0.5,]


# 定义inception块
class Inception(nn.Module):
    """Inception 块, c1, c2, c3, c4 为每条路径的输出通道数"""
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 路径1, 单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 路径2, 1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 路径3, 1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 路径4, 3x3最大池化层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1) # 【N, C, H, W】的数据结构，在通道维度上进行拼接
    
# 定义全局平均池化
class GlobalAvgPool2d(nn.Module):
    """全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现"""
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
    
# 定义展平层
class FlattenLayer(nn.Module):
    """展平层用于将多维张量展平成一维张量"""
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1) # 【N, C, H, W】 -> 【N, C*H*W】
    
# 定义GoogLeNet模型
class GoogLeNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GoogLeNet, self).__init__()
        # 第一模块 output_shape = (N, 64, 112, 112)
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 第二模块 output_shape = (N, 192, 56, 56)
        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 第三模块 output_shape = (N, 480, 28, 28)
        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                Inception(256, 128, (128, 192), (32, 96), 64),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 第四模块 output_shape = (N, 832, 14, 14)
        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                Inception(512, 160, (112, 224), (24, 64), 64),
                                Inception(512, 128, (128, 256), (24, 64), 64),
                                Inception(512, 112, (144, 288), (32, 64), 64),
                                Inception(528, 256, (160, 320), (32, 128), 128),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 第五模块 output_shape = (N, 832, 7, 7)
        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                Inception(832, 384, (192, 384), (48, 128), 128),
                                GlobalAvgPool2d())
        # 第六模块 output_shape = (N, num_classes)
        self.output = nn.Sequential(FlattenLayer(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(1024, num_classes))
        
          # 权重初始化
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
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.output(x)
        return x

# 定义配置类
class Config(object):
    """配置类"""
    def __init__(self):
        self.vgg_name = 'GoogLeNet_v1'
        self.num_epochs = 500
        self.in_channels = 1
        self.num_classes = 10
        self.batch_size = 512
        self.patience = 500
        self.lr = 0.001
        self.device = "cuda:2"
        self.plot_titles = "GoogLeNet_v1"
        self.save_path = os.path.join(root_dir, 'models', self.vgg_name)
        self.logs_path = os.path.join(root_dir, 'logs', self.vgg_name)
        self.class_list = text_labels_fashion_mnist
        self.dataset_name = "FashionMNIST"
    
    def __call__(self):
        return self.vgg_name, self.cfg, self.in_channels, self.num_classes, self.batch_size, self.patience, self.lr, self.device, self.save_path, self.logs_path, self.plot_titles


def kk_data_transform():
    """数据预处理"""
    return  {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

if __name__ == "__main__":
    config = Config()

    # 数据加载
    train_loader, valid_loader = kk_load_data(os.path.join(root_dir, 'data', "FashionMNIST"), config.batch_size, FashionMNIST, kk_data_transform(), num_works=4)

    # 模型定义
    model = GoogLeNet(config.in_channels, config.num_classes)

    # 损失函数，优化器，学习率
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=3e-6)

    # 模型训练
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler)
    trainer.train_iter(train_loader, valid_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 模型测试
    trainer.test(valid_loader)

