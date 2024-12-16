#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-02 15:25
# @Desc   : GoogLeNet_v2 模型图像分类
# --------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from kk_libraries.kk_functions import get_device, kk_ImageClassifierTrainer
from kk_libraries.kk_dataprocess import kk_load_data
from kk_libraries.kk_constants import mean, std, text_labels_cifar10, text_labels_fashion_mnist

mean=[0.5,]
std=[0.5,]

# Inception 模块
class Inception2(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        """c1, c2, c3, c4 为每条路径的输出通道数"""
        super(Inception2, self).__init__()
        # 第一条路径: 1x1 卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p1_bn = nn.BatchNorm2d(c1)
        
        # 第二条路径: 1x1 卷积 -> 3x3 卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_1_bn = nn.BatchNorm2d(c2[0])
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p2_2_bn = nn.BatchNorm2d(c2[1])
        
        # 第三条路径: 1x1 卷积 -> 两个3x3卷积 (替代5x5)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_1_bn = nn.BatchNorm2d(c3[0])
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1)
        self.p3_2_bn = nn.BatchNorm2d(c3[1])
        self.p3_3 = nn.Conv2d(c3[1], c3[2], kernel_size=3, padding=1)
        self.p3_3_bn = nn.BatchNorm2d(c3[2])
        
        # 第四条路径: 3x3最大池化 -> 1x1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        self.p4_2_bn = nn.BatchNorm2d(c4)

    def forward(self, x):
        # 路径1
        p1 = F.relu(self.p1_bn(self.p1_1(x)))
        
        # 路径2
        p2 = F.relu(self.p2_1_bn(self.p2_1(x)))
        p2 = F.relu(self.p2_2_bn(self.p2_2(p2)))
        
        # 路径3
        p3 = F.relu(self.p3_1_bn(self.p3_1(x)))
        p3 = F.relu(self.p3_2_bn(self.p3_2(p3)))
        p3 = F.relu(self.p3_3_bn(self.p3_3(p3)))
        
        # 路径4
        p4 = self.p4_1(x)
        p4 = F.relu(self.p4_2_bn(self.p4_2(p4)))
        
        # 拼接四条路径的输出
        return torch.cat((p1, p2, p3, p4), dim=1)

class GoogLeNetV2(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(GoogLeNetV2, self).__init__()
        # 第一阶段：标准卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 第二阶段：标准卷积层
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 第三阶段：2个Inception模块
        self.inception3a = Inception2(192, 64, [96, 128], [16, 32, 32], 32)
        self.inception3b = Inception2(256, 128, [128, 192], [32, 96, 96], 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 第四阶段：5个Inception模块
        self.inception4a = Inception2(480, 192, [96, 208], [16, 48, 64], 64)
        self.inception4b = Inception2(528, 160, [112, 224], [24, 64, 64], 64)
        self.inception4c = Inception2(512, 128, [128, 256], [24, 64, 64], 64)
        self.inception4d = Inception2(512, 112, [144, 288], [32, 64, 64], 64)
        self.inception4e = Inception2(528, 256, [160, 320], [32, 128, 128], 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 第五阶段：2个Inception模块
        self.inception5a = Inception2(832, 256, [160, 320], [32, 128, 128], 128)
        self.inception5b = Inception2(832, 384, [192, 384], [48, 128, 128], 128)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout和分类器
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
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
        # 第一阶段
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # 第二阶段
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # 第三阶段
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        # 第四阶段
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        # 第五阶段
        x = self.inception5a(x)
        x = self.inception5b(x)
        # 分类器
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# 配置类
class Config(object):
    def __init__(self):
        self.model_name = 'GoogLeNet_v2'
        self.num_epochs = 500
        self.in_channels = 1
        self.num_classes = 10
        self.batch_size = 256
        self.patience = 30
        self.lr = 0.001
        self.device = "cuda:1"
        self.plot_titles = "GoogLeNet_v2"
        self.save_path = os.path.join(root_dir, 'models', self.model_name)
        self.logs_path = os.path.join(root_dir, 'logs', self.model_name)
        self.class_list = text_labels_fashion_mnist
        self.dataset_name = "FashionMNIST"

    def __call__(self):
        return self.model_name, self.cfg, self.in_channels, self.num_classes, \
                self.batch_size, self.patience, self.lr, self.device, self.save_path, \
                self.logs_path, self.plot_titles


class LabelSmoothingCrossEntropy(nn.Module):
    """googlenet v2 标签平滑交叉熵损失实现"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        # 将真实标签转换为one-hot编码
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        # 应用标签平滑
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        loss = (-smooth_one_hot * F.log_softmax(pred, dim=1)).sum(dim=1).mean()
        return loss


def kk_data_transform():
    """数据预处理"""
    return  {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.CenterCrop(224),
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
    model = GoogLeNetV2(config.in_channels, config.num_classes)

    # 损失函数，优化器，学习率
    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=16, min_lr=3e-6) 

    # 训练
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler)
    trainer.train_iter(train_loader, valid_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 测试
    trainer.test(valid_loader)
