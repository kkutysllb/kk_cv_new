#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-02 17:33
# @Desc   : GoogLeNet_v3 模型 图像分类
# 这个实现是GoogLeNet v3的简化版本，适合CIFAR10这样的小型数据集。
# 原始的GoogLeNet v3更大更复杂，包含更多的Inception模块和辅助分类器。
# 如果需要处理更大的图像（如ImageNet），可以增加更多的Inception模块和调整网络结构。
# --------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from kk_libraries.kk_functions import get_device, kk_ImageClassifierTrainer
from kk_libraries.kk_dataprocess import kk_load_data
from kk_libraries.kk_constants import mean, std, text_labels_cifar10

# 定义Inception3模块
class Inception3(nn.Module):
    """Inception3 模块"""
    def __init__(self, in_channels):
        super(Inception3, self).__init__()
        
        # 分支1: 1x1卷积
        self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        
        # 分支2: 1x1卷积 -> 5x5卷积
        self.branch2_1 = nn.Conv2d(in_channels, 48, kernel_size=1)
        self.branch2_2 = nn.Conv2d(48, 64, kernel_size=5, padding=2)
        
        # 分支3: 1x1卷积 -> 3x3卷积 -> 3x3卷积
        self.branch3_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch3_3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        
        # 分支4: 平均池化 -> 1x1卷积
        self.branch4_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = nn.Conv2d(in_channels, 32, kernel_size=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2_1 = nn.BatchNorm2d(48)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.bn3_2 = nn.BatchNorm2d(96)
        self.bn3_3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(32)

    def forward(self, x):
        # 分支1
        branch1 = F.relu(self.bn1(self.branch1(x)))
        
        # 分支2
        branch2 = F.relu(self.bn2_1(self.branch2_1(x)))
        branch2 = F.relu(self.bn2_2(self.branch2_2(branch2)))
        
        # 分支3
        branch3 = F.relu(self.bn3_1(self.branch3_1(x)))
        branch3 = F.relu(self.bn3_2(self.branch3_2(branch3)))
        branch3 = F.relu(self.bn3_3(self.branch3_3(branch3)))
        
        # 分支4
        branch4 = self.branch4_1(x)
        branch4 = F.relu(self.bn4(self.branch4_2(branch4)))
        
        # 将所有分支在通道维度上连接
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class GoogLeNetV3(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(GoogLeNetV3, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 最大池化
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(80, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        
        # 最大池化
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception模块
        self.inception1 = Inception3(192)  # 输出通道: 64+64+96+32=256
        self.inception2 = Inception3(256)  # 输出通道: 64+64+96+32=256
        self.inception3 = Inception3(256)  # 输出通道: 64+64+96+32=256
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        # 卷积层
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        # Inception模块
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类器
        x = self.classifier(x)
        return x
    
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

# 定义配置类
class Config(object):
    """配置类"""
    def __init__(self):
        self.model_name = 'GoogLeNet_v3'
        self.num_epochs = 200
        self.in_channels = 3
        self.num_classes = 10
        self.batch_size = 256
        self.patience = 500
        self.lr = 0.001
        self.device = get_device()
        self.plot_titles = "GoogLeNet_v3"
        self.save_path = os.path.join(root_dir, 'models', self.model_name)
        self.logs_path = os.path.join(root_dir, 'logs', self.model_name)
        self.class_list = text_labels_cifar10
        self.dataset_name = "CIFAR10"

    def __call__(self):
        return self.model_name, self.cfg, self.in_channels, self.num_classes, self.batch_size, self.patience, self.lr, self.device, self.save_path, self.logs_path, self.plot_titles

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
        "train": transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

if __name__ == "__main__":
    config = Config()

    # 加载数据
    train_loader, valid_loader = kk_load_data(os.path.join(root_dir, 'data', 'CIFAR10'), config.batch_size, CIFAR10, kk_data_transform(), num_works=4)

    # 模型定义
    model = GoogLeNetV3(config.in_channels, config.num_classes)
    # 损失函数、优化器、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.55, patience=100, min_lr=1e-5)

    # 训练模型
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler, train_loader, valid_loader)
    trainer.train_iter(train_loader, valid_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))
    
    # 测试
    trainer.test(valid_loader)
