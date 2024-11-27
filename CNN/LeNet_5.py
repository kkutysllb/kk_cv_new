#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-11-26 10:15
# @Desc   : LeNet-5网络手写数字识别
# --------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from kk_libraries.kk_functions import kk_animator_train_evaluate, get_device
from kk_libraries.kk_dataprocess import kk_load_data, kk_predict_gray_labels
from kk_libraries.kk_constants import text_labels_mnist


class LeNet_5(nn.Module):
    """LeNet-5网络"""
    def __init__(self, in_channels, num_classes):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Sigmoid()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid()
        )
        self.out = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


# 定义数据预处理方法
def kk_data_transform():
    return {
        'train': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        'valid': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        }


if __name__ == "__main__":
    # 数据加载
    data_path = os.path.join(parent_dir, "data/MNIST")
    train_loader, valid_loader, test_loader = kk_load_data(data_path, ratio=0.9, batch_size=64, DataSets=MNIST, transform=kk_data_transform())
    
    # 模型构建
    model = LeNet_5(in_channels=1, num_classes=10)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 超参数
    epochs = 100
    device = get_device()
    
    # 训练
    kk_animator_train_evaluate(model, train_loader, valid_loader, criterion, optimizer, epochs, device)
    
    # 测试
    kk_predict_gray_labels(model, test_loader, text_labels_mnist, device)
    