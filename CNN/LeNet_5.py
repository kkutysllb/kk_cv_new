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

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


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

if __name__ == "__main__":
    print(parent_dir)