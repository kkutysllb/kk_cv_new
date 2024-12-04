#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-03 12:19
# @Desc   : DenseNet模型图像分类 
#
# 创建DenseNet-121
# model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)
#
# 创建DenseNet-169
# model = DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)
#
# 创建DenseNet-201
# model = DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)
#
# 创建DenseNet-264
# model = DenseNet(Bottleneck, [6,12,64,48], growth_rate=32)
# --------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from kk_libraries.kk_functions import get_device, kk_ImageClassifierTrainer
from kk_libraries.kk_dataprocess import kk_load_data
from kk_libraries.kk_constants import text_labels_cifar10, mean, std


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inner_channels = 4 * growth_rate  # BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner_channels)
        self.conv2 = nn.Conv2d(inner_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.avg_pool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, num_blocks, growth_rate=32, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        # 第一个卷积层
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks
        self.dense1 = self._make_dense_block(block, num_channels, num_blocks[0])
        num_channels += num_blocks[0] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans1 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense2 = self._make_dense_block(block, num_channels, num_blocks[1])
        num_channels += num_blocks[1] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans2 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense3 = self._make_dense_block(block, num_channels, num_blocks[2])
        num_channels += num_blocks[2] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans3 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense4 = self._make_dense_block(block, num_channels, num_blocks[3])
        num_channels += num_blocks[3] * growth_rate

        # 最后的BN和分类器
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_dense_block(self, block, in_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(in_channels + i * self.growth_rate, self.growth_rate))
        return nn.Sequential(*layers)

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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = self.trans3(out)
        out = self.dense4(out)

        out = self.bn_final(out)
        out = self.relu_final(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class Config(object):
    def __init__(self): 
        self.model_name = 'DenseNet_121'
        self.num_epochs = 200
        self.lr = 0.001
        self.batch_size = 512
        self.device = get_device()
        self.in_channels = 3
        self.num_classes = 10
        self.patience = None
        self.save_path = os.path.join(root_dir, 'models', self.model_name)
        self.logs_path = os.path.join(root_dir, 'logs', self.model_name)
        self.plot_titles = self.model_name
        self.class_list = text_labels_cifar10
        self.dataset_name = "CIFAR10"

    def __call__(self):
        return self.model_name, self.cfg, self.in_channels, self.num_classes, self.batch_size, \
               self.patience, self.lr, self.device, self.save_path, self.logs_path, self.plot_titles, \
               self.class_list, self.dataset_name


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
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }


if __name__ == "__main__":
    config = Config()

    # 数据加载
    train_loader, test_loader = kk_load_data(os.path.join(root_dir, 'data', 'CIFAR10'), config.batch_size, CIFAR10, kk_data_transform())

    # 模型初始化 (DenseNet-121)
    model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)
    
    # 损失函数、优化器、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.55, patience=100, min_lr=1e-5)

    # 训练  
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler)
    trainer.train_model(train_loader, test_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 测试
    trainer.test(test_loader) 