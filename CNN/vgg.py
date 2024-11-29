#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-11-29 15:11
# @Desc   : vgg模型图像分类
# --------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from kk_libraries.kk_dataprocess import kk_load_data
from kk_libraries.kk_functions import kk_ImageClassifierTrainer, get_device
from kk_libraries.kk_constants import text_labels_cifar10, mean, std


vgg_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 模型定义
class VGG(nn.Module):
    """带BN层的VGG模型"""
    def __init__(self, vgg_name, cfg, in_channels=3, num_classes=10):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.output_layer = nn.Linear(512 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.output_layer(x)
        return x
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
    
class Config:
    """配置类"""
    def __init__(self):
        self.vgg_name = 'VGG16'
        self.cfg = vgg_cfg
        self.num_epochs = 50
        self.in_channels = 3
        self.num_classes = 10
        self.batch_size = 512
        self.patience = 500
        self.lr = 0.001
        self.device = get_device()
        self.save_path = os.path.join(parent_dir, 'models', self.vgg_name)
        self.logs_path = os.path.join(parent_dir, 'logs', self.vgg_name)
    
    def __call__(self):
        return self.vgg_name, self.cfg, self.in_channels, self.num_classes
    
def kk_data_transform():
    """数据变换"""
    return {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
   

if __name__ == "__main__":
    # 数据加载
    train_loader, valid_loader = kk_load_data(os.path.join(parent_dir, 'data', 'CIFAR10'), Config.batch_size, CIFAR10, kk_data_transform(), num_works=4)

    # 模型实例
    config = Config()
    model = VGG(vgg_name=config.vgg_name, cfg=config.cfg, in_channels=config.in_channels, num_classes=config.num_classes)

    # 损失函数，优化器，学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.5)

    # 训练模型
    trainer = kk_ImageClassifierTrainer(model, criterion, optimizer, scheduler, config.device, config.save_path, config.logs_path)
    trainer.train(train_loader, valid_loader)
    trainer.plot_training_curves(xaxis=range(1, len(trainer.train_losses) + 1))

    # 测试
    trainer.test(valid_loader)