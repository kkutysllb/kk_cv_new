#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-10-14 21:15
# @Desc   : VGG_KAN 模型 图像分类
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
class VGG_KAN(nn.Module):
    """Knowledge Augmented Network based on VGG architecture"""
    def __init__(self, vgg_name, cfg, in_channels=3, num_classes=10):
        super(VGG_KAN, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(cfg[vgg_name])
        
        # Knowledge Enhancement Module
        self.knowledge_module = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Attention Module
        self.attention = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
        
        self._initialize_weights()

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Knowledge enhancement
        k = self.knowledge_module(x)
        
        # Attention mechanism
        att = self.attention(k)
        x = x * att + k
        
        # Global average pooling
        x = torch.mean(x, dim=(2, 3))
        
        # Classification
        x = self.output_layer(x)
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
    
class Config(object):
    """配置类"""
    def __init__(self):
        self.vgg_name = 'VGG16'
        self.cfg = vgg_cfg
        self.num_epochs = 200
        self.in_channels = 3
        self.num_classes = 10
        self.batch_size = 512
        self.patience = None
        self.lr = 0.001
        self.device = get_device()
        self.plot_titles = "VGG16_KAN"
        self.save_path = os.path.join(parent_dir, 'models', self.plot_titles)
        self.logs_path = os.path.join(parent_dir, 'logs', self.plot_titles)
        self.class_list = text_labels_cifar10
        self.dataset_name = "CIFAR10"
    def __call__(self):
        return self.vgg_name, self.cfg, self.in_channels, self.num_classes, self.batch_size, self.patience, self.lr, self.device, self.save_path, self.logs_path, self.plot_titles
    
def kk_data_transform():
    """数据变换"""
    return {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
   

if __name__ == "__main__":
    config = Config()
    # 数据加载
    train_loader, valid_loader = kk_load_data(os.path.join(parent_dir, 'data', 'CIFAR10'), config.batch_size, CIFAR10, kk_data_transform(), num_works=4)

    # 模型实例
    
    model = VGG_KAN(vgg_name=config.vgg_name, cfg=config.cfg, in_channels=config.in_channels, num_classes=config.num_classes)

    # 损失函数，优化器，学习率调度器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-5)

    # 训练模型
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler=scheduler)
    trainer.train_iter(train_loader, valid_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 测试
    trainer.test(valid_loader)