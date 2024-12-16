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
from torchvision.datasets import CIFAR10, FashionMNIST
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from kk_libraries.kk_dataprocess import kk_load_data
from kk_libraries.kk_functions import kk_ImageClassifierTrainer, get_device
from kk_libraries.kk_constants import text_labels_cifar10, text_labels_fashion_mnist


mean = [0.5, ]
std = [0.5, ]
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
        self.output_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        
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
        x = self.features(x)
        x = torch.mean(x, dim=(2, 3))
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
    
class Config(object):
    """配置类"""
    def __init__(self):
        self.vgg_name = 'VGG16'
        self.cfg = vgg_cfg
        self.num_epochs = 500
        self.in_channels = 1
        self.num_classes = 10
        self.batch_size = 256
        self.patience = 60
        self.lr = 0.001
        self.device = "cuda:0"
        self.plot_titles = self.vgg_name
        self.save_path = os.path.join(parent_dir, 'models', self.vgg_name)
        self.logs_path = os.path.join(parent_dir, 'logs', self.vgg_name)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        self.class_list = text_labels_fashion_mnist
        self.dataset_name = "FashionMNIST"
    
    def __call__(self):
        return self.vgg_name, self.cfg, self.in_channels, self.num_classes, self.batch_size, \
        self.patience, self.lr, self.device, self.save_path, self.logs_path, self.plot_titles, \
        self.class_list, self.dataset_name
    
def kk_data_transform():
    """数据变换"""
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
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
    train_loader, valid_loader = kk_load_data(os.path.join(parent_dir, 'data', 'FashionMNIST'), config.batch_size, FashionMNIST, kk_data_transform(), num_works=4)

    # 模型实例
    
    model = VGG(vgg_name=config.vgg_name, cfg=config.cfg, in_channels=config.in_channels, num_classes=config.num_classes)

    # 损失函数，优化器，学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, min_lr=3e-6)

    # 训练模型
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler)
    trainer.train_iter(train_loader, valid_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 测试
    trainer.test(valid_loader)
