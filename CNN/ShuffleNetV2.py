#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-03 12:19
# @Desc   : ShuffleNetV2模型图像分类 
#
# ShuffleNetV2的四个模型变体：
# 0.5x: [4, 8, 4], channels=48
# 1.0x: [4, 8, 4], channels=116
# 1.5x: [4, 8, 4], channels=176
# 2.0x: [4, 8, 4], channels=244
# --------------------------------------------------------
"""
import torch
import torch.nn as nn
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


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    
    # 重塑张量以进行通道混洗
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    
    return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        
        branch_features = out_channels // 2
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                # dw
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                # pw
                nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            # pw
            nn.Conv2d(in_channels if stride > 1 else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            # pw
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)  # 通道分割
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)  # 通道混洗
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024], num_classes=10):
        super(ShuffleNetV2, self).__init__()
        
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        # 初始卷积层
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建网络阶段
        self.stages = nn.ModuleList([])
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for i, (repeats, output_channels) in enumerate(zip(stages_repeats, self._stage_out_channels[1:-1])):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            self.stages.append(nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channels, num_classes)

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
        x = self.conv1(x)
        x = self.maxpool(x)
        
        for stage in self.stages:
            x = stage(x)
            
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class Config(object):
    def __init__(self): 
        self.model_name = 'ShuffleNetV2_1x'
        self.num_epochs = 500
        self.lr = 0.001
        self.batch_size = 128
        self.device = "cuda:7"
        self.in_channels = 3
        self.num_classes = 10
        self.patience = 300
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
    train_loader, test_loader = kk_load_data(os.path.join(root_dir, 'data', 'CIFAR10'), config.batch_size, CIFAR10, kk_data_transform())

    # 模型初始化 (ShuffleNetV2 1.0x)
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                        stages_out_channels=[24, 116, 232, 464, 1024],
                        num_classes=config.num_classes)
    
    # 损失函数、优化器、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.55, patience=100, min_lr=1e-5)

    # 训练  
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler=None)
    trainer.train_iter(train_loader, test_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 测试
    trainer.test(test_loader) 