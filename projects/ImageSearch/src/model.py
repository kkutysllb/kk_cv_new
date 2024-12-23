#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-12-08 21:15
# @Desc   : 采用ResNet50模型迁移学习训练猫狗分类模型
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from projects.ImageSearch.src.dataset import kk_Dataset
from kk_libraries.kk_functions import kk_ImageClassifierTrainer, get_device
from kk_libraries.kk_dataprocess import kk_get_data_mean_stdv2, kk_loader_train, kk_loader_test


# 模型定义
class DogCatModel(nn.Module):
    """构建猫狗分类迁移学习模型"""
    def __init__(self, num_classes=2):
        super(DogCatModel, self).__init__()
        model = models.resnet50(pretrained=True)
        for param in model.parameters(): # 冻结模型参数，不参与训练
            param.requires_grad = False
        model.fc.in_features = 2048 # 修改全连接层输入特征数为1024， 原始为2048
        model.fc = nn.Linear(model.fc.in_features, out_features=512) # 修改全连接层输出特征数为512
        model.activation = nn.ReLU(inplace=True)
        model.out = nn.Linear(in_features=512, out_features=num_classes) # 修改全连接层输出特征数为2， 即猫狗二分类
        self.kk_resnet50 = model
        
    def forward(self, x):
        x = self.kk_resnet50(x)
        return x
    
class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = "DogCatModel"
        self.num_epochs = 500
        self.batch_size = 256
        self.lr = 0.001
        self.device = "cuda:2"
        self.patience = 1000
        self.save_path = os.path.join(root_dir, "models", "DogCatModel")
        self.logs_path = os.path.join(root_dir, "logs", "DogCatModel")
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        self.dataset_name = "DogCat"
        self.class_list = ["cat", "dog"]
        self.plot_titles = self.model_name
        self.in_channels = 3
        self.num_classes = 2
    

        
def kk_train_data_transform():
    """数据预处理"""
    mean, _, std = kk_get_data_mean_stdv2(os.path.join(root_dir, "data", "dogcat"))
    return transforms.Compose([transforms.RandomResizedCrop(256),
                             transforms.CenterCrop(224),
                             transforms.RandomHorizontalFlip(0.5),
                             transforms.RandomRotation(15),
                             transforms.ToTensor(), 
                             transforms.Normalize(mean, std)])
    
def kk_test_data_transform():
    """数据预处理"""
    mean, _, std = kk_get_data_mean_stdv2(os.path.join(root_dir, "data", "dogcat"))
    return transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(), 
                             transforms.Normalize(mean, std)])


if __name__ == "__main__":
    config = Config()

    # 数据加载
    train_loader, valid_loader = kk_loader_train(os.path.join(root_dir, "data", "dogcat", "train"), config.batch_size, transform=kk_train_data_transform())
    test_loader = kk_loader_test(os.path.join(root_dir, "data", "dogcat", "test"), config.batch_size, transform=kk_test_data_transform())

    # 模型实例化
    model = DogCatModel(num_classes=config.num_classes)

    # 损失函数、优化器，调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=3e-6)

    # 模型训练
    trainer = kk_ImageClassifierTrainer(config, model, criterion, optimizer, scheduler)
    trainer.train_iter(train_loader, valid_loader)
    trainer.plot_training_curves(xaixs=range(1, len(trainer.train_losses) + 1))

    # 模型测试
    trainer.test(test_loader)

