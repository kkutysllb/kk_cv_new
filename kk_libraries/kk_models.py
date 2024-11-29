#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
# @Author kkutysllb


import torch
import torch.nn as nn
import torch.nn.functional as F


def kk_init_weights_relu(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def kk_init_weights_sigmoid(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class kk_MLP_Classification(nn.Module):
    """自定义多层感知机模型"""

    def __init__(self, num_inputs, num_outputs, n_hiddens1, n_hiddens2):
        super(kk_MLP_Classification, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(num_inputs, n_hiddens1),
                                    nn.BatchNorm1d(n_hiddens1))
        self.layer2 = nn.Sequential(nn.Linear(n_hiddens1, n_hiddens2),
                                    nn.BatchNorm1d(n_hiddens2))
        self.out = nn.Linear(n_hiddens2, num_outputs)

        # 权重参数初始化
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weigt, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.out(x)
        return x


def kk_softmax(x):
    """定义softmax函数"""
    x_exp = torch.exp(x)
    partition = torch.sum(x_exp, dim=1, keepdim=True)
    return x_exp / partition


def kk_softmax_classifier(num_inputs, num_outputs):
    """定义softmax分类器"""
    net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))

    def init_weights(m):
        """定义初始化权重的方法"""
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.01)

    net.apply(init_weights)
    return net


def kk_linear_regression(in_features):
    """构建模型并返回"""
    return nn.Sequential(
        nn.Linear(in_features, 1)
    )


def kk_conv_models():
    """构建自定义卷积网络模型"""
    layer1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5),
        nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
    )
    layer2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3),
        nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
    )
    layer3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3),
        nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
    )
    layer4 = nn.Sequential(
        nn.Flatten(), nn.Linear(in_features=128 * 2 * 2, out_features=128), nn.ReLU()
    )
    layer5 = nn.Sequential(nn.Linear(in_features=128, out_features=10))
    layer6 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_features=128, out_features=10))

    net1 = nn.Sequential(layer1, layer2, layer3, layer4, layer5)
    net2 = nn.Sequential(layer1, layer2, layer3, layer6)
    return net1, net2


class kk_CNNNet(nn.Module):
    """自定义CNN网络"""

    def __init__(self, in_channels, out_classes):
        super(kk_CNNNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=5),
                                    nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3),
                                    nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3),
                                    nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(in_features=128 * 2 * 2, out_features=256), nn.ReLU())
        self.fc2 = nn.Linear(in_features=256, out_features=out_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class kk_CNNNet_GAP(nn.Module):
    """自定义带全局平均池化的CNN网络"""

    def __init__(self, in_channels, out_classes):
        super(kk_CNNNet_GAP, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=5),
                                    nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3),
                                    nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3),
                                    nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(),
                                    nn.Linear(in_features=128, out_features=out_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class kk_LeNet5(nn.Module):
    """复现LeNet5模型"""

    def __init__(self, in_channels, num_classes):
        super(kk_LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_features=16 * 5 * 5, out_features=120),
                                 nn.Sigmoid()
                                 )
        self.fc2 = nn.Sequential(nn.Linear(in_features=120, out_features=84),
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


class kk_AlexNet(nn.Module):
    """复现AlexNet网络"""

    def __init__(self, in_channels, num_classes):
        super(kk_AlexNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer4 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(in_features=256 * 6 * 6, out_features=512), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(in_features=512, out_features=256), nn.ReLU())
        self.out = nn.Linear(in_features=256, out_features=num_classes)

        # 权重参数初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.out(x)
        return x


vgg11_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
vgg13_arch = ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512))
vgg16_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
vgg19_arch = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))

vgg_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class kk_VGG(nn.Module):
    """复现VGG16经典网络"""

    def __init__(self, vgg_name, cfg, num_classes):
        super(kk_VGG, self).__init__()
        self.features = self._mark_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.outlayer = nn.Linear(512 * 1 * 1, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.outlayer(x)
        return x

    def _mark_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def kk_vgg_block(num_convs, in_channels, out_channels):
    """定义VGG块, 包括卷积层数，输入通道数和输出通道数三个参数"""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def kk_vgg_net(conv_arch, in_channels, out_classes):
    """定义VGG网络"""
    conv_blks = []

    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(kk_vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 512), nn.ReLU(),
        nn.Linear(512, out_classes)
    )


def kk_small_vgg_net(arch, input_channels):
    """
    简化的vgg网络
    :param arch: 网络架构参数，一个序列数据，每个元素是一个元组: (卷积层个数, 输出通道数)
    :param input_channels: 初始图片数据通道数
    :return: vgg_net
    """
    blks = []
    for (num_convs, out_channels) in arch:
        blks.append(kk_vgg_block(num_convs, input_channels, out_channels))
        input_channels = out_channels

    return nn.Sequential(
        *blks, nn.Flatten(),
        nn.Linear(512, 10)
    )


class kk_Inception(nn.Module):
    """GoogLeNet的Inception块定义， c1---c4是每条路径上的输出通道"""

    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(kk_Inception, self).__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3,1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大池化层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2((self.p4_1(x))))
        # 在通道维度上进行合并输出
        return torch.cat([p1, p2, p3, p4], dim=1)


class kk_GoogLeNet(nn.Module):
    """复现GoogLeNet网络"""

    def __init__(self, in_channels, num_classes):
        super(kk_GoogLeNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(192),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer3 = nn.Sequential(kk_Inception(192, 64, (96, 128), (16, 32), 32),
                                    kk_Inception(256, 128, (128, 192), (32, 96), 64),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer4 = nn.Sequential(kk_Inception(480, 192, (96, 208), (16, 48), 64),
                                    kk_Inception(512, 160, (112, 224), (24, 64), 64),
                                    kk_Inception(512, 128, (128, 256), (24, 64), 64),
                                    kk_Inception(512, 112, (128, 288), (32, 64), 64),
                                    kk_Inception(528, 256, (160, 320), (32, 128), 128),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer5 = nn.Sequential(kk_Inception(832, 256, (160, 320), (32, 128), 128),
                                    kk_Inception(832, 384, (192, 384), (48, 128), 128),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        x = self.out(x)
        return x


class kk_Residual(nn.Module):
    """定义残差块"""

    def __init__(self, in_channels, num_channels, use_1x1conv=False, strides=1):
        super(kk_Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y += x
        return F.relu(y)


def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    """定义残差网络结构"""
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(kk_Residual(in_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(kk_Residual(num_channels, num_channels))
    return blk


def kk_resnet18(in_channels, num_classes):
    """复现resnet18网络"""
    bn1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    bn2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    bn3 = nn.Sequential(*resnet_block(64, 128, 2))
    bn4 = nn.Sequential(*resnet_block(128, 256, 2))
    bn5 = nn.Sequential(*resnet_block(256, 512, 2))
    return nn.Sequential(bn1, bn2, bn3, bn4, bn5,
                         nn.AdaptiveAvgPool2d((1, 1)),
                         nn.Flatten(), nn.Linear(512, num_classes))


